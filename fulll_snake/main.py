import sys
import os
import re
from snake_game import *
from agent import DQNAgent, PPOAgent
import matplotlib.pyplot as plt
from itertools import count
import numpy as np
import torch
import torch.multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import time
import pygame

def list_models(directory):
    return [f for f in os.listdir(directory) if f.endswith('.pth')]

def extract_episode_number(filename):
    # Try different patterns to extract episode number
    patterns = [
        r"model_ep(\d+)_",  # Standard pattern
        r"agent\d+_gen\d+_ep(\d+)_",  # Agent generation pattern
        r"best_overall_gen\d+_ep(\d+)_",  # Best overall pattern
        r"best_final_ep(\d+)_"  # Best final pattern
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            return int(match.group(1))
    return 0  # Default to 0 if no episode number found

def train_snake(agent_id, num_episodes, device, population=None, generation=0):
    state_dim = 18
    action_dim = 4
    
    # Initialize agent with CUDA support
    if population is not None and len(population) > 0:
        # Select parents based on fitness
        parents = sorted(population, key=lambda x: x.fitness, reverse=True)[:2]
        if len(parents) == 2:
            # Create child through crossover
            agent = parents[0].crossover(parents[1])
            # Apply mutation
            agent.mutate(mutation_rate=0.1, mutation_scale=0.1)
        else:
            agent = parents[0].clone()
            agent.mutate(mutation_rate=0.2, mutation_scale=0.2)
    else:
        agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            gamma=0.99,
            lr=1e-4,
            clip_epsilon=0.2,
            k_epochs=8,
            device=device
        )
    
    episode = 0
    best_score = 0
    best_steps = 0
    best_food = 0
    no_improvement_count = 0
    max_no_improvement = 50
    last_reset_episode = 0
    min_episodes_between_reset = 20  # Minimum episodes before allowing another reset
    
    # Track top 4 performances
    top_performances = []  # List of tuples (score, steps, food, episode)
    
    while episode < num_episodes:
        game_over = False
        game_close = False
        
        x1 = dis_width / 2
        y1 = dis_height / 2
        x1_change = 0
        y1_change = 0
        snake_List = []
        Length_of_snake = 1
        
        seed = 42 + agent_id + (generation * 1000)  # Different seed for each agent and generation
        random.seed(seed)
        foodx, foody = respawn_food(snake_List)
        
        total_loss = 0
        steps = 0
        total_reward = 0
        total_food_eaten = 0
        
        prev_distance = abs(x1 - foodx) + abs(y1 - foody)
        
        while not game_close:
            state = get_state(snake_List, Point(foodx, foody), Point(x1, y1), (x1_change, y1_change))
            action = agent.select_action(state)
            
            if action == 0 and x1_change == 0:  # Move left
                x1_change = -cell_size
                y1_change = 0
            elif action == 1 and x1_change == 0:  # Move right
                x1_change = cell_size
                y1_change = 0
            elif action == 2 and y1_change == 0:  # Move up
                y1_change = -cell_size
                x1_change = 0
            elif action == 3 and y1_change == 0:  # Move down
                y1_change = cell_size
                x1_change = 0
                
            if x1 >= dis_width or x1 < 0 or y1 >= dis_height or y1 < 0:
                game_close = True
            x1 += x1_change
            y1 += y1_change
            
            snake_Head = [x1, y1]
            snake_List.append(snake_Head)
            if len(snake_List) > Length_of_snake:
                del snake_List[0]
                
            for x in snake_List[:-1]:
                if x == snake_Head:
                    game_close = True
                    
            is_wall_collision = x1 >= dis_width or x1 < 0 or y1 >= dis_height or y1 < 0
            is_self_collision = any(x == snake_Head for x in snake_List[:-1])
            
            new_distance = abs(x1 - foodx) + abs(y1 - foody)
            reward = calculate_reward(prev_distance, new_distance, snake_Head, Point(foodx, foody), game_close, is_wall_collision, is_self_collision, snake_List)
            total_reward += reward
            prev_distance = new_distance
            
            if snake_Head == [foodx, foody]:
                foodx, foody = respawn_food(snake_List)
                Length_of_snake += 1
                total_food_eaten += 1
                
            next_state = get_state(snake_List, Point(foodx, foody), Point(x1, y1), (x1_change, y1_change))
            done = float(game_close)
            
            agent.store_transition(reward, done)
            loss = agent.optimize_model()
            if loss is not None:
                total_loss += loss
            steps += 1
            
        score = Length_of_snake - 1
        agent.update_fitness(score, steps, total_food_eaten)
        print(f'Generation {generation} - Agent {agent_id} - Episode {episode} - Score: {score}, Total Loss: {total_loss:.2f}, Total Reward: {total_reward:.2f}, Steps: {steps}, Food Eaten: {total_food_eaten}, Fitness: {agent.fitness:.2f}')
        
        # Update best performances
        current_performance = (score, steps, total_food_eaten, episode)
        top_performances.append(current_performance)
        top_performances.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)  # Sort by score, then steps, then food
        top_performances = top_performances[:4]  # Keep only top 4
        
        # Save model if it's in top 4
        if current_performance in top_performances:
            # Delete old model files if we have more than 4
            model_files = [f for f in os.listdir("./models") if f.startswith(f"agent{agent_id}_gen{generation}")]
            if len(model_files) >= 4:
                # Sort by episode number and remove oldest
                model_files.sort(key=lambda x: extract_episode_number(x))
                os.remove(os.path.join("./models", model_files[0]))
            
            # Save new model
            agent.save_model(episode, f"agent{agent_id}_gen{generation}")
            print(f"Saved model for episode {episode} (Score: {score}, Steps: {steps}, Food: {total_food_eaten})")
        
        if score > best_score:
            best_score = score
            best_steps = steps
            best_food = total_food_eaten
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            
        # Instead of rebasing, reset the agent with fresh parameters if it's struggling
        if no_improvement_count >= max_no_improvement and (episode - last_reset_episode) >= min_episodes_between_reset:
            print(f"\nAgent {agent_id} resetting with fresh parameters")
            
            # Create a new agent with the same architecture but fresh parameters
            agent = PPOAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                gamma=0.99,
                lr=1e-4,
                clip_epsilon=0.2,
                k_epochs=8,
                device=device
            )
            
            no_improvement_count = 0
            last_reset_episode = episode
            print(f"Agent {agent_id} reset and continuing training")
            
        episode += 1
    
    return agent, best_score

def game_loop(agent, model_file, episode, headless=False):
    if model_file:
        agent.load_model(model_file)
    
    best_score = 0
    no_improvement_count = 0
    max_no_improvement = 50
    
    while True:
        game_over = False
        game_close = False
        
        x1 = dis_width / 2
        y1 = dis_height / 2
        x1_change = 0
        y1_change = 0
        snake_List = []
        Length_of_snake = 1
        
        foodx, foody = respawn_food(snake_List)
        
        total_loss = 0
        steps = 0
        total_reward = 0
        total_food_eaten = 0
        
        prev_distance = abs(x1 - foodx) + abs(y1 - foody)
        
        while not game_close:
            if not headless:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        quit()
            
            state = get_state(snake_List, Point(foodx, foody), Point(x1, y1), (x1_change, y1_change))
            action = agent.select_action(state)
            
            if action == 0 and x1_change == 0:  # Move left
                x1_change = -cell_size
                y1_change = 0
            elif action == 1 and x1_change == 0:  # Move right
                x1_change = cell_size
                y1_change = 0
            elif action == 2 and y1_change == 0:  # Move up
                y1_change = -cell_size
                x1_change = 0
            elif action == 3 and y1_change == 0:  # Move down
                y1_change = cell_size
                x1_change = 0
                
            if x1 >= dis_width or x1 < 0 or y1 >= dis_height or y1 < 0:
                game_close = True
            x1 += x1_change
            y1 += y1_change
            
            snake_Head = [x1, y1]
            snake_List.append(snake_Head)
            if len(snake_List) > Length_of_snake:
                del snake_List[0]
                
            for x in snake_List[:-1]:
                if x == snake_Head:
                    game_close = True
                    
            is_wall_collision = x1 >= dis_width or x1 < 0 or y1 >= dis_height or y1 < 0
            is_self_collision = any(x == snake_Head for x in snake_List[:-1])
            
            new_distance = abs(x1 - foodx) + abs(y1 - foody)
            reward = calculate_reward(prev_distance, new_distance, snake_Head, Point(foodx, foody), game_close, is_wall_collision, is_self_collision, snake_List)
            total_reward += reward
            prev_distance = new_distance
            
            if snake_Head == [foodx, foody]:
                foodx, foody = respawn_food(snake_List)
                Length_of_snake += 1
                total_food_eaten += 1
                
            next_state = get_state(snake_List, Point(foodx, foody), Point(x1, y1), (x1_change, y1_change))
            done = float(game_close)
            
            agent.store_transition(reward, done)
            loss = agent.optimize_model()
            if loss is not None:
                total_loss += loss
            steps += 1
            
            if not headless:
                dis.fill(black)
                pygame.draw.rect(dis, green, [foodx, foody, cell_size, cell_size])
                for x in snake_List:
                    pygame.draw.rect(dis, red, [x[0], x[1], cell_size, cell_size])
                pygame.display.update()
                clock.tick(30)
        
        score = Length_of_snake - 1
        print(f'Episode {episode} - Score: {score}, Total Loss: {total_loss:.2f}, Total Reward: {total_reward:.2f}, Steps: {steps}, Food Eaten: {total_food_eaten}')
        
        if score > best_score:
            best_score = score
            no_improvement_count = 0
            agent.save_model(episode)
        else:
            no_improvement_count += 1
            
        if no_improvement_count >= max_no_improvement:
            print(f"No improvement for {max_no_improvement} episodes. Stopping training.")
            break
            
        episode += 1

def main():
    state_dim = 18
    action_dim = 4
    
    # Check for CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Terminal menu
    setup_complete = False
    while not setup_complete:
        print("Select an option:")
        print("1. Start a new game")
        print("2. Load a saved model")
        print("3. Start headless training")
        print("4. Start parallel training with evolution")
        choice = input("Enter 1, 2, 3, or 4: ")
        
        if choice == "1":
            model_file = None
            headless = False
            setup_complete = True
            game_loop(PPOAgent(state_dim, action_dim, device=device), model_file, 0, headless)
        elif choice == "2":
            models = list_models("./models")
            if not models:
                print("No saved models found. Starting a new game.")
                model_file = None
                headless = False
                setup_complete = True
                game_loop(PPOAgent(state_dim, action_dim, device=device), model_file, 0, headless)
            else:
                print("Available models:")
                for i, model in enumerate(models):
                    print(f"{i + 1}. {model}")
                model_choice = int(input("Enter the number of the model to load: ")) - 1
                if 0 <= model_choice < len(models):
                    model_file = os.path.join("./models", models[model_choice])
                    episode = extract_episode_number(models[model_choice])
                    print(f"Starting from episode {episode}")
                    headless = False
                    setup_complete = True
                    game_loop(PPOAgent(state_dim, action_dim, device=device), model_file, episode, headless)
                else:
                    print("Invalid choice. Please enter a number corresponding to a model.")
        elif choice == "3":
            model_file = None
            headless = True
            setup_complete = True
            game_loop(PPOAgent(state_dim, action_dim, device=device), model_file, 0, headless)
        elif choice == "4":
            setup_complete = True
            num_agents = int(input("Enter number of parallel agents to train: "))
            num_episodes = int(input("Enter number of episodes per agent: "))
            num_generations = int(input("Enter number of generations: "))
            
            population = []
            best_overall_score = 0
            best_overall_agent = None
            
            for generation in range(num_generations):
                print(f"\nStarting Generation {generation + 1}/{num_generations}")
                
                # Create a pool of processes
                with ProcessPoolExecutor(max_workers=num_agents) as executor:
                    # Submit training tasks
                    futures = [executor.submit(train_snake, i, num_episodes, device, population, generation) 
                             for i in range(num_agents)]
                    
                    # Collect results
                    generation_results = []
                    for future in futures:
                        agent, best_score = future.result()
                        generation_results.append((agent, best_score))
                        if best_score > best_overall_score:
                            best_overall_score = best_score
                            best_overall_agent = agent
                            best_overall_agent.save_model(num_episodes, f"best_overall_gen{generation}")
                
                # Update population for next generation
                population = [agent for agent, _ in generation_results]
                
                # Print generation summary
                avg_fitness = sum(agent.fitness for agent in population) / len(population)
                max_fitness = max(agent.fitness for agent in population)
                print(f"\nGeneration {generation + 1} Summary:")
                print(f"Average Fitness: {avg_fitness:.2f}")
                print(f"Best Fitness: {max_fitness:.2f}")
                print(f"Best Overall Score: {best_overall_score}")
            
            print("\nTraining Complete!")
            print(f"Best Overall Score: {best_overall_score}")
            if best_overall_agent:
                best_overall_agent.save_model(num_episodes, "best_final")
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")

if __name__ == "__main__":
    # Set start method for multiprocessing
    mp.set_start_method('spawn')
    main()
