import sys
import os
import re
from snake_game import *
from DQN import DQNAgent
from PPO import PPOAgent
import matplotlib.pyplot as plt
from itertools import count
import numpy as np
import torch


Should_Render = False #Set to True to render the game, False is for headless training



def list_models(directory):
    return [f for f in os.listdir(directory) if f.endswith('.pth')]

def extract_episode_number(filename):
    match = re.search(r"model_ep(\d+)_", filename)
    if match:
        return int(match.group(1))
    return None

def main():
    state_dim = 19  # Number of features in the state representation
    action_dim = 4  # Number of possible actions (left, right, up, down)

    agent = PPOAgent(state_dim, action_dim)

    model_directory = "./models"

    # Ensure the model directory exists
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    episode = 0

    # Terminal menu
    setup_complete = False
    while not setup_complete:
        print("Select an option:")
        print("1. Start a new game")
        print("2. Load a saved model")
        choice = input("Enter 1 or 2: ")

        if choice == "1":
            model_file = None
            setup_complete = True
        elif choice == "2":
            models = list_models(model_directory)
            if not models:
                print("No saved models found. Starting a new game.")
                model_file = None
                setup_complete = True
            else:
                print("Available models:")
                for i, model in enumerate(models):
                    print(f"{i + 1}. {model}")
                model_choice = int(input("Enter the number of the model to load: ")) - 1
                if 0 <= model_choice < len(models):
                    model_file = os.path.join(model_directory, models[model_choice])
                    episode = extract_episode_number(models[model_choice])
                    setup_complete = True
                else:
                    print("Invalid choice. Please enter a number corresponding to a model.")
        else:
            print("Invalid choice. Please enter 1 or 2.")

    game_loop(agent, model_file, episode)


def game_loop(agent, model_file=None, start_episode=0, render=Should_Render):

    # ✅ Detect and set device (GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")

    # Move the agent's model to the detected device
    agent.model.to(device)
    agent.device = device

    if model_file:
        agent.load_model(model_file)


    episode = start_episode
    best_reward = -float('inf')  # ✅ Initialize to very low reward

    seed = 42  # Initial random seed

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    line1, = ax1.plot([], [], lw=2, label='Total Food Eaten')
    avg_line1, = ax1.plot([], [], lw=2, linestyle='--', label='Average Food Eaten')
    line2, = ax2.plot([], [], lw=2, label='Total Reward')
    avg_line2, = ax2.plot([], [], lw=2, linestyle='--', label='Average Reward')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Food Eaten')
    ax1.legend()
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Total Reward')
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.legend()

    xdata, food_data, reward_data = [], [], []
    food_avg_data, reward_avg_data = [], []

    def update_plot(x, food, reward):
        window_size = 1000  # Rolling window size

        xdata.append(x)
        food_data.append(food)
        reward_data.append(reward)

        # ✅ Drop old points if window is full
        if len(xdata) > window_size:
            xdata.pop(0)
            food_data.pop(0)
            reward_data.pop(0)
            food_avg_data.pop(0)
            reward_avg_data.pop(0)

        # ✅ Calculate rolling average over current window
        food_avg = np.mean(food_data)
        reward_avg = np.mean(reward_data)
        food_avg_data.append(food_avg)
        reward_avg_data.append(reward_avg)

        line1.set_data(xdata, food_data)
        avg_line1.set_data(xdata, food_avg_data)
        line2.set_data(xdata, reward_data)
        avg_line2.set_data(xdata, reward_avg_data)

        ax1.relim()
        ax1.autoscale_view()
        ax2.relim()
        ax2.autoscale_view()
        plt.draw()
        plt.pause(0.001)


    


    while True:
        game_close = False
        x1 = dis_width / 2
        y1 = dis_height / 2
        x1_change = 0
        y1_change = 0

        snake_List = []
        Length_of_snake = 1

        random.seed(seed)
        foodx, foody = respawn_food(snake_List)
        seed += 1

        total_reward = 0
        total_food_eaten = 0
        steps = 0
        total_msa = 0
        prev_distance = abs(x1 - foodx) + abs(y1 - foody)

        for t in count():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

            state, food_seen_flags = get_state(snake_List, Point(foodx, foody), Point(x1, y1), (x1_change, y1_change))

            action = agent.select_action(state)

            # Move snake
            if action == 0 and x1_change == 0:
                x1_change = -cell_size
                y1_change = 0
            elif action == 1 and x1_change == 0:
                x1_change = cell_size
                y1_change = 0
            elif action == 2 and y1_change == 0:
                y1_change = -cell_size
                x1_change = 0
            elif action == 3 and y1_change == 0:
                y1_change = cell_size
                x1_change = 0

            if x1 >= dis_width or x1 < 0 or y1 >= dis_height or y1 < 0:
                game_close = True

            x1 += x1_change
            y1 += y1_change
            # Always update snake logic
            snake_Head = [x1, y1]
            snake_List.append(snake_Head)
            if len(snake_List) > Length_of_snake:
                del snake_List[0]

            for x in snake_List[:-1]:
                if x == snake_Head:
                    game_close = True

            # Only render visuals if needed
            if render:
                dis.fill(blue)
                pygame.draw.rect(dis, green, [foodx, foody, cell_size, cell_size])
                our_snake(cell_size, snake_List)
                display_score(Length_of_snake - 1)
                pygame.display.update()


            # Calculate reward
            is_wall_collision = x1 >= dis_width or x1 < 0 or y1 >= dis_height or y1 < 0
            is_self_collision = any(x == snake_Head for x in snake_List[:-1])

            new_distance = abs(x1 - foodx) + abs(y1 - foody)
            # Calculate food_visible flag (1 if food seen in any raycast)
            food_visible = any(food_seen_flags)

            reward = calculate_reward(prev_distance, new_distance, snake_Head, Point(foodx, foody), game_close, is_wall_collision, is_self_collision, food_visible)

            total_reward += reward
            prev_distance = new_distance

            if snake_Head == [foodx, foody]:
                foodx, foody = respawn_food(snake_List)
                Length_of_snake += 1
                total_food_eaten += 1

            done = float(game_close)
            agent.store_transition(reward, done)
            steps += 1
            if render:
                clock.tick(snake_speed)

            if game_close:
                avg_msa = total_msa / steps if steps > 0 else 0
                print(f'Episode {episode} - Score: {Length_of_snake - 1}, Total Reward: {total_reward:.2f}, Steps: {steps}, Avg MSA: {avg_msa:.2f}, Food Eaten: {total_food_eaten}')
                
                # ✅ Optimize here after death
                agent.optimize_model()
                
                # ✅ Save model cleanly
                best_reward = agent.save_model(episode, total_reward, best_reward)
                
                # ✅ Update plot
                update_plot(episode, total_food_eaten, total_reward)
                
                episode += 1
                break


        print("Food location:", (foodx, foody))
        print("Snake body positions:", snake_List)

if __name__ == "__main__":
    main()
