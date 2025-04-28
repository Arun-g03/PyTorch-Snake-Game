import sys
import os
import re
import csv
import numpy as np
from snake_game import *
from agent_tf import DQNAgent
from itertools import count

def list_models(directory):
    return [f for f in os.listdir(directory) if f.endswith('.weights.h5')]

def extract_episode_number(filename):
    match = re.search(r"model_ep(\d+)_", filename)
    if match:
        return int(match.group(1))
    return None

def main():
    state_dim = 15  # Number of features in the state representation
    action_dim = 4  # Number of possible actions (left, right, up, down)

    agent = DQNAgent(state_dim, action_dim)
    model_directory = "./models"
    history_file = "training_history.csv"

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

    game_loop(agent, history_file, model_file, episode)

def game_loop(agent, history_file, model_file=None, start_episode=0):
    if model_file:
        agent.load_model(model_file)

    epsilon = 0.9  # Exploration rate
    epsilon_decay = 0.995
    min_epsilon = 0.01
    beta_start = 0.4
    beta_frames = 10000

    episode = start_episode
    seed = 42  # You can change this seed value as needed

    if not os.path.exists(history_file):
        with open(history_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Episode", "Food Eaten", "Total Reward"])

    while True:
        game_over = False
        game_close = False

        x1 = dis_width / 2
        y1 = dis_height / 2

        x1_change = 0
        y1_change = 0

        snake_List = []
        Length_of_snake = 1

        random.seed(seed)
        foodx, foody = respawn_food(snake_List)
        seed += 1  # Increment the seed for the next episode

        total_loss = 0
        steps = 0
        total_reward = 0  # Track total reward per episode
        total_food_eaten = 0  # Track the total food eaten per episode

        prev_distance = abs(x1 - foodx) + abs(y1 - foody)

        for t in count():
            # Linearly anneal beta from beta_start to 1.0 over beta_frames frames
            beta = min(1.0, beta_start + t * (1.0 - beta_start) / beta_frames)

            # State representation: relative positions and wall indicators
            state = get_state(snake_List, Point(foodx, foody), Point(x1, y1), (x1_change, y1_change))
            action = agent.select_action(state, epsilon)
            
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

            # Determine if the snake died by hitting a wall or itself
            is_wall_collision = x1 >= dis_width or x1 < 0 or y1 >= dis_height or y1 < 0
            is_self_collision = any(x == snake_Head for x in snake_List[:-1])

            new_distance = abs(x1 - foodx) + abs(y1 - foody)
            reward = calculate_reward(prev_distance, new_distance, snake_Head, Point(foodx, foody), game_close, is_wall_collision, is_self_collision)
            total_reward += reward
            prev_distance = new_distance

            if snake_Head == [foodx, foody]:
                foodx, foody = respawn_food(snake_List)
                Length_of_snake += 1
                total_food_eaten += 1  # Increment food eaten

            next_state = get_state(snake_List, Point(foodx, foody), Point(x1, y1), (x1_change, y1_change))
            done = float(game_close)

            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.optimize_model(beta)

            if loss is not None:
                total_loss += loss
            steps += 1

            epsilon = max(min_epsilon, epsilon * epsilon_decay)

            if game_close:
                print(f'Episode {episode} - Score: {Length_of_snake - 1}, Total Loss: {total_loss:.2f}, Total Reward: {total_reward:.2f}, Steps: {steps}, Food Eaten: {total_food_eaten}')
                with open(history_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([episode, total_food_eaten, total_reward])
                agent.save_model(episode)  # Save the model after each episode
                episode += 1
                break

if __name__ == "__main__":
    main()
