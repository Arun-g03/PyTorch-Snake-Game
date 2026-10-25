import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for config.py
import config
import os
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch

from snake_game import SnakeGame, ACTION_DIM
import checkpoints
from DQN import DQNAgent
from evolution import evolve
from PPO import PPOAgent

PLOT_EVERY = config.PLOT_EVERY
AVG_WINDOW = config.AVG_WINDOW
PLOT_WINDOW = config.PLOT_WINDOW


def make_agent(state_dim, device, num_envs=1):
    """Build the agent selected by config.MODEL_TYPE."""
    agents = {"ppo": PPOAgent, "dqn": DQNAgent}
    if config.MODEL_TYPE not in agents:
        raise ValueError(f"config.MODEL_TYPE must be one of {sorted(agents)}, got {config.MODEL_TYPE!r}")
    return agents[config.MODEL_TYPE](state_dim, ACTION_DIM, device=device, num_envs=num_envs)


def ask_yes_no(prompt, default=False):
    answer = input(f"{prompt} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
    return default if not answer else answer.startswith('y')


def pick_model():
    """Let the user choose a saved model. Returns its path, or None if there are none / invalid."""
    models = checkpoints.list_checkpoints(config.MODEL_TYPE)
    if not models:
        print(f"No saved {config.MODEL_TYPE} models found in {config.MODEL_DIR}.")
        return None
    print("Available models:")
    for i, model in enumerate(models):
        print(f"{i + 1}. {model}")
    try:
        index = int(input("Enter the number of the model to load: ")) - 1
    except ValueError:
        index = -1
    if 0 <= index < len(models):
        return os.path.join(config.MODEL_DIR, models[index])
    print("Invalid choice.")
    return None


def choose_mode():
    """Terminal menu. Returns (mode, model_file) with mode in {'train', 'watch', 'evolve'}."""
    while True:
        print("Select an option:")
        print("1. Train a new model")
        print("2. Continue training a saved model")
        print("3. Watch a saved model play")
        print("4. Evolve a population of PPO agents in parallel")
        choice = input("Enter 1, 2, 3 or 4: ").strip()

        if choice == "1":
            return "train", None
        if choice == "4":
            return "evolve", None
        if choice in ("2", "3"):
            model_file = pick_model()
            if model_file:
                return ("train" if choice == "2" else "watch"), model_file
        else:
            print("Invalid choice. Please enter 1, 2, 3 or 4.")


def main():
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    mode, model_file = choose_mode()
    if mode == "evolve":
        run_evolution()
        return
    render = True if mode == "watch" else ask_yes_no("Show the game window while training? (slower)")

    device = torch.device('cuda' if config.DEVICE == 'auto' and torch.cuda.is_available() else
                          'cpu' if config.DEVICE == 'auto' else config.DEVICE)
    print(f"[INFO] Using device: {device}, model type: {config.MODEL_TYPE}")

    # Training plays several games at once; watching (or a visible window) uses a single game
    num_envs = 1 if (render or mode == "watch") else config.NUM_ENVS
    games = [SnakeGame(render=render and i == 0) for i in range(num_envs)]
    agent = make_agent(games[0].state_dim, device, num_envs)

    episode, best_score = 0, -float('inf')
    if model_file:
        try:
            episode, best_score = agent.load_model(model_file)
        except RuntimeError as e:
            print(f"Could not load {model_file} (saved with a different network shape?): {e}")
            return

    if mode == "watch":
        watch(agent, games[0])
    else:
        train(agent, games, episode, best_score, render)


def ask_int(prompt, default):
    answer = input(f"{prompt} [{default}]: ").strip()
    try:
        return int(answer) if answer else default
    except ValueError:
        return default


def run_evolution():
    if config.MODEL_TYPE != "ppo":
        print(f"Evolution trains PPO agents; set MODEL_TYPE = \"ppo\" in config.py (it is {config.MODEL_TYPE!r}).")
        return
    num_agents = ask_int("Number of parallel agents", config.EVO_AGENTS)
    generations = ask_int("Number of generations", config.EVO_GENERATIONS)
    episodes = ask_int("Episodes per agent per generation", config.EVO_EPISODES)
    evolve(num_agents, generations, episodes)


def watch(agent, game):
    """Play with the loaded model (no learning) until the window is closed or Ctrl+C."""
    print("Watching the model play - close the window or press Ctrl+C to stop.")
    episode = 0
    try:
        while True:
            state = game.reset()
            done = False
            while not done:
                state, _, done, info = game.step(agent.act(state, greedy=True))
                if not game.render():
                    return
            print(f"Game {episode} - Score: {info['score']}, Steps: {info['steps']}")
            episode += 1
    except KeyboardInterrupt:
        pass
    finally:
        game.close()


def make_plot():
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    lines = {
        'food': ax1.plot([], [], lw=2, label='Total Food Eaten')[0],
        'food_avg': ax1.plot([], [], lw=2, linestyle='--', label='Average Food Eaten')[0],
        'reward': ax2.plot([], [], lw=2, label='Total Reward')[0],
        'reward_avg': ax2.plot([], [], lw=2, linestyle='--', label='Average Reward')[0],
    }
    ax1.set_xlabel('Episode (this run)')
    ax1.set_ylabel('Food Eaten')
    ax1.legend()
    ax2.set_xlabel('Episode (this run)')
    ax2.set_ylabel('Total Reward')
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.legend()
    return (ax1, ax2), lines


def train(agent, games, start_episode=0, best_score=-float('inf'), render=False):
    (ax1, ax2), lines = make_plot()

    xdata = deque(maxlen=PLOT_WINDOW)
    food_data = deque(maxlen=PLOT_WINDOW)
    reward_data = deque(maxlen=PLOT_WINDOW)
    food_avg_data = deque(maxlen=PLOT_WINDOW)
    reward_avg_data = deque(maxlen=PLOT_WINDOW)
    recent_food = deque(maxlen=AVG_WINDOW)
    recent_reward = deque(maxlen=AVG_WINDOW)

    def update_plot():
        lines['food'].set_data(xdata, food_data)
        lines['food_avg'].set_data(xdata, food_avg_data)
        lines['reward'].set_data(xdata, reward_data)
        lines['reward_avg'].set_data(xdata, reward_avg_data)
        for ax in (ax1, ax2):
            ax.relim()
            ax.autoscale_view()
        plt.pause(0.001)

    episode = start_episode  # lifetime episode number, continues from a loaded checkpoint
    run_episodes = 0         # episodes finished in this run - the plot's x axis starts from zero
    states = np.stack([game.reset() for game in games])
    total_rewards = [0.0] * len(games)
    try:
        while True:
            actions = agent.select_actions(states)  # one batched forward pass for every game

            for i, game in enumerate(games):
                next_state, reward, done, info = game.step(int(actions[i]))
                agent.observe(i, states[i], actions[i], reward, next_state, done)
                total_rewards[i] += reward

                if done:
                    print(f"Episode {episode} - Score: {info['score']}, Total Reward: {total_rewards[i]:.2f}, "
                          f"Steps: {info['steps']}")

                    recent_food.append(info['score'])
                    recent_reward.append(total_rewards[i])
                    xdata.append(run_episodes)
                    food_data.append(info['score'])
                    reward_data.append(total_rewards[i])
                    food_avg_data.append(np.mean(recent_food))
                    reward_avg_data.append(np.mean(recent_reward))
                    if run_episodes % PLOT_EVERY == 0:
                        update_plot()

                    # Best model is judged on the rolling average, not a single lucky episode
                    best_score = agent.save_model(episode, float(np.mean(recent_reward)), best_score)
                    episode += 1
                    run_episodes += 1
                    total_rewards[i] = 0.0
                    next_state = game.reset()

                states[i] = next_state

            agent.update(states)

            if render and not games[0].render():
                return
    except KeyboardInterrupt:
        print("\nInterrupted, saving latest model...")
    finally:
        agent.save_model(episode, -float('inf'), best_score, force=True)
        for game in games:
            game.close()


if __name__ == "__main__":
    main()
