import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for config.py
import config
import os
import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch

from snake_game import ACTION_DIM
from vec_game import VecSnakeGame
import checkpoints
from DQN import DQNAgent
from evolution import evolve
from PPO import PPOAgent

PLOT_SECONDS = config.PLOT_SECONDS
PRINT_SECONDS = config.PRINT_SECONDS
HILO_FRACTION = config.HILO_FRACTION
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

    # Training simulates many games at once; watching (or a visible window) uses a single game
    num_envs = 1 if (render or mode == "watch") else resolve_num_envs(device)
    env = VecSnakeGame(num_envs, device=device, render=render)
    agent = make_agent(env.state_dim, device, num_envs)
    if mode != "watch":
        print(f"[INFO] Simulating {num_envs} game{'s' if num_envs != 1 else ''} in parallel")

    episode, best_score = 0, -float('inf')
    if model_file:
        try:
            episode, best_score = agent.load_model(model_file)
        except RuntimeError as e:
            print(f"Could not load {model_file} (saved with a different network shape?): {e}")
            return

    if mode == "watch":
        watch(agent, env)
    else:
        train(agent, env, episode, best_score, render)


def resolve_num_envs(device):
    if config.NUM_ENVS == "auto":
        return 256 if torch.device(device).type == "cuda" else 16
    return int(config.NUM_ENVS)


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


def watch(agent, env):
    """Play with the loaded model (no learning) until the window is closed or Ctrl+C."""
    print("Watching the model play - close the window or press Ctrl+C to stop.")
    games = 0
    state = env.reset()
    try:
        while True:
            state, _, done, info = env.step(torch.tensor([agent.act(state, greedy=True)], device=env.device))
            if not env.render():
                return
            if bool(done[0]):
                print(f"Game {games} - Score: {int(info['score'][0])}, Steps: {int(info['steps'][0])}")
                games += 1
                state = env.reset_done(done, state)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


def tail_means(values, fraction=HILO_FRACTION):
    """(low, high): the mean of the worst and of the best `fraction` of `values` (at least one each).

    Unlike the plain min / max, one lucky or unlucky episode cannot move it, and it shows whether the
    whole spread of results is improving - including the weak runs.
    """
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    k = max(1, int(len(ordered) * fraction))
    return float(ordered[:k].mean()), float(ordered[-k:].mean())


def make_plot():
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    lines = {
        'food': ax1.plot([], [], lw=2, label='Total Food Eaten')[0],
        'food_avg': ax1.plot([], [], lw=2, linestyle='--', label='Average Food Eaten')[0],
        'reward': ax2.plot([], [], lw=2, label='Total Reward')[0],
        'reward_avg': ax2.plot([], [], lw=2, linestyle='--', label='Average Reward')[0],
    }
    # High / low over the same rolling window as the average, drawn thin and dotted in the average's colour
    window = f'Avg of top / bottom {HILO_FRACTION:.0%} (last {AVG_WINDOW})'
    for key, ax, avg in (('food', ax1, 'food_avg'), ('reward', ax2, 'reward_avg')):
        color = lines[avg].get_color()
        lines[f'{key}_hi'] = ax.plot([], [], lw=1, linestyle=':', color=color, label=window)[0]
        lines[f'{key}_lo'] = ax.plot([], [], lw=1, linestyle=':', color=color)[0]
    ax1.set_xlabel('Episode (this run)')
    ax1.set_ylabel('Food Eaten')
    ax1.legend()
    ax2.set_xlabel('Episode (this run)')
    ax2.set_ylabel('Total Reward')
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.legend()
    return (ax1, ax2), lines


def train(agent, env, start_episode=0, best_score=-float('inf'), render=False):
    (ax1, ax2), lines = make_plot()

    xdata = deque(maxlen=PLOT_WINDOW)
    food_data = deque(maxlen=PLOT_WINDOW)
    reward_data = deque(maxlen=PLOT_WINDOW)
    food_avg_data = deque(maxlen=PLOT_WINDOW)
    reward_avg_data = deque(maxlen=PLOT_WINDOW)
    food_hi_data, food_lo_data = deque(maxlen=PLOT_WINDOW), deque(maxlen=PLOT_WINDOW)
    reward_hi_data, reward_lo_data = deque(maxlen=PLOT_WINDOW), deque(maxlen=PLOT_WINDOW)
    recent_food = deque(maxlen=AVG_WINDOW)
    recent_reward = deque(maxlen=AVG_WINDOW)

    def update_plot():
        lines['food'].set_data(xdata, food_data)
        lines['food_avg'].set_data(xdata, food_avg_data)
        lines['reward'].set_data(xdata, reward_data)
        lines['reward_avg'].set_data(xdata, reward_avg_data)
        lines['food_hi'].set_data(xdata, food_hi_data)
        lines['food_lo'].set_data(xdata, food_lo_data)
        lines['reward_hi'].set_data(xdata, reward_hi_data)
        lines['reward_lo'].set_data(xdata, reward_lo_data)
        for ax in (ax1, ax2):
            ax.relim()
            ax.autoscale_view()
        plt.pause(0.001)

    episode = start_episode  # lifetime episode number, continues from a loaded checkpoint
    run_episodes = 0         # episodes finished in this run - the plot's x axis starts from zero
    states = env.reset()
    total_rewards = torch.zeros(env.n, device=env.device)
    env_steps = 0
    started = last_plot = last_print = time.monotonic()
    steps_at_print = 0
    try:
        while True:
            actions = agent.select_actions(states)  # one batched forward pass for every game
            next_states, rewards, dones, info = env.step(actions)
            agent.observe(states, actions, rewards, next_states, dones)
            total_rewards += rewards
            env_steps += env.n

            if dones.any():
                rows = dones.nonzero()[:, 0]
                scores, returns = info['score'][rows].tolist(), total_rewards[rows].tolist()
                total_rewards[rows] = 0
                for score, ret in zip(scores, returns):
                    recent_food.append(score)
                    recent_reward.append(ret)
                    xdata.append(run_episodes)
                    food_data.append(score)
                    reward_data.append(ret)
                    run_episodes += 1
                finished = len(scores)
                food_avg_data.extend([float(np.mean(recent_food))] * finished)
                reward_avg_data.extend([float(np.mean(recent_reward))] * finished)
                food_lo, food_hi = tail_means(recent_food)
                reward_lo, reward_hi = tail_means(recent_reward)
                food_lo_data.extend([food_lo] * finished)
                food_hi_data.extend([food_hi] * finished)
                reward_lo_data.extend([reward_lo] * finished)
                reward_hi_data.extend([reward_hi] * finished)

                # Save when this batch of episodes crossed a SAVE_EVERY boundary. The best model is
                # judged on the rolling average, not a single lucky episode.
                crossed = (episode + finished) // config.SAVE_EVERY > episode // config.SAVE_EVERY
                episode += finished
                if crossed:
                    best_score = agent.save_model(episode, float(np.mean(recent_reward)), best_score, force=True)

                states = env.reset_done(dones, next_states)
            else:
                states = next_states
            agent.update(states)

            now = time.monotonic()
            if now - last_print >= PRINT_SECONDS and recent_food:
                rate = (env_steps - steps_at_print) / (now - last_print)
                food_lo, food_hi = tail_means(recent_food)
                reward_lo, reward_hi = tail_means(recent_reward)
                print(f"Episode {episode} | food avg {np.mean(recent_food):.1f} "
                      f"(low {food_lo:.1f}, high {food_hi:.1f}) | "
                      f"reward avg {np.mean(recent_reward):.2f} "
                      f"(low {reward_lo:.2f}, high {reward_hi:.2f}) | {rate:,.0f} steps/s")
                last_print, steps_at_print = now, env_steps
            if now - last_plot >= PLOT_SECONDS and xdata:
                update_plot()
                last_plot = time.monotonic()

            if render and not env.render():
                return
    except KeyboardInterrupt:
        print("\nInterrupted, saving latest model...")
    finally:
        agent.save_model(episode, -float('inf'), best_score, force=True)
        env.close()


if __name__ == "__main__":
    main()
