"""
Parallel evolution of PPO agents.

Each generation, `num_agents` agents train in separate processes for a fixed number of episodes. The
fittest two are bred (random per-weight crossover) and mutated to form the next generation, with the
fittest agent carried over unchanged. Only a new overall best is written to disk
(models/ptorch_evo_ppo.pth), so the run costs one save per record instead of one per episode.
"""
import multiprocessing as mp
import random
import sys
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for config.py
import config
import numpy as np
import torch

import checkpoints
from PPO import PPOAgent
from snake_game import ACTION_DIM
from vec_game import VecSnakeGame


def crossover(a, b):
    """Child weights: each weight is taken from parent `a` or `b` with equal probability."""
    return {k: np.where(np.random.random(a[k].shape) < 0.5, a[k], b[k]) for k in a}


def mutate(weights, rate, scale):
    """Add Gaussian noise to each weight tensor with probability `rate`."""
    out = {}
    for k, w in weights.items():
        out[k] = w + np.random.standard_normal(w.shape).astype(w.dtype) * scale if random.random() < rate else w
    return out


def train_agent(agent_id, generation, num_episodes, weights=None):
    """Worker process: train one agent for `num_episodes`. Returns (agent_id, fitness, mean_reward, weights).

    Fitness is the mean score over the last `config.EVO_FITNESS_WINDOW` episodes.
    """
    torch.set_num_threads(1)  # one thread per worker; the workers are the parallelism
    env = VecSnakeGame(config.EVO_ENVS, device='cpu')

    def new_agent():
        return PPOAgent(env.state_dim, ACTION_DIM, device='cpu', num_envs=env.n)

    agent = new_agent()
    if weights is not None:
        agent.set_weights(weights)

    scores = deque(maxlen=config.EVO_FITNESS_WINDOW)
    rewards = deque(maxlen=config.EVO_FITNESS_WINDOW)
    best_score, since_best, episodes = -1, 0, 0
    states = env.reset()
    total_rewards = torch.zeros(env.n)

    while episodes < num_episodes:
        actions = agent.select_actions(states)
        next_states, step_rewards, dones, info = env.step(actions)
        agent.observe(states, actions, step_rewards, next_states, dones)
        total_rewards += step_rewards

        if dones.any():
            rows = dones.nonzero()[:, 0]
            for score, ret in zip(info['score'][rows].tolist(), total_rewards[rows].tolist()):
                scores.append(score)
                rewards.append(ret)
                episodes += 1
                if score > best_score:
                    best_score, since_best = score, 0
                else:
                    since_best += 1
                if episodes % 50 == 0:
                    print(f"  gen {generation} agent {agent_id}: episode {episodes}/{num_episodes}, "
                          f"recent avg score {np.mean(scores):.2f}", flush=True)
            total_rewards[rows] = 0
            states = env.reset_done(dones, next_states)

            # A stalled agent starts over with fresh weights instead of wasting the rest of its budget
            if config.EVO_RESET_AFTER and since_best >= config.EVO_RESET_AFTER and episodes < num_episodes:
                agent = new_agent()
                best_score, since_best = -1, 0
                print(f"  gen {generation} agent {agent_id}: no improvement, restarting from fresh weights")
        else:
            states = next_states
        agent.update(states)

    return agent_id, float(np.mean(scores)), float(np.mean(rewards)), agent.get_weights()


def evolve(num_agents, generations, episodes):
    """Run the whole evolution. Returns the best fitness reached."""
    ctx = mp.get_context('spawn')  # CUDA- and Windows-safe; workers always run on the CPU
    population = [None] * num_agents  # None = start from fresh weights
    best_fitness = -float('inf')

    try:
        for generation in range(generations):
            print(f"\nGeneration {generation + 1}/{generations}")
            with ProcessPoolExecutor(max_workers=num_agents, mp_context=ctx) as pool:
                futures = [pool.submit(train_agent, i, generation, episodes, population[i])
                           for i in range(num_agents)]
                results = [f.result() for f in futures]

            results.sort(key=lambda r: (r[1], r[2]), reverse=True)  # fitness, then mean reward
            fitnesses = [r[1] for r in results]
            print(f"Generation {generation + 1} summary: best fitness {fitnesses[0]:.2f}, "
                  f"average {np.mean(fitnesses):.2f}")

            if fitnesses[0] > best_fitness:
                best_fitness = fitnesses[0]
                checkpoints.save_weights(results[0][3], "evo", "ppo", (generation + 1) * episodes, best_fitness)

            # Next generation: keep the best agent as is, breed the top two for the rest
            parents = [r[3] for r in results[:2]]
            population = [parents[0]]
            for _ in range(num_agents - 1):
                child = crossover(parents[0], parents[1]) if len(parents) > 1 else parents[0]
                population.append(mutate(child, config.EVO_MUTATION_RATE, config.EVO_MUTATION_SCALE))
    except KeyboardInterrupt:
        print("\nInterrupted.")

    print(f"\nEvolution finished. Best fitness: {best_fitness:.2f}")
    return best_fitness
