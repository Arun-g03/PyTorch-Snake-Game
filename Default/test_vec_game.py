"""
Checks that VecSnakeGame behaves exactly like the single-game reference (snake_game.SnakeGame).

Both games are fed the same actions and the same food positions (recorded from the reference games and
replayed into the vectorised one), then every state, reward and done flag is compared.

Run:  python Default/test_vec_game.py
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for config.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch

import snake_game as ref
from vec_game import CELLS, VecSnakeGame, W


class ReplayVec(VecSnakeGame):
    """VecSnakeGame whose food comes from per-game queues instead of its own random generator."""
    def __init__(self, queues, **kwargs):
        self.queues = queues
        super().__init__(**kwargs)

    def _spawn_food(self, rows):
        cells = []
        for row in rows.tolist():
            food = self.queues[row].pop(0)
            cells.append(-1 if food is None else food[1] * W + food[0])
        return torch.tensor(cells, dtype=torch.long, device=self.device)


def make_pair(num_envs, device='cpu'):
    queues = [[] for _ in range(num_envs)]
    games = []
    for i in range(num_envs):
        g = ref.SnakeGame(seed=i)
        original = g._spawn_food

        def recording(i=i, original=original):
            food = original()
            queues[i].append(food)
            return food
        g._spawn_food = recording
        g.reset()                       # re-draw once with the hook installed, so the first food is recorded
        games.append(g)
    return games, ReplayVec(queues, num_envs=num_envs, device=device)


def safe_greedy(game, rng):
    """Mostly sensible moves, so snakes get long and the flood fill is exercised."""
    best = None
    for a in (0, 1, 2):
        dx, dy = game.heading
        if a == 1:
            dx, dy = -dy, dx
        elif a == 2:
            dx, dy = dy, -dx
        c = (game.head[0] + dx, game.head[1] + dy)
        ok = 0 <= c[0] < ref.grid_w and 0 <= c[1] < ref.grid_h and (c not in game.body or c == game.snake[0])
        if ok:
            d = abs(c[0] - game.food[0]) + abs(c[1] - game.food[1]) if game.food else 0
            if best is None or d + rng.random() * 3 < best[0]:
                best = (d, a)
    return best[1] if best else rng.randrange(3)


def run(num_envs, steps, policy, device='cpu'):
    rng = random.Random(1)
    games, vec = make_pair(num_envs, device)
    states = vec.get_state().cpu().numpy()
    for i, g in enumerate(games):
        assert np.array_equal(states[i], g.get_state()), f"initial state differs (game {i})"

    longest = compared = 0
    for t in range(steps):
        actions = [policy(g, rng) for g in games]
        outs = [g.step(a) for g, a in zip(games, actions)]
        v_states, v_rewards, v_dones, v_info = vec.step(torch.tensor(actions, device=device))
        v_states, v_rewards, v_dones = v_states.cpu().numpy(), v_rewards.cpu().numpy(), v_dones.cpu().numpy()

        for i, (s, r, d, info) in enumerate(outs):
            assert np.array_equal(v_states[i], s), f"state differs at step {t}, game {i}"
            assert v_rewards[i] == np.float32(r), f"reward differs at step {t}, game {i}: {v_rewards[i]} vs {r}"
            assert bool(v_dones[i]) == d, f"done differs at step {t}, game {i}"
            assert int(v_info['score'][i]) == info['score'] and int(v_info['steps'][i]) == info['steps']
            longest = max(longest, len(games[i].snake))
            compared += 1

        # Reset the references first: that is what records the new food the vectorised game then replays
        fresh = {i: games[i].reset() for i, (_, _, d, _) in enumerate(outs) if d}
        next_states = vec.reset_done(torch.from_numpy(v_dones).to(device), torch.from_numpy(v_states).to(device))
        next_states = next_states.cpu().numpy()
        for i, state in fresh.items():
            assert np.array_equal(next_states[i], state), f"reset state differs at step {t}, game {i}"
    return compared, longest


if __name__ == "__main__":
    for name, policy in (("random", lambda g, rng: rng.randrange(3)), ("safe-greedy", safe_greedy)):
        compared, longest = run(num_envs=6, steps=3000, policy=policy)
        print(f"{name:12s} OK - {compared} game-steps identical, longest snake {longest}")
