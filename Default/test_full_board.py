"""
Checks the "snake fills the whole board" case on a tiny 4x4 board, in both implementations.

A scripted player follows a route that visits every cell, so it eventually eats the last piece of food.
Both the reference game and the vectorised game must then report the same final state, end the episode,
and handle having no food left.

Run:  python Default/test_full_board.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for config.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

config.DIS_WIDTH = config.DIS_HEIGHT = 4 * config.CELL_SIZE  # must be set before the game modules are imported

import numpy as np
import torch

import snake_game as ref
from test_vec_game import make_pair

W, H = ref.grid_w, ref.grid_h
assert (W, H) == (4, 4), (W, H)


def hamiltonian_cycle():
    """A closed route through every cell of the 4x4 board: along the top, serpentine down, back up column 0."""
    cells = [(x, 0) for x in range(W)]
    for y in range(1, H):
        xs = range(W - 1, 0, -1) if y % 2 == 1 else range(1, W)
        cells += [(x, y) for x in xs]
    cells += [(0, y) for y in range(H - 1, 0, -1)]
    assert len(cells) == W * H and len(set(cells)) == W * H
    return cells


CYCLE = hamiltonian_cycle()
NEXT = {a: b for a, b in zip(CYCLE, CYCLE[1:] + CYCLE[:1])}
# The snake starts heading east and cannot turn around, so use the direction of travel that allows it
start = (W // 2, H // 2)
if NEXT[start] == (start[0] - 1, start[1]):
    NEXT = {b: a for a, b in NEXT.items()}


def follow_route(game):
    """Relative action that moves the head to the next cell on the route."""
    target = NEXT[game.head]
    want = (target[0] - game.head[0], target[1] - game.head[1])
    hx, hy = game.heading
    if want == (-hy, hx):
        return 1
    if want == (hy, -hx):
        return 2
    return 0


def main(num_envs=3, steps=900):
    games, vec = make_pair(num_envs)
    wins = 0
    for t in range(steps):
        actions = [follow_route(g) for g in games]
        outs = [g.step(a) for g, a in zip(games, actions)]
        v_states, v_rewards, v_dones, v_info = vec.step(torch.tensor(actions))

        for i, (s, r, d, info) in enumerate(outs):
            assert np.array_equal(v_states[i].numpy(), s), f"state differs at step {t}, game {i}"
            assert v_rewards[i].item() == np.float32(r) and bool(v_dones[i]) == d, f"result differs at step {t}"
            if len(games[i].snake) == W * H:        # the board is full
                assert games[i].food is None and d, "a full board must end the episode with no food left"
                assert int(vec.food[i]) == -1
                assert r == ref.REWARD_FOOD, "the last piece of food is rewarded like any other"
                wins += 1

        fresh = {i: games[i].reset() for i, (_, _, d, _) in enumerate(outs) if d}
        nxt = vec.reset_done(v_dones, v_states)
        for i, state in fresh.items():
            assert np.array_equal(nxt[i].numpy(), state), f"reset state differs at step {t}, game {i}"
    assert wins >= num_envs, f"only {wins} full boards reached"
    print(f"full board OK - {wins} games filled the 4x4 board; reference and vectorised results identical")


if __name__ == "__main__":
    main()
