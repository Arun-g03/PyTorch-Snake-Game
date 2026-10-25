"""
Many snake games simulated at once as tensors (works on the CPU or the GPU).

Same rules, rewards and 44-value state as the single-game reference in snake_game.py, but every game
advances in one batch of tensor operations, so adding games is almost free and, on a GPU, training
never has to leave the device. test_vec_game.py checks that the two implementations agree exactly.

Board representation: `timer[n, cell]` is the number of further ticks that cell stays part of snake n
(head = length, ..., tail = 1, empty = 0). Moving without eating subtracts 1 from every timer and
stamps the new head with the length, which moves the tail forward with no list handling at all.
The extra last column is an always-empty sentinel used for cells outside the board.
"""
import numpy as np
import torch

import config  # repo root is put on sys.path by main.py
from snake_game import (HEADINGS, RAY_DIRECTIONS, REWARD_CLOSER, REWARD_DEATH, REWARD_FARTHER,
                        REWARD_FOOD, STEP_LIMIT_FACTOR, SnakeGame, cell_size, dis_height, dis_width,
                        grid_h, grid_w, snake_speed)

W, H = grid_w, grid_h
CELLS = W * H
MAX_STEPS = max(W, H)
RAY_LEN = max(W, H)

# Relative action -> change of heading index (headings are ordered clockwise: N, E, S, W)
TURN = torch.tensor([0, 1, -1])  # straight, right, left


def _build_tables():
    """Cells along each of the 8 rays from every head cell, and the wall distance on each ray."""
    ray_cells = np.full((CELLS, len(RAY_DIRECTIONS), RAY_LEN), CELLS, dtype=np.int64)  # CELLS = off board
    wall = np.zeros((CELLS, len(RAY_DIRECTIONS)), dtype=np.float64)
    for hy in range(H):
        for hx in range(W):
            head = hy * W + hx
            for d, (dx, dy) in enumerate(RAY_DIRECTIONS):
                x, y, k = hx + dx, hy + dy, 1
                while 0 <= x < W and 0 <= y < H:
                    ray_cells[head, d, k - 1] = y * W + x
                    x, y, k = x + dx, y + dy, k + 1
                wall[head, d] = k / MAX_STEPS  # k is the first step that leaves the board
    return ray_cells, wall


_RAY_CELLS_NP, _WALL_NP = _build_tables()


class VecSnakeGame:
    state_dim = SnakeGame.state_dim

    def __init__(self, num_envs, device='cpu', seed=None, render=False):
        self.n = num_envs
        self.device = torch.device(device)
        self.gen = torch.Generator(device=self.device)
        if seed is not None:
            self.gen.manual_seed(seed)
        else:
            self.gen.seed()

        self.ray_cells = torch.from_numpy(_RAY_CELLS_NP).to(self.device)
        self.wall_table = torch.from_numpy(_WALL_NP).to(self.device)
        self.heading_delta = torch.tensor(HEADINGS, dtype=torch.long, device=self.device)  # [4, 2]
        self.turn = TURN.to(self.device)
        self.arange = torch.arange(num_envs, device=self.device)

        self.timer = torch.zeros((num_envs, CELLS + 1), dtype=torch.int32, device=self.device)
        zeros = lambda: torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.head_x, self.head_y = zeros(), zeros()
        self.heading, self.length = zeros(), zeros()
        self.score, self.steps, self.steps_since_food = zeros(), zeros(), zeros()
        self.food = zeros()       # flat cell index, -1 when the board is full
        self.distance = zeros()   # Manhattan distance head -> food

        self.render_enabled = render
        self._pygame = self._screen = self._clock = self._font = None
        self.reset()

    # ------------------------------------------------------------------ resetting
    def reset(self):
        """Reset every game; returns the initial states [N, state_dim]."""
        self._reset_rows(self.arange)
        return self.get_state()

    def _reset_rows(self, rows):
        self.timer[rows] = 0
        self.head_x[rows], self.head_y[rows] = W // 2, H // 2
        self.heading[rows] = 1  # east
        self.length[rows] = 1
        self.timer[rows, (H // 2) * W + W // 2] = 1
        self.score[rows] = self.steps[rows] = self.steps_since_food[rows] = 0
        self.food[rows] = self._spawn_food(rows)
        self.distance[rows] = self._food_distance(rows)

    def reset_done(self, dones, next_states):
        """Reset the finished games; returns the states to act on next (fresh for finished games)."""
        rows = dones.nonzero()[:, 0]
        if len(rows) == 0:
            return next_states
        self._reset_rows(rows)
        states = next_states.clone()
        states[rows] = self.get_state(rows)
        return states

    def _spawn_food(self, rows):
        """Uniformly random free cell for each of `rows` (-1 if the board is full)."""
        free = self.timer[rows, :CELLS] == 0
        any_free = free.any(1)
        weights = free.float()
        weights[~any_free] = 1.0  # multinomial needs a valid distribution; the result is discarded below
        choice = torch.multinomial(weights, 1, generator=self.gen)[:, 0]
        return torch.where(any_free, choice, torch.full_like(choice, -1))

    def _food_distance(self, rows):
        food = self.food[rows]
        fx, fy = food % W, food // W
        dist = (self.head_x[rows] - fx).abs() + (self.head_y[rows] - fy).abs()
        return torch.where(food >= 0, dist, torch.zeros_like(dist))

    # ------------------------------------------------------------------ stepping
    def step(self, actions):
        """Advance every game one tick.

        Returns (next_states, rewards, dones, info). For finished games `next_states` is their final
        state; call reset_done() to start new games in their place.
        """
        self.heading = (self.heading + self.turn[actions]) % 4
        delta = self.heading_delta[self.heading]
        nx, ny = self.head_x + delta[:, 0], self.head_y + delta[:, 1]
        inside = (nx >= 0) & (nx < W) & (ny >= 0) & (ny < H)
        new_cell = torch.where(inside, ny * W + nx, torch.full_like(nx, CELLS))
        ate = inside & (new_cell == self.food)

        self.steps += 1
        self.steps_since_food += 1

        # The tail cell (timer == 1) is vacated this tick unless we grow
        t_new = self.timer.gather(1, new_cell[:, None])[:, 0]
        hit_self = inside & ((t_new > 1) | ((t_new == 1) & ate))
        died = ~inside | hit_self
        alive = ~died

        # Alive snakes that did not eat move their tail forward; every alive snake stamps its new head
        shrink = (alive & ~ate).to(torch.int32)[:, None]
        self.timer.sub_(shrink).clamp_(min=0)
        self.length = self.length + (alive & ate).long()
        rows = alive.nonzero()[:, 0]
        self.timer[rows, new_cell[rows]] = self.length[rows].to(torch.int32)
        self.head_x = torch.where(alive, nx, self.head_x)
        self.head_y = torch.where(alive, ny, self.head_y)

        grew = alive & ate
        self.score += grew.long()
        self.steps_since_food = torch.where(grew, torch.zeros_like(self.steps_since_food), self.steps_since_food)
        grown_rows = grew.nonzero()[:, 0]
        if len(grown_rows):
            self.food[grown_rows] = self._spawn_food(grown_rows)

        won = alive & (self.food < 0)
        stalled = alive & ~won & (self.steps_since_food > STEP_LIMIT_FACTOR * self.length)

        new_distance = self._food_distance(self.arange)
        reward = torch.zeros(self.n, dtype=torch.float32, device=self.device)
        reward = torch.where(new_distance > self.distance, torch.full_like(reward, REWARD_FARTHER), reward)
        reward = torch.where(new_distance < self.distance, torch.full_like(reward, REWARD_CLOSER), reward)
        reward = torch.where(ate & ~died, torch.full_like(reward, REWARD_FOOD), reward)
        reward = torch.where(died | stalled, torch.full_like(reward, REWARD_DEATH), reward)
        self.distance = new_distance

        dones = died | stalled | won
        info = {"score": self.score.clone(), "steps": self.steps.clone(), "died": died, "stalled": stalled}
        return self.get_state(), reward, dones, info

    # ------------------------------------------------------------------ observation
    def get_state(self, rows=None):
        """The 44-value state for every game (or just `rows`), as float32 [N, state_dim]."""
        sel = slice(None) if rows is None else rows
        timer = self.timer[sel]
        hx, hy, heading = self.head_x[sel], self.head_y[sel], self.heading[sel]
        food, length = self.food[sel], self.length[sel]
        n = timer.shape[0]
        ar = torch.arange(n, device=self.device)
        head_cell = hy * W + hx

        # Rays: nearest body cell and the food (if it lies exactly on a ray) in each of the 8 directions
        cells = self.ray_cells[head_cell]                                  # [n, 8, K]
        on_body = (timer > 0)[ar[:, None, None], cells]                    # [n, 8, K]
        has_body = on_body.any(2)
        body_k = on_body.to(torch.int8).argmax(2) + 1
        body = torch.where(has_body, body_k.double() / MAX_STEPS, torch.ones(1, dtype=torch.float64, device=self.device))
        on_food = cells == food[:, None, None]
        has_food = on_food.any(2)
        food_k = on_food.to(torch.int8).argmax(2) + 1
        food_ray = torch.where(has_food, 1.0 / food_k.double(), torch.zeros(1, dtype=torch.float64, device=self.device))
        wall = self.wall_table[head_cell]

        heading_onehot = torch.nn.functional.one_hot(heading, 4).double()
        length_norm = (length.double() / CELLS)[:, None]

        # Danger for the three moves: ahead / right / left of the heading
        move_headings = torch.stack([heading, (heading + 1) % 4, (heading - 1) % 4], 1)   # [n, 3]
        d = self.heading_delta[move_headings]                                              # [n, 3, 2]
        cx, cy = hx[:, None] + d[..., 0], hy[:, None] + d[..., 1]
        inside = (cx >= 0) & (cx < W) & (cy >= 0) & (cy < H)
        cell = torch.where(inside, cy * W + cx, torch.full_like(cx, CELLS))
        t = timer.gather(1, cell)
        is_wall = ~inside
        is_body = inside & (t > 1)  # the tail (timer 1) is vacated this tick, so it is not a danger
        free_space = self._free_space(timer, length, cx, cy, ~(is_wall | is_body))

        # Food relative to the head in the snake's own frame (forward / right)
        fwd = self.heading_delta[heading]
        fx, fy = fwd[:, 0], fwd[:, 1]
        rx, ry = -fy, fx
        ox, oy = food % W - hx, food // W - hy
        ahead, right = ox * fx + oy * fy, ox * rx + oy * ry
        has = (food >= 0)
        zero = torch.zeros(1, dtype=torch.float64, device=self.device)
        food_dir = torch.stack([
            (ahead > 0).double(), (ahead < 0).double(), (right > 0).double(), (right < 0).double(),
            ahead.double() / MAX_STEPS, right.double() / MAX_STEPS], 1)
        food_dir = torch.where(has[:, None], food_dir, zero)

        state = torch.cat([wall, body, food_ray, heading_onehot, length_norm,
                           is_wall.double(), is_body.double(), free_space, food_dir], 1)
        return state.to(torch.float32)

    def _free_space(self, timer, length, cx, cy, valid):
        """Fraction (0-1) of the room the snake needs that is reachable from each candidate cell.

        Flood fill by repeated dilation of a boolean grid. A game stops growing once it has reached
        `length + 1` cells or can grow no further, so the work stays small on an open board.
        """
        n = timer.shape[0]
        out = torch.zeros((n, 3), dtype=torch.float64, device=self.device)
        env, move = valid.nonzero(as_tuple=True)
        if len(env) == 0:
            return out
        passable = (timer[:, :CELLS] <= 1).reshape(n, H, W)[env]            # free cells and the vacating tail
        reach = torch.zeros_like(passable)
        reach[torch.arange(len(env), device=self.device), cy[env, move], cx[env, move]] = True
        need = length[env] + 1
        count = torch.ones(len(env), dtype=torch.long, device=self.device)
        active = count < need

        while True:
            sub = active.nonzero()[:, 0]
            if len(sub) == 0:
                break
            r = reach[sub]
            grown = r.clone()
            grown[:, 1:, :] |= r[:, :-1, :]
            grown[:, :-1, :] |= r[:, 1:, :]
            grown[:, :, 1:] |= r[:, :, :-1]
            grown[:, :, :-1] |= r[:, :, 1:]
            grown &= passable[sub]
            new_count = grown.sum((1, 2))
            stuck = new_count == count[sub]
            reach[sub] = grown
            count[sub] = new_count
            active[sub] = (new_count < need[sub]) & ~stuck

        out[env, move] = torch.minimum(count, need).double() / need.double()
        return out

    # ------------------------------------------------------------------ drawing (first game only)
    def render(self, env=0):
        """Draw one game and tick the clock. Returns False once the window is closed."""
        if self._pygame is None:
            import pygame
            pygame.init()
            self._pygame = pygame
            self._screen = pygame.display.set_mode((dis_width, dis_height))
            pygame.display.set_caption('Snake Game')
            self._clock = pygame.time.Clock()
            self._font = pygame.font.SysFont(None, 35)
        pygame = self._pygame

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return False

        self._screen.fill((50, 153, 213))
        food = int(self.food[env])
        if food >= 0:
            pygame.draw.rect(self._screen, (0, 255, 0),
                             [(food % W) * cell_size, (food // W) * cell_size, cell_size, cell_size])
        for cell in (self.timer[env, :CELLS] > 0).nonzero()[:, 0].tolist():
            pygame.draw.rect(self._screen, (0, 0, 0),
                             [(cell % W) * cell_size, (cell // W) * cell_size, cell_size, cell_size])
        self._screen.blit(self._font.render(f"Your Score: {int(self.score[env])}", True, (0, 0, 0)), [0, 0])
        pygame.display.update()
        self._clock.tick(snake_speed)
        return True

    def close(self):
        if self._pygame is not None:
            self._pygame.quit()
            self._pygame = None
