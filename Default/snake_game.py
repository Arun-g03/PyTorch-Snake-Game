"""
The actual game logic is implemented in this file.

Positions are stored in grid cells (col, row); pygame is only imported and
initialised when rendering is requested, so headless training never opens a window.
"""
import random
import numpy as np

import config  # repo root is put on sys.path by main.py

# Colors
white = (255, 255, 255)
yellow = (255, 255, 102)
black = (0, 0, 0)
red = (213, 50, 80)
green = (0, 255, 0)
blue = (50, 153, 213)

# Display / grid dimensions (see config.py)
dis_width = config.DIS_WIDTH
dis_height = config.DIS_HEIGHT
cell_size = config.CELL_SIZE
snake_speed = config.SNAKE_SPEED
grid_w = dis_width // cell_size
grid_h = dis_height // cell_size

# Relative actions: the snake can never reverse into itself
STRAIGHT, TURN_RIGHT, TURN_LEFT = 0, 1, 2
ACTION_DIM = 3

# Rewards (see config.py)
REWARD_FOOD = config.REWARD_FOOD
REWARD_DEATH = config.REWARD_DEATH
REWARD_CLOSER = config.REWARD_CLOSER
REWARD_FARTHER = config.REWARD_FARTHER
STEP_LIMIT_FACTOR = config.STEP_LIMIT_FACTOR

# Headings in (dx, dy) grid units, clockwise on screen: N, E, S, W
HEADINGS = [(0, -1), (1, 0), (0, 1), (-1, 0)]

# 8 raycast directions: N, NE, E, SE, S, SW, W, NW
RAY_DIRECTIONS = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]


def calculate_reward(prev_distance, new_distance, ate_food, died):
    """Reward for one step: food and death dominate, distance shaping is symmetric."""
    if died:
        return REWARD_DEATH
    if ate_food:
        return REWARD_FOOD
    if new_distance < prev_distance:
        return REWARD_CLOSER
    if new_distance > prev_distance:
        return REWARD_FARTHER
    return 0.0


class SnakeGame:
    # wall/body/food rays + heading + length + (wall, body, free-space) x (ahead, right, left)
    # + food direction (4 flags + 2 signed offsets)
    state_dim = len(RAY_DIRECTIONS) * 3 + len(HEADINGS) + 1 + 9 + 6

    def __init__(self, render=False, seed=None):
        self.rng = random.Random(seed)
        self.render_enabled = render
        self._pygame = None
        self._screen = None
        self._clock = None
        self._font = None
        self.reset()

    def reset(self):
        self.head = (grid_w // 2, grid_h // 2)
        self.snake = [self.head]  # tail first, head last
        self.body = {self.head}
        self.heading = (1, 0)
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self.food = self._spawn_food()
        self.distance = self._food_distance()
        return self.get_state()

    def _spawn_food(self):
        free = [(c, r) for c in range(grid_w) for r in range(grid_h) if (c, r) not in self.body]
        return self.rng.choice(free) if free else None

    def _food_distance(self):
        if self.food is None:
            return 0
        return abs(self.head[0] - self.food[0]) + abs(self.head[1] - self.food[1])

    def step(self, action):
        """Advance one tick. Returns (state, reward, done, info)."""
        dx, dy = self.heading
        if action == TURN_RIGHT:
            dx, dy = -dy, dx
        elif action == TURN_LEFT:
            dx, dy = dy, -dx
        self.heading = (dx, dy)

        new_head = (self.head[0] + dx, self.head[1] + dy)
        ate = new_head == self.food
        tail = self.snake[0]
        self.steps += 1
        self.steps_since_food += 1

        hit_wall = not (0 <= new_head[0] < grid_w and 0 <= new_head[1] < grid_h)
        # The tail cell is vacated this tick, unless we grow
        hit_self = new_head in self.body and (ate or new_head != tail)
        died = hit_wall or hit_self

        if not died:
            if not ate:
                self.snake.pop(0)
                self.body.discard(tail)
            self.snake.append(new_head)
            self.body.add(new_head)
            self.head = new_head
            if ate:
                self.score += 1
                self.steps_since_food = 0
                self.food = self._spawn_food()

        won = not died and self.food is None
        stalled = (not died and not won
                   and self.steps_since_food > STEP_LIMIT_FACTOR * len(self.snake))

        new_distance = self._food_distance()
        reward = calculate_reward(self.distance, new_distance, ate and not died, died or stalled)
        self.distance = new_distance

        done = died or stalled or won
        info = {"score": self.score, "steps": self.steps, "died": died, "stalled": stalled}
        return self.get_state(), reward, done, info

    def get_state(self):
        """Wall/body distance and food closeness along 8 rays, one-hot heading, length."""
        hx, hy = self.head
        max_steps = max(grid_w, grid_h)
        wall, body, food = [], [], []

        for dx, dy in RAY_DIRECTIONS:
            x, y, k = hx + dx, hy + dy, 1
            body_k = food_k = None
            while 0 <= x < grid_w and 0 <= y < grid_h:
                if body_k is None and (x, y) in self.body:
                    body_k = k
                if food_k is None and (x, y) == self.food:
                    food_k = k
                x, y, k = x + dx, y + dy, k + 1
            wall.append(k / max_steps)
            body.append(1.0 if body_k is None else body_k / max_steps)
            food.append(0.0 if food_k is None else 1.0 / food_k)

        heading = [1.0 if h == self.heading else 0.0 for h in HEADINGS]
        length_norm = len(self.snake) / (grid_w * grid_h)

        # For each move (ahead / right / left of the heading): is the next cell a wall, is it the
        # snake's own body, and how much free space is reachable from it (0 = a dead end that traps us)
        fx, fy = self.heading
        rx, ry = -fy, fx  # right-hand vector (clockwise on screen)
        wall_danger, body_danger, free_space = [], [], []
        for dx, dy in ((fx, fy), (rx, ry), (-rx, -ry)):
            cell = (hx + dx, hy + dy)
            is_wall = not (0 <= cell[0] < grid_w and 0 <= cell[1] < grid_h)
            is_body = cell in self.body and cell != self.snake[0]  # the tail cell is vacated this tick
            wall_danger.append(float(is_wall))
            body_danger.append(float(is_body))
            free_space.append(0.0 if is_wall or is_body else self._free_space(cell))
        danger = wall_danger + body_danger + free_space

        # Food relative to the head, in the snake's own frame (forward / right), so the agent always
        # knows which way to turn - the rays alone only see food that lies exactly on one of them.
        food_dir = [0.0] * 6
        if self.food is not None:
            ox, oy = self.food[0] - hx, self.food[1] - hy
            ahead = ox * fx + oy * fy
            right = ox * rx + oy * ry
            food_dir = [float(ahead > 0), float(ahead < 0), float(right > 0), float(right < 0),
                        ahead / max_steps, right / max_steps]

        return np.array(wall + body + food + heading + [length_norm] + danger + food_dir, dtype=np.float32)

    def _free_space(self, start):
        """Fraction (0-1) of the space the snake needs that is reachable from `start`.

        Flood fill that stops once there is room for the whole snake plus one, so it stays cheap on an
        open board and only does real work when the snake is close to trapping itself.
        """
        need = len(self.snake) + 1
        blocked = self.body - {self.snake[0]}  # the tail moves away
        seen = {start}
        stack = [start]
        while stack and len(seen) < need:
            x, y = stack.pop()
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if (0 <= nx < grid_w and 0 <= ny < grid_h
                        and (nx, ny) not in seen and (nx, ny) not in blocked):
                    seen.add((nx, ny))
                    stack.append((nx, ny))
        return min(1.0, len(seen) / need)

    def render(self):
        """Draw the current frame and tick the clock. Returns False once the window is closed."""
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

        self._screen.fill(blue)
        if self.food is not None:
            pygame.draw.rect(self._screen, green,
                             [self.food[0] * cell_size, self.food[1] * cell_size, cell_size, cell_size])
        for c, r in self.snake:
            pygame.draw.rect(self._screen, black, [c * cell_size, r * cell_size, cell_size, cell_size])
        self._screen.blit(self._font.render(f"Your Score: {self.score}", True, black), [0, 0])
        pygame.display.update()
        self._clock.tick(snake_speed)
        return True

    def close(self):
        if self._pygame is not None:
            self._pygame.quit()
            self._pygame = None
