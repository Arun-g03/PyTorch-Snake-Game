import pygame
import random
import numpy as np
"""
The actual game logic is implemented in this file.
"""
# Define the Point class
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# Define the Direction enum if not already defined
class Direction:
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3

# Initialize Pygame
pygame.init()

# Define colors
white = (255, 255, 255)
yellow = (255, 255, 102)
black = (0, 0, 0)
red = (213, 50, 80)
green = (0, 255, 0)
blue = (50, 153, 213)

# Set display dimensions
dis_width = 400
dis_height = 400

# Create display
dis = pygame.display.set_mode((dis_width, dis_height))
pygame.display.set_caption('Snake Game')

# Define clock and cell size
clock = pygame.time.Clock()
cell_size = 20  # Cell size variable
snake_speed = 15

# Define font styles
font_style = pygame.font.SysFont(None, 50)
score_font = pygame.font.SysFont(None, 35)

def our_snake(cell_size, snake_List):
    for x in snake_List:
        pygame.draw.rect(dis, black, [x[0], x[1], cell_size, cell_size])

def display_score(score):
    value = score_font.render("Your Score: " + str(score), True, black)
    dis.blit(value, [0, 0])

# Improved reward function to encourage eating food and avoid dying
def calculate_reward(prev_distance, new_distance, snake_head, food, game_close, is_wall_collision, is_self_collision):
    if game_close:
        if is_wall_collision or is_self_collision:
            return -500  # Penalize death
    elif snake_head == food:
        return 100  # Big reward for eating
    else:
        distance_change = prev_distance - new_distance
        if distance_change > 0:
            return 5  # Reward for getting closer
        elif distance_change < 0:
            return -2  # Mild penalty for getting farther
        else:
            return 2  # Tiny reward just for surviving and moving


import math

def get_state(snake_list, food, head, direction):
    def danger_in_direction(point, snake_list):
        """Returns 1 if the point is dangerous (wall or body), 0 otherwise."""
        if point.x < 0 or point.x >= dis_width or point.y < 0 or point.y >= dis_height:
            return 1
        if [point.x, point.y] in snake_list:
            return 1
        return 0

    point_l = Point(head.x - cell_size, head.y)
    point_r = Point(head.x + cell_size, head.y)
    point_u = Point(head.x, head.y - cell_size)
    point_d = Point(head.x, head.y + cell_size)

    dir_l = direction == Direction.LEFT
    dir_r = direction == Direction.RIGHT
    dir_u = direction == Direction.UP
    dir_d = direction == Direction.DOWN

    # Multi-tile vision (1, 2, 3 steps ahead)
    danger_straight = []
    danger_right = []
    danger_left = []

    moves = {
        "straight": (x_dir := x_direction(dir_l, dir_r), y_dir := y_direction(dir_u, dir_d)),
        "right": (y_dir, -x_dir),
        "left": (-y_dir, x_dir)
    }

    for move_type, (x_step, y_step) in moves.items():
        for distance in range(1, 4):  # look 1, 2, 3 tiles ahead
            check_point = Point(head.x + x_step * cell_size * distance, head.y + y_step * cell_size * distance)
            danger = danger_in_direction(check_point, snake_list)
            if move_type == "straight":
                danger_straight.append(danger)
            elif move_type == "right":
                danger_right.append(danger)
            elif move_type == "left":
                danger_left.append(danger)

    # Food direction angle
    dx = food.x - head.x
    dy = food.y - head.y
    food_angle = math.atan2(dy, dx) / math.pi  # normalize between -1 and 1

    # Distance to walls (normalized 0–1)
    distance_left = head.x / dis_width
    distance_right = (dis_width - head.x) / dis_width
    distance_top = head.y / dis_height
    distance_bottom = (dis_height - head.y) / dis_height

    # Snake length normalized
    snake_length_norm = len(snake_list) / (dis_width * dis_height / (cell_size * cell_size))

    # Movement direction encoding
    move_x = 1 if dir_r else -1 if dir_l else 0
    move_y = 1 if dir_d else -1 if dir_u else 0

    state = [
        *danger_straight,
        *danger_right,
        *danger_left,
        food_angle,
        distance_left,
        distance_right,
        distance_top,
        distance_bottom,
        snake_length_norm,
        move_x,
        move_y
    ]

    return np.array(state, dtype=np.float32)

def x_direction(dir_l, dir_r):
    return -1 if dir_l else 1 if dir_r else 0

def y_direction(dir_u, dir_d):
    return -1 if dir_u else 1 if dir_d else 0

def respawn_food(snake_List):
    while True:
        foodx = round(random.randrange(0, dis_width - cell_size) / cell_size) * cell_size
        foody = round(random.randrange(0, dis_height - cell_size) / cell_size) * cell_size
        if [foodx, foody] not in snake_List:
            break
    return foodx, foody

def get_grid_cells():
    cells = []
    for x in range(0, dis_width, cell_size):
        for y in range(0, dis_height, cell_size):
            cells.append((x, y))
    return cells

def main():
    # Initialize game state
    snake_list = [[dis_width // 2, dis_height // 2]]
    food = respawn_food(snake_list)
    grid_cells = get_grid_cells()

    print("Grid cells:", grid_cells)
    print("Food location:", food)
    print("Snake body positions:", snake_list)

    # Main game loop would go here...

if __name__ == "__main__":
    main()
