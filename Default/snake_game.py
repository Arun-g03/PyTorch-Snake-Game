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
def calculate_reward(prev_distance, new_distance, snake_head, food, game_close, is_wall_collision, is_self_collision, food_visible):
    if game_close:
        if is_wall_collision or is_self_collision:
            return -500  # Heavy death penalty

    if snake_head == [food.x, food.y]:
        return 150  # Big reward for eating food

    if food_visible:
        distance_change = prev_distance - new_distance
        if distance_change > 0:
            return 10  # ✅ Moving closer to food
        elif distance_change < 0:
            return -5  # ❌ Moving farther away from food

    return 2  # ✅ Small reward for surviving



import math

def get_state(snake_list, food, head, direction):
    def raycast(start, dx, dy, snake_list):
        """Returns (normalized_distance, food_seen) along a ray."""
        distance = 0
        x, y = start.x, start.y
        food_seen = 0

        while 0 <= x < dis_width and 0 <= y < dis_height:
            if [x, y] in snake_list:
                break
            if x == food.x and y == food.y:
                food_seen = 1
                break
            x += dx * cell_size
            y += dy * cell_size
            distance += 1

        max_distance = max(dis_width, dis_height) // cell_size
        normalized_distance = distance / max_distance
        return normalized_distance, food_seen

    # Raycast in 8 directions
    directions = [
        (0, -1),  # N
        (1, -1),  # NE
        (1, 0),   # E
        (1, 1),   # SE
        (0, 1),   # S
        (-1, 1),  # SW
        (-1, 0),  # W
        (-1, -1)  # NW
    ]

    obstacle_distances = []
    food_seen_flags = []

    for dx, dy in directions:
        dist, food_seen = raycast(head, dx, dy, snake_list)
        obstacle_distances.append(dist)
        food_seen_flags.append(food_seen)

    # Movement direction encoding
    move_x = 0
    move_y = 0
    if direction[0] > 0:
        move_x = 1
    elif direction[0] < 0:
        move_x = -1
    if direction[1] > 0:
        move_y = 1
    elif direction[1] < 0:
        move_y = -1

    # Snake length normalized
    snake_length_norm = len(snake_list) / (dis_width * dis_height / (cell_size * cell_size))

    # Final state
    state = obstacle_distances + food_seen_flags + [move_x, move_y, snake_length_norm]

    return np.array(state, dtype=np.float32), food_seen_flags


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
