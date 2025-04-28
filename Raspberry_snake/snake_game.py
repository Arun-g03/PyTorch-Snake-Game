import random
import numpy as np

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

# Set display dimensions
dis_width = 400
dis_height = 400

# Define clock and cell size
cell_size = 20  # Cell size variable

# Improved reward function to encourage eating food and avoid dying
def calculate_reward(prev_distance, new_distance, snake_head, food, game_close, is_wall_collision, is_self_collision):
    if game_close:
        if is_wall_collision or is_self_collision:
            return -1000  # Penalty for dying by hitting a wall or itself
        
    elif snake_head == food:
        return 100  # Reward for eating food
    else:
        # Reward based on the change in distance to the food
        distance_change = prev_distance - new_distance
        return 2.5 if distance_change > 0 else -10  # Positive reward for getting closer, negative for moving away

def get_state(snake_list, food, head, direction):
    point_l = Point(head.x - cell_size, head.y)
    point_r = Point(head.x + cell_size, head.y)
    point_u = Point(head.x, head.y - cell_size)
    point_d = Point(head.x, head.y + cell_size)
    
    dir_l = direction == Direction.LEFT
    dir_r = direction == Direction.RIGHT
    dir_u = direction == Direction.UP
    dir_d = direction == Direction.DOWN

    # Check if the snake's head is near a wall
    is_wall_left = int(head.x <= cell_size)
    is_wall_right = int(head.x >= dis_width - 2 * cell_size)
    is_wall_top = int(head.y <= cell_size)
    is_wall_bottom = int(head.y >= dis_height - 2 * cell_size)

    state = [
        # Danger straight
        (dir_r and Point(head.x + cell_size, head.y) in snake_list) or
        (dir_l and Point(head.x - cell_size, head.y) in snake_list) or
        (dir_u and Point(head.x, head.y - cell_size) in snake_list) or
        (dir_d and Point(head.x, head.y + cell_size) in snake_list),

        # Danger right
        (dir_u and Point(head.x + cell_size, head.y) in snake_list) or
        (dir_d and Point(head.x - cell_size, head.y) in snake_list) or
        (dir_l and Point(head.x, head.y - cell_size) in snake_list) or
        (dir_r and Point(head.x, head.y + cell_size) in snake_list),

        # Danger left
        (dir_d and Point(head.x + cell_size, head.y) in snake_list) or
        (dir_u and Point(head.x - cell_size, head.y) in snake_list) or
        (dir_r and Point(head.x, head.y - cell_size) in snake_list) or
        (dir_l and Point(head.x, head.y + cell_size) in snake_list),

        # Move direction
        dir_l,
        dir_r,
        dir_u,
        dir_d,
        
        # Food location 
        food.x < head.x,  # food left
        food.x > head.x,  # food right
        food.y < head.x,  # food up
        food.y > head.x,  # food down

        # Wall proximity
        is_wall_left,
        is_wall_right,
        is_wall_top,
        is_wall_bottom
    ]

    return np.array(state, dtype=int)

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
