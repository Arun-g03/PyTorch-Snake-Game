import pygame
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
        pygame.draw.rect(dis, red, [x[0], x[1], cell_size, cell_size])

def display_score(score):
    value = score_font.render("Your Score: " + str(score), True, white)
    dis.blit(value, [0, 0])

# Improved reward function to encourage eating food and avoid dying
def calculate_reward(prev_distance, new_distance, snake_head, food, game_close, is_wall_collision, is_self_collision, snake_list):
    if game_close:
        if is_wall_collision:
            return -100  # Penalty for hitting wall
        elif is_self_collision:
            return -150  # Higher penalty for self-collision
        
    elif snake_head == food:
        # Reward for eating food increases with snake length
        snake_length = len(snake_list)
        return 100 + (snake_length * 10)  # Base reward + bonus based on length
    else:
        # Reward based on the change in distance to the food
        distance_change = prev_distance - new_distance
        if distance_change > 0:
            return 5  # Reward for getting closer
        elif distance_change < 0:
            return -10  # Penalty for moving away
        else:
            return -1  # Small penalty for no progress

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

    # Calculate relative food position
    food_dx = (food.x - head.x) / dis_width
    food_dy = (food.y - head.y) / dis_height

    # Calculate snake body proximity
    body_left = any(pos[0] == head.x - cell_size and pos[1] == head.y for pos in snake_list)
    body_right = any(pos[0] == head.x + cell_size and pos[1] == head.y for pos in snake_list)
    body_up = any(pos[0] == head.x and pos[1] == head.y - cell_size for pos in snake_list)
    body_down = any(pos[0] == head.x and pos[1] == head.y + cell_size for pos in snake_list)

    # Calculate snake length
    snake_length = len(snake_list)

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
        
        # Food location (normalized)
        food_dx,
        food_dy,

        # Wall proximity
        is_wall_left,
        is_wall_right,
        is_wall_top,
        is_wall_bottom,

        # Body proximity
        body_left,
        body_right,
        body_up,
        body_down,

        # Snake length (normalized)
        snake_length / 100.0  # Normalize to [0,1] range
    ]

    return np.array(state, dtype=float)

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
