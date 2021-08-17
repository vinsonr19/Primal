from pygame import init
from pygame.display import set_mode


init()

# Define constants
WIDTH = 50
HEIGHT = 50
TOTAL_SQUARES = WIDTH*HEIGHT
ENEMY_INFLUENCE = 4
FRIENDLY_INFLUENCE = 4
GRID_PIXEL_SIZE = 15
SPACE_BETWEEN_PIXELS = 0
TOTAL = GRID_PIXEL_SIZE+SPACE_BETWEEN_PIXELS
SCREEN_WIDTH = WIDTH*TOTAL+1
SCREEN_HEIGHT = HEIGHT*TOTAL+1
MOVES = ((1,0), (0, -1), (-1, 0), (0, 1))
NUM_ENEMIES = WIDTH//2
NUM_ALLIES = HEIGHT//2
NUM_ASSETS = 12

# Define costs
move_cost = 3
ally_influence_change = -1.9
enemy_influence_change = 10
mixed_influence_change = 3

# Define colors
WHITE = (255,255,255) # blank pixel
BLACK = (0,0,0) # background/wall
RED = (255,0,0) # bad guy
BAD_INFLUENCE = (255,100,100)
GREEN = (0,255,0) # ally location
ALLY_INFLUENCE = (100,255,100)
BLUE = (0,0,255) # goal
PURPLE = (255,0,255) # final path
MIXED = ((255+100)//2, (100+255)//2, (100+100)//2) # close to both good and bad guys
CLEMSON_ORANGE = (245, 102, 0) # assets
REGALIA = (82,45,128) # goals

# Dictionary from states to colors
state_dict = {0:WHITE, 1:GREEN, 2:PURPLE, 5:ALLY_INFLUENCE, 3:RED, 4:BAD_INFLUENCE, 6:BLUE, 7:MIXED, 8:CLEMSON_ORANGE, 9:REGALIA}

# Initialize background
screen = set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
screen.fill(BLACK)

# Set hz (1/fps)
hz = 1/15
