from pygame import init
from pygame.display import set_mode


init()

SPACE = []
NORMAL_STATES = []

# RL constants
LEARNING_RATE       = 2.5e-4
GAMMA               = 0.99
GAE_LAMBDA          = 0.95
PPO_EPSILON         = 0.2
CRITIC_DISCOUNT     = 0.05
VALID_DISCOUNT      = 0.5
BLOCKING_DISCOUNT   = 0.5
ON_GOAL_DISCOUNT    = 0.4
ENTROPY_BETA        = 0.01
PPO_STEPS           = 64
MINI_BATCH_SIZE     = 32
PPO_EPOCHS          = 5#10
TEST_FREQ           = 20
NUM_TESTS           = 10
TRAINING_EPISODES   = 10000
MAX_GRAD_NORM       = 0.5
SUPERVISED_CUTOFF   = 200



# Define constants
WIDTH = 30
HEIGHT = 30
TOTAL_SQUARES = WIDTH*HEIGHT
ENEMY_INFLUENCE = 5
FRIENDLY_INFLUENCE = 3#7
GRID_PIXEL_SIZE = 12
SPACE_BETWEEN_PIXELS = 0
TOTAL = GRID_PIXEL_SIZE+SPACE_BETWEEN_PIXELS
SCREEN_WIDTH = WIDTH*TOTAL+1
SCREEN_HEIGHT = HEIGHT*TOTAL+1
MOVES = ((1,0), (0, -1), (-1, 0), (0, 1), (1, 1), (1, -1), (-1, 1), (-1, -1))
NUM_ENEMIES = WIDTH//6
NUM_ALLIES = HEIGHT//6
NUM_ASSETS = 10
COMM_RANGE = 100


# state_radius = 4
state_shape = (WIDTH,HEIGHT,7)
goal_shape = (2,1)
num_actions = 9


# Define costs
move_cost = 3
ally_influence_change = -2
enemy_influence_change = 10
mixed_influence_change = 3
scary_unknown_change = 5
invalid_move = 20
collision_penalty = 20
stay_off_goal = 3
el_fin = 20

# Define colors
BLANK = (255,255,255) # blank pixel
BLACK = (0,0,0) # background/wall
BAD_GUY = (255,0,0) # bad guy
BAD_INFLUENCE = (255,100,100)
ALLY = (0,255,0) # ally location
ALLY_INFLUENCE = (100,222,100)
#GOAL = (0,0,255) # goal
PURPLE = (255,0,255) # final path
MIXED = ((255+100)//2, (100+255)//2, (100+100)//2) # close to both good and bad guys
ASSET = (245, 102, 0) # assets - CLEMSON_ORANGE
GOAL = (82,45,128) # goals

# Dictionary from states to colors
state_dict = {0:BLANK, 1:ALLY, 5:ALLY_INFLUENCE, 3:BAD_GUY, 4:BAD_INFLUENCE, 7:MIXED, 8:ASSET, 9:GOAL}

# Initialize background
screen = set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
screen.fill(BLACK)

# Set hz (1/fps)
hz = 1/15
