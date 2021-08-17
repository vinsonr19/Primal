import numpy as np
import constants
from constants import *
from pixel_class import *
from M_Star import *


def create_adj_mat():
    adj_mat = np.zeros((TOTAL_SQUARES, TOTAL_SQUARES))

    for i in range(0, WIDTH):
        for j in range(0, HEIGHT):
            index_from = locationtuple_to_index((i, j))
            for m in MOVES:
                if 0<=i+m[0]<WIDTH and 0<=j+m[1]<HEIGHT:
                    index_to = locationtuple_to_index((i + m[0], j + m[1]))
                    if SPACE[i + m[0], j + m[1]].state not in [1,3]:
                        adj_mat[index_from, index_to] = 1
                        
    return(adj_mat)

def create_assets_and_goals():
    assets = []
    goals = []

    for _ in range(NUM_ASSETS):
        occupied = True
        while occupied:
            new_loc = (randint(0, WIDTH), randint(0, HEIGHT))
            if not new_loc in assets and not new_loc in goals:
                assets.append(new_loc)
                occupied = False
        occupied = True
        while occupied:
            new_loc = (randint(0, WIDTH), randint(0, HEIGHT))
            if not new_loc in assets and not new_loc in goals:
                goals.append(new_loc)
                occupied = False



    for i,j in goals:
        normal_states[i,j] = 9
        SPACE[i,j].change_state(9)
        
    return(assets, goals)
    
def create_enemies(assets, goals):
    #add enemies with threat squares
    enemies = [(np.random.randint(0,WIDTH), np.random.randint(0,HEIGHT)) for i in range(NUM_ENEMIES)]

    for pos in enemies:
        cur_influence = ENEMY_INFLUENCE + np.random.randint(-ENEMY_INFLUENCE+1, 3)
        if pos in assets or pos in goals:
            enemies.remove(pos)
            continue
        normal_states[pos[0], pos[1]] = 3
        SPACE[pos[0], pos[1]].change_state(3)
        for i in range(-cur_influence, cur_influence+1):
            for j in range(-cur_influence, cur_influence+1):
                try:
                    # if i != 0 or j != 0:
                    if 1 <= abs(i) + abs(j) <= FRIENDLY_INFLUENCE:
                        if 0<=pos[0]+i<WIDTH and 0<=pos[1]+j<HEIGHT:
                            if (pos[0] + i, pos[1] + j) in goals or (pos[0] + i, pos[1] + j) in assets or SPACE[pos[0] + i, pos[1] + j].state in [3,9]:
                                if SPACE[pos[0] + i, pos[1] + j].state != 3:
                                    normal_states[pos[0] + i, pos[1] + j] = 4
                                continue
                            normal_states[pos[0] + i, pos[1] + j] = 4
                            SPACE[pos[0] + i, pos[1] + j].change_state(4)
                except:
                    continue
                
    return(enemies)
    
def create_allies(assets, goals, enemies):
    #add allies with help squares
    allies = [(np.random.randint(0,WIDTH), np.random.randint(0,HEIGHT)) for i in range(NUM_ALLIES)]
    for pos in allies:
        if pos in enemies or SPACE[pos].state in [3,4,8,9] or pos in assets or pos in goals:
            allies.remove(pos)

    ally_influence_radii = []

    for pos in allies:
        cur_influence = FRIENDLY_INFLUENCE + np.random.randint(-FRIENDLY_INFLUENCE+1, 2)
        ally_influence_radii.append(cur_influence)
        normal_states[pos[0], pos[1]] = 1
        SPACE[pos[0], pos[1]].change_state(1)
        for i in range(-cur_influence, cur_influence+1):
            for j in range(-cur_influence, cur_influence+1):
                try:
                    if 1 <= abs(i) + abs(j) <= FRIENDLY_INFLUENCE:
                        if 0<=pos[0]+i<WIDTH and 0<=pos[1]+j<HEIGHT:
                            if (pos[0] + i, pos[1] + j) in goals or SPACE[pos[0] + i, pos[1] + j].state in [1,3,9]:
                                if SPACE[pos[0] + i, pos[1] + j].state not in [3,9]:
                                    if SPACE[pos[0] + i, pos[1] + j].state in [4,7]:
                                        normal_states[pos[0] + i, pos[1] + j] = 7
                                    else:
                                        normal_states[pos[0] + i, pos[1] + j] = 5
                                continue
                            if SPACE[pos[0] + i, pos[1] + j].state in [4,7]:
                                normal_states[pos[0] + i, pos[1] + j] = 7
                                SPACE[pos[0] + i, pos[1] + j].change_state(7)
                            else:
                                normal_states[pos[0] + i, pos[1] + j] = 5
                                SPACE[pos[0] + i, pos[1] + j].change_state(5)
                except:
                    continue
                
    return(allies, ally_influence_radii)

def create_cost_mat():
    cost_mat = np.zeros((TOTAL_SQUARES, TOTAL_SQUARES))

    for i in range(0, WIDTH):
        for j in range(0, HEIGHT):
            index_from = locationtuple_to_index((i, j))
            for m in MOVES:
                index_to = locationtuple_to_index((i + m[0], j + m[1]))
                try:
                    if SPACE[i + m[0], j + m[1]].state in [0]:
                        cost_mat[index_from, index_to] = move_cost
                    elif SPACE[i + m[0], j + m[1]].state in [1, 5]:
                        cost_mat[index_from, index_to] = move_cost + ally_influence_change
                    elif SPACE[i + m[0], j + m[1]].state in [3, 4]:
                        cost_mat[index_from, index_to] = move_cost + enemy_influence_change
                    elif SPACE[i + m[0], j + m[1]].state in [9]:
                        cost_mat[index_from, index_to] = move_cost-0.1
                    elif SPACE[i + m[0], j + m[1]].state == 7:
                        cost_mat[index_from, index_to] = move_cost + mixed_influence_change
                except:
                    continue
                
    return(cost_mat)
    
def setup_space():
    global SPACE
    SPACE = np.array([[Pixel(i,j) for j in range(HEIGHT)] for i in range(WIDTH)])
    constants.SPACE = SPACE

    global normal_states
    normal_states = np.array([[SPACE[i,j].state for j in range(HEIGHT)] for i in range(WIDTH)])
    constants.NORMAL_STATES = normal_states

    adj_mat = create_adj_mat()

    assets, goals = create_assets_and_goals()


    #add enemies with threat squares
    enemies = create_enemies(assets, goals)


    #add allies with help squares
    allies, ally_influence_radii = create_allies(assets, goals, enemies)
                

    cost_mat = create_cost_mat()
    
    return(adj_mat, assets, goals, enemies, allies, cost_mat, ally_influence_radii)