import numpy as np
from pixel_class import *
from real_constants import *
from pygame.event import get
from pygame.locals import K_ESCAPE, KEYDOWN, QUIT, K_r
from pygame.display import flip
from pygame import quit
from real_M_Star import *
from time import sleep


#######
# Have separate A* algorithms that runs for those that flicker that treats other (local) assets as obstacles
#######


def check_for_collisions(new_state, current_state):
    set_of_robots = set()

    collided = {}

    cur_robot_index = 0


    for robot_loc in new_state:
        if robot_loc in set_of_robots:
            # print(robot_loc)
            collided[robot_loc] = [cur_robot_index]
            for j in range(cur_robot_index):
                if new_state[j] == robot_loc and j not in collided.get(robot_loc):
                    collided[robot_loc].append(j)
        else:
            set_of_robots.add(robot_loc)
        cur_robot_index += 1


    i = 0
    for robot_num1 in range(len(new_state)):
        for robot_num2 in range(len(current_state)):
            if robot_num1 == robot_num2:
                continue
            if new_state[robot_num1] == current_state[robot_num2] and new_state[robot_num2] == current_state[robot_num1]:
                robot1_found = False
                robot2_found = False
                for key in collided.keys():
                    if robot_num1 in collided[key]:
                        robot1_found = True
                    if robot_num2 in collided[key]:
                        robot2_found = True

                if not robot1_found or not robot2_found:
                    collided[i] = []

                    if robot1_found == False:
                        collided[i].append(robot_num1)
                    if robot2_found == False:
                        collided[i].append(robot_num2)
                    i += 1

    return collided


def m_star_primal():
    #define our space
    global SPACE
    SPACE = np.array([[Pixel(i,j) for j in range(HEIGHT)] for i in range(WIDTH)])

    global normal_states
    normal_states = np.array([[SPACE[i,j].state for j in range(HEIGHT)] for i in range(WIDTH)])

    adj_mat = np.zeros((TOTAL_SQUARES, TOTAL_SQUARES))

    for i in range(0, WIDTH):
        for j in range(0, HEIGHT):
            index_from = state_to_index((i, j))
            for m in MOVES:
                if 0<=i+m[0]<WIDTH and 0<=j+m[1]<HEIGHT:
                    index_to = state_to_index((i + m[0], j + m[1]))
                    if SPACE[i + m[0], j + m[1]].state not in [1,3]:
                        adj_mat[index_from, index_to] = 1

    assets = []
    goals = []

    for _ in range(NUM_ASSETS):
        stuff = True
        while stuff:
            new_loc = (randint(0, WIDTH), randint(0, HEIGHT))
            if not new_loc in assets and not new_loc in goals:
                assets.append(new_loc)
                stuff = False
        stuff = True
        while stuff:
            new_loc = (randint(0, WIDTH), randint(0, HEIGHT))
            if not new_loc in assets and not new_loc in goals:
                goals.append(new_loc)
                stuff = False



    for i,j in goals:
        normal_states[i,j] = 9
        SPACE[i,j].change_state(9)


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


    #add allies with help squares
    allies = [(np.random.randint(0,WIDTH), np.random.randint(0,HEIGHT)) for i in range(NUM_ALLIES)]
    for pos in allies:
        if pos in enemies or SPACE[pos].state in [3,4,8,9] or pos in assets or pos in goals:
            allies.remove(pos)

    for pos in allies:
        cur_influence = FRIENDLY_INFLUENCE + np.random.randint(-FRIENDLY_INFLUENCE+1, 2)
        normal_states[pos[0], pos[1]] = 1
        SPACE[pos[0], pos[1]].change_state(1)
        for i in range(-cur_influence, cur_influence+1):
            for j in range(-cur_influence, cur_influence+1):
                try:
                    # if i != 0 or j != 0:
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


    cost_mat = np.zeros((TOTAL_SQUARES, TOTAL_SQUARES))

    for i in range(0, WIDTH):
        for j in range(0, HEIGHT):
            index_from = state_to_index((i, j))
            for m in MOVES:
                index_to = state_to_index((i + m[0], j + m[1]))
                try:
                    if SPACE[i + m[0], j + m[1]].state == 0:
                        cost_mat[index_from, index_to] = move_cost
                    elif SPACE[i + m[0], j + m[1]].state in [1, 5]:
                        cost_mat[index_from, index_to] = move_cost + ally_influence_change
                    elif SPACE[i + m[0], j + m[1]].state in [3, 4]:
                        cost_mat[index_from, index_to] = move_cost + enemy_influence_change
                    elif SPACE[i + m[0], j + m[1]].state == 6:
                        cost_mat[index_from, index_to] = 0
                    elif SPACE[i + m[0], j + m[1]].state == 7:
                        cost_mat[index_from, index_to] = move_cost + mixed_influence_change
                except:
                    continue

    for i,j in assets:
        SPACE[i,j].change_state(8)

    flip()


    solved = False
    optimal_paths = [[] for i in range(NUM_ASSETS)]
    for i in range(NUM_ASSETS):
        optimal_paths[i] = A_Star(cost_mat, SPACE, assets[i], goals[i], Dijkstra=False)

    max_len = 0
    for i in range(NUM_ASSETS):
        if len(optimal_paths[i]) > max_len:
            max_len = len(optimal_paths[i])

    for i in range(NUM_ASSETS):
        if len(optimal_paths[i]) < max_len:
            while len(optimal_paths[i]) < max_len:
                optimal_paths[i].append(optimal_paths[i][-1])

    length_of_all = len(optimal_paths[0])
    for i in range((NUM_ASSETS)):
        if len(optimal_paths[i]) != length_of_all:
            breakpoint()

    collided_set = set([])

    for j in range(length_of_all-1):
        system_state = [optimal_paths[i][j] for i in range(NUM_ASSETS)]
        new_system_state = [optimal_paths[i][j+1] for i in range(NUM_ASSETS)]

        cur_collided = check_for_collisions(new_system_state, system_state)
        for pos in cur_collided.keys():
            cur_list = cur_collided[pos]
            for thing in cur_list:
                collided_set.add(thing)

    new_set = set([i for i in range(NUM_ASSETS)])
    C_bar = new_set - collided_set
    print(C_bar)

    breakpoint()




    return optimal_paths




    #find optimal path
    #cost of moving 2
    #cost of entering enemy territory is 5
    #cost of moving in friendly territory 1
if __name__ == '__main__':
    solved = False
    running = True
    while running:
        for event in get():
            if event.type == QUIT:
                quit()
                running = False
                break
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    quit()
                    running = False
                    break
                elif event.key == K_r:
                    solved = False
        if not solved:
            optimal_paths = m_star_primal()
            solved = True
            path_lengths = [len(optimal_paths[i]) for i in range(NUM_ASSETS)]
            max_length = max(path_lengths)
        for i in range(WIDTH):
            for j in range(HEIGHT):
                SPACE[i,j].change_state(normal_states[i,j])

        # breakpoint()

        for i in range(NUM_ASSETS):
            SPACE[optimal_paths[i][0][0], optimal_paths[i][0][1]].change_state(8)

        for step in range(1, max_length):
            for asset in range(NUM_ASSETS):
                if step >= path_lengths[asset]:
                    SPACE[optimal_paths[asset][-1][0], optimal_paths[asset][-1][1]].change_state(8)
                    continue
                SPACE[optimal_paths[asset][step-1][0], optimal_paths[asset][step-1][1]].change_state(normal_states[optimal_paths[asset][step-1][0], optimal_paths[asset][step-1][1]])
                SPACE[optimal_paths[asset][step][0], optimal_paths[asset][step][1]].change_state(8)
            for i in range(WIDTH):
                for j in range(HEIGHT):
                    SPACE[i,j].show()
            flip()
            # sleep(.2)
        # sleep(0.5)
