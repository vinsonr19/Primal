import numpy as np
from space_setup import setup_space
from pixel_class import *
from constants import *
from pygame.event import get
from pygame.locals import K_ESCAPE, KEYDOWN, QUIT, K_r
from pygame.display import flip
from pygame.time import delay
from pygame import quit
from M_Star import *
from RL_Stuff import *
from collections import deque



def check_for_collisions(new_state, current_state, goals):
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

    at_goals = []

    for collision in collided:
        for robot in collided[collision]:
            if current_state[robot] == goals[robot]:
                at_goals.append((collision, robot))

    for collision, robot in at_goals:
        collided[collision].remove(robot)

    for collision, _ in at_goals:
        if len(collided[collision]) == 0:
            collided.pop(collision)

    return collided


def find_new_move(cost_mat, area, robot_current_loc, robot_new_loc, make_random=False):
    def neighbor_getter(index, tried_index):
        legal_indices = []
        for pos in MOVES:
            cur_state = index_to_state(index)
            new_state = (cur_state[0]+pos[0], cur_state[1]+pos[1])
            new_index = state_to_index(new_state)
            if new_index != tried_index and 0<=new_state[0]<WIDTH and 0<=new_state[1]<HEIGHT:
                    if area[new_state[0], new_state[1]].state not in [1,3,8]:
                        legal_indices.append(new_index)
        return legal_indices

    cur_index = state_to_index(robot_current_loc)
    tried_index = state_to_index(robot_new_loc)

    new_inds = neighbor_getter(cur_index, tried_index)

    if make_random == False:
        min_cost = np.infty
        min_cost_move = 0
        for neighbor in new_inds:
            if cost_mat[cur_index, neighbor] < min_cost:
                min_cost = cost_mat[cur_index, neighbor]
                min_cost_move = neighbor
    else:
        if uniform(0,1) < 0.2:
            return robot_current_loc
        return index_to_state(new_inds[randint(0, len(new_inds))])

    return index_to_state(min_cost_move)


def m_star_primal(solve=True, show=True):
    #define our space
    adj_mat, assets, goals, enemies, allies, cost_mat, _ = setup_space()


    if not solve and not show:
        return assets, goals, constants.NORMAL_STATES, cost_mat

    for i,j in assets:
        SPACE[i,j].change_state(8)

    if show:
        flip()

    naive_paths = [A_Star(cost_mat, SPACE, assets[i], goals[i], []) for i in range(NUM_ASSETS)]
    good_assets = set()
    colliding_assets = set()

    for i, list1 in enumerate(naive_paths):
        for j, list2 in enumerate(naive_paths):
            if i!=j:
                max_length = max(len(list1), len(list2))
                for pos in range(max_length):
                    try:
                        if list1[pos] == list2[pos]:
                            colliding_assets.add(i)
                            colliding_assets.add(j)
                    except:
                        pass

    for i in range(NUM_ASSETS):
        if not i in colliding_assets:
            good_assets.add(i)


    optimal_paths = [[] for i in range(NUM_ASSETS)]
    for i in good_assets:
        optimal_paths[i] = naive_paths[i]

    if show:
        print('{} assets didn\'t collide, I really really hapy'.format(len(good_assets)))

    for i in colliding_assets:
        start_pos = assets[i]
        goal_pos = goals[i]
        optimal_paths[i] = A_Star(cost_mat, SPACE, start_pos, goal_pos, optimal_paths)

    return optimal_paths, assets, goals, constants.NORMAL_STATES, cost_mat



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
            optimal_paths, assets, goals, constants.NORMAL_STATES, cost_mat = m_star_primal()
            solved = True
            path_lengths = [len(optimal_paths[i]) for i in range(NUM_ASSETS)]
            max_length = max(path_lengths)
            num_collisions = 0
            collision_list = []
            for i, list1 in enumerate(optimal_paths):
                for j, list2 in enumerate(optimal_paths):
                    if i!=j:
                        for pos in range(max_length):
                            try:
                                if list1[pos] == list2[pos]:
                                    # i,j bad
                                    num_collisions+=1
                                    collision_list.append((list1[pos], pos, i, j))
                                elif pos > 0 and (list1[pos-1]==list2[pos] and list1[pos]==list2[pos-1]):
                                    num_collisions+=1
                                    collision_list.append((list1[pos], pos, i, j))

                            except:
                                pass

            print('{} collisions'.format(num_collisions//2))
            print(collision_list)
            # if num_collisions > 0:
            #     breakpoint()


        for i in range(WIDTH):
            for j in range(HEIGHT):
                SPACE[i,j].change_state(constants.NORMAL_STATES[i,j])

        for i in range(NUM_ASSETS):
            SPACE[optimal_paths[i][0][0], optimal_paths[i][0][1]].change_state(8)

        for step in range(1, max_length):
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
            if step==2:
                delay(1000)
            for asset in range(NUM_ASSETS):
                if step >= path_lengths[asset]:
                    SPACE[optimal_paths[asset][-1][0], optimal_paths[asset][-1][1]].change_state(8)
                    continue
                SPACE[optimal_paths[asset][step-1][0], optimal_paths[asset][step-1][1]].change_state(constants.NORMAL_STATES[optimal_paths[asset][step-1][0], optimal_paths[asset][step-1][1]])
                SPACE[optimal_paths[asset][step][0], optimal_paths[asset][step][1]].change_state(8)
            for i in range(WIDTH):
                for j in range(HEIGHT):
                    SPACE[i,j].show()
            delay(80)
            flip()
