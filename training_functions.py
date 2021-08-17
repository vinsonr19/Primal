from main import *
import numpy as np
from pixel_class import *
from constants import *
from M_Star import *
from RL_Stuff import *
from collections import deque
import os
import tensorflow as tf
from numba import njit
from tester import tester
import json

@njit()
def dist(pos1, pos2):
    return max(abs(pos1[0]-pos2[0]), abs(pos1[1]-pos2[1]))


def path_to_actions(paths):
    actions = [[] for path in paths]
    for asset, path in enumerate(paths):
        for i in range(len(path)-1):
            movement = (path[i+1][0]-path[i][0], path[i+1][1]-path[i][1])
            if movement == MOVES[0]:
                actions[asset].append(0)
            elif movement == MOVES[1]:
                actions[asset].append(1)
            elif movement == MOVES[2]:
                actions[asset].append(2)
            elif movement == MOVES[3]:
                actions[asset].append(3)
            else:
                actions[asset].append(4)
    return actions


def space_to_global_state(area, positions):
    global_state = area.copy()
    for i,j in positions:
        global_state[i,j] = 8
    return global_state


# 0:out of bounds, 1: empty space, 2:other assets, 3:allies, 4:bad guys, 5:other goal, 6:our goal
@njit()
def find_local_state(pos_x, pos_y, goal, global_state):
    space_near_asset = np.zeros(shape=(state_shape[0], state_shape[1]))
    for x in range(0,WIDTH):
        for y in range(0,HEIGHT):
            space_state = global_state[x,y]
            if space_state in [0,4,5,7] or (x==0 and y==0): # assume we don't know influence areas right now (for the 2nd part we're always in middle so who cares)
                space_near_asset[x,y] = 1
            elif (x,y) == goal: # if it's our goal assume no one is on it (if they are then I'd suggest they move)
                space_near_asset[x,y] = 6
            elif space_state == 8: # other assets
                space_near_asset[x,y] = 2
            elif space_state == 1: # allies
                space_near_asset[x,y] = 3
            elif space_state == 3: # bad guys
                space_near_asset[x,y] = 4
            elif space_state == 8: # others' goals (if we're here this should be a goal space and not ours)
                space_near_asset[x,y] = 5
    return space_near_asset


def state_maker(pos_x, pos_y, goal, global_state):
    space_near_asset = find_local_state(pos_x, pos_y, goal, global_state)
    state = np.array([[[1 if space_near_asset[j,k] == i else 0 for k in range(state_shape[0])] for j in range(state_shape[1])] for i in range(state_shape[2])]).reshape((1, *state_shape)).astype('float32')
    return state


def find_collisions(asset_positions, old_positions):
    collision_list = [False for _ in asset_positions]
    for asset1 in range(NUM_ASSETS):
        for asset2 in range(asset1+1, NUM_ASSETS):
            if asset_positions[asset1] == asset_positions[asset2] or (old_positions[asset1] == asset_positions[asset2] and asset_positions[asset1] == old_positions[asset2]):
                collision_list[asset1] = True
                collision_list[asset2] = True
    return collision_list


def get_reward(asset_positions, cost_mat, cur_valids, cur_actions, goals, collision_list, done, step):
    rewards_list = [0 for _ in range(NUM_ASSETS)]
    for asset in range(NUM_ASSETS):
        r = -cost_mat[asset_positions[asset][0]][asset_positions[asset][1]]
        try:
            if not cur_valids[asset][-1][int(cur_actions[asset][-1])]:
                r -= invalid_move
        except:
            breakpoint()
        if cur_actions[asset][-1] == 4 and asset_positions[asset] != goals[asset]:
            r -= stay_off_goal
        if collision_list[asset] > 1:
            r -= collision_penalty
        if done:
            r += el_fin
        elif step == PPO_STEPS-1 and asset_positions[asset] != goals[asset]:
            r -= 100
        r -= (abs(asset_positions[asset][0] - goals[asset][0]) + abs(asset_positions[asset][1] - goals[asset][1]))/WIDTH
        rewards_list[asset] = r
    return rewards_list


def show_off_primal(num):

    def sample(policy):
        if np.random.rand() < 1:
            return np.random.choice(5, 1, p=policy.numpy().squeeze())[0]
        return np.argmax(policy.numpy().squeeze())

    model = ActorCriticModel(state_shape, goal_shape, num_actions)
    model.load_weights('saved_weights/my_weights_{}'.format(num))

    assets, goals, normal_states, cost_mat, SPACE = m_star_primal(solve=False, show=True, return_space=True)


    step = 0

    asset_positions = assets
    done = False
    if asset_positions == goals:
        done = True

    assets_output = [[asset] for asset in assets]

    global_state = space_to_global_state(normal_states.copy(), asset_positions)

    while not done:
        for asset, (i,j) in enumerate(asset_positions):

            # create state/goal
            state = state_maker(i, j, goals[asset], global_state)
            goal = np.array([goals[asset][0]-i, goals[asset][1]-j]).reshape((1,2))

            # call the model to obtain an action
            if step == 0: # initializes the model
                model.call((state, goal))
            policy, value = model((state, goal))
            action = np.argmax(policy)

            # is valid
            valid = [1 for i in range(num_actions)]
            for k, (x,y) in enumerate(MOVES):
                if i+x<0 or i+x>=WIDTH or j+y<0 or j+y>=HEIGHT:
                    valid[k] = 0
                elif global_state[i+x, j+y] in [1,3,8]:
                    valid[k] = 0

            if valid[action] == 1: # if a legal move
                if action != 4: # if it's 4 we don't move
                    asset_positions[asset] = (asset_positions[asset][0]+MOVES[action][0], asset_positions[asset][1]+MOVES[action][1])
            else:
                policy = policy.numpy()
                policy[0,action] = 0
                action = np.argmax(policy)
                if action == 4:
                    policy[0,action] = 0
                    action = np.argmax(policy)
                if valid[action] == 1:
                    if action != 4:
                        asset_positions[asset] = (asset_positions[asset][0]+MOVES[action][0], asset_positions[asset][1]+MOVES[action][1])

            assets_output[asset].append(asset_positions[asset])

        # update global_state, check if done
        global_state = space_to_global_state(normal_states.copy(), asset_positions)
        if asset_positions == goals:
            done = True


        step += 1
        # SHOW OFF MY RESULTS
        for i in range(WIDTH):
            for j in range(HEIGHT):
                SPACE[i,j].change_state(global_state[i,j])


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
        if step==1:
            delay(1000)
        for i in range(WIDTH):
            for j in range(HEIGHT):
                SPACE[i,j].show()
        # delay(5)
        flip()


    output_dict = {}
    output_dict['assets'] = assets_output
    output_dict['friendlies'] = allies
    output_dict['enemies'] = enemies
    output_dict['friendly_influence_radii'] = ally_influence_radii
    output_dict['enemy_influence_radii'] = enemy_influence_radii

    print('This took {} steps to finish!'.format(step))
    delay(5000)

    with open('output.json', 'w') as f:
        json.dump(output_dict, f, indent=4)

    return


def scuffed_a_star(asset_positions, goals, normal_states, cost_mat, allies):

    num_good = len(asset_positions) + len(allies)

    # initialize asset/ally vision of space
    spaces = [[[(0, -1) for j in range(HEIGHT)] for i in range(WIDTH)] for _ in range(num_good)] # state,timestep
    t = 0
    done = False
    optimal_paths = [[] for i in range(NUM_ASSETS)]
    while not done:
        position_list = asset_positions+allies

        # find buckets
        buckets = bucket_and_state_maker(asset_positions, allies, normal_states, goals, return_only_buckets=True)
        buckets = list(buckets)
        optimal_paths = [[] for _ in range(NUM_ASSETS)]

        bucket_state_list = []
        cur_cost_list = []
        for bucket in buckets:
            # update individual spaces AND combine bucket vision
            cur_cost_mat = np.zeros((TOTAL_SQUARES, TOTAL_SQUARES))
            # hacky way of accessing first element of set
            for i in bucket:
                bucket_state = spaces[i]
                break
            # update each pixel to the most recent information for all assets/allies in the bucket
            for i in range(WIDTH):
                for j in range(HEIGHT):
                    for asset in bucket:
                        if dist(position_list[asset], (i,j)) <= COMM_RANGE:
                            spaces[asset][i][j] = (normal_states[i][j], t)
                        if bucket_state[i][j][1] < spaces[asset][i][j][1]:
                            bucket_state[i][j] = spaces[asset][i][j]
                    # NOTE: must change if we want allies/enemies to move
                    # make cost matrices
                    if bucket_state[i][j][1] != -1:
                        cur_cost_mat[i][j] = cost_mat[i][j]
                    else:
                        cur_cost_mat[i][j] = scary_unknown_change
            # store state, cost_mat
            bucket_state_list.append(bucket_state)
            cur_cost_list.append(cur_cost_mat)

        # do astar
        max_moves = COMM_RANGE
        naive_paths = [[] for i in range(NUM_ASSETS)]
        for i, bucket in enumerate(buckets):
            for asset in bucket:
                bucket_state = bucket_state_list[i]
                cur_cost_mat = cur_cost_list[i]
                if asset < NUM_ASSETS and asset_positions[asset] != goals[asset]:
                    path = A_Star_Low_Comms(cur_cost_mat, bucket_state, asset_positions[asset], goals[asset], [])
                    naive_paths[asset] = path[:min(len(path), max_moves)]



        # creates a dictionary that tracks all current locations of assets
        # appends all valid moves to optimal_paths for each assets
        # if an asset collides, it stops iterating over its path
        # optimal paths will not contain the starting location of an asset
        position_tracker = dict()
        collision_list = []
        ##### I THINK WE NEED TO CHECK FOR COLLISIONS STILL
        def update_paths():
            # function to update paths for cleaner code
            for future_time, move in enumerate(naive_paths[asset]):
                try:
                    if move not in position_tracker[t + future_time + 1]:
                        position_tracker[t + future_time + 1].append(move)
                        optimal_paths[asset].append(move)
                    else:
                        collision_list.append(asset, bucket_num)
                        break
                except:
                    position_tracker[t + future_time + 1] = []
                    position_tracker[t + future_time + 1].append(move)
                    optimal_paths[asset].append(move)


        for i, bucket in enumerate(buckets):
            for asset in bucket:
                update_paths()


        max_tries = len(collision_list) * 3
        # continues cycling through collision_list until it is empty
        # or a quitting criteria is met
        # if we are going to use a dictionary to track asset poitions, it may
        # be better to take that in to account in A_Star, rather than here
        while len(collision_list) != 0 and max_tries > 0:
            asset, bucket_num = collision_list.pop(0)

            cur_cost_mat = cur_cost_list[bucket_num]

            # this does not scale well, as it essentially blocks out
            # previous assets' paths
            for future_time_step in range(t + 1, t + max_moves + 1):
                for asset_location in position_tracker[future_time_step]:
                    cur_cost_mat[asset_location[0]][asset_location[1]] = collision_penalty

            path = A_Star_Low_Comms(cur_cost_mat, bucket_state, optimal_paths[asset][-1], goals[asset], [])
            remaining_moves = t + max_moves - len(optimal_paths[asset])
            naive_paths[asset] = path[:min(len(path), remaining_moves)]

            update_paths()

            max_tries -= 1

        if max_tries == 0 and len(collision_list) != 0:
            print("Sadge: couldn't resolve collisions in time")



        done = True
        for i in range(NUM_ASSETS):
            asset_positions[i] = optimal_paths[i][-1]
            if asset_positions[i] != goals[i]:
                done = False
        t += max_moves

    return optimal_paths
