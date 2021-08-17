from main import *
from space_setup import setup_space
import numpy as np
from pixel_class import *
import constants
from constants import *
from M_Star import *
from RL_Stuff import *
from collections import deque
import os
import tensorflow as tf
from numba import njit
from tester import tester
from supervised import space_to_global_state, my_subfunction, state_maker, rewards_getter
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


def show_off_primal(num):

    model = ActorCriticModel(state_shape, goal_shape, num_actions)
    model.load_weights('saved_weights/my_weights_{}'.format(num))

    # define our space
    adj_mat, assets, goals, enemies, allies, cost_mat, ally_influence_radii = setup_space()

    for i,j in assets:
        SPACE[i,j].change_state(8)

    flip()

    def sample(policy):
        if np.random.rand() < 1:
            return np.random.choice(5, 1, p=policy.numpy().squeeze())[0]
        return np.argmax(policy.numpy().squeeze())

    step = 0

    asset_positions = assets
    done = False
    if asset_positions == goals:
        done = True

    assets_output = [[asset] for asset in assets]

    global_state = space_to_global_state(constants.NORMAL_STATES.copy(), asset_positions)

    while not done:
        for asset, (i,j) in enumerate(asset_positions):

            # create state/goal
            state = state_maker(i, j, goals[asset], global_state)
            goal = np.array([goals[asset][0]-i, goals[asset][1]-j]).reshape((1,2))

            # call the model to obtain an action
            if step == 0:
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
        global_state = space_to_global_state(constants.NORMAL_STATES.copy(), asset_positions)
        #visuals here?
        #save asset locations?
        #not sure best way to do this part
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


if __name__ == '__main__':

    CONTINUE_TRAINING = False
    model = ActorCriticModel(state_shape, goal_shape, num_actions)
    flatten = lambda t: [item for sublist in t for item in sublist]

    for episode in range(TRAINING_EPISODES):
        if (episode % 3 > 0 and episode > 50) or episode > SUPERVISED_CUTOFF:

            asset_positions, goals, constants.NORMAL_STATES, cost_mat = m_star_primal(solve=False, show=False)
            global_state = space_to_global_state(constants.NORMAL_STATES.copy(), asset_positions)

            done = False
            step = 0
            next_values = [0 for _ in range(NUM_ASSETS)]

            # initialize lists to keep track of individual assets throughout the episode
            cur_states = [[] for _ in range(NUM_ASSETS)]
            cur_policies = [[] for _ in range(NUM_ASSETS)]
            cur_values = [[] for _ in range(NUM_ASSETS)]
            cur_goals = [[] for _ in range(NUM_ASSETS)]
            # cur_goal_guesses = [[] for _ in range(NUM_ASSETS)]
            cur_actions = [[] for _ in range(NUM_ASSETS)]
            cur_valids = [[] for _ in range(NUM_ASSETS)]
            cur_rewards = [[] for _ in range(NUM_ASSETS)]
            masks = []
            cur_advantages = [[] for _ in range(NUM_ASSETS)]
            cur_returns = [[] for _ in range(NUM_ASSETS)]


            while not done and step < PPO_STEPS:

                for asset, (i,j) in enumerate(asset_positions):

                    # create state/goal
                    state = state_maker(i, j, goals[asset], global_state)
                    goal = np.array([goals[asset][0]-i, goals[asset][1]-j]).reshape((1,2))

                    # call the model to obtain an action
                    if episode == 0 and step == 0:
                        model.call((state, goal))
                        if CONTINUE_TRAINING:
                            model.load_weights('my_weights')
                    # policy, value, goal_guess = model((state, goal))
                    policy, value = model((state, goal))
                    action = model.sample(policy)

                    # is valid
                    valid = [1 for i in range(num_actions)]
                    for k, (x,y) in enumerate(MOVES):
                        if i+x<0 or i+x>=WIDTH or j+y<0 or j+y>=HEIGHT:
                            valid[k] = 0
                        elif global_state[i+x, j+y] in [1,3,8]:
                            valid[k] = 0

                    # dont need to store next state, check later after all moves
                    # store state, policy, value, goal, action, valids
                    cur_states[asset].append(state)
                    # cur_policies[asset].append(policy)
                    cur_values[asset].append(value)
                    cur_goals[asset].append(goal)
                    # cur_goal_guesses[asset].append(goal_guess)
                    cur_actions[asset].append(action)
                    cur_valids[asset].append(valid)


                # make all actions
                for asset in range(NUM_ASSETS):
                    action = cur_actions[asset][-1][0]
                    if cur_valids[asset][-1][action]: # if a legal move
                        if cur_actions[asset][-1] != 8: # if it's 4 we don't move
                            try:
                                asset_positions[asset] = (asset_positions[asset][0]+MOVES[action][0], asset_positions[asset][1]+MOVES[action][1])
                            except:
                                breakpoint()
                        

                # update global_state, check if done
                global_state = space_to_global_state(constants.NORMAL_STATES.copy(), asset_positions)
                if asset_positions == goals:
                    done = True
                masks.append(1-done)


                # get rewards (each created individually)
                # first, find collisions
                collision_dictionary = dict()
                for i,j in asset_positions:
                    if (i,j) in collision_dictionary.keys():
                        collision_dictionary[(i,j)] += 1
                    else:
                        collision_dictionary[(i,j)] = 1

                for asset in range(NUM_ASSETS):
                    r = -cost_mat[asset_positions[asset][0]][asset_positions[asset][1]]
                    try:
                        if not cur_valids[asset][-1][int(cur_actions[asset][-1])]:
                            r -= invalid_move
                    except:
                        breakpoint()
                    if cur_actions[asset][-1] == 4 and asset_positions[asset] != goals[asset]:
                        r -= stay_off_goal
                    if collision_dictionary[asset_positions[asset]] > 1:
                        r -= collision_penalty
                    if done:
                        r += el_fin
                    elif step == PPO_STEPS-1 and asset_positions[asset] != goals[asset]:
                        r -= 100
                    r -= (abs(asset_positions[asset][0] - goals[asset][0]) + abs(asset_positions[asset][1] - goals[asset][1]))/WIDTH

                    cur_rewards[asset].append(r)


                step += 1

            try:
                for asset, (i,j) in enumerate(asset_positions):
                    new_state = state_maker(i, j, goals[asset], global_state)
                    new_goal = np.array([[goals[asset][0]-i, goals[asset][1]-j]])
                    # _, next_value, _ = model((new_state, new_goal))
                    _, next_value = model((new_state, new_goal))
                    next_values[asset] = next_value

            except:
                print('Death at Next Values - antipog')
                breakpoint()

            returns = np.array([compute_gae(next_values[asset], cur_rewards[asset], masks, cur_values[asset]) for asset in range(NUM_ASSETS)])


            # states = sum(cur_states)
            # policies = sum(cur_policies)
            # values = sum(cur_values)
            # goals = sum(cur_goals)
            # actions = sum(cur_actions)
            # is_valid = sum(cur_valids)
            # rewards = sum(cur_rewards)

            advantages = normalize(returns-cur_values)

            # breakpoint()
            # on_goal = [[1 if np.array_equal(my_goal,np.array([[0,0]])) else 0 for my_goal in thing] for thing in cur_goals]

            cur_states = flatten(cur_states)
            returns = flatten(returns)
            advantages = flatten(advantages)
            cur_valids = flatten(cur_valids)
            cur_goals = flatten(cur_goals)
            # cur_goal_guesses = flatten(cur_goal_guesses)
            # on_goal = flatten(on_goal)



            model.train(cur_states, returns, advantages, cur_valids, cur_goals)#, on_goal, cur_goal_guesses)

        ##########################################################################################################################################

        else:

            try:
                optimal_paths, asset_positions, goals, constants.NORMAL_STATES, cost_mat = m_star_primal(solve=True, show=False)
                global_state = space_to_global_state(constants.NORMAL_STATES.copy(), asset_positions)

                beginning_positions = asset_positions.copy()

                correct_actions = path_to_actions(optimal_paths.copy())

                done = False
                step = 0
                next_values = [0 for _ in range(NUM_ASSETS)]

                # initialize lists to keep track of individual assets throughout the episode
                cur_states = [[] for _ in range(NUM_ASSETS)]
                cur_policies = [[] for _ in range(NUM_ASSETS)]
                cur_values = [[] for _ in range(NUM_ASSETS)]
                cur_goals = [[] for _ in range(NUM_ASSETS)]
                # cur_goal_guesses = [[] for _ in range(NUM_ASSETS)]
                cur_actions = [[] for _ in range(NUM_ASSETS)]
                cur_valids = [[] for _ in range(NUM_ASSETS)]
                cur_rewards = [[] for _ in range(NUM_ASSETS)]
                masks = []
                cur_advantages = [[] for _ in range(NUM_ASSETS)]
                cur_returns = [[] for _ in range(NUM_ASSETS)]


                while not done:

                    for asset, (i,j) in enumerate(asset_positions):

                        # create state/goal
                        state = state_maker(i, j, goals[asset], global_state)
                        goal = np.array([goals[asset][0]-i, goals[asset][1]-j]).reshape((1,2))

                        # call the model to obtain the model's chosen policy/value
                        if episode == 0 and step == 0:
                            model.call((state, goal))
                            if CONTINUE_TRAINING:
                                model.load_weights('my_weights')

                        # policy, value, goal_guess = model((state, goal))
                        policy, value = model((state, goal))

                        try:
                            action = correct_actions[asset][step]
                        except:
                            if asset_positions[asset] == goals[asset]:
                                action = 4
                            elif abs(asset_positions[asset][0]-goals[asset][0]) + abs(asset_positions[asset][1]-goals[asset][1]) > 1:
                                print(asset)
                                breakpoint()


                        # is valid
                        valid = [1 for i in range(num_actions)]
                        for k, (x,y) in enumerate(MOVES):
                            if i+x<0 or i+x>=WIDTH or j+y<0 or j+y>=HEIGHT:
                                valid[k] = 0
                            elif global_state[i+x, j+y] in [1,3,8]:
                                valid[k] = 0

                        # dont need to store next state, check later after all moves
                        # store state, policy, value, goal, action, valids
                        cur_states[asset].append(state)
                        cur_policies[asset].append(policy)
                        cur_values[asset].append(value)
                        cur_goals[asset].append(goal)
                        # cur_goal_guesses[asset].append(goal_guess)
                        cur_actions[asset].append(action)
                        cur_valids[asset].append(valid)


                    # make all actions
                    for asset in range(NUM_ASSETS):
                        action = cur_actions[asset][-1]
                        if action != 4: # if it's 4 we don't move
                            asset_positions[asset] = (asset_positions[asset][0]+MOVES[action][0], asset_positions[asset][1]+MOVES[action][1])

                    # update global_state, check if done
                    global_state = space_to_global_state(constants.NORMAL_STATES.copy(), asset_positions)
                    if asset_positions == goals:
                        done = True
                    masks.append(1-done)


                    # get rewards (each created individually)
                    # first, find collisions
                    collision_dictionary = dict()
                    for i,j in asset_positions:
                        if (i,j) in collision_dictionary.keys():
                            collision_dictionary[(i,j)] += 1
                        else:
                            collision_dictionary[(i,j)] = 1

                    for asset in range(NUM_ASSETS):
                        r = -cost_mat[asset_positions[asset][0]][asset_positions[asset][1]]
                        if not cur_valids[asset][-1][cur_actions[asset][-1]]:
                            r -= invalid_move
                        if cur_actions[asset][-1] == 4 and asset_positions[asset] != goals[asset]:
                            r -= stay_off_goal
                        if collision_dictionary[asset_positions[asset]] > 1:
                            r -= collision_penalty
                        if done:
                            r += el_fin
                        #elif step == PPO_STEPS-1 and asset_positions[asset] != goals[asset]:
                            #r -= 100
                        r -= (abs(asset_positions[asset][0] - goals[asset][0]) + abs(asset_positions[asset][1] - goals[asset][1]))/WIDTH

                        cur_rewards[asset].append(r / 33)


                    step += 1

                # get next_values
                try:
                    for asset, (i,j) in enumerate(asset_positions):
                        new_state = state_maker(i, j, goals[asset], global_state)
                        new_goal = np.array([[goals[asset][0]-i, goals[asset][1]-j]])
                        # _, next_value, _ = model((new_state, new_goal))
                        _, next_value = model((new_state, new_goal))
                        next_values[asset] = next_value

                except:
                    print('Death at Next Values - antipog')
                    breakpoint()

                returns = flatten([compute_gae(next_values[asset], cur_rewards[asset], masks, cur_values[asset]) for asset in range(NUM_ASSETS)])
                # advantages = normalize(returns-cur_values)

                cur_states = flatten(cur_states)
                cur_goals = flatten(cur_goals)
                cur_actions = flatten(cur_actions)

                model.train_imitation(cur_states, cur_goals, cur_actions, returns)

            except:
                continue

        if episode%10 == 0:
            print('Saving...')
            model.save_weights('/saved_weights/my_weights_{}'.format(episode+1))
            print('Testing...')
            tester(model, '/saved_weights/my_weights_{}'.format(episode+1))
            print('Finished Testing!')
        print('Epsiode {} is finished!'.format(episode+1))
