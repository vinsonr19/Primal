from main import m_star_primal
from training_functions import *
import numpy as np
from pixel_class import *
from constants import *
from M_Star import *
from RL_Stuff import *
from tester import tester



if __name__ == '__main__':

    CONTINUE_TRAINING = False
    model = ActorCriticModel(state_shape, goal_shape, num_actions)
    flatten = lambda t: [item for sublist in t for item in sublist]

    for episode in range(TRAINING_EPISODES):
        if (episode % 3 > 0 and episode > 50) or episode > SUPERVISED_CUTOFF:

            asset_positions, goals, normal_states, cost_mat = m_star_primal(solve=False, show=False)
            global_state = space_to_global_state(normal_states.copy(), asset_positions)
            allies = []
            for i in range(WIDTH):
                for j in range(HEIGHT):
                    if global_state[i, j] == 1:
                        allies.append((i, j))

            done = False
            step = 0
            next_values = [0 for _ in range(NUM_ASSETS)]

            # initialize lists to keep track of individual assets throughout the episode
            cur_states = [[] for _ in range(NUM_ASSETS)]
            cur_policies = [[] for _ in range(NUM_ASSETS)]
            cur_values = [[] for _ in range(NUM_ASSETS)]
            cur_goals = [[] for _ in range(NUM_ASSETS)]
            cur_actions = [[] for _ in range(NUM_ASSETS)]
            cur_valids = [[] for _ in range(NUM_ASSETS)]
            cur_rewards = [[] for _ in range(NUM_ASSETS)]
            masks = []
            cur_advantages = [[] for _ in range(NUM_ASSETS)]
            cur_returns = [[] for _ in range(NUM_ASSETS)]


            while not done and step < PPO_STEPS:
                old_positions = asset_positions.copy()

                state_list, buckets = bucket_and_state_maker(asset_positions, allies, normal_states, goals)

                for asset, (i,j) in enumerate(asset_positions):

                    # create state/goal
                    # breakpoint()
                    state = state_list[asset]
                    goal = np.array([goals[asset][0]-i, goals[asset][1]-j]).reshape((1,2))

                    # call the model to obtain an action
                    if episode == 0 and step == 0:
                        model.call((state, goal))
                        if CONTINUE_TRAINING:
                            model.load_weights('my_weights')
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
                    # store state, value, goal, action, valids
                    cur_states[asset].append(state)
                    cur_values[asset].append(value)
                    cur_goals[asset].append(goal)
                    cur_actions[asset].append(action)
                    cur_valids[asset].append(valid)


                # make all actions
                for asset in range(NUM_ASSETS):
                    action = cur_actions[asset][-1][0]
                    if cur_valids[asset][-1][action]: # if a legal move
                        if cur_actions[asset][-1] != 4: # if it's 4 we don't move
                            asset_positions[asset] = (asset_positions[asset][0]+MOVES[action][0], asset_positions[asset][1]+MOVES[action][1])

                # update global_state, check if done
                global_state = space_to_global_state(normal_states.copy(), asset_positions)
                if asset_positions == goals:
                    done = True
                masks.append(1-done)

                # find collisions
                collision_list = find_collisions(asset_positions, old_positions)

                # compute and store rewards
                rewards_list = get_reward(asset_positions, cost_mat, cur_valids, cur_actions, goals, collision_list, done, step)
                for asset in range(NUM_ASSETS):
                    cur_rewards[asset].append(rewards_list[asset])

                step += 1

            for asset, (i,j) in enumerate(asset_positions):
                new_state = state_maker(i, j, goals[asset], global_state)
                new_goal = np.array([[goals[asset][0]-i, goals[asset][1]-j]])
                _, next_value = model((new_state, new_goal))
                next_values[asset] = next_value

            returns = np.array([compute_gae(next_values[asset], cur_rewards[asset], masks, cur_values[asset]) for asset in range(NUM_ASSETS)])
            advantages = normalize(returns-cur_values)

            cur_states = flatten(cur_states)
            returns = flatten(returns)
            advantages = flatten(advantages)
            cur_valids = flatten(cur_valids)
            cur_goals = flatten(cur_goals)

            model.train(cur_states, returns, advantages, cur_valids, cur_goals)

        ##########################################################################################################################################

        else:

            # try:
            asset_positions, goals, normal_states, cost_mat = m_star_primal(solve=False, show=False)
            global_state = space_to_global_state(normal_states.copy(), asset_positions)
            allies = []
            for i in range(WIDTH):
                for j in range(HEIGHT):
                    if global_state[i, j] == 1:
                        allies.append((i, j))

            optimal_paths = scuffed_a_star(asset_positions, goals, normal_states, cost_mat, allies)
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
            cur_actions = [[] for _ in range(NUM_ASSETS)]
            cur_valids = [[] for _ in range(NUM_ASSETS)]
            cur_rewards = [[] for _ in range(NUM_ASSETS)]
            masks = []
            cur_advantages = [[] for _ in range(NUM_ASSETS)]
            cur_returns = [[] for _ in range(NUM_ASSETS)]


            while not done:
                state_list, buckets = bucket_and_state_maker(asset_position, allies, normal_states, goals)
                for asset, (i,j) in enumerate(asset_positions):

                    # create state/goal
                    state = state_list[asset]
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
                    cur_actions[asset].append(action)
                    cur_valids[asset].append(valid)


                # make all actions
                for asset in range(NUM_ASSETS):
                    action = cur_actions[asset][-1]
                    if action != 4: # if it's 4 we don't move
                        asset_positions[asset] = (asset_positions[asset][0]+MOVES[action][0], asset_positions[asset][1]+MOVES[action][1])

                # update global_state, check if done
                global_state = space_to_global_state(normal_states.copy(), asset_positions)
                if asset_positions == goals:
                    done = True
                masks.append(1-done)


            # find collisions
            collision_list = [False for _ in assets]

            # compute and store rewards
            rewards_list = get_reward(asset_positions, cost_mat, cur_valids, cur_actions, goals, collision_list, done, step)
            for asset in range(NUM_ASSETS):
                cur_rewards[asset].append(rewards_list[asset])

            step += 1

            # get next_values
            for asset, (i,j) in enumerate(asset_positions):
                new_state = state_maker(i, j, goals[asset], global_state)
                new_goal = np.array([[goals[asset][0]-i, goals[asset][1]-j]])
                _, next_value = model((new_state, new_goal))
                next_values[asset] = next_value

            returns = flatten([compute_gae(next_values[asset], cur_rewards[asset], masks, cur_values[asset]) for asset in range(NUM_ASSETS)])
            cur_states = flatten(cur_states)
            cur_goals = flatten(cur_goals)
            cur_actions = flatten(cur_actions)

            model.train_imitation(cur_states, cur_goals, cur_actions, returns)

            # except:
            #     print('oh no')
            #     breakpoint()
            #     continue

        if episode%10 == 0:
            print('Saving...')
            model.save_weights('/saved_weights/low_comm_weights_{}'.format(episode+1))
            print('Testing...')
            tester(model, '/saved_weights/low_comm_weights_{}'.format(episode+1))
            print('Finished Testing!')
        print('Epsiode {} is finished!'.format(episode+1))
