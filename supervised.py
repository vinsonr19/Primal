from main import *
import numpy as np
from pixel_class import *
from constants import *
from M_Star import *
from RL_Stuff import *
from collections import deque
import os

import tensorflow as tf
# import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers
from numba import njit

njit()
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


@njit()
def rewards_getter(valid, goal, action):
    reward = 0
    stay_still = num_actions - 1
    stay_still_penalty = .5
    move_penalty = .3
    illegal_action_penalty = 2
    
    if action == stay_still:
        if goal == (0,0):
            return(reward)
        else:
            reward -= stay_still_penalty
            return(reward)

    else:
        reward -= move_penalty

        if valid == 0:
            reward -= illegal_action_penalty

    return(reward)


def space_to_global_state(area, positions):
    global_state = area.copy()
    for i,j in positions:
        global_state[i,j] = 8
    return global_state


# 0:out of bounds, 1: empty space, 2:other assets, 3:allies, 4:bad guys, 5:other goal, 6:our goal
@njit()
def my_subfunction(pos_x, pos_y, goal, global_state):
    space_near_asset = np.zeros(shape=(state_shape[0], state_shape[1]))
    for x in range(0,WIDTH):
        for y in range(0,HEIGHT):
            space_state = global_state[x,y]
            if space_state in [0,4,5,7]: # assume we don't know influence areas right now (for the 2nd part we're always in middle so who cares)
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
    space_near_asset = my_subfunction(pos_x, pos_y, goal, global_state)
    state = np.array([[[1 if space_near_asset[j,k] == i else 0 for k in range(state_shape[0])] for j in range(state_shape[1])] for i in range(state_shape[2])]).reshape((1, *state_shape)).astype('float32')
    return state


def create_supervised_nn():
    state_input = keras.Input(shape=state_shape, name='state')
    goal_input = keras.Input(shape=goal_shape, name='goal')

    # fancy input stuff
    convlayer1 = layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same', activation='swish')(state_input)
    convlayer2 = layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same', activation='swish')(convlayer1)
    pool1 = layers.MaxPooling2D(pool_size=(2,2))(convlayer2)
    convlayer3 = layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same', activation='swish')(pool1)
    convlayer4 = layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', activation='swish')(convlayer3)
    pool2 = layers.MaxPooling2D(pool_size=(2,2))(convlayer4)
    convlayer5 = layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', activation='swish')(pool2)
    flattenlayer = layers.Flatten()(convlayer5)

    dropout = layers.Dropout(0.1)(flattenlayer) # get rid of info to speed up our lives! and overfitting blah blah blah

    # goal input
    flat_goal = layers.Flatten()(goal_input)
    goal_layer1 = layers.Dense(12, activation='swish')(flat_goal)
    goal_layer2 = layers.Dense(12, activation='swish')(goal_layer1)

    concat = layers.concatenate([dropout, goal_layer2])

    # post concatenation
    dense1 = layers.Dense(128, activation='swish')(concat)
    dense2 = layers.Dense(128, activation='swish')(dense1)
    # LSTM = layers.LSTM(128, activation='swish', recurrent_activation='swish', return_sequences=False)
    dense3 = layers.Dense(128, activation='swish')(dense2)

    # policy output
    policy_dense1 = layers.Dense(128, activation='swish')(dense3)
    policy_dense2 = layers.Dense(64, activation='swish')(policy_dense1)
    policy_output = layers.Dense(num_actions, activation='softmax', name='act')(policy_dense2)

    model = tf.keras.Model(inputs=[state_input, goal_input], outputs=[policy_output])

    model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    # keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)
    # model.summary()

    return model


def show_off_supervised():

    model = create_supervised_nn()
    model.load_weights('saved_weights/supervised_weights_141')

    # define our space
    adj_mat, assets, goals, enemies, allies, cost_mat, _ = setup_space()

    for i,j in assets:
        SPACE[i,j].change_state(8)

    flip()

    step = 0

    asset_positions = assets
    done = False
    if asset_positions == goals:
        done = True

    global_state = space_to_global_state(normal_states.copy(), asset_positions)

    while not done:
        for asset, (i,j) in enumerate(asset_positions):

            # create state/goal
            state = state_maker(i, j, goals[asset], global_state)
            goal = np.array([goals[asset][0]-i, goals[asset][1]-j]).reshape((1,2))

            # call the model to obtain an action
            if step == 0:
                model((state, goal))
            policy = model((state, goal))
            # print(policy)
            # action = model.sample(policy)
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



        # update global_state, check if done
        global_state = space_to_global_state(normal_states.copy(), asset_positions)
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
        delay(100)
        flip()



    delay(5000)
    print('This took {} steps to finish!'.format(step))
    return



if __name__ == '__main__':

    CONTINUE_TRAINING = True
    START_EPISODE = 1

    cwd = os.getcwd()
    try:
        os.chdir(cwd + '\\saved_weights')
    except:
        os.mkdir(cwd + '\\saved_weights')
        os.chdir(cwd + '\\saved_weights')
    cwd = os.getcwd()


    model = create_supervised_nn()


    for episode in range(TRAINING_EPISODES):

        try:
            optimal_paths, asset_positions, goals, normal_states, cost_mat = m_star_primal(solve=True, show=False)
            global_state = space_to_global_state(normal_states.copy(), asset_positions)

            beginning_positions = asset_positions.copy()

            correct_actions = path_to_actions(optimal_paths.copy())

            done = False
            step = 0
            next_values = [0 for _ in range(NUM_ASSETS)]

            # initialize lists to keep track of individual assets throughout the episode
            cur_states = [[] for _ in range(NUM_ASSETS)]
            cur_actions = [[] for _ in range(NUM_ASSETS)]
            cur_goals = [[] for _ in range(NUM_ASSETS)]

            while not done:

                for asset, (i,j) in enumerate(asset_positions):

                    # create state/goal
                    state = state_maker(i, j, goals[asset], global_state)
                    goal = np.array([goals[asset][0]-i, goals[asset][1]-j]).reshape((2,1))

                    # call the model to obtain the model's chosen policy/value
                    if episode == 0 and step == 0 and CONTINUE_TRAINING:
                        model.load_weights('supervised_weights')

                    try:
                        action = correct_actions[asset][step]
                    except:
                        if asset_positions[asset] == goals[asset]:
                            action = 4
                        elif abs(asset_positions[asset][0]-goals[asset][0]) + abs(asset_positions[asset][1]-goals[asset][1]) > 1:
                            print(asset)
                            breakpoint()


                    stored_action = np.zeros((5))
                    stored_action[action] = 1

                    if stored_action[4] == 1:
                        if random.rand() <= 0.3:
                            cur_states[asset].append(state)
                            cur_goals[asset].append(goal)
                            cur_actions[asset].append(stored_action)
                    else:
                        cur_states[asset].append(state)
                        cur_goals[asset].append(goal)
                        cur_actions[asset].append(stored_action)


                # make all actions
                for asset in range(NUM_ASSETS):
                    action = cur_actions[asset][-1]
                    if action[4] != 1: # if it's 4 we don't move
                        asset_positions[asset] = (asset_positions[asset][0]+MOVES[np.argmax(action)][0], asset_positions[asset][1]+MOVES[np.argmax(action)][1])

                # update global_state, check if done
                global_state = space_to_global_state(normal_states.copy(), asset_positions)
                if asset_positions == goals:
                    done = True


                step += 1


        except:
            continue


        train_states = np.array([item.reshape(30,30,7) for sublist in cur_states for item in sublist])
        train_actions = np.array([item for sublist in cur_actions for item in sublist])
        train_goals = np.array([item for sublist in cur_goals for item in sublist])

        model.fit({'state':train_states, 'goal':train_goals}, {'act':train_actions}, epochs=3, verbose=0, shuffle=True)

        if episode % 10 == 0:
            print('Saving...')
            model.save_weights('supervised_weights_{}'.format(episode+START_EPISODE))
        print('Episode {} is finished!'.format(episode+1))
