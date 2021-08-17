from main import *
from RL_Stuff import *
from constants import *
from numpy import mean
from numba import njit
from supervised import space_to_global_state, my_subfunction, state_maker

def tester(model, model_name, num_tests=20):

    print('Testing model {}...'.format(model_name.split('_')[-1]))

    TOTAL_COST = [0 for i in range(num_tests)]
    l1_norm = [0 for i in range(num_tests)]


    for trial in range(num_tests):

        CUR_COST = 0
        working = True

        while working:
            try:
                optimal_paths, asset_positions, goals, normal_states, cost_mat = m_star_primal(solve=True, show=False)
                global_state = space_to_global_state(normal_states.copy(), asset_positions)
                working = False
            except:
                pass

        best_cost = 0
        for my_list in optimal_paths:
            for t, (i,j) in enumerate(my_list):
                if t > 0:
                    best_cost += cost_mat[i][j]

        done = False
        step = 0


        while not done and step < PPO_STEPS:
            for asset, (i,j) in enumerate(asset_positions):

                # create state/goal
                state = state_maker(i, j, goals[asset], global_state)
                goal = np.array([goals[asset][0]-i, goals[asset][1]-j]).reshape((1,2))

                # call the model to obtain an action
                if step == 0:
                    model.call((state, goal))
                # policy, value, goal_guess = model((state, goal))
                policy, value = model((state, goal))
                # action = model.sample(policy)[0]
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
                    else:
                        CUR_COST += invalid_move


            # update global_state, check if done
            global_state = space_to_global_state(normal_states.copy(), asset_positions)
            #visuals here?
            #save asset locations?
            #not sure best way to do this part
            if asset_positions == goals:
                done = True

            for asset in range(NUM_ASSETS):
                CUR_COST += cost_mat[asset_positions[asset][0]][asset_positions[asset][1]]

            step += 1

            for event in get():
                if event.type == QUIT:
                    quit()
                    running = False
                    break



        for asset in range(NUM_ASSETS):
            l1_norm[trial] += abs(asset_positions[asset][0]-goals[asset][0]) + abs(asset_positions[asset][1]-goals[asset][1])

        TOTAL_COST[trial] = round(CUR_COST/best_cost, 4)

    print('Saving results...')

    string1 = '\nThe total costs beyond optimality over the trials were: \n'
    for cost in TOTAL_COST:
        string1 += str(cost) + '\n'

    string2 = '\nThe distances from goals over the trials were:\n'
    for cost in l1_norm:
        string2 += str(cost) + '\n'

    file1 = open('{}.txt'.format(model_name), 'w')
    file1.write('The average cost beyond optimality was {} across {} trials.'.format(round(np.mean(TOTAL_COST), 2), num_tests))
    file1.writelines([string1, string2])
    file1.close()




if __name__ == '__main__':
    model = ActorCriticModel(state_shape, goal_shape, num_actions)
    model.load_weights('saved_weights/my_weights_1')
    tester(model, 'saved_weights/my_weights_1')

    i = 1
    while True:
        model.load_weights('saved_weights/my_weights_{}1'.format(i))
        tester(model, 'saved_weights/my_weights_{}1'.format(i))
        i+=1
