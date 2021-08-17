from pygame.event import get
from pygame.mouse import get_pos, get_pressed
from pygame.display import flip
from pygame.locals import K_SPACE, K_ESCAPE, KEYDOWN, QUIT, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, K_9, K_r, K_q
from numpy.random import randint, uniform, shuffle
from numpy import infty, array, unique, argmin, zeros
from math import sqrt
from queue import PriorityQueue
import constants
from constants import *


def dist(loc1, loc2):
    distance = ((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)**.5
    return(distance)

def locationtuple_to_index(pos):
    i,j = pos
    index = i * WIDTH + j
    return index

def index_to_locationtuple(index):
    return divmod(index, WIDTH)

def reconstruct_path(cur_index, came_from):
        total_path = [cur_index]
        while came_from[cur_index][0] != None:
            cur_index = came_from[cur_index][0]
            total_path.insert(0, cur_index)
        pos_path = [index_to_locationtuple(i) for i in total_path]
        return pos_path



def neighbor_getter(index, future_locs, previous_locs, failed = False):
    legal_indices = []
    for pos in MOVES:
        cur_loctup = index_to_locationtuple(index)
        new_loctup = (cur_loctup[0]+pos[0], cur_loctup[1]+pos[1])
        new_index = locationtuple_to_index(new_loctup)
        valid = False
        if 0<=new_loctup[0]<WIDTH and 0<=new_loctup[1]<HEIGHT:
            # not a physical ally or enemy
            if constants.SPACE[new_loctup[0], new_loctup[1]].state not in [1,3]:
                # verify no asset is at the new location
                if new_loctup not in future_locs:
                    valid = True
                    # check to see if the asset who left our target square
                    # moved to our current location.  This would be a pass
                    # through collision
                    if new_loctup in previous_locs:
                        if previous_locs[new_loctup] == cur_loctup:
                            print("Pass through collision from {0} to {1}".format(cur_loctup, new_loctup))
                            valid = False
                elif failed == True:
                    print("Head on collision from {0} to {1}".format(cur_loctup, new_loctup))
            elif failed == True:
                print("Ally or enemy from {0} to {1}".format(cur_loctup, new_loctup))
                
        elif failed == True:
            print("Edge of board from {0} to {1}".format(cur_loctup, new_loctup))
                
        if valid:
            legal_indices.append(new_index)
    return legal_indices

def neighbor_getter_best(index, cost_matrix, g_score, goal):
    h = lambda cur_pos: (abs(goal[0]-cur_pos[0])**2 + abs(goal[1]-cur_pos[1])**2)**0.5
    best_move = []
    best_move_g_score = 10000
    for pos in MOVES:
        cur_loctup = index_to_locationtuple(index)
        new_loctup = (cur_loctup[0]+pos[0], cur_loctup[1]+pos[1])
        new_index = locationtuple_to_index(new_loctup)
        valid = False
        if 0<=new_loctup[0]<WIDTH and 0<=new_loctup[1]<HEIGHT:
            # not a physical ally or enemy
            if constants.SPACE[new_loctup[0], new_loctup[1]].state not in [1,3]:
                # verify no asset is at the new location
                temp_f_score = g_score[index] + cost_matrix[index, new_index]
                if temp_g_score < best_move_g_score:
                    best_move = new_index
                    best_move_g_score = temp_g_score
                
    return best_move, best_move_g_score

#def neighbor_getter_MSTAR(index)


# A_Star is an algorithm based on Dijkstra's Algorithm with a heuristic that
# tells us how to choose which node in the open set is most worth checking by
# utilizing a function that guesses how far away from the end each node is
# in order to do a "best-first" search
# https://en.wikipedia.org/wiki/A*_search_algorithm
def A_Star(cost_matrix, area, start, goal, optimal_paths, Dijkstra=True):

    h = lambda cur_pos: (abs(goal[0]-cur_pos[0])**2 + abs(goal[1]-cur_pos[1])**2)**0.5

    count = 0
    start_index = locationtuple_to_index(start)

    open_set = PriorityQueue()
    if Dijkstra:
        open_set.put((0, count, locationtuple_to_index(start)))
    else:
        open_set.put((0, -count, locationtuple_to_index(start)))

    in_open_set = {j:False for j in range(TOTAL_SQUARES)}
    in_open_set[start_index] = True

    # [last_index, count, cur_timestep]
    came_from = {j:[None, 0, 0] for j in range(TOTAL_SQUARES)}

    g_score = {j:infty for j in range(TOTAL_SQUARES)}
    g_score[start_index] = 0

    f_score = {j:0 for j in range(TOTAL_SQUARES)}
    f_score[start_index] = h(start)

    prev_index = -1

    running = True
    solving = True
    solved = False
    while running:
        if solving:
            if open_set.empty():
                print('cry')
                print('we should never see this as there are practically no *actual* obstacles, hence why we cry')
                raise ValueError

            _, _, cur_index = open_set.get()
            cur_timestep = came_from[cur_index][2]
            current = index_to_locationtuple(cur_index)
            in_open_set[cur_index] = False

            if current == goal:
                move_list = reconstruct_path(cur_index, came_from)
                return move_list

            neighbors = neighbor_getter(cur_index, prev_index, cur_timestep, area)
            shuffle(neighbors)
            for neighbor in neighbors:
                temp_g_score = g_score[cur_index] + cost_matrix[cur_index, neighbor]

                if temp_g_score < g_score[neighbor]:
                    came_from[neighbor] = [cur_index, came_from[cur_index][1]+1, cur_timestep+1]
                    g_score[neighbor] = temp_g_score
                    f_score[neighbor] = temp_g_score + h(index_to_locationtuple(neighbor))
                    if not in_open_set[neighbor]:
                        count += 1
                        if Dijkstra:
                            open_set.put((f_score[neighbor], count, neighbor))
                        else:
                            open_set.put((f_score[neighbor], -count, neighbor))
                        in_open_set[neighbor] = True
            prev_index = cur_index



# low comms state maker
def bucket_maker(assets, allies, goals):

    # find which assets and allies can communicate
    all_good_guys = [i for i,x in enumerate(assets + allies)]
    assets = [tuple(asset) for asset in assets]
    comms_dict = {asset:set([asset]) for asset in all_good_guys}
    for asset in all_good_guys:
        for asset2 in all_good_guys:
            if asset >= NUM_ASSETS:
                asset_loc = allies[asset - NUM_ASSETS]
            else:
                asset_loc = assets[asset]
            if asset2 >= NUM_ASSETS:
                asset2_loc = allies[asset2 - NUM_ASSETS]
            else:
                asset2_loc = assets[asset2]
            if dist(asset_loc, asset2_loc) <= COMM_RANGE:
                temp_union = comms_dict[asset].union(comms_dict[asset2])
                for i in temp_union:
                    comms_dict[i] = temp_union

    # create the buckets
    buckets = []
    for asset in all_good_guys:
        cur_set = comms_dict[asset]
        if cur_set not in buckets:
            buckets.append(cur_set)

    return buckets

def map_maker(buckets, assets, allies, enemies):
    # create a state for each bucket
    blank_state = array([[-1 for j in range(HEIGHT)] for i in range(WIDTH)])
    for asset in assets:
        blank_state[asset[0], asset[1]] = 8
    for ally in allies:
        blank_state[ally[0], ally[1]] = 1
    for enemy in enemies:
        blank_state[enemy[0], enemy[1]] = 3
    asset_states = [0 for asset in range(NUM_ASSETS)]
    for bucket in buckets:
        cur_state = blank_state.copy()
        for asset in bucket:
            if asset < NUM_ASSETS:
                cur_loc = assets[asset]
            else:
                cur_loc = allies[asset - NUM_ASSETS]
            for i in range(max(0, cur_loc[0]-COMM_RANGE), min(WIDTH-1, cur_loc[0]+COMM_RANGE)):
                for j in range(max(0, cur_loc[1]-COMM_RANGE), min(HEIGHT-1, cur_loc[1]+COMM_RANGE)):
                    cur_state[i][j] = constants.NORMAL_STATES[i][j]
                    
        for asset in bucket:
            if asset < NUM_ASSETS:
                #idx = assets.index(asset)
                asset_states[asset] = cur_state#state_maker(assets[asset][0], assets[asset][1], goals[asset], cur_state)
                
    return(asset_states)
    
def cost_matrix_maker(maps):
    cost_mats = []
    for cur_map in maps:
        
        cost_mat = zeros((TOTAL_SQUARES, TOTAL_SQUARES))
    
        for i in range(0, WIDTH):
            for j in range(0, HEIGHT):
                index_from = locationtuple_to_index((i, j))
                for m in MOVES:
                    index_to = locationtuple_to_index((i + m[0], j + m[1]))
                    try:
                        if cur_map[i, j] == -1 or cur_map[i + m[0], j + m[1]] == -1:
                            cost_mat[index_from, index_to] = move_cost + enemy_influence_change
                        elif constants.SPACE[i + m[0], j + m[1]].state in [0]:
                            cost_mat[index_from, index_to] = move_cost
                        elif constants.SPACE[i + m[0], j + m[1]].state in [1, 5]:
                            cost_mat[index_from, index_to] = move_cost + ally_influence_change
                        elif constants.SPACE[i + m[0], j + m[1]].state in [3, 4]:
                            cost_mat[index_from, index_to] = move_cost + enemy_influence_change
                        elif constants.SPACE[i + m[0], j + m[1]].state in [9]:
                            cost_mat[index_from, index_to] = move_cost-0.1
                        elif constants.SPACE[i + m[0], j + m[1]].state == 7:
                            cost_mat[index_from, index_to] = move_cost + mixed_influence_change
                    except:
                        continue
                    
        cost_mats.append(cost_mat)
        
    return(cost_mats)
    
    
def A_Star_Varried_Comms(cost_matrix, start, goal, future_locations, previous_locations, starting_timestep, failed = False, idx = -1):
    h = lambda cur_pos: (abs(goal[0]-cur_pos[0])**2 + abs(goal[1]-cur_pos[1])**2)**0.5
    
    count = 0
    start_index = locationtuple_to_index(start)
    
    open_set = PriorityQueue()
    
    in_open_set = {j:False for j in range(TOTAL_SQUARES)}
    in_open_set[start_index] = True
    
    came_from = {j:[None, starting_timestep] for j in range(TOTAL_SQUARES)}
    
    g_score = {j:infty for j in range(TOTAL_SQUARES)}
    g_score[start_index] = 0
    
    f_score = {j:0 for j in range(TOTAL_SQUARES)}
    f_score[start_index] = h(start)
    
    prev_index = -1
    
    open_set.put((0, start_index))
    
    solving = True
    
    while solving:
        if open_set.empty():
            return(False)
            
        cur_f_score, cur_index = open_set.get()
        cur_timestep = came_from[cur_index][1]
        cur_loc = index_to_locationtuple(cur_index)
        in_open_set[cur_index] = False
        
        if cur_loc == goal:
            move_list = reconstruct_path(cur_index, came_from)
            return(move_list)
        
        try:
            neighbors = neighbor_getter(cur_index, future_locations[cur_timestep], previous_locations[cur_timestep], failed = failed)
        except:
            neighbors = neighbor_getter(cur_index, dict(), dict(), failed = failed)
            
        shuffle(neighbors)
        
        for neighbor in neighbors:
            temp_g_score = g_score[cur_index] + cost_matrix[cur_index, neighbor]

            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = [cur_index, cur_timestep+1]
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + h(index_to_locationtuple(neighbor))
                if not in_open_set[neighbor]:
                    count += 1
                    open_set.put((f_score[neighbor], neighbor))
                    in_open_set[neighbor] = True



def M_Star_Varied_Comms(cost_matrix, assets, allies, goals, enemies):
    # group close enough allies
    buckets = bucket_maker(assets, allies, goals)
    
    # update knowledge
    asset_maps = map_maker(buckets, assets, allies, enemies)
    
    # cost_matrices
    cost_mats = cost_matrix_maker(asset_maps)
    
    # will store where assets are moving to 
    future_locations = []
    
    # will store where assets came from
    previous_locations = []
    
    # store solutions
    optimal_paths = [[asset] for asset in assets]
    
    solved = False
    
    timestep = 0
    
    while not solved:
        for idx, asset in enumerate(assets):
            optimal_path = optimal_paths[idx]
            if optimal_path[-1] == goals[idx]:
                continue
            
            path = A_Star_Varried_Comms(cost_mats[idx], optimal_path[-1] , goals[idx], future_locations, previous_locations, timestep)
            delay = 0
            
            while(path == False and delay <= 100):
                delay += 1
                print("Delaying for {}".format(delay))
                path = A_Star_Varried_Comms(cost_mats[idx], optimal_path[-1] , goals[idx], future_locations, previous_locations, timestep + delay)
                
            
            if delay >= 100:
                path = A_Star_Varried_Comms(cost_mats[idx], optimal_path[-1] , goals[idx], dict(), dict(), 0)
                if path == False:
                    print("No path found")
                
                print("Infinite")
                raise ValueError
                
            for d in range(delay):
                path.inset(0, optimal_path[-1])
            
            path_len = len(path)
            
            #try:
            #    path = A_Star_Varried_Comms(cost_mats[idx], optimal_path[-1] , goals[idx], future_locations, previous_locations, timestep)
            #except:
            #    loc_index = optimal_path[-1]
            #    optimal_path.append(loc_index)
            #    future_locations.append(dict())
            #    previous_locations.append(dict())
            #    print("Timestep is {}".format(timestep))
            #    print("Length is {}".format(len(future_locations)))
            #    future_locations[timestep + 1][loc_index] = 1
            #    previous_locations[timestep + 1][loc_index] = loc_index
                    
            #    continue
            
            
            
            # update based on either path len (if it reaches the goal in less than COMM_RANGE)
            # or COMM_RANGE
            
            for step, loc_index in enumerate(path[:min(path_len, COMM_RANGE)]):
                # make sure future_locations is long enough to put locations in to
                if len(future_locations) <= step + timestep:
                    future_locations.append(dict())
                    previous_locations.append(dict())
                if step + timestep == 0:
                    continue
                future_locations[step + timestep][loc_index] = 1
                previous_locations[step + timestep][loc_index] = optimal_path[-1]
                optimal_path.append(loc_index)
                
            if path_len < COMM_RANGE:
                for step in range(COMM_RANGE - path_len):
                    if (len(future_locations)) <= step + timestep + path_len:
                        future_locations.append(dict())
                        previous_locations.append(dict())
                    future_locations[step + timestep + path_len][optimal_path[-1]] = 1
                    previous_locations[step + timestep][optimal_path[-1]] = optimal_path[-1]
                    
        solved = True            
        for idx, asset in enumerate(assets):
            optimal_path = optimal_paths[idx]
            if optimal_path[-1] == goals[idx]:
                continue
            else:
                solved = False
                break
            
        buckets = bucket_maker(assets, allies, goals)
        asset_maps = map_maker(buckets, assets, allies, enemies)
        cost_mats = cost_matrix_maker(asset_maps)
        timestep += COMM_RANGE
        
    return(optimal_paths)


def Actual_MSTAR_Varried_Comms(cost_matrix, assets, allies, goals):
    # create bucketed vision
    # iterate through each timestep
    # on each timestep find the best next move for each asset
    # unless they have previously collided, then find all moves
    # check for collisions
    # add collided assets to the collision list, go back, and re-expand
    
    visited = dict()
    
    visited[allies] = dict()
    visited[allies]['cost'] = 0
    visited[allies]['back_set'] = []
    visited[allies]['collision_set'] = set()
    visited[allies]['back_ptr'] == None
    
    
    
    
    start = allies
    
    
    
    open_set = PriorityQueue()
    open_set.put((visited[allies]['cost'], allies))
    
    while open_set.empty() == False:
        _,current = open_set.get()
        
        if current == goals:
            return(reconstruct_path(current))
        
        for neighbor in neighbor_getter_MSTAR(current):
            if neighbor not in visited:
                visited[neighbor] = dict()
                visited[neighbor]['back_set'] = []
                visited[neighbor]['cost'] = infty
                visited[neighbor]['collision_set'] = set()
                
            visited[neighbor]['back_set'].append(current)
            visited[neighbor]['collision_set'].union(collision_detector(neighbor))
            back_prop(current, visited[neighbor]['collision_set'], open_set)
            
            move_cost = visited[current]['cost'] + f(visited[current], visited[neighbor])
            if visited[neighbor]['collision_set'] == set() and move_cost < visited[neighbor]['cost']:
                visited[neighbor]['cost'] = move_cost
                visited[neighbor]['back_ptr'] = current
                open_set.put(move_cost, neighbor)
    
    
        
    
        
    
                    
                    
                    
                
                
                
    
    


