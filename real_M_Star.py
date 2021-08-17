from pygame.event import get
from pygame.mouse import get_pos, get_pressed
from pygame.display import flip
from pygame.locals import K_SPACE, K_ESCAPE, KEYDOWN, QUIT, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, K_9, K_r, K_q
from numpy.random import randint, uniform, shuffle
from numpy import infty, array, unique, argmin, zeros
from math import sqrt
from queue import PriorityQueue
from real_constants import *

def state_to_index(pos):
    i,j = pos
    index = i * WIDTH + j
    return index

def index_to_state(index):
    return divmod(index, WIDTH)


# A_Star with multiple dimensions
def N_dim_A_Star(cost_matrix, area, starts, goals, Dijkstra=False):

    def reconstruct_path(cur_index, came_from):
        total_path = [cur_index]
        while came_from[cur_index][0] != None:
            cur_index = came_from[cur_index][0]
            total_path.insert(0, cur_index)
        pos_path = [index_to_state(i) for i in total_path]
        return pos_path

    def neighbor_getter(index, last_index):
        legal_indices = []
        for pos in MOVES:
            cur_state = index_to_state(index)
            new_state = (cur_state[0]+pos[0], cur_state[1]+pos[1])
            new_index = state_to_index(new_state)
            if new_index != last_index and 0<=new_state[0]<WIDTH and 0<=new_state[1]<HEIGHT:
                    if area[new_state[0], new_state[1]].state not in [1,3]:
                        legal_indices.append(new_index)
        return legal_indices

    h = lambda cur_pos: (abs(goal[0]-cur_pos[0])**2 + abs(goal[1]-cur_pos[1])**2)**0.5

    count = 0
    start_index = [state_to_index(start) for start in starts]

    open_set = PriorityQueue()
    if Dijkstra:
        open_set.put((0, count, ([state_to_index(start) for start in starts], 0)))
    else:
        open_set.put((0, -count, ([state_to_index(start) for start in starts], 0)))

    in_open_set = zeros(shape=[TOTAL_SQUARES]*len(starts))
    in_open_set[start_index] = True

    came_from = [[None, 0] for j in range(TOTAL_SQUARES)]

    g_score = [infty for j in range(TOTAL_SQUARES)]
    g_score[start_index] = 0

    f_score = [0 for j in range(TOTAL_SQUARES)]
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
                breakpoint()
                return False

            _, _, cur_index = open_set.get()
            current, time_step = index_to_state(cur_index)
            in_open_set[cur_index] = False

            if current == goal:
                move_list = reconstruct_path(cur_index, came_from)
                return move_list

            neighbors = neighbor_getter(cur_index, prev_index)
            shuffle(neighbors)
            for neighbor in neighbors:
                temp_g_score = g_score[cur_index] + cost_matrix[cur_index, neighbor]

                if temp_g_score < g_score[neighbor]:
                    came_from[neighbor] = [cur_index, came_from[cur_index][1]+1]
                    g_score[neighbor] = temp_g_score
                    f_score[neighbor] = temp_g_score + h(index_to_state(neighbor))
                    if not in_open_set[neighbor]:
                        count += 1
                        if Dijkstra:
                            open_set.put((f_score[neighbor], count, neighbor))
                        else:
                            open_set.put((f_score[neighbor], -count, neighbor))
                        in_open_set[neighbor] = True
            prev_index = cur_index


# A_Star is an algorithm based on Dijkstra's Algorithm with a heuristic that
# tells us how to choose which node in the open set is most worth checking by
# utilizing a function that guesses how far away from the end each node is
# in order to do a "best-first" search
# https://en.wikipedia.org/wiki/A*_search_algorithm
def A_Star(cost_matrix, area, start, goal, Dijkstra=False):

    def reconstruct_path(cur_index, came_from):
        total_path = [cur_index]
        while came_from[cur_index][0] != None:
            cur_index = came_from[cur_index][0]
            total_path.insert(0, cur_index)
        pos_path = [index_to_state(i) for i in total_path]
        return pos_path

    def neighbor_getter(index, last_index):
        legal_indices = []
        for pos in MOVES:
            cur_state = index_to_state(index)
            new_state = (cur_state[0]+pos[0], cur_state[1]+pos[1])
            new_index = state_to_index(new_state)
            if new_index != last_index and 0<=new_state[0]<WIDTH and 0<=new_state[1]<HEIGHT:
                    if area[new_state[0], new_state[1]].state not in [1,3]:
                        legal_indices.append(new_index)
        return legal_indices

    h = lambda cur_pos: (abs(goal[0]-cur_pos[0])**2 + abs(goal[1]-cur_pos[1])**2)**0.5

    count = 0
    start_index = state_to_index(start)

    open_set = PriorityQueue()
    if Dijkstra:
        open_set.put((0, count, state_to_index(start)))
    else:
        open_set.put((0, -count, state_to_index(start)))

    in_open_set = [False for j in range(TOTAL_SQUARES)]
    in_open_set[start_index] = True

    came_from = [[None, 0] for j in range(TOTAL_SQUARES)]

    g_score = [infty for j in range(TOTAL_SQUARES)]
    g_score[start_index] = 0

    f_score = [0 for j in range(TOTAL_SQUARES)]
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
                breakpoint()
                return False

            _, _, cur_index = open_set.get()
            current = index_to_state(cur_index)
            in_open_set[cur_index] = False

            if current == goal:
                move_list = reconstruct_path(cur_index, came_from)
                return move_list

            neighbors = neighbor_getter(cur_index, prev_index)
            shuffle(neighbors)
            for neighbor in neighbors:
                temp_g_score = g_score[cur_index] + cost_matrix[cur_index, neighbor]

                if temp_g_score < g_score[neighbor]:
                    came_from[neighbor] = [cur_index, came_from[cur_index][1]+1]
                    g_score[neighbor] = temp_g_score
                    f_score[neighbor] = temp_g_score + h(index_to_state(neighbor))
                    if not in_open_set[neighbor]:
                        count += 1
                        if Dijkstra:
                            open_set.put((f_score[neighbor], count, neighbor))
                        else:
                            open_set.put((f_score[neighbor], -count, neighbor))
                        in_open_set[neighbor] = True
            prev_index = cur_index
