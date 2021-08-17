from pygame.event import get
from pygame.mouse import get_pos, get_pressed
from pygame.display import flip
from pygame.locals import K_SPACE, K_ESCAPE, KEYDOWN, QUIT, K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, K_9, K_r, K_q
from numpy.random import randint
from numpy import infty, array, unique
from math import sqrt
from queue import PriorityQueue
from constants import *

def state_to_index(pos):
    i,j = pos
    index = i * WIDTH + j
    return index

def index_to_state(index):
    # print(index)
    return divmod(index, WIDTH)

# A_Star is an algorithm based on Dijkstra's Algorithm with a heuristic that
# tells us how to choose which node in the open set is most worth checking by
# utilizing a function that guesses how far away from the end each node is
# in order to do a "best-first" search
# https://en.wikipedia.org/wiki/A*_search_algorithm
def A_Star(adjacency_matrix, cost_matrix, area, start, goal, Dijkstra=False):

    def reconstruct_path(cur_index, came_from):
        if cur_index != state_to_index(goal):
            total_path = [cur_index]
        else:
            total_path = []
        while came_from[cur_index] != None:
            cur_index = came_from[cur_index]
            if cur_index != state_to_index(goal):
                total_path.insert(0, cur_index)
        pos_path = [index_to_state(i) for i in total_path]
        return pos_path

    def neighbor_getter(index, adjacency_matrix, last_index):
        legal_indices = []
        # for i in range(TOTAL_SQUARES):
        for pos in MOVES:
            cur_state = index_to_state(index)
            new_state = (cur_state[0]+pos[0], cur_state[1]+pos[1])
            new_index = state_to_index(new_state)
            if 0<=new_index<TOTAL_SQUARES and adjacency_matrix[index, new_index] == 1 and new_index != last_index:
                legal_indices.append(new_index)
        return legal_indices

    h = lambda cur_pos: abs(goal[0]-cur_pos[0]) + abs(goal[1]-cur_pos[1])

    count = 0

    open_set = PriorityQueue()
    if Dijkstra:
        open_set.put((0, count, state_to_index(start)))
    else:
        open_set.put((0, -count, state_to_index(start)))

    in_open_set = [False for j in range(TOTAL_SQUARES)]
    in_open_set[0] = True

    came_from = [None for j in range(TOTAL_SQUARES)]

    g_score = [infty for j in range(TOTAL_SQUARES)]
    g_score[0] = 0

    f_score = [0 for j in range(TOTAL_SQUARES)]
    f_score[0] = h(start)

    prev_index = -1

    running = True
    solving = True
    solved = False
    while running:
        # breakpoint()
        if solving:
            if open_set.empty():
                breakpoint()
                print('cry')
                print('we should never see this as there are practically no *actual* obstacles, hence why we cry')
                raise ValueError

            _, _, cur_index = open_set.get()
            current = index_to_state(cur_index)
            in_open_set[cur_index] = False

            if current == goal:
                move_list = reconstruct_path(cur_index, came_from)
                return move_list

            neighbors = neighbor_getter(cur_index, adjacency_matrix, prev_index)
            for neighbor in neighbors:
                temp_g_score = g_score[cur_index] + cost_matrix[cur_index, neighbor]

                if temp_g_score < g_score[neighbor]:
                    came_from[neighbor] = cur_index
                    g_score[neighbor] = temp_g_score
                    f_score[neighbor] = temp_g_score + h(index_to_state(neighbor))
                    if not in_open_set[neighbor]:
                        count += 1
                        if Dijkstra:
                            open_set.put((f_score[neighbor], count, neighbor))
                        else:
                            open_set.put((f_score[neighbor], -count, neighbor))
                        in_open_set[neighbor] = True
            # breakpoint()
            prev_index = cur_index
