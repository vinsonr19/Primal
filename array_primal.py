import numpy as np
from pixel_class import *
from constants import *
from pygame.event import get
from pygame.locals import K_ESCAPE, KEYDOWN, QUIT, K_r
from pygame.display import flip
from pygame import quit
from A_Star import *
from time import sleep


def array_primal():
    #define our space
    global SPACE
    SPACE = np.array([[Pixel(i,j) for j in range(HEIGHT)] for i in range(WIDTH)])

    adj_mat = np.zeros((TOTAL_SQUARES, TOTAL_SQUARES))

    for i in range(0, WIDTH):
        for j in range(0, HEIGHT):
            index_from = state_to_index((i, j))
            for m in MOVES:
                if 0<=i+m[0]<WIDTH and 0<=j+m[1]<HEIGHT:
                    index_to = state_to_index((i + m[0], j + m[1]))
                    if SPACE[i + m[0], j + m[1]].state not in [1,3]:
                        adj_mat[index_from, index_to] = 1


    #define a start position
    start_pos = (0,0)
    #define a goal
    goal_pos = (WIDTH-1, HEIGHT-1)
    SPACE[goal_pos[0], goal_pos[1]].change_state(6)

    #add enemies with threat squares
    enemies = [(np.random.randint(0,WIDTH), np.random.randint(0,HEIGHT)) for i in range(NUM_ENEMIES)]

    for pos in enemies:
        cur_influence = ENEMY_INFLUENCE + np.random.randint(-ENEMY_INFLUENCE+1, 3)
        SPACE[pos[0], pos[1]].change_state(3)
        for i in range(-cur_influence, cur_influence+1):
            for j in range(-cur_influence, cur_influence+1):
                try:
                    # if i != 0 or j != 0:
                    if 1 <= abs(i) + abs(j) <= FRIENDLY_INFLUENCE:
                        if 0<=pos[0]+i<WIDTH and 0<=pos[1]+j<HEIGHT:
                            if (pos[0] + i, pos[1] + j) == start_pos or (pos[0] + i, pos[1] + j) == goal_pos or SPACE[pos[0] + i, pos[1] + j].state == 3:
                                continue
                            SPACE[pos[0] + i, pos[1] + j].change_state(4)
                except:
                    continue


    #add allies with help squares
    allies = [(np.random.randint(0,WIDTH), np.random.randint(0,HEIGHT)) for i in range(NUM_ALLIES)]
    for pos in allies:
        if pos in enemies or SPACE[pos].state == 4:
            allies.remove(pos)

    for pos in allies:
        cur_influence = FRIENDLY_INFLUENCE + np.random.randint(-FRIENDLY_INFLUENCE+1, 2)
        SPACE[pos[0], pos[1]].change_state(1)
        for i in range(-cur_influence, cur_influence+1):
            for j in range(-cur_influence, cur_influence+1):
                try:
                    # if i != 0 or j != 0:
                    if 1 <= abs(i) + abs(j) <= FRIENDLY_INFLUENCE:
                        if 0<=pos[0]+i<WIDTH and 0<=pos[1]+j<HEIGHT:
                            if (pos[0] + i, pos[1] + j) == start_pos or (pos[0] + i, pos[1] + j) == goal_pos or SPACE[pos[0] + i, pos[1] + j].state in [1,3]:
                                continue
                            if SPACE[pos[0] + i, pos[1] + j].state in [4,7]:
                                SPACE[pos[0] + i, pos[1] + j].change_state(7)
                            else:
                                SPACE[pos[0] + i, pos[1] + j].change_state(5)
                except:
                    continue

    cost_mat = np.zeros((TOTAL_SQUARES, TOTAL_SQUARES))
    flip()

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

    return A_Star(adj_mat, cost_mat, SPACE, start_pos, goal_pos, Dijkstra=False)

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
            final_path = array_primal()
            # print(final_path)
            # breakpoint()
            sleep(2)
            for i,j in final_path:
                screen.fill(BLACK)
                for a in range(WIDTH):
                    for b in range(HEIGHT):
                        SPACE[a,b].show()
                SPACE[i,j].change_state(2)
                flip()
                # sleep(0.15)
                solved = True
