from space_setup import *
import constants
from M_Star import *

if __name__ == '__main__':
    
    
    def test_success(op, goals):
        count = 0
        for idx, o in enumerate(op):
            if o[-1] == goals[idx]:
                count += 1
                
        if count == len(goals):
            return(True)
        else:
            return(False)
        
    
    count = 0
    for i in range(0, 1000):
        
        adj_mat, assets, goals, enemies, allies, cost_mat, ally_influence_radii = setup_space()
        
        try:
            op = M_Star_Varied_Comms(cost_mat, assets, allies, goals, enemies)
            count += 1
        except:
            continue
        
        
        print(count)
    
    
    
    
    breakpoint()