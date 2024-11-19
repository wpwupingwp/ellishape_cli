import numpy as np

def is_completed_chain_code(chain_code, start_point):
    end_point = np.array(start_point)
    for direction in chain_code:
        direction = direction % 8
        if direction == 0:
            end_point += np.array([0, 1])
        elif direction == 1:
            end_point += np.array([-1, 1])
        elif direction == 2:
            end_point += np.array([-1, 0])
        elif direction == 3:
            end_point += np.array([-1, -1])
        elif direction == 4:
            end_point += np.array([0, -1])
        elif direction == 5:
            end_point += np.array([1, -1])
        elif direction == 6:
            end_point += np.array([1, 0])
        elif direction == 7:
            end_point += np.array([1, 1])
    
    distance = np.sqrt(np.sum((np.array(start_point) - end_point) ** 2))
    
    is_closed = distance <= 2
    
    if is_closed:
        print('Chain code is closed.')
    else:
        print('Chain code is not closed.')
    
    return is_closed,end_point
