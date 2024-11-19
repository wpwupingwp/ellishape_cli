import numpy as np
def code2axis(chain_code, start_point):
    end_point = start_point
    axis=np.zeros((len(chain_code)+1,2))
    axis[0,:]=start_point
    i=0
    for code in chain_code:
        i=i+1
        direction = code % 8
        if direction == 0:
            end_point = np.add(end_point , [0, 1])
        elif direction == 7:
            end_point = np.add(end_point , [1, 1])
        elif direction == 6:
            end_point = np.add(end_point , [1, 0])
        elif direction == 5:
            end_point = np.add(end_point , [1, -1])
        elif direction == 4:
            end_point = np.add(end_point , [0, -1])
        elif direction == 3:
            end_point = np.add(end_point , [-1, -1])
        elif direction == 2:
            end_point = np.add(end_point , [-1, 0])
        elif direction == 1:
            end_point = np.add(end_point , [-1, 1])
        # print(end_point)
        axis[i,:]=end_point
    return axis