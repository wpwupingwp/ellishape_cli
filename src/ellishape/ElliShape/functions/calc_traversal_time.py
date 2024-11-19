import numpy as np
def calc_traversal_time(ai):
    t_ = 0
    if(np.isscalar(ai)):
        t = np.zeros(1)
        t_ = t_ + 1 + ((np.sqrt(2) - 1) / 2) * (1 - (-1) ** ai)
        t[0] = t_
    else:  
        t = np.zeros(ai.shape[0])
        for i in range(ai.shape[0]):
            t_ = t_ + 1 + ((np.sqrt(2) - 1) / 2) * (1 - (-1) ** ai[i])
            t[i] = t_
    return t
