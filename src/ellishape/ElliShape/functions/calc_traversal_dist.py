
import numpy as np
def calc_traversal_dist(ai):
    x_ = 0
    y_ = 0
    if(np.isscalar(ai)):
        p = np.zeros((1, 2))
        x_ += np.sign(6 - ai) * np.sign(2 - ai)
        y_ += np.sign(4 - ai) * np.sign(ai)
        p[0, 0] = x_
        p[0, 1] = y_  
    else:  
        p = np.zeros((ai.shape[0], 2))
        for i in range(ai.shape[0]):
            x_ += np.sign(6 - ai[i]) * np.sign(2 - ai[i])
            y_ += np.sign(4 - ai[i]) * np.sign(ai[i])
            p[i, 0] = x_
            p[i, 1] = y_
    return p
