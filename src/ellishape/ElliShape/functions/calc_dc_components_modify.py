import numpy as np
from calc_traversal_time import calc_traversal_time
from calc_traversal_dist import calc_traversal_dist


def calc_dc_components_modify(ai, mode):
    k = ai.shape[0]
    t = calc_traversal_time(ai)
    d = calc_traversal_dist(ai)
    Tk = t[k-1]
    
    if mode == 0:
        edist = d[k-1, 0]**2 + d[k-1, 1]**2
        if edist > 2:
            print('error chaincode, not close form')
            return None, None, None, None
        else:
            if edist > 0:
                vect = [-d[k-1, 0], -d[k-1, 1]]
                if vect[0] == 1 and vect[1] == 0:
                    ai = np.append(ai, 0)
                elif vect[0] == 1 and vect[1] == 1:
                    ai = np.append(ai, 1)
                elif vect[0] == 0 and vect[1] == 1:
                    ai = np.append(ai, 2)
                elif vect[0] == -1 and vect[1] == 1:
                    ai = np.append(ai, 3)
                elif vect[0] == -1 and vect[1] == 0:
                    ai = np.append(ai, 4)
                elif vect[0] == -1 and vect[1] == -1:
                    ai = np.append(ai, 5)
                elif vect[0] == 0 and vect[1] == -1:
                    ai = np.append(ai, 6)
                elif vect[0] == 1 and vect[1] == -1:
                    ai = np.append(ai, 7)

    elif mode == 1:
        if d[k-1, 0] != 0:
            exp1 = np.zeros(abs(d[k-1, 0])) + 2 * np.sign(d[k-1, 0]) + 2 * np.sign(d[k-1, 0]) * np.sign(d[k-1, 0])
            ai = np.append(ai, exp1)
        if d[-1, 1] != 0:
            exp2 = np.zeros(abs(d[k-1, 1])) + 2 + 2 * np.sign(d[k-1, 1]) + 2 * np.sign(d[k-1, 1]) * np.sign(d[k-1, 1])
            ai = np.append(ai, exp2)

    elif mode == 2:
        if d[k-1, 0] != 0 or d[k-1, 1] != 0:
            exp = (ai + 4) % 8
            ai = np.append(ai, exp)
    k = ai.shape[0]
    t = calc_traversal_time(ai)
    d = calc_traversal_dist(ai)
    T = t[k-1]
    
    sum_a0 = 0
    sum_c0 = 0
    for p in range(1, k):
        if p >=1:
            dp_prev = d[p - 1,:]
        else:
            dp_prev = np.zeros(2)
        delta_t = calc_traversal_time(ai[p])
        sum_a0 += (d[p, 0] + dp_prev[0]) * delta_t / 2
        sum_c0 += (d[p, 1] + dp_prev[1]) * delta_t / 2
    
    A0 = sum_a0 / T
    C0 = sum_c0 / T
    
    return A0, C0, Tk, T
