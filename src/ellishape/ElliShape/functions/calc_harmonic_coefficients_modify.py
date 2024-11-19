import numpy as np
from calc_traversal_dist import calc_traversal_dist
from calc_traversal_time import calc_traversal_time
def calc_harmonic_coefficients_modify(ai, n, mode):
    k = ai.shape[0]
    d = calc_traversal_dist(ai)
    if mode == 0:
        edist = d[k-1, 0] ** 2 + d[k-1, 1] ** 2
        if edist > 2:
            print('error chaincode, not close form')
            return None
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
        if d[k-1, 1] != 0:
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
    two_n_pi = 2 * n * np.pi

    sigma_a = 0
    sigma_b = 0
    sigma_c = 0
    sigma_d = 0

    for p in range(k):
        if p >= 1:
            tp_prev = t[p-1]
            dp_prev = d[p-1]
        else:
            tp_prev = 0
            dp_prev = np.zeros(2)

        delta_d = calc_traversal_dist(ai[p])
        # print(delta_d)
        delta_x = delta_d[:,0]
        delta_y = delta_d[:,1]
        delta_t = calc_traversal_time(ai[p])

        q_x = delta_x / delta_t
        q_y = delta_y / delta_t

        sigma_a += two_n_pi * (d[p, 0] * np.sin(two_n_pi * t[p] / T) - dp_prev[0] * np.sin(two_n_pi * tp_prev / T)) / T
        sigma_a += q_x * (np.cos(two_n_pi * t[p] / T) - np.cos(two_n_pi * tp_prev / T))
        sigma_b -= two_n_pi * (d[p, 0] * np.cos(two_n_pi * t[p] / T) - dp_prev[0] * np.cos(two_n_pi * tp_prev / T)) / T
        sigma_b += q_x * (np.sin(two_n_pi * t[p] / T) - np.sin(two_n_pi * tp_prev / T))
        sigma_c += two_n_pi * (d[p, 1] * np.sin(two_n_pi * t[p] / T) - dp_prev[1] * np.sin(two_n_pi * tp_prev / T)) / T
        sigma_c += q_y * (np.cos(two_n_pi * t[p] / T) - np.cos(two_n_pi * tp_prev / T))
        sigma_d -= two_n_pi * (d[p, 1] * np.cos(two_n_pi * t[p] / T) - dp_prev[1] * np.cos(two_n_pi * tp_prev / T)) / T
        sigma_d += q_y * (np.sin(two_n_pi * t[p] / T) - np.sin(two_n_pi * tp_prev / T))


    if n == 0:
        print("n must be non-zero")
    
    try:
        r = T / (2 * n ** 2 * np.pi ** 2)
    except ZeroDivisionError:
        r = np.inf


    a = r * sigma_a
    b = r * sigma_b
    c = r * sigma_c
    d = r * sigma_d

    return [a, b, c, d]
