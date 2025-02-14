#!/usr/bin/python3
import argparse
import csv
from _ast import arg
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from ellishape_cli.global_vars import log
from memoization import cached


code_axis_map = {
    0: [0, 1],
    1: [-1, 1],
    2: [-1, 0],
    3: [-1, -1],
    4: [0, -1],
    5: [1, -1],
    6: [1, 0],
    7: [1, 1]
}
# pixel direction
axis_code_map = {
    (1, 0): 0,
    (1, 1): 1,
    (0, 1): 2,
    (-1, 1): 3,
    (-1, 0): 4,
    (-1, -1): 5,
    (0, -1): 6,
    (1, -1): 7
}


def get_options():
    # options for normalization, open by default
    ro = True
    sc = True
    re = True
    y_sy = True
    x_sy = True
    sta = True
    trans = True
    option = (ro, sc, re, y_sy, x_sy, sta, trans)
    return option


def direction2code(chain_code: np.array, start_point: np.array):
    end_point = np.array(start_point)
    for direction in chain_code:
        direction = direction % 8  # Ensure direction is within bounds
        end_point += code_axis_map[direction]
    return end_point


def code2axis(chain_code, start_point):
    axis = np.zeros((len(chain_code) + 1, 2))
    axis[0, :] = start_point
    end_point = start_point

    for i, code in enumerate(chain_code, start=1):
        direction = code % 8
        end_point = np.add(end_point, code_axis_map[direction])
        axis[i, :] = end_point
    return axis


def check_input(input_file: Path, encode='utf-8'):
    n = 0
    if not input_file.exists():
        log.error('Input file does not exist: {}'.format(input_file))
        raise SystemExit(-1)
    try:
        line = ''
        with open(input_file, 'r', encoding=encode) as _:
            for n, line in enumerate(_):
                pass
    except UnicodeDecodeError:
        log.error(f'Encode error found in {n} line, please convert it to utf-8')
        log.error(line)
        raise SystemExit(-2)


def gui_chain_code_func(axis_info, origin_ori):
    nrow, ncol = axis_info.shape
    # print(nrow)
    # print(ncol)
    idxall = np.where(axis_info == 255)
    # idxall=np.transpose(idxall)
    # print(idxall)
    numoftotalpoints = len(idxall[0])

    rowidxst = origin_ori[0]
    colidxst = origin_ori[1]
    origin_ori = [rowidxst, colidxst]

    flags = np.zeros((nrow, ncol))
    contour_points = np.zeros((numoftotalpoints, 2))
    # print(contour_points)
    backword_points = np.zeros((numoftotalpoints, 2))
    chain_code_ori = np.zeros(numoftotalpoints, dtype=int)

    numofpoints = 0
    numofpoints_pre = 0
    contour_points[numofpoints] = origin_ori
    flags[rowidxst, colidxst] = 1
    numofbackword = 0
    backwordflag = False

    while numofpoints < numoftotalpoints - 1:
        fatecount = 8
        # print(numofpoints)
        rowidxpre, colidxpre = contour_points[numofpoints]  # 前一个点的坐标

        rowidx = int(min(max(rowidxpre, 0), nrow - 1))  #
        colidx = int(min(max(colidxpre + 1, 0), ncol - 1))
        # print(rowidx)
        # print(colidx)
        # print(rowidxst)
        # print(colidxst)

        if rowidx == rowidxst and colidx == colidxst and numofpoints > 1:
            numofpoints += 1
            contour_points[numofpoints] = [rowidx, colidx]
            chain_code_ori[numofpoints - 1] = 0
            break

        # print(axis_info[rowidx, colidx])
        # print(flags[rowidx, colidx])
        if axis_info[rowidx, colidx] == 255 and flags[rowidx, colidx] == 0:
            numofpoints += 1
            # print(numofpoints)
            contour_points[numofpoints] = [rowidx, colidx]
            flags[rowidx, colidx] = 1
            chain_code_ori[numofpoints - 1] = 0
        else:
            fatecount -= 1
            rowidx = int(min(max(rowidxpre - 1, 0), nrow - 1))
            colidx = int(min(max(colidxpre + 1, 0), ncol - 1))
            if rowidx == origin_ori[0] and colidx == origin_ori[
                1] and numofpoints > 1:
                numofpoints += 1
                contour_points[numofpoints] = [rowidx, colidx]
                chain_code_ori[numofpoints - 1] = 1
                break
            # print(rowidx)
            # print(colidx)
            # print(axis_info[colidx,rowidx])
            # print(flags[rowidx, colidx])
            if axis_info[rowidx, colidx] == 255 and flags[rowidx, colidx] == 0:

                numofpoints += 1
                contour_points[numofpoints] = [rowidx, colidx]
                flags[rowidx, colidx] = 1
                chain_code_ori[numofpoints - 1] = 1
            else:
                fatecount -= 1
                rowidx = int(min(max(rowidxpre - 1, 0), nrow - 1))
                colidx = int(min(max(colidxpre, 0), ncol - 1))
                if rowidx == rowidxst and colidx == colidxst and numofpoints > 1:
                    numofpoints += 1
                    contour_points[numofpoints] = [rowidx, colidx]
                    chain_code_ori[numofpoints - 1] = 2
                    break
                if axis_info[rowidx, colidx] == 255 and flags[
                    rowidx, colidx] == 0:
                    numofpoints += 1
                    contour_points[numofpoints] = [rowidx, colidx]
                    flags[rowidx, colidx] = 1
                    chain_code_ori[numofpoints - 1] = 2
                else:
                    fatecount -= 1
                    rowidx = int(min(max(rowidxpre - 1, 0), nrow - 1))
                    colidx = int(min(max(colidxpre - 1, 0), ncol - 1))
                    if rowidx == rowidxst and colidx == colidxst and numofpoints > 1:
                        numofpoints += 1
                        contour_points[numofpoints] = [rowidx, colidx]
                        chain_code_ori[numofpoints - 1] = 3
                        break
                    if axis_info[rowidx, colidx] == 255 and flags[
                        rowidx, colidx] == 0:
                        numofpoints += 1
                        contour_points[numofpoints] = [rowidx, colidx]
                        flags[rowidx, colidx] = 1
                        chain_code_ori[numofpoints - 1] = 3
                    else:
                        fatecount -= 1
                        rowidx = int(min(max(rowidxpre, 0), nrow - 1))
                        colidx = int(min(max(colidxpre - 1, 0), ncol - 1))
                        if rowidx == rowidxst and colidx == colidxst and numofpoints > 2:
                            numofpoints += 1
                            contour_points[numofpoints] = [rowidx, colidx]
                            chain_code_ori[numofpoints - 1] = 4
                            break
                        if axis_info[rowidx, colidx] == 255 and flags[
                            rowidx, colidx] == 0:
                            numofpoints += 1
                            contour_points[numofpoints] = [rowidx, colidx]
                            flags[rowidx, colidx] = 1
                            chain_code_ori[numofpoints - 1] = 4
                        else:
                            fatecount -= 1
                            rowidx = int(min(max(rowidxpre + 1, 0), nrow - 1))
                            colidx = int(min(max(colidxpre - 1, 0), ncol - 1))
                            if rowidx == rowidxst and colidx == colidxst and numofpoints > 1:
                                numofpoints += 1
                                contour_points[numofpoints] = [rowidx, colidx]
                                chain_code_ori[numofpoints - 1] = 5
                                break
                            if axis_info[rowidx, colidx] == 255 and flags[
                                rowidx, colidx] == 0:
                                numofpoints += 1
                                contour_points[numofpoints] = [rowidx, colidx]
                                flags[rowidx, colidx] = 1
                                chain_code_ori[numofpoints - 1] = 5
                            else:
                                fatecount -= 1
                                rowidx = int(
                                    min(max(rowidxpre + 1, 0), nrow - 1))
                                colidx = int(min(max(colidxpre, 0), ncol - 1))
                                if rowidx == rowidxst and colidx == colidxst and numofpoints > 1:
                                    numofpoints += 1
                                    contour_points[numofpoints] = [rowidx,
                                                                   colidx]
                                    chain_code_ori[numofpoints - 1] = 6
                                    break
                                if axis_info[rowidx, colidx] == 255 and flags[
                                    rowidx, colidx] == 0:
                                    numofpoints += 1
                                    contour_points[numofpoints] = [rowidx,
                                                                   colidx]
                                    flags[rowidx, colidx] = 1
                                    chain_code_ori[numofpoints - 1] = 6
                                else:
                                    fatecount -= 1
                                    rowidx = int(
                                        min(max(rowidxpre + 1, 0), nrow - 1))
                                    colidx = int(
                                        min(max(colidxpre + 1, 0), ncol - 1))
                                    if rowidx == rowidxst and colidx == colidxst and numofpoints > 1:
                                        numofpoints += 1
                                        contour_points[numofpoints] = [rowidx,
                                                                       colidx]
                                        chain_code_ori[numofpoints - 1] = 7
                                        break
                                    if axis_info[rowidx, colidx] == 255 and \
                                            flags[rowidx, colidx] == 0:
                                        numofpoints += 1
                                        contour_points[numofpoints] = [rowidx,
                                                                       colidx]
                                        flags[rowidx, colidx] = 1
                                        chain_code_ori[numofpoints - 1] = 7
                                    else:
                                        fatecount -= 1

        if fatecount == 0:
            rowidxpre = int(contour_points[numofpoints, 0])
            colidxpre = int(contour_points[numofpoints, 1])
            # print(rowidxpre)
            # print(colidxpre)
            axis_info[rowidxpre, colidxpre] = 0
            if numofbackword == 0 and numofpoints > numofpoints_pre:
                numofpoints_pre = numofpoints
                backwordflag = 1
            numofpoints -= 1
            if numofpoints < 1:
                break
            numofbackword += 1
            if numofbackword > 20:
                break
            else:
                if backwordflag == 1:
                    backword_points[numofbackword] = [rowidxpre, colidxpre]
        else:
            numofbackword = 0
            backwordflag = False

    chain_code = chain_code_ori[:numofpoints]
    # endpoint = backword_points[0]
    origin = origin_ori
    # print(endpoint)
    # print((np.where(flags == 1)))
    log.debug(f'{chain_code_ori=} {chain_code_ori.shape=}')
    log.debug(f'{chain_code=} {chain_code.shape=}')
    return chain_code, origin


@cached
def calc_traversal_dist(ai):
    x_ = 0
    y_ = 0
    if np.isscalar(ai):
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


def is_completed_chain_code(chain_code, start_point):
    # todo: why 2
    close_threshold = 2
    end_point = np.array(start_point)
    for direction in chain_code:
        direction = direction % 8  # Ensure direction is within bounds
        end_point += code_axis_map[direction]
    distance = np.sqrt(np.sum((np.array(start_point) - end_point) ** 2))
    log.debug(f'{distance=}')
    is_closed = (distance <= close_threshold)
    return is_closed, end_point


def compute_harmonic_coefficients(ai, harmonic_index):
    return calc_harmonic_coefficients_modify(ai, harmonic_index+1, 0)


@cached
def fourier_approx_norm_modify(ai, n, m, normalized, mode, option):
    EPS = 1e-10
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)

    # parallel
    with ProcessPoolExecutor() as executor:
        # todo: send_bytes too expensive
        results = list(executor.map(compute_harmonic_coefficients, [ai] * n, range(n)))

    for i, harmonic_coeff in enumerate(results):
        a[i] = harmonic_coeff[0].item()
        b[i] = harmonic_coeff[1].item()
        c[i] = harmonic_coeff[2].item()
        d[i] = harmonic_coeff[3].item()

    A0, C0, Tk, T = calc_dc_components_modify(ai, 0)

    log.debug(f'{A0=}, {C0=}, {Tk=}, {T=}')
    # Normalization procedure
    if normalized:
        ro, sc, re, y_sy, x_sy, sta, trans = option
        # ro, sc, re, y_sy, x_sy, sta, trans = [False]*7
        # Remove DC components
        if trans:
            A0 = 0
            C0 = 0

        if re:
            CrossProduct = a[0] * d[0] - c[0] * b[0]
            if CrossProduct < 0:
                b = -b
                d = -d

        tan_theta2 = 2 * (a[0] * b[0] + c[0] * d[0]) / (
                a[0] ** 2 + c[0] ** 2 - b[0] ** 2 - d[0] ** 2)
        theta1 = 0.5 * np.arctan(tan_theta2)
        if theta1 < 0:
            theta1 += np.pi / 2
        sin_2theta = np.sin(2 * theta1)
        cos_2theta = np.cos(2 * theta1)
        cos_theta_square = (1 + cos_2theta) / 2
        sin_theta_square = (1 - cos_2theta) / 2

        axis_theta1 = (a[0] ** 2 + c[0] ** 2) * cos_theta_square + (
                a[0] * b[0] + c[0] * d[0]) * sin_2theta + (
                              b[0] ** 2 + d[0] ** 2) * sin_theta_square
        axis_theta2 = (a[0] ** 2 + c[0] ** 2) * sin_theta_square - (
                a[0] * b[0] + c[0] * d[0]) * sin_2theta + (
                              b[0] ** 2 + d[0] ** 2) * cos_theta_square

        # print(axis_theta1)
        # print(axis_theta2)
        if axis_theta1 < axis_theta2:
            theta1 += np.pi / 2

        costh1 = np.cos(theta1)
        sinth1 = np.sin(theta1)
        a_star_1 = costh1 * a[0] + sinth1 * b[0]
        c_star_1 = costh1 * c[0] + sinth1 * d[0]
        psi1 = np.arctan(np.abs(c_star_1 / a_star_1))

        if c_star_1 > 0 > a_star_1:
            psi1 = np.pi - psi1
        if c_star_1 < 0 and a_star_1 < 0:
            psi1 = np.pi + psi1
        if c_star_1 < 0 < a_star_1:
            psi1 = np.pi * 2 - psi1

        E = np.sqrt(a_star_1 ** 2 + c_star_1 ** 2)
        # print(E)

        if sc:
            a /= E
            b /= E
            c /= E
            d /= E

        cospsi1 = np.cos(psi1)
        sinpsi1 = np.sin(psi1)
        normalized_all = np.zeros((n, 4))

        if ro:
            for i in range(n):
                normalized = np.dot([[cospsi1, sinpsi1], [-sinpsi1, cospsi1]],
                                    [[a[i], b[i]], [c[i], d[i]]])
                # print(normalized.reshape(1, 4))
                normalized_all[i] = normalized.reshape(1, 4)
            a = normalized_all[:, 0]
            b = normalized_all[:, 1]
            c = normalized_all[:, 2]
            d = normalized_all[:, 3]

        normalized_all_1 = np.zeros((n, 4))

        if sta:
            for i in range(n):
                normalized_1 = np.dot([[a[i], b[i]], [c[i], d[i]]], [
                    [np.cos(theta1 * (i + 1)), -np.sin(theta1 * (i + 1))],
                    [np.sin(theta1 * (i + 1)), np.cos(theta1 * (i + 1))]])
                normalized_all_1[i, :] = normalized_1.reshape(1, 4)
                # print(normalized_1)
            a = normalized_all_1[:, 0]
            b = normalized_all_1[:, 1]
            c = normalized_all_1[:, 2]
            d = normalized_all_1[:, 3]

            # todo: new?
            # if a[0] < 0:
            #     a = -a
            #     d = -d

        if y_sy:
            if n > 1:
                if a[1] < -EPS:
                    for i in range(1, n):
                        signval = (-1) ** (((i + 1) % 2) + 1)
                        a[i] = signval * a[i]
                        d[i] = signval * d[i]
                        signval = (-1) ** ((i + 1) % 2)
                        b[i] = signval * b[i]
                        c[i] = signval * c[i]

        if x_sy:
            if n > 1:
                if c[1] < -EPS:
                    b[1:] *= -1
                    c[1:] *= -1

    output = np.zeros((m, 2))

    for t in range(m):
        x_ = 0.0
        y_ = 0.0
        for i in range(n):
            x_ += (a[i] * np.cos(2 * (i + 1) * np.pi * (t) * Tk / (m - 1) / T) +
                   b[i] * np.sin(2 * (i + 1) * np.pi * (t) * Tk / (m - 1) / T))
            y_ += (c[i] * np.cos(2 * (i + 1) * np.pi * (t) * Tk / (m - 1) / T) +
                   d[i] * np.sin(2 * (i + 1) * np.pi * (t) * Tk / (m - 1) / T))
        # remove numpy DeprecationWarning
        output[t, 0] = np.array(A0 + x_).item()
        output[t, 1] = np.array(C0 + y_).item()

    return output, a, b, c, d


@cached
def calc_traversal_time(ai):
    t_ = 0
    if np.isscalar(ai):
        t = np.zeros(1)
        t_ = t_ + 1 + ((np.sqrt(2) - 1) / 2) * (1 - (-1) ** ai)
        t[0] = t_
    else:
        t = np.zeros(ai.shape[0])
        for i in range(ai.shape[0]):
            t_ = t_ + 1 + ((np.sqrt(2) - 1) / 2) * (1 - (-1) ** ai[i])
            t[i] = t_
    return t


def calc_harmonic_coefficients_modify(ai, n, mode):
    k = ai.shape[0]
    d = calc_traversal_dist(ai)
    if mode == 0:
        edist = d[k - 1, 0] ** 2 + d[k - 1, 1] ** 2
        if edist > 2:
            log.error('error chaincode, not close form')
            raise SystemExit(-1)
        else:
            if edist > 0:
                vect = (-d[k - 1, 0], -d[k - 1, 1])
                ai = np.append(ai, axis_code_map[vect])

    elif mode == 1:
        if d[k - 1, 0] != 0:
            exp1 = np.zeros(abs(d[k - 1, 0])) + 2 * np.sign(
                d[k - 1, 0]) + 2 * np.sign(d[k - 1, 0]) * np.sign(d[k - 1, 0])
            ai = np.append(ai, exp1)
        if d[k - 1, 1] != 0:
            exp2 = np.zeros(abs(d[k - 1, 1])) + 2 + 2 * np.sign(
                d[k - 1, 1]) + 2 * np.sign(d[k - 1, 1]) * np.sign(d[k - 1, 1])
            ai = np.append(ai, exp2)

    elif mode == 2:
        if d[k - 1, 0] != 0 or d[k - 1, 1] != 0:
            exp = (ai + 4) % 8
            ai = np.append(ai, exp)

    k = ai.shape[0]
    t = calc_traversal_time(ai)
    d = calc_traversal_dist(ai)
    T = t[k - 1]
    two_n_pi = 2 * n * np.pi

    sigma_a = 0
    sigma_b = 0
    sigma_c = 0
    sigma_d = 0

    for p in range(k):
        if p >= 1:
            tp_prev = t[p - 1]
            dp_prev = d[p - 1]
        else:
            tp_prev = 0
            dp_prev = np.zeros(2)

        delta_d = calc_traversal_dist(ai[p])
        # print(delta_d)
        delta_x = delta_d[:, 0]
        delta_y = delta_d[:, 1]
        delta_t = calc_traversal_time(ai[p])

        q_x = delta_x / delta_t
        q_y = delta_y / delta_t

        sigma_a += two_n_pi * (
                d[p, 0] * np.sin(two_n_pi * t[p] / T) - dp_prev[0] * np.sin(
            two_n_pi * tp_prev / T)) / T
        sigma_a += q_x * (np.cos(two_n_pi * t[p] / T) - np.cos(
            two_n_pi * tp_prev / T))
        sigma_b -= two_n_pi * (
                d[p, 0] * np.cos(two_n_pi * t[p] / T) - dp_prev[0] * np.cos(
            two_n_pi * tp_prev / T)) / T
        sigma_b += q_x * (np.sin(two_n_pi * t[p] / T) - np.sin(
            two_n_pi * tp_prev / T))
        sigma_c += two_n_pi * (
                d[p, 1] * np.sin(two_n_pi * t[p] / T) - dp_prev[1] * np.sin(
            two_n_pi * tp_prev / T)) / T
        sigma_c += q_y * (np.cos(two_n_pi * t[p] / T) - np.cos(
            two_n_pi * tp_prev / T))
        sigma_d -= two_n_pi * (
                d[p, 1] * np.cos(two_n_pi * t[p] / T) - dp_prev[1] * np.cos(
            two_n_pi * tp_prev / T)) / T
        sigma_d += q_y * (np.sin(two_n_pi * t[p] / T) - np.sin(
            two_n_pi * tp_prev / T))

    if n == 0:
        log.error('n must be non-zero')

    try:
        r = T / (2 * n ** 2 * np.pi ** 2)
    except ZeroDivisionError:
        r = np.inf

    a = r * sigma_a
    b = r * sigma_b
    c = r * sigma_c
    d = r * sigma_d
    return a, b, c, d



def calc_dc_components_modify(ai, mode):
    k = ai.shape[0]
    t = calc_traversal_time(ai)
    d = calc_traversal_dist(ai)
    Tk = t[k-1]

    if mode == 0:
        edist = d[k - 1, 0] ** 2 + d[k - 1, 1] ** 2
        if edist > 2:
            log.error('error chaincode, not close form')
            return None, None, None, None
        else:
            if edist > 0:
                vect = (-d[k - 1, 0], -d[k - 1, 1])
                # ai2 = ai.copy()
                ai = np.append(ai, axis_code_map[vect])

                # if vect[0] == 1 and vect[1] == 0:
                #     ai2 = np.append(ai2, 0)
                # elif vect[0] == 1 and vect[1] == 1:
                #     ai2 = np.append(ai2, 1)
                # elif vect[0] == 0 and vect[1] == 1:
                #     ai2 = np.append(ai2, 2)
                # elif vect[0] == -1 and vect[1] == 1:
                #     ai2 = np.append(ai2, 3)
                # elif vect[0] == -1 and vect[1] == 0:
                #     ai2 = np.append(ai2, 4)
                # elif vect[0] == -1 and vect[1] == -1:
                #     ai2 = np.append(ai2, 5)
                # elif vect[0] == 0 and vect[1] == -1:
                #     ai2 = np.append(ai2, 6)
                # elif vect[0] == 1 and vect[1] == -1:
                #     ai2 = np.append(ai2, 7)
                #     # todo: remove
                # log.warning('test')
                # assert np.linalg.norm(ai2 - ai) == 0

    elif mode == 1:
        if d[k - 1, 0] != 0:
            exp1 = np.zeros(abs(d[k - 1, 0])) + 2 * np.sign(
                d[k - 1, 0]) + 2 * np.sign(d[k - 1, 0]) * np.sign(d[k - 1, 0])
            ai = np.append(ai, exp1)
        if d[-1, 1] != 0:
            exp2 = np.zeros(abs(d[k - 1, 1])) + 2 + 2 * np.sign(
                d[k - 1, 1]) + 2 * np.sign(d[k - 1, 1]) * np.sign(d[k - 1, 1])
            ai = np.append(ai, exp2)

    elif mode == 2:
        if d[k - 1, 0] != 0 or d[k - 1, 1] != 0:
            exp = (ai + 4) % 8
            ai = np.append(ai, exp)
    k = ai.shape[0]
    t = calc_traversal_time(ai)
    d = calc_traversal_dist(ai)
    T = t[k - 1]

    sum_a0 = 0
    sum_c0 = 0
    for p in range(1, k):
        if p >= 1:
            dp_prev = d[p - 1, :]
        else:
            dp_prev = np.zeros(2)
        delta_t = calc_traversal_time(ai[p])
        sum_a0 += (d[p, 0] + dp_prev[0]) * delta_t / 2
        sum_c0 += (d[p, 1] + dp_prev[1]) * delta_t / 2

    A0 = sum_a0 / T
    C0 = sum_c0 / T

    return A0, C0, Tk, T


def get_chain_code(img_file: Path) -> (np.ndarray|None, np.ndarray|None):
    # read color images and convert to gray
    # binary
    img = cv2.imread(str(img_file), cv2.IMREAD_COLOR)
    # todo: resize?
    # img = cv2.resize(img, None, fx=0.125, fy=0.125)
    log.info(f'Image size: {img.shape}')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
    # img_result = np.zeros_like(gray, dtype=np.uint8)
    img_result = np.zeros_like(gray)
    log.debug(f'{gray=}')
    _, gray_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # find contours
    contours, _ = cv2.findContours(gray_bin, cv2.RETR_EXTERNAL,
                                   # cv2.CHAIN_APPROX_SIMPLE)
                                   cv2.CHAIN_APPROX_NONE)
    if not contours:
        log.error('Cannot find boundary in the image file')
        return None
    max_contour = max(contours, key=cv2.contourArea)
    # m_ = max_contour.copy()
    max_contour = np.reshape(max_contour, (
        max_contour.shape[0], max_contour.shape[2]))
    log.debug(f'{max_contour[0][0]=}')
    log.debug(f'{max_contour.shape=}')
    # log.debug(f'{max_contour=}')
    # draw for chain code, thickness must be equal to 1
    cv2.drawContours(img_result, [max_contour], -1, 255, thickness=1)
    # cv2.imshow('a', img_result)
    # cv2.waitKey()
    # cv2.imshow('gray', gray)
    max_contour[:, [0, 1]] = max_contour[:, [1, 0]]
    log.debug(f'{max_contour[0][0]=}')
    log.debug(f'{max_contour.shape=}')
    # log.debug(f'{max_contour=}')
    boundary = max_contour
    log.debug(f'{max_contour[0].shape}')
    # chaincode, origin = gui_chain_code_func(gray, max_contour[0])
    chaincode, origin = gui_chain_code_func(img_result, max_contour[0])
    log.debug(f'{chaincode.shape=}')
    log.debug(f'Chaincode shape: {chaincode.shape}')
    if len(chaincode) == 0:
        log.error('Cannot generate chain code from the image')
        return None

    log.debug(f'{boundary[0]=}')
    is_closed, endpoint = is_completed_chain_code(chaincode, boundary[0])

    if not is_closed:
        log.error(f'Chain code is not closed')
        log.error(f'Chain code length: {chaincode.shape[0]}')
        return None, None
    else:
        log.debug(f'Chain code is closed')
        log.debug(f'{chaincode=}')
        log.debug(f'{chaincode.shape=}')
    return chaincode, boundary


def get_efd_from_chain_code(chain_code, n_order):
    pass


def get_ef_from_contour(contour, n_order):
    pass


def eft_to_curve(efd: np.ndarray, n_order: int, n_dots=512, A0=0, C0=0):
    """
    Convert efd coefficients to dots of curve
    Args:
        efd: N*4 matrix
        n_order: int
        A0:
        C0:
    Returns:
        dots: [n_dots*2] matrix
    """
    t = np.linspace(0, 1.0, num=n_dots, endpoint=False)  # [M] M: number of points
    n = np.arange(1, n_order + 1).reshape(
        (-1, 1))  # efficients selected to reconstruct
    x_t0 = np.sum(
        a * np.cos(2 * n[0] * np.pi * t) + b * np.sin(2 * n[0] * np.pi * t), axis=0)
    y_t0 = np.sum(
        c * np.cos(2 * n[0] * np.pi * t) + d * np.sin(2 * n[0] * np.pi * t), axis=0)
    pass


def output_csv(dots, a, b, c, d, arg):
    n_harmonic = arg.n_harmonic
    n_dots = arg.n_dots
    input_file = Path(arg.input)
    out_file = Path(arg.out)

    t = np.transpose([a, b, c, d])
    Hs = np.reshape(t, (1, -1))
    coffs = [["filepath"]]
    cols = n_harmonic * 4
    matrix = []
    for col in range(1, cols + 1):
        letter = chr(ord('a') + (col - 1) % 4)
        number = str((col - 1) // 4 + 1)
        matrix.append(letter + number)

    coffs[0].extend(matrix)
    coffs.append([str(input_file.absolute())])
    coffs[1].extend(Hs.flatten().tolist())

    # 1000 *2
    header2 = ['filepath']
    for i in range(1, n_dots+1):
        header2.extend([f'x{i}', f'y{i}'])
    xy = [input_file.absolute(),]
    xy.extend(dots.flatten().tolist())

    # FFT coordinate
    out_file2 = out_file.with_suffix('.dot.csv')

    if out_file.exists():
        with open(out_file, 'a', encoding='utf-8', newline='') as out:
            writer = csv.writer(out)
            writer.writerows(coffs[1:])
        with open(out_file2, 'a', encoding='utf-8', newline='') as out:
            writer = csv.writer(out)
            writer.writerow(xy)
    else:
        with open(out_file, 'a', encoding='utf-8', newline='') as out:
            writer = csv.writer(out)
            writer.writerows(coffs)
        with open(out_file2, 'a', encoding='utf-8', newline='') as out:
            writer = csv.writer(out)
            writer.writerow(header2)
            writer.writerow(xy)
    return out_file


def plot_result(efd_result, max_contour, arg) -> Path:
    out_img_file = arg.out.with_suffix('.out.png')
    n_harmonic = arg.n_harmonic
    # n_dots = arg.n_dots + 1000
    from matplotlib import pyplot as plt
    n_dots = arg.n_dots
    dots, a, b, c, d = efd_result
    a = np.reshape(a, (n_harmonic, 1))
    b = np.reshape(b, (n_harmonic, 1))
    c = np.reshape(c, (n_harmonic, 1))
    d = np.reshape(d, (n_harmonic, 1))
    canvas = cv2.imread(str(arg.input), cv2.IMREAD_COLOR)
    canvas = cv2.resize(canvas, (canvas.shape[1]//4, canvas.shape[0]//4))
    # ax = plt.subplot2grid((2, canvas.shape[0]//2), (canvas.shape[0], 2%canvas.shape[0]//2))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    # ax1 = plt.subplot2grid((1,2), (0,0))
    # height, width = 1024, 1024
    # canvas1 = np.zeros((height, width, 3))
    # canvas2 = np.copy(canvas1)
    # cv2.polylines(canvas, [chain_points], False, (255, 0, 0), 2)
    # A0=C0=01000
    ax1.set_title('contour')
    ax1.set_aspect('equal')
    ax1.plot(max_contour[:, 0], max_contour[:, 1], 'o', linewidth=2)
    # cv2.drawContours(canvas, [max_contour], 0, (0, 255, 0),
    #                  thickness=5)
    t = np.linspace(0, 1.0, num=n_dots, endpoint=False)  # [M] M: number of points
    n = np.arange(1, n_harmonic + 1).reshape(
        (-1, 1))  # efficients selected to reconstruct
    x_t0 = np.sum(
        a * np.cos(2 * n[0] * np.pi * t) + b * np.sin(2 * n[0] * np.pi * t), axis=0)
    y_t0 = np.sum(
        c * np.cos(2 * n[0] * np.pi * t) + d * np.sin(2 * n[0] * np.pi * t), axis=0)
    x_t = np.sum(
        a * np.cos(2 * n * np.pi * t) + b * np.sin(2 * n * np.pi * t), axis=0)
    y_t = np.sum(
        c * np.cos(2 * n * np.pi * t) + d * np.sin(2 * n * np.pi * t), axis=0)
    # ax2= plt.subplot2grid((1,2), (0,1))
    ax2.set_title('Normalized ellipse fourier coefficients')
    ax2.set_aspect('equal')
    # ax2.set_xlim(-2, 2)
    # ax2.set_ylim(-2, 2)
    ax2.plot(x_t0, y_t0, 'b--', linewidth=1)
    ax2.plot(x_t, y_t, 'r', linewidth=2)
    ax2.plot(x_t[0], y_t[0], 'bo', linewidth=1, alpha=0.5)
    # ax2.annotate('$x_0,y_0$',
    #              xy=(x_t[1], y_t[1]), xytext=(x_t[0], y_t[0]),
    #              arrowprops=dict(arrowstyle="->",
    #                              connectionstyle="arc3,rad=.5",
    #                              fc="blue", ec="blue",
    #                              alpha=0.5,
    #                              lw=1)
    #              )
    # ax.imshow(canvas)
    plt.savefig(out_img_file)
    # # # todo: for verify
    # from pyefd import elliptic_fourier_descriptors, plot_efd
    # coeff_other = elliptic_fourier_descriptors(max_contour, normalize=True,
    #                                               order=n_harmonic)
    # coeff_us = np.concatenate([a,b,c,d], axis=1)
    # a2, b2, c2, d2 = coeff_other.T
    # a2 = np.reshape(a2, (n_harmonic, 1))
    # b2 = np.reshape(b2, (n_harmonic, 1))
    # c2 = np.reshape(c2, (n_harmonic, 1))
    # d2 = np.reshape(d2, (n_harmonic, 1))
    # x_t2 = np.sum(
    #     a2 * np.cos(2 * n[0] * np.pi * t) + b2 * np.sin(2 * n[0] * np.pi * t), axis=0)
    # y_t2 = np.sum(
    #     c2 * np.cos(2 * n[0] * np.pi * t) + d2 * np.sin(2 * n[0] * np.pi * t), axis=0)
    # ax2.plot(x_t2, y_t2, 'b--')
    # plt.show()
    # fig2 = plt.figure(2, (20, 10))
    # plot_efd(coeff_us, n=n_dots)
    # plt.savefig(out_img_file.with_suffix('.out2.png'))
    # fig3 = plt.figure(3, (20, 10))
    # plot_efd(coeff_other, n=n_dots)
    # plt.savefig(out_img_file.with_suffix('.out3.png'))
    # plot_efd(coeff_us, n=n_dots)
    return out_img_file


def parse_args():
    arg = argparse.ArgumentParser(description='ElliShape cli')
    arg.add_argument('-i', '-input', dest='input',
                      help='input grayscale image with white as foreground',
                      required=True)
    arg.add_argument('-n', '-n_harmonic', dest='n_harmonic',
                      default=35, type=int, help='number of harmonic rank')
    arg.add_argument('-n_dots', type=int, default=512,
                     help='number of output dots')
    arg.add_argument('-out', help='output csv file')
    arg.add_argument('-out_image', action='store_true',
                     help='output result image')
    return arg.parse_args()


def cli_main():
    # one leaf per image
    arg = parse_args()
    arg.input = Path(arg.input).absolute()
    log.info(f'Input {arg.input}')
    if arg.out is None:
        arg.out = arg.input.with_suffix('.csv')
    else:
        arg.out = Path(arg.out).absolute()

    chain_code_result, max_contour = get_chain_code(arg.input)
    if chain_code_result is None:
        log.error('Quit')
        return -1
    option = get_options()
    efd_result = fourier_approx_norm_modify(
        chain_code_result, arg.n_harmonic, arg.n_dots, 1, 0, option)
    dots, a, b, c, d =  efd_result
    output_csv(dots, a, b, c, d, arg)
    if arg.out_image:
        out_img_file = plot_result(efd_result, max_contour, arg)
        log.info(f'Output data: {arg.out}')
        log.info(f'Output data: {arg.out.with_suffix(".2.csv")}')
        log.info(f'Output image: {out_img_file}')
        log.debug('Write: contour')
        log.debug('Green: boundary')
        log.debug('Blue: chain code')
        log.debug('Red: chain code approximate')
        log.debug('Yellow: chain code approximate with normalization')
    log.info('Done')


if __name__ == '__main__':
    cli_main()
