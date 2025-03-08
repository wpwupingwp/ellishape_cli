#!/usr/bin/python3
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from timeit import default_timer as timer
from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt

from ellishape_cli.global_vars import log

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


def check_input_csv(input_file: Path, encode='utf-8'):
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


def get_max_contour(gray):
    _, gray_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # find contours
    contours, _ = cv2.findContours(gray_bin, cv2.RETR_EXTERNAL,
                                   # cv2.CHAIN_APPROX_SIMPLE)
                                   cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    max_contour = max(contours, key=cv2.contourArea)
    # 3D to 2D
    max_contour = np.reshape(max_contour, (
        max_contour.shape[0], max_contour.shape[2]))
    log.debug(f'{max_contour[0][0]=}')
    log.debug(f'{max_contour.shape=}')
    return max_contour


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


def normalize(efd, ro=True, sc=True, re=True, y_sy=True, x_sy=True, sta=True,
              trans=True):
    EPS = 1e-10
    a, b, c, d, A0, C0 = efd
    n = a.shape[0]
    # ro, sc, re, y_sy, x_sy, sta, trans = [False]*7
    # Remove DC components
    if trans:
        log.debug('trans')
        A0 = 0
        C0 = 0

    if re:
        CrossProduct = a[0] * d[0] - c[0] * b[0]
        if CrossProduct < 0:
            log.debug('re')
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

    if sc:
        log.debug('sc')
        a /= E
        b /= E
        c /= E
        d /= E

    cospsi1 = np.cos(psi1)
    sinpsi1 = np.sin(psi1)
    normalized_all = np.zeros((n, 4))

    if ro:
        log.debug('ro')
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
        log.debug('sta')
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

    if y_sy:
        if n > 1:
            if a[1] < -EPS:
                log.debug('y_sy')
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
                log.debug('x_sy')
                b[1:] *= -1
                c[1:] *= -1

    log.debug([efd[0][0], efd[1][0], efd[2][0], efd[3][0]])
    log.debug(normalized_all[0])
    log.debug(normalized_all_1[0])
    log.info(f'Theta: {np.rad2deg(theta1):.2f}\u00b0')
    log.info(f'Psi: {np.rad2deg(psi1):.2f}\u00b0')
    log.info(f'E: {E.item():.2f}')
    return a, b, c, d, A0, C0


def get_curve_old(a, b, c, d, A0, C0, Tk, T, n, m):
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

    return output


def get_chain_code(gray, max_contour) -> (np.ndarray|None):
    img_result = np.zeros_like(gray)
    cv2.drawContours(img_result, [max_contour], -1, 255, thickness=1)
    # cv2.imshow('a', img_result)
    # cv2.waitKey()
    # cv2.imshow('gray', gray)
    max_contour[:, [0, 1]] = max_contour[:, [1, 0]]
    # log.debug(f'{max_contour=}')
    boundary = max_contour
    log.debug(f'{max_contour[0].shape}')
    # chaincode, origin = gui_chain_code_func(gray, max_contour[0])
    chaincode, origin = gui_chain_code_func(img_result, max_contour[0])
    max_contour[:, [0, 1]] = max_contour[:, [1, 0]]
    log.debug(f'{chaincode.shape=}')
    log.debug(f'Chaincode shape: {chaincode.shape}')
    if len(chaincode) == 0:
        log.error('Cannot generate chain code from the image')
        return None

    log.debug(f'{boundary[0]=}')
    is_closed, endpoint = is_completed_chain_code(chaincode, boundary[0])

    if not is_closed:
        log.error('Chain code is not closed')
        log.error(f'Chain code length: {chaincode.shape[0]}')
        return None
    else:
        log.debug('Chain code is closed')
        log.debug(f'{chaincode=}')
        log.debug(f'{chaincode.shape=}')
    return chaincode


def calc_traversal_dist(ai):
    x_ = 0
    y_ = 0
    m = len(ai)
    p = np.zeros((m, 2))  # Initialize position array
    dp = np.zeros((m, 2))  # Initialize displacement array

    for i in range(m):
        dx_ = np.sign(6 - ai[i]) * np.sign(2 - ai[i])
        dy_ = np.sign(4 - ai[i]) * np.sign(ai[i])
        x_ += dx_
        y_ += dy_
        p[i, 0] = x_
        p[i, 1] = y_
        dp[i, 0] = dx_
        dp[i, 1] = dy_

    return p, dp


def calc_traversal_dist_old(ai):
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


def calc_traversal_time(ai):
    t_ = 0
    m = len(ai)
    t = np.zeros(m)
    dt = np.zeros(m)

    for i in range(m):
        dt_ = 1 + ((np.sqrt(2) - 1) / 2) * (1 - (-1) ** ai[i])
        t_ += dt_
        t[i] = t_
        dt[i] = dt_
    return t, dt


def calc_traversal_time_old(ai):
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


def calc_harmonic_coefficients_modify_old(ai, n, mode):
    k = ai.shape[0]
    d = calc_traversal_dist_old(ai)
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
    t = calc_traversal_time_old(ai)
    d = calc_traversal_dist_old(ai)
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


def calc_harmonic_coefficients_modify(ai, n, mode):
    """
    This function calculates the n-th set of four harmonic coefficients.
    The output is [an, bn, cn, dn].
    """
    if mode == 0:
        k = len(ai)
        d = calc_traversal_dist_old(ai)
        edist = d[-1, 0] ** 2 + d[-1, 1] ** 2
        if edist > 2:
            print("Error: Chain code is not closed.")
            return None
        elif edist > 0:
            vect = [-d[-1, 0], -d[-1, 1]]
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

    # Maximum length of chain code
    k = len(ai)

    # Traversal time
    t, dt = calc_traversal_time(ai)

    # Traversal distance
    _, dd = calc_traversal_dist(ai)

    # Basic period of the chain code
    T = t[-1]

    # Store this value to make computation faster
    two_n_pi = 2 * n * np.pi

    # Compute harmonic coefficients: an, bn, cn, dn
    delta_x = dd[0, 0]
    delta_y = dd[0, 1]
    delta_t = dt[0]
    q_x = delta_x / delta_t
    q_y = delta_y / delta_t
    cosp = np.cos(two_n_pi * t[0] / T)
    sinp = np.sin(two_n_pi * t[0] / T)
    sigma_a = q_x * (cosp - 1)
    sigma_b = q_x * sinp
    sigma_c = q_y * (cosp - 1)
    sigma_d = q_y * sinp

    for p in range(1, k):
        delta_x = dd[p, 0]
        delta_y = dd[p, 1]
        delta_t = dt[p]
        q_x = delta_x / delta_t
        q_y = delta_y / delta_t
        cost = np.cos(two_n_pi * t[p] / T)
        sint = np.sin(two_n_pi * t[p] / T)

        sigma_a += q_x * (cost - cosp)
        sigma_b += q_x * (sint - sinp)
        sigma_c += q_y * (cost - cosp)
        sigma_d += q_y * (sint - sinp)

        cosp = cost
        sinp = sint

    r = T / (2 * n ** 2 * np.pi ** 2)

    a = r * sigma_a
    b = r * sigma_b
    c = r * sigma_c
    d = r * sigma_d

    # Assign to output
    # output = np.array([a, b, c, d])
    return a, b, c, d


def calc_dc_components_modify(ai, mode):
    k = ai.shape[0]
    t = calc_traversal_time_old(ai)
    d = calc_traversal_dist_old(ai)
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
    t = calc_traversal_time_old(ai)
    d = calc_traversal_dist_old(ai)
    T = t[k - 1]

    sum_a0 = 0
    sum_c0 = 0
    for p in range(1, k):
        if p >= 1:
            dp_prev = d[p - 1, :]
        else:
            dp_prev = np.zeros(2)
        delta_t = calc_traversal_time_old(ai[p])
        sum_a0 += (d[p, 0] + dp_prev[0]) * delta_t / 2
        sum_c0 += (d[p, 1] + dp_prev[1]) * delta_t / 2

    A0 = sum_a0 / T
    C0 = sum_c0 / T

    return A0, C0, Tk, T


def get_efd_from_chain_code(chain_code, n_order):
    a = np.zeros(n_order)
    b = np.zeros(n_order)
    c = np.zeros(n_order)
    d = np.zeros(n_order)

    # parallel
    with ProcessPoolExecutor() as pool:
        # todo: send_bytes too expensive
        results = list(pool.map(compute_harmonic_coefficients,
                                [chain_code] * n_order, range(n_order)))

    for i, harmonic_coeff in enumerate(results):
        a[i] = harmonic_coeff[0].item()
        b[i] = harmonic_coeff[1].item()
        c[i] = harmonic_coeff[2].item()
        d[i] = harmonic_coeff[3].item()

    A0, C0, Tk, T = calc_dc_components_modify(chain_code, 0)

    log.debug(f'{A0=}, {C0=}, {Tk=}, {T=}')
    return a, b, c, d, A0, C0, Tk, T


def get_efd_from_contour(contour, n_order):
    # link final to start ?
    contour = np.concatenate([contour, contour[[0], :]], axis=0)
    dxy = np.diff(contour, axis=0)
    dt = np.sqrt((dxy ** 2).sum(axis=1))
    t = np.concatenate([([0.0]), np.cumsum(dt)])
    # total length
    T = t[-1]
    phi = (2 * np.pi * t) / T
    orders = np.arange(1, n_order + 1)
    consts = T / (2 * orders * orders * np.pi * np.pi)
    phi = phi * orders.reshape((n_order, -1))

    d_cos_phi = np.cos(phi[:, 1:]) - np.cos(phi[:, :-1])
    d_sin_phi = np.sin(phi[:, 1:]) - np.sin(phi[:, :-1])

    a = consts * np.sum((dxy[:, 0] / dt) * d_cos_phi, axis=1)
    b = consts * np.sum((dxy[:, 0] / dt) * d_sin_phi, axis=1)
    c = consts * np.sum((dxy[:, 1] / dt) * d_cos_phi, axis=1)
    d = consts * np.sum((dxy[:, 1] / dt) * d_sin_phi, axis=1)
    return a, b, c, d, 0, 0, 0, 0


def get_curve_from_efd(a, b, c, d, n_order: int, n_dots: int, A0=0, C0=0):
    """
    Convert efd coefficients to dots of curve
    Args:
        a, b, c, d: [n_order*1] array
        n_order: how many order of coefficients to use
        A0:
        C0:
    Returns:
        dots: [n_dots*2] matrix
    """
    log.debug(f'{A0=} {C0=}')
    log.debug([a[0], b[0], c[0], d[0]])
    assert a.shape[0] == b.shape[0] == c.shape[0] == d.shape[0]
    total_order = a.shape[0]
    # similar to ShyBoy233's
    a = np.reshape(a, (total_order, 1))
    b = np.reshape(b, (total_order, 1))
    c = np.reshape(c, (total_order, 1))
    d = np.reshape(d, (total_order, 1))
    log.debug([a[0], b[0], c[0], d[0]])
    t = np.linspace(0, 1.0, num=n_dots, endpoint=False)
    n = np.arange(1, n_order + 1).reshape((-1, 1))
    x_t = A0 + np.sum(
        a[:n_order] * np.cos(2 * n * np.pi * t) +
        b[:n_order] * np.sin(2 * n * np.pi * t),
        axis=0)
    y_t = C0 + np.sum(
        c[:n_order] * np.cos(2 * n * np.pi * t) +
        d[:n_order] * np.sin(2 * n * np.pi * t),
        axis=0)
    log.debug(f'{x_t[0]=} {y_t[0]=}')
    dots = np.concatenate([x_t.reshape(-1, 1), y_t.reshape(-1, 1)], axis=1)
    return dots


def output_csv(input_file, out_file, dots, a, b, c, d, n_order, n_dots):
    efd_file = out_file
    dot_file = efd_file.with_suffix('.dot.csv')

    # t = np.transpose([a, b, c, d])
    # Hs = np.reshape(t, (1, -1))
    Hs = np.column_stack((a, b, c, d)).ravel()
    efd_header = ["filepath"] + [
        f"{chr(ord('a') + (col - 1) % 4)}{(col - 1) // 4 + 1}"
        for col in range(1, n_order * 4 + 1)
    ]
    dot_header = ['filepath'] + [
        f"{axis}{i}" for i in range(1, n_dots + 1) for axis in ('x', 'y')
    ]
    efd_data = [str(input_file.resolve())] + Hs.tolist()
    dot_data = [str(input_file.resolve())] + dots.ravel().tolist()

    if efd_file.exists():
        log.info('Append data to existed file')
        with open(efd_file, 'a', encoding='utf-8', newline='') as out1:
            writer = csv.writer(out1)
            writer.writerow(efd_data)
        with open(dot_file, 'a', encoding='utf-8', newline='') as out2:
            writer = csv.writer(out2)
            writer.writerow(dot_data)
    else:
        with open(efd_file, 'a', encoding='utf-8', newline='') as out1:
            writer = csv.writer(out1)
            writer.writerow(efd_header)
            writer.writerow(efd_data)
        with open(dot_file, 'a', encoding='utf-8', newline='') as out2:
            writer = csv.writer(out2)
            writer.writerow(dot_header)
            writer.writerow(dot_data)
    return efd_file


def plot_result(out_file, efd_result, max_contour, dots_t, n_dots) -> Path:
    out_img_file = out_file.with_suffix('.out.png')
    # n_order = arg.n_order
    a, b, c, d, A0, C0 = efd_result
    # efd = np.concatenate([a,b,c,d], axis=1)
    # canvas = cv2.imread(str(arg.input), cv2.IMREAD_COLOR)
    # canvas = cv2.resize(canvas, (canvas.shape[1]//4, canvas.shape[0]//4))
    # ax = plt.subplot2grid((2, canvas.shape[0]//2), (canvas.shape[0], 2%canvas.shape[0]//2))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.set_title('contour')
    ax1.set_aspect('equal')
    ax1.plot(max_contour[:, 0], max_contour[:, 1], 'o', linewidth=1)
    ax2.set_title('Normalized ellipse fourier coefficients')
    ax2.set_aspect('equal')
    # ax2.set_xlim(-2, 2)
    # ax2.set_ylim(-2, 2)
    dots_0 = get_curve_from_efd(a, b, c, d, 1, n_dots=n_dots)
    # dots_t = get_curve_from_efd(a, b, c, d, n_order, n_dots=arg.n_dots)
    ax2.plot(dots_0[:, 0], dots_0[:, 1], 'b--', linewidth=1)
    ax2.plot(dots_t[:, 0], dots_t[:, 1], 'r', linewidth=2)
    ax2.plot(dots_t[0, 0], dots_t[0, 1], 'bo', linewidth=1, alpha=0.5)
    plt.savefig(out_img_file)
    # todo: for verify
    # from pyefd import elliptic_fourier_descriptors, plot_efd
    # coeff_other = elliptic_fourier_descriptors(max_contour, normalize=True,
    #                                               order=n_order)
    # a = np.reshape(a, (n_order, 1))
    # b = np.reshape(b, (n_order, 1))
    # c = np.reshape(c, (n_order, 1))
    # d = np.reshape(d, (n_order, 1))
    # coeff_us = np.concatenate([a,b,c,d], axis=1)
    # a2, b2, c2, d2 = coeff_other.T
    # dots_2 = eft_to_curve(a2, b2, c2, d2, n_order, n_dots=512)
    # dots_3 = eft_to_curve(a2, b2, c2, d2, 1, n_dots=512)
    # ax2.plot(dots_2[:, 0], dots_2[:, 1], 'y')
    # ax2.plot(dots_3[:, 0], dots_3[:, 1], 'y--')
    # ax2.plot(dots_3[0, 0], dots_3[0, 1], 'co', linewidth=1, alpha=0.5)
    # plt.show()
    # fig2 = plt.figure(2, (20, 10))
    # plot_efd(coeff_us, n=n_dots)
    # plt.savefig(out_img_file.with_suffix('.out2.png'))
    # fig3 = plt.figure(3, (20, 10))
    # plot_efd(coeff_other, n=n_dots)
    # plt.savefig(out_img_file.with_suffix('.out3.png'))
    # plot_efd(coeff_us, n=n_dots)
    return out_img_file


def get_args():
    arg = argparse.ArgumentParser(description='ElliShape cli')
    arg.add_argument('-i', '-input', dest='input',
                      help='input grayscale image with white as foreground')
    # no headers for easily use in Linux
    arg.add_argument('-I', '-input_list', dest='input_list',
                     help='input list with each line for filename')
    arg.add_argument('-n', '-n_order', dest='n_order',
                      default=64, type=int, help='number of EFD orders')
    arg.add_argument('-N', '-n_dots', dest='n_dots',type=int, default=512,
                     help='number of output dots')
    arg.add_argument('-method', choices=('chain_code', 'dots'), default='dots')
    arg.add_argument('-skip_normalize', action='store_true')
    arg.add_argument('-out', help='output csv file')
    arg.add_argument('-out_image', action='store_true',
                     help='output result image')
    return arg


def check_input(filename: str|Path) ->Path:
    i = Path(filename).resolve()
    if not i.exists() or not i.is_file():
        log.error(f'Cannot find input {i} or it is not a valid file')
        raise SystemExit(-1)
    return i


def init_args(arg_):
    arg = arg_.parse_args()
    log.info(vars(arg))
    if arg.input is None and arg.input_list is None:
        log.critical('Empty input')
        arg_.print_usage()
        raise SystemExit(-1)
    if arg.input and arg.input_list:
        log.warning('Ignore "-input" due to "-input_list')
        arg.input = None
    if arg.input_list:
        arg.input_list = check_input(arg.input_list)
        input_list = arg.input_list.read_text().splitlines()
        arg.input_file_list = [check_input(i) for i in input_list]
        if len(input_list) == 0:
            log.critical(f'Input list {arg.input_list} is empty')
            arg_.print_usage()
            raise SystemExit(-1)
        log.info(f'Input list {arg.input_list} with {len(input_list)} files')
    if arg.input:
        arg.input = check_input(arg.input)
        arg.input_file_list = [arg.input, ]
        log.info(f'Input {arg.input}')
    if arg.method == 'chain_code' and arg.input_list is not None:
        log.error('Batch mode on chain code method may cause program dead!')
        log.error('Continue?')
        if input().lower().strip().startswith('y'):
            pass
        else:
            raise SystemExit(-1)
    if arg.out is None:
        if arg.input is None:
            arg.out = arg.input_list.parent / 'out.csv'
        else:
            arg.out = arg.input.with_suffix('.csv')
    else:
        arg.out = Path(arg.out).resolve()
    if arg.out.exists():
        log.warning(f'Output {arg.out} exists.')
    return arg


def run_main(input_file, out_file, method, n_order, n_dots, skip_normalize,
             out_image):
    # read and convert input
    # todo: resize?
    # img = cv2.resize(img, None, fx=0.125, fy=0.125)
    gray = cv2.imread(str(input_file), cv2.IMREAD_GRAYSCALE)
    log.info(f'Image size: {gray.shape}')
    # get chain_code from contour
    log.info('Finding contours')
    max_contour = get_max_contour(gray)
    if max_contour is None:
        log.error('Cannot find boundary in the image file')
        return -1
    else:
        log.info('Biggest contour found')
    if method == 'chain_code':
        chain_code_result = get_chain_code(gray, max_contour)
        log.info('Got valid chain code')
        if chain_code_result is None:
            log.error('Quit')
            return -1
        # get efd
        efd_result = get_efd_from_chain_code(chain_code_result, n_order)
    else:
        efd_result = get_efd_from_contour(max_contour, n_order)
    log.info('Got efd')
    a, b, c, d, A0, C0, Tk, T = efd_result
    if not skip_normalize:
        normalized_efd = normalize([a, b, c, d, A0, C0])
        log.info('Efd normalized')
    else:
        log.warning('Skip normalization')
        normalized_efd = [a, b, c, d, A0, C0]
    # draw
    a_new, b_new, c_new, d_new, A0_new, C0_new = normalized_efd
    if method == 'chain_code':
        dots = get_curve_old(a_new, b_new, c_new, d_new, A0_new, C0_new, Tk, T,
                             n_order, n_dots)
    else:
        dots = get_curve_from_efd(a_new, b_new, c_new, d_new, n_order, n_dots)
    log.info('Reconstructed curve')
    # plt.plot(dots[:, 0], dots[:, 1], 'r')
    # plt.show()
    output_csv(input_file, out_file, dots, a_new, b_new, c_new, d_new, n_order,
               n_dots)
    log.info(f'Output data: {out_file.resolve()}')
    log.info(f'Output data: {out_file.with_suffix(".dot.csv").resolve()}')
    if out_image:
        out_img_file = plot_result(out_file, normalized_efd, max_contour, dots,
                                   n_dots)
        log.info(f'Output image: {out_img_file.resolve()}')
    return


def cli_main():
    start = timer()
    arg_ = get_args()
    arg = init_args(arg_)
    common_params = (arg.out, arg.method, arg.n_order, arg.n_dots,
                     arg.skip_normalize, arg.out_image)
    if arg.input_list is None:
        run_main(arg.input_file_list[0], *common_params)
    else:
        with ProcessPoolExecutor() as pool:
            for i in arg.input_file_list:
                pool.submit(run_main, i, *common_params)
    end = timer()
    log.info(f'Done with {end-start:.3f} seconds')


if __name__ == '__main__':
    cli_main()
