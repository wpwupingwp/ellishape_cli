from scipy.optimize import brute, minimize, minimize_scalar
from ellishape_cli.global_vars import log
from ellishape_cli.cli import get_curve_from_efd, normalize
from ellishape_cli.tree import get_distance_matrix2, read_csv
from sys import argv
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
from pathlib import Path
from timeit import default_timer as timer


def min_shape_distance_old(phi, A_dots, B_efd, n_dots):
    # slow due to get dots from efd
    B_efd_rotated = rotate_efd(B_efd, phi.item())
    B_a2, B_b2, B_c2, B_d2 = np.hsplit(B_efd_rotated, 4)
    # print(B_a2[0], )
    B_dots = get_curve_from_efd(B_a2, B_b2, B_c2, B_d2, B_a2.shape[0], n_dots)
    data = np.vstack([A_dots.ravel(), B_dots.ravel()])
    diff = get_distance_matrix2(data)[0][1]
    return diff


def min_shape_distance_old2(phi, A_dots, B_dots):
    # slow due to pdist and squareform
    # n_dots * n_samples
    B_dots_rotated = rotate_dots(B_dots, phi.item())
    data = np.vstack([A_dots.ravel(), B_dots_rotated.ravel()])
    dist_matrix = get_distance_matrix2(data)
    diff = dist_matrix[0][1]
    return diff


def min_shape_distance(phi, A_dots, B_dots):
    B_dots_rotated = rotate_dots(B_dots, phi.item())
    # n_dots * n_samples
    m, n = B_dots.shape
    factor = np.sqrt(1/(m))
    diff = np.linalg.norm(A_dots.ravel()-B_dots_rotated.ravel()) * factor
    return diff


def min_shape_distance2(phi, A_dots, B_dots):
    B_dots_rotated = rotate_dots(B_dots, phi.item())
    # data = np.vstack([A_dots.ravel(), B_dots_rotated.ravel()])
    for i in np.arange(128):
        B_dots_rotated_rolled = np.roll(B_dots_rotated, -i, axis=0)
        data = np.vstack([A_dots.ravel(), B_dots_rotated_rolled.ravel()])
        exit -1
    diff = get_distance_matrix2(data)
    x = np.min(squareform(diff))
    # diff = get_distance_matrix2(data)[0][1]
    return x



# def rotate_efd2(efd: np.ndarray, angle: float):
#     """
#     Args:
#         efd: n*4 matrix
#         angle: rad
#     Returns:
#         new_efd: n*4 matrix
#     """
#     # [[a, b, c, d]]
#     # print(efd.shape)
#     rotate_matrix = np.array([[np.cos(angle), -1*np.sin(angle)],
#                               [np.sin(angle), np.cos(angle)]])
#     # [[a, b], [c, d]]
#     efd_matrix = efd.reshape(-1, 2, 2)
#     # print(efd[0])
#     # print(efd_matrix[0])
#     new_efd = np.dot(rotate_matrix, efd_matrix).reshape(-1, 4)
#     return new_efd


def rotate_efd(efd: np.ndarray, angle: float):
    """
    Args:
        efd: n*4 matrix
        angle: rad
    Returns:
        new_efd: n*4 matrix
    """
    # [[a, b, c, d]]
    # todo: simplify
    # todo: rotate orientation
    angle *= -1
    rotate_matrix = np.array([[np.cos(angle), -1*np.sin(angle)],
                              [np.sin(angle), np.cos(angle)]])
    # [[a, b], [c, d]]
    efd_matrix = efd.reshape(-1, 2, 2)
    # temp = np.dot(rotate_matrix, efd_matrix)
    # return temp.reshape(-1, 4)
    # print('old', efd_matrix.shape, efd_matrix[0])
    # [[a, c], [b, d]]
    tmp_matrix = efd_matrix.transpose(0, 2, 1)
    # print('new', dots_matrix.shape, dots_matrix[0])
    # [[a, b, c, d]]
    new_efd = np.dot(tmp_matrix, rotate_matrix).reshape(-1, 4)
    # print(new_efd.shape)
    # [[a, c], [b, d]]
    new_efd = new_efd.reshape(-1, 2, 2)
    # [[a, b], [c, d]]
    new_efd = new_efd.transpose(0, 2, 1)
    # [[a, b, c, d]]
    new_efd = new_efd.reshape(-1, 4)
    return new_efd


def rotate_dots(dots: np.ndarray, angle: float):
    """
    rotate clockwise
    Args:
        dots: n*2 matrix
        angle: rad
    Returns:
        new_dots: n*2 matrix
    """
    angle *= -1
    dots_matrix = dots.reshape(-1, 2)
    rotate_matrix = np.array([[np.cos(angle), -1*np.sin(angle)],
                              [np.sin(angle), np.cos(angle)]])
    new_dots = np.dot(dots_matrix, rotate_matrix).reshape(-1, 2)
    return new_dots


def calibrate(A_dots, B_dots, method='Powell'):
    x0 = np.array([0])
    if method == 'Bounded':
        result = minimize_scalar(min_shape_distance, args=(A_dots, B_dots),
                                  method='Bounded',
                                  bounds=(0, np.pi*2))
    else:
        result = minimize(min_shape_distance, x0, args=(A_dots, B_dots),
                           method=method,
                           bounds=[(0, np.pi*2)])
    return result


def calibrate2(A_dots, B_efd, n_dots, method='Powell'):
    x0 = np.array([0, 0])
    if method == 'Bounded':
        result2 = minimize_scalar(min_shape_distance2, args=(A_dots, B_efd),
                                  method='Bounded',
                                  bounds=(0, np.pi*2))
    else:
        result2 = minimize(min_shape_distance2, x0, args=(A_dots, B_efd),
                           method=method,
                           bounds=[(0, np.pi*2), [0, n_dots]])
    return result2


def use_brute(A_dots, B_dots, B_efd, n_dots):
    start = timer()
    # brute_result = brute(min_shape_distance_old, args=(A_dots, B_efd, n_dots), Ns=360,
    brute_result = brute(min_shape_distance, args=(A_dots, B_dots), Ns=360,
                         ranges=[(0, np.pi * 2)], workers=8, full_output=True)
    end = timer()
    log.warning(f'Brute force on angle cost {end-start:.6f} seconds')
    x_result, y_result, x_list, y_list = brute_result
    log.debug(x_result)
    log.debug(y_result)
    phi = x_result[0].item()
    dist = y_result
    rotated_efd = rotate_efd(B_efd, phi)
    B_a2, B_b2, B_c2, B_d2 = np.hsplit(rotated_efd, 4)
    dots2_2 = get_curve_from_efd(B_a2, B_b2, B_c2, B_d2, B_a2.shape[0], n_dots)
    dots2_3 = rotate_dots(B_dots, phi)
    log.info(f'\trotated efd->dots vs rotated dots: {np.sum(dots2_2-dots2_3)}')
    log.info(f'\tRotate B -{np.rad2deg(np.pi*2-phi):.6f}\u00b0')
    log.info(f'\tMinimize distance {dist:.6f}')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.set_title('Distance')
    ax2.set_title('Shape')
    ax1.plot(x_list, y_list)
    ax1.plot(x_result, y_result, 'ro')
    ax1.set_xticks(np.arange(0, 9) * np.pi / 4,
                   [rf'$\frac{{{i}\pi}}{{4}}$' for i in range(9)])
    ax1.set_xlabel('Degree')
    ax1.set_ylabel('Euclidean distance')

    ax2.plot(A_dots[:, 0], A_dots[:, 1], 'b', label='A', linewidth=2, alpha=0.8)
    ax2.plot(B_dots[:, 0], B_dots[:, 1], 'g-.', label='B')
    ax2.plot(dots2_2[:, 0], dots2_2[:, 1], 'r-', label='B rotated', linewidth=2,
             alpha=0.8)
    ax2.legend()
    ax2.set_aspect('equal')
    plt.show()
    return phi, dist


def main():
    input_file = Path(argv[1]).resolve()
    log.info(f'Input {input_file}')
    names, data = read_csv(input_file)
    if len(names) == 0:
        log.error('Empty input')
        raise SystemExit(-1)

    n_dots = 256
    A_efd = data[0].reshape(-1, 4).astype(np.float64)
    # B_efd = data[1].reshape(-1, 4).astype(np.float64)
    B_efd = A_efd.copy()

    deg = 90.005
    rad = np.deg2rad(deg)
    log.info(f'Rotate {np.rad2deg(rad):.6f}\u00b0')
    B_efd = rotate_efd(B_efd, rad)

    A_a, A_b, A_c, A_d = np.hsplit(A_efd, 4)
    A_dots = get_curve_from_efd(A_a, A_b, A_c, A_d, A_a.shape[0], n_dots)
    B_a, B_b, B_c, B_d = np.hsplit(B_efd, 4)
    B_dots = get_curve_from_efd(B_a, B_b, B_c, B_d, B_a.shape[0], n_dots)

    # min_shape_distance2(np.array([rad]), A_dots, B_dots)

    phi, dist = use_brute(A_dots, B_dots, B_efd, n_dots)
    # log.info(f'Brute result: {phi}, {dist}')
    for m in ('Bounded', 'Powell'):
        # for m in ('Powell',):
        start2 = timer()
        try:
            result2 = calibrate(A_dots, B_dots, method=m)
            phi = result2.x.item()
            dist = result2.fun
            end2 = timer()
            log.warning(f'{m} method on angle cost {end2-start2:.6f} seconds')
            # log.info(result2.message)
            # log.info(result2)
            log.info(f'\t{result2.nit} iterations')
            log.info(f'\tRotate B -{np.rad2deg(np.pi*2-phi):.6f}\u00b0')
            log.info(f'\tMinimize distance {dist:.6f}')
        except Exception as e:
            # raise
            log.error(f'{m} failed')
            log.error(e)
            continue
        # log.info(result2)
    log.info(f'Rotate before vs after: '
             f'{np.sum(B_efd - rotate_efd(B_efd, -deg))}')
    return


if __name__ == '__main__':
    main()
