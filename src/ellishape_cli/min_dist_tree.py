from scipy.optimize import brute, minimize, minimize_scalar
from scipy import optimize
from ellishape_cli.global_vars import log
from ellishape_cli.cli import get_curve_from_efd, normalize
from ellishape_cli.tree import get_distance_matrix2, read_csv
from sys import argv
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
from pathlib import Path
from timeit import default_timer as timer
from os import cpu_count


def min_dist_on_angle_old(phi, A_dots, B_efd, n_dots):
    # slow due to get dots from efd
    B_efd_rotated = rotate_efd(B_efd, phi.item())
    B_a2, B_b2, B_c2, B_d2 = np.hsplit(B_efd_rotated, 4)
    # print(B_a2[0], )
    B_dots = get_curve_from_efd(B_a2, B_b2, B_c2, B_d2, B_a2.shape[0], n_dots)
    data = np.vstack([A_dots.ravel(), B_dots.ravel()])
    diff = get_distance_matrix2(data)[0][1]
    return diff


def min_dist_on_angle_old2(phi, A_dots, B_dots):
    # slow due to pdist and squareform
    # n_dots * n_samples
    B_dots_rotated = rotate_dots(B_dots, phi.item())
    data = np.vstack([A_dots.ravel(), B_dots_rotated.ravel()])
    dist_matrix = get_distance_matrix2(data)
    diff = dist_matrix[0][1]
    return diff


def min_dist_on_angle(phi, A_dots, B_dots):
    # only consider angle
    B_dots_rotated = rotate_dots(B_dots, phi.item())
    # n_dots * n_samples
    m, n = B_dots.shape
    factor = np.sqrt(1/(m))
    diff = np.linalg.norm(A_dots.ravel()-B_dots_rotated.ravel()) * factor
    # diff = np.linalg.norm(np.roll(A_dots.ravel(), 64)-B_dots_rotated.ravel()) * factor
    # A2 = A_dots.ravel()
    # B2 = B_dots_rotated.ravel()
    # for i in range(m):
    #     new = np.roll(B2, i)
    #     print(B2.shape, new.shape)
    #     diff2 = np.linalg.norm(A2-new) * factor
    #     print(diff, diff2)
    return diff


def min_dist_on_offset(A_dots, B_dots):
    # only consider offset
    a = timer()
    A_ = A_dots.ravel()
    B_ = B_dots.ravel()
    # n_dots
    n = A_dots.shape[0]
    # (x, y)
    m = n * 2
    # todo: should be 1
    factor = np.sqrt(1/n)
    B_roll_matrix = np.empty((n, m))
    for i in range(0, n):
        B_roll_matrix[i] = np.roll(B_, -i*2)
    # print(B_roll_matrix2.shape, B_roll_matrix.shape)
    dist = np.linalg.norm(A_-B_roll_matrix, axis=1) * factor
    offset = np.argmin(dist)
    b = timer()
    log.warning(f'Min dist on offset cost {b-a:.6f} seconds')
    row_indice = (np.arange(m)+np.arange(m)[::2, None]) % m
    B_roll_matrix2 = B_[row_indice].transpose(0, 1)
    dist2 = np.linalg.norm(A_-B_roll_matrix2, axis=1) * factor
    offset2 = np.argmin(dist2)
    c = timer()
    log.error(f'Min dist on offset cost {c-b:.6f} seconds')
    # print(np.sum(B_roll_matrix2-B_roll_matrix))
    Ak1 = np.fft.fft(A_dots[:,0])
    Ak2 = np.fft.fft(A_dots[:,1])
    Bk1 = np.fft.fft(B_dots[:,0])
    Bk2 = np.fft.fft(B_dots[:,1])
    Cxx = np.fft.ifft(Ak1*np.conj(Bk1)).real
    Cyy = np.fft.ifft(Ak2*np.conj(Bk2)).real
    delta = -2*Cxx-2*Cyy+np.sum(
        np.power(A_dots.ravel(), 2)+np.power(B_dots.ravel(), 2)
    )
    delta = np.sqrt(delta/n)
    offset3 = np.argmin(delta)
    dist3 = delta[offset3]
    # fft direction
    offset3 = n - offset3
    d = timer()
    log.critical(f'Min dist on offset cost {d-c:.6f} seconds')
    # print('offset,dist', offset, dist[offset], offset2, dist2[offset2], offset3, delta[offset3], flush=True)
    return offset, dist[offset]


def min_dist(x0, A_dots, B_dots):
    angle, offset = x0
    B_dots_rotated = rotate_dots(B_dots, angle)
    # n_dots * n_samples
    m, n = B_dots.shape
    factor = np.sqrt(1/(m))
    B_dots_rotated_offset = np.roll(B_dots_rotated, -offset, axis=0)
    diff = np.linalg.norm(A_dots.ravel()-B_dots_rotated_offset.ravel()) * factor
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
        result = minimize_scalar(min_dist_on_angle, args=(A_dots, B_dots),
                                 method='Bounded',
                                 bounds=(0, np.pi*2))
    else:
        result = minimize(min_dist_on_angle, x0, args=(A_dots, B_dots),
                          method=method,
                          bounds=[(0, np.pi*2)])
    return result


def calibrate2(A_dots, B_dots, method='Powell'):
    # x0 = (np.pi, A_dots.shape[0]//2)
    x0 = (0, 0)
    result2 = minimize(min_dist, x0, args=(A_dots, B_dots),
                       method=method,
                       bounds=[(0, np.pi*2), [0, A_dots.shape[0]]])
    return result2


def use_brute(A_dots, B_dots):
    start = timer()
    # brute_result = brute(min_shape_distance_old, args=(A_dots, B_efd, n_dots), Ns=360,
    brute_result = brute(min_dist_on_angle, args=(A_dots, B_dots), Ns=360,
                         ranges=[(0, np.pi * 2)], workers=cpu_count(), 
                         full_output=True)
    end = timer()
    log.warning(f'Brute force on angle cost {end-start:.6f} seconds')
    x_result, y_result, x_list, y_list = brute_result
    log.debug(x_result)
    log.debug(y_result)
    phi = x_result[0].item()
    dist = y_result
    log.info(f'\tRotate B {np.rad2deg(phi):.6f}\u00b0')
    log.info(f'\tMinimize distance {dist:.6f}')
    return phi, dist, brute_result


def use_brute2(A_dots, B_dots):
    start = timer()
    # brute_result = brute(min_shape_distance_old, args=(A_dots, B_efd, n_dots), Ns=360,
    brute_result = brute(min_dist, args=(A_dots, B_dots), Ns=360,
                         ranges=[(0, np.pi * 2), (0, A_dots.shape[0])],
                         workers=cpu_count(),
                         full_output=True)
    end = timer()
    log.warning(f'Brute force on angle cost {end-start:.6f} seconds')
    x_result, y_result, x_list, y_list = brute_result
    phi, offset = x_result.tolist()
    dist = y_result
    log.info(f'\tRotate B {np.rad2deg(phi):.6f}\u00b0')
    log.info(f'\tOffset B {dist} dots')
    log.info(f'\tMinimize distance {dist:.6f}')
    return phi, dist, brute_result


def plot(brute_result, A_dots, B_dots, B_dots2, phi, n_dots):
    font_settings = {'legend.fontsize': 'large', 'axes.labelsize': 'x-large',
                     'xtick.labelsize': 'large', 'ytick.labelsize': 'large',
                     'axes.titlesize': 'xx-large'}
    plt.rcParams.update(font_settings)
    x_result, y_result, x_list, y_list = brute_result

    B_dots3 = rotate_dots(B_dots, phi)
    log.info(f'rotated efd->dots vs rotated dots: {np.sum(B_dots2-B_dots3)}')

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    plt.tight_layout(pad=5)
    axs[0, 0].set_title('Angle-Distance (offset=0)')
    axs[0, 1].set_title('Shape')
    axs[1, 0].set_title('Offset-Distance (angle=0)')
    axs[1, 1].set_title('Offset-Distance (angle=best)')
    axs[0, 0].plot(x_list, y_list)
    axs[0, 0].plot(x_result, y_result, 'ro')
    axs[0, 0].set_xticks(np.arange(0, 9) * np.pi / 4,
                   [rf'$\frac{{{i}\pi}}{{4}}$' for i in range(9)])
    axs[0, 0].set_xlabel('Degree')
    axs[0, 0].set_ylabel('Euclidean distance')

    axs[0, 1].plot(A_dots[:, 0], A_dots[:, 1], 'b', label='A', linewidth=2, alpha=0.8)
    axs[0, 1].plot(B_dots[:, 0], B_dots[:, 1], 'g-.', label='B')
    axs[0, 1].plot(B_dots2[:, 0], B_dots2[:, 1], 'r-', label='B rotated', linewidth=2,
             alpha=0.8)
    axs[0, 1].legend()
    axs[0, 1].set_aspect('equal')

    x_ = np.arange(n_dots)
    factor = np.sqrt(1/n_dots)
    y3 = [factor*np.linalg.norm(A_dots.ravel()-np.roll(B_dots.ravel(), -i*2)) for i in range(n_dots)]
    y4 = [factor*np.linalg.norm(A_dots.ravel()-np.roll(B_dots2.ravel(), -i*2)) for i in range(n_dots)]
    x3_min, y3_min = min_dist_on_offset(A_dots, B_dots)
    x4_min, y4_min = min_dist_on_offset(A_dots, B_dots2)
    # axs[1, 0].plot(x3, y3, 'ro', linewidth=1)
    # axs[1, 1].plot(x3, y4, 'go', linewidth=1)
    axs[1, 0].plot(x_, y3)
    axs[1, 0].plot(x3_min, y3_min, 'ro', label=f'{x3_min}, {y3_min}')
    axs[1, 1].plot(x_, y4)
    axs[1, 1].plot(x4_min, y4_min, 'ro', label=f'{x4_min}, {y4_min}')
    axs[1, 0].set_xlabel('Dot offset')
    axs[1, 1].set_xlabel('Dot offset')
    axs[1, 0].set_ylabel('Distance')
    axs[1, 1].set_ylabel('Distance')
    plt.show()
    return


def plot_3d(A_dots, B_dots):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import colormaps  as cm
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    cmap = cm.get_cmap('coolwarm').copy()
    m_dots = 360
    # x = np.linspace(0, np.pi*2, m_dots)
    x = np.linspace(np.pi*2, 0, m_dots)
    y = np.arange(0, m_dots)
    # y = np.arange(m_dots, 0, -1)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((m_dots, m_dots))
    for i, ix in enumerate(x):
        for j, jy in enumerate(y):
            # _ = np.roll(B_dots_rotated, -1*jy)
            _ = np.roll(B_dots, -1*jy)
            _2 = rotate_dots(_, ix)
            Z[i][j] = np.linalg.norm(A_dots.ravel()-_2.ravel())
    print(np.argmin(np.argmin(Z, axis=0)), np.min(np.min(Z, axis=0)))
    print(np.argmin(np.argmin(Z, axis=1)), np.min(np.min(Z, axis=1)))
    print(np.unravel_index(np.argmin(Z), Z.shape), np.min(Z))
    # surf = ax.plot_surface(X, Y, Z, cmap='bwr')
    surf = ax.plot_surface(X, Y, Z, cmap=cmap, vmin=Z.min()+0.01)
    # ax.contourf(X, Y, Z, zdir='z', offset=0, cmap=cmap)
    # ax.contourf(X, Y, Z, zdir='x', offset=0, cmap=cmap)
    # ax.contourf(X, Y, Z, zdir='y', offset=0, cmap=cmap)
    ax.set_xlabel('Rotate angle')
    ax.set_ylabel('Offset dots')
    ax.set_zlabel('Distance')
    plt.colorbar(surf)
    plt.ion()
    plt.show()
    input('Press any key to continue')
    return


def plot_3d_v2(A_dots, B_dots):
    import plotly.graph_objects as go

    m_dots = 360
    # x = np.linspace(0, np.pi*2, m_dots)
    x = np.linspace(0, np.pi*2, m_dots)
    y = np.arange(0, m_dots)
    # y = np.arange(m_dots, 0, -1)
    Z = np.zeros((m_dots, m_dots))
    for i, ix in enumerate(x):
        for j, jy in enumerate(y):
            # _ = np.roll(B_dots_rotated, -1*jy)
            _ = rotate_dots(B_dots, ix)
            _2 = np.roll(_, -1*jy, axis=0)
            Z[i][j] = np.linalg.norm(A_dots.ravel()-_2.ravel()) * np.sqrt(1/A_dots.shape[0])
    dist_min = np.min(Z)
    log.warning(f'Found {len(Z[Z==dist_min])} best dots')
    print(np.argmin(np.argmin(Z, axis=1)), np.min(np.min(Z, axis=1)))
    print(np.argmin(np.argmin(Z, axis=0)), np.min(np.min(Z, axis=0)))
    print('x,y,z', np.unravel_index(np.argmin(Z), Z.shape), np.min(Z))
    x_min, y_min = np.unravel_index(np.argmin(Z), Z.shape)
    log.critical(f'Min distance {dist_min} on rotate {np.rad2deg(x[x_min]):.6f}\u00b0 and offset {y[y_min]}')
    fig = go.Figure(data=[go.Surface(x=x, y=y, z=Z, colorscale='Portland')])
    # fig.update_traces(
    #     contours_z=dict(show=True, usecolormap=True, highlightcolor='limegreen',
    #                     project_z=True))
    # fig.update_traces(
    #     contours_x=dict(show=True, usecolormap=True, highlightcolor='limegreen',
    #                     project_x=True))
    # fig.update_traces(
    #     contours_y=dict(show=True, usecolormap=True, highlightcolor='limegreen',
    #                     project_y=True))
    fig.update_layout(
        xaxis=dict(title_text='Angle'),
        yaxis=dict(title_text='Offset'),
        title_text=f'Min: x={np.rad2deg(x[x_min])}, y={y[y_min]}, dist={dist_min}',
        font_size=13,
        title_x=0.5,
    )
    fig.show()
    # exit(-1)
    return


def only_find_best_angle(A_dots, B_dots, B_efd, n_dots, deg):
    phi, dist, brute_result = use_brute(A_dots, B_dots)
    B_dots_rotated = rotate_dots(B_dots, phi)
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
            log.info(f'\tRotate B {np.rad2deg(np.pi*2-phi):.6f}\u00b0')
            log.info(f'\tMinimize distance {dist:.6f}')
        except Exception as e:
            # raise
            log.error(f'{m} failed')
            log.error(e)
            continue
        # log.info(result2)
    plot(brute_result, A_dots, B_dots, B_dots_rotated, phi, n_dots)
    log.info(f'Rotate before vs after: '
             f'{np.sum(B_efd - rotate_efd(B_efd, -deg))}')
    return

def p_res(res):
    phi, shift = res.x.tolist()
    dist = res.fun
    log.info(f'\t{res.nit} iterations {res.success}')
    log.info(f'\tRotate B {np.rad2deg(phi):.6f}\u00b0')
    log.info(f'\tOffset B {shift} dots')
    log.info(f'\tMinimize distance {dist:.6f}')


def find_best(A_dots, B_dots, B_efd, deg, offset):
    use_brute2(A_dots, B_dots)
    bounds = [(0, np.pi * 2), (0, A_dots.shape[0])]
    a = timer()
    res = optimize.dual_annealing(min_dist, bounds, args=(A_dots, B_dots))
    b = timer()
    log.warning(f'Dual annealing {b-a:.6f} seconds')
    p_res(res)

    a = timer()
    res = optimize.differential_evolution(min_dist, bounds, args=(A_dots, B_dots))
    b = timer()
    log.warning(f'Differential evolution {b-a:.6f} seconds')
    p_res(res)

    a = timer()
    res = optimize.shgo(min_dist, bounds, args=(A_dots, B_dots))
    b = timer()
    log.warning(f'Shgo {b-a:.6f} seconds')
    p_res(res)

    # a = timer()
    # res = optimize.basinhopping(min_dist,
    #                             (0, 0),
    #                             minimizer_kwargs=dict(method='BFGS',
    #                                                   args=[A_dots, B_dots])
    #                             )
    #
    # b = timer()
    # log.warning(f'Basin hopping {b-a:.6f} seconds')
    # p_res(res)

    for m in ('Powell', 'Nelder-Mead', 'TNC', 'L-BFGS-B'):
        start2 = timer()
        try:
            result2 = calibrate2(A_dots, B_dots, method=m)
            phi, shift = result2.x.tolist()
            dist = result2.fun
            end2 = timer()
            log.warning(f'{m} method cost {end2-start2:.6f} seconds')
            # log.info(result2.message)
            # log.info(result2)
            log.info(f'\t{result2.nit} iterations')
            log.info(f'\tRotate B {np.rad2deg(phi):.6f}\u00b0')
            log.info(f'\tOffset B {shift} dots')
            log.info(f'\tMinimize distance {dist:.6f}')
        except Exception as e:
            # raise
            log.error(f'{m} failed')
            log.error(e)
            raise
            continue
        # log.info(result2)
    log.info(f'Before vs after: '
             f'{np.sum(B_efd - np.roll(rotate_efd(B_efd, -deg), -offset, axis=1))}')
    return


def main():
    input_file = Path(argv[1]).resolve()
    log.info(f'Input {input_file}')
    names, data = read_csv(input_file)
    if len(names) == 0:
        log.error('Empty input')
        raise SystemExit(-1)

    n_dots = 360
    A_efd = data[0].reshape(-1, 4).astype(np.float64)
    B_efd = data[1].reshape(-1, 4).astype(np.float64)
    # B_efd = A_efd.copy()

    offset = 0
    deg = 0
    rad = np.deg2rad(deg)
    log.info(f'Rotate {np.rad2deg(rad):.6f}\u00b0')
    B_efd = rotate_efd(B_efd, rad)

    A_a, A_b, A_c, A_d = np.hsplit(A_efd, 4)
    A_dots = get_curve_from_efd(A_a, A_b, A_c, A_d, A_a.shape[0], n_dots)
    B_a, B_b, B_c, B_d = np.hsplit(B_efd, 4)
    B_dots = get_curve_from_efd(B_a, B_b, B_c, B_d, B_a.shape[0], n_dots)

    B_dots = np.roll(B_dots, offset, axis=1)
    log.info(f'Offset {offset} dots')

    # plot_3d(A_dots, B_dots)
    plot_3d_v2(A_dots, B_dots)
    # min_dist_on_angle(np.array([rad]), A_dots, B_dots)
    # min_dist_on_offset(A_dots, B_dots)

    # only_find_best_angle(A_dots, B_dots, B_efd, n_dots, deg)
    find_best(A_dots, B_dots, B_efd, deg, offset)
    return


if __name__ == '__main__':
    main()
