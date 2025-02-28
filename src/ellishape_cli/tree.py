from pathlib import Path
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from timeit import default_timer as timer

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import brute, minimize, minimize_scalar

from skbio import DistanceMatrix as DistanceMatrix2
from skbio.tree import nj

from ellishape_cli.cli import check_input_csv
from ellishape_cli.global_vars import log

from src.ellishape_cli.cli import normalize

# contour coordinates
h_calc = cv2.createHausdorffDistanceExtractor()
s_calc = cv2.createShapeContextDistanceExtractor()
pool = ProcessPoolExecutor()


def parse_args():
    arg = argparse.ArgumentParser()
    arg.add_argument('-i', '-input', dest='input', required=True,
                     help='input csv of value matrix, first column for names')
    arg.add_argument('-kind', help="input csv of sample's kind, "
                                   "format: sample name,kind name")
    arg.add_argument('-o', '-output', dest='output', default='out',
                     help='output prefix')
    arg.add_argument('-h_dist', action='store_true',
                     help='calculate Hausdorff distance')
    arg.add_argument('-s_dist', action='store_true',
                     help='calculate shape context distance')
    debug = arg.add_argument_group('Debug')
    debug.add_argument('-pca', action='store_true')
    debug.add_argument('-no_factor', action='store_true',
                       help='set factor of euclidean distance matrix to 1')
    return arg.parse_args()


def read_csv(input_file: Path, simple_name=True):
    # read input csv
    # sample name, values
    raw_data = np.loadtxt(input_file, delimiter=',', dtype=str, quotechar='"')
    data = raw_data[1:, 1:]
    names = raw_data[1:, 0:1].flatten().tolist()
    if simple_name:
        simple_names = []
        s2 = set()
        for i in names:
            simple = Path(i).stem
            if simple in s2:
                log.warning(f'Found duplicate names: {i} -> {simple}')
                raise SystemExit(-2)
            s2.add(simple)
            simple_names.append(simple)
    else:
        simple_names = names
    return simple_names, data


def read_kind_csv(category_csv: Path, sample_names: list) -> (dict, list):
    name_kind = dict()
    raw_data = np.loadtxt(category_csv, delimiter=',', dtype=str, quotechar='"')
    kind_list = raw_data[1:, 1:].flatten().tolist()
    with open(category_csv, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            name, kind = line.rstrip().split(',')
            simple_name = Path(name).stem
            name_kind[simple_name] = kind
    kinds_set = set(name_kind.values())
    kinds = sorted(list(kinds_set))
    # check missing
    missing1 = set(sample_names).difference(set(name_kind.keys()))
    missing2 = set(name_kind.keys()).difference(set(sample_names))
    if len(missing1) > 0:
        log.warning(f'{len(missing1)} found in sample list but not in kind list')
        log.info(f'{"\n".join(missing1)}')
        log.critical('Continue?')
        if not input().lower().startswith('y'):
            raise SystemExit(-3)
    if len(missing2) > 0:
        log.warning(f'{len(missing2)} found in kind list but not in sample list')
        log.info(f'{"\n".join(missing2)}')
        log.critical('Continue?')
        if not input().lower().startswith('y'):
            raise SystemExit(-3)
    return name_kind, kinds, kind_list


def tril_to_matrix(old_matrix: list[list[float]]) -> np.ndarray[float]:
    # dtype=float
    width = len(old_matrix)
    new = np.zeros((width, width), dtype=float)
    for idx, row in enumerate(old_matrix):
        new[idx, :] = np.pad(row, (0, width-len(row)))
    new += new.T
    np.fill_diagonal(new, 0.0)
    return new


def matrix_to_tril(old_matrix: np.ndarray[float]) -> list[list[float]]:
    # old_matrix: 2D matrix
    result = []
    m, n = old_matrix.shape
    assert m==n
    for i in range(m):
        row = []
        for j in range(n):
            if j <= i:
                row.append(old_matrix[i, j])
        result.append(row)
    return result


def matrix2csv(m_name: str, names: list, matrix: np.ndarray,
               out_path: Path) -> Path:
    precision = 10
    out_file = out_path.parent / f'{out_path.name}-{m_name}.matrix.csv'
    header = ['Name'] + names
    col_name = np.array([names]).T
    matrix_s = np.strings.mod(f'%.{precision}f', matrix)
    data = np.hstack([col_name, matrix_s])
    all_data = np.vstack([header, data])
    np.savetxt(out_file, all_data, fmt='%s', delimiter=',')
    log.info(f'Output matrix {out_file}')
    return out_file


def get_distance(a_name: str, b_name: str, a_raw: np.array, b_raw: np.array,
                 get_h_dist=False, get_s_dist=False):
    pair_name = f'{a_name}-{b_name}'
    a = a_raw.reshape(-1, 1, 2)
    b = b_raw.reshape(-1, 1, 2)
    if a_name == b_name:
        return pair_name, 0, 0
    # Euclidean distance * sqrt(len(dots))
    # minus = a - b
    # e_dist = np.sqrt(
    #     np.sum(
    #         np.sum(
    #             np.power(minus, 2), axis=1)
    #         /len(minus))
    # )
    # e_dist = np.mean(np.linalg.norm(minus, axis=1))
    s_dist, h_dist = 0, 0
    # Frobenius distance
    # if get_f_dist:
    #     f_dist = np.linalg.norm(minus)
    # s_dist shows long branch between square and rotated square
    if get_h_dist:
        h_dist = h_calc.computeDistance(a, b)
    if get_s_dist:
        s_dist = s_calc.computeDistance(a, b)
    # log.debug(f'{e_dist=:.2f} {h_dist=:.2f} {s_dist=:.2f}')
    log.info(f'{pair_name} done')
    return pair_name, h_dist, s_dist


# def get_task(names, data, h, s):
#     for i in range(len(data)):
#         for j in range(i + 1):
#             a_name = names[i]
#             b_name = names[j]
#             a = data[i].copy()
#             b = data[j].copy()
#             yield a_name, b_name, a, b, h, s
#
#
def get_distance_matrix2(data, no_factor=False):
    # samples, dots*2
    log.debug('Start calculating euclidean distance')
    m, n = data.shape
    data = data.reshape((m, -1)).astype(np.float64)
    s_pdist = pdist(data)
    # todo: cosine?
    if no_factor:
        log.warning('Set factor of euclidean distance matrix to 1')
        factor = 1
    else:
        factor = np.sqrt(1/(n/2))
    # factor = 1
    s_pdist2 = s_pdist * factor
    result = squareform(s_pdist2)
    return result


def get_distance_matrix(names, data, get_s_dist: bool, get_h_dist: bool):
    # slow and use large mem
    h_dist_matrix, s_dist_matrix = [], []
    name_result = dict()
    # parallel
    futures = []
    data = data.astype(np.float64)
    with ProcessPoolExecutor() as executor:
        for i in range(len(data)):
            for j in range(i+1):
                a_name = names[i]
                b_name = names[j]
                a = data[i]
                b = data[j]
                future = executor.submit(
                    get_distance, a_name, b_name, a, b,
                    get_h_dist, get_s_dist)
                futures.append(future)
    for r in futures:
        result = r.result()
        pair_name, h_dist, s_dist = result
        name_result[pair_name] = [h_dist, s_dist]

    # read result
    for i in range(len(data)):
        h_dist_list, s_dist_list = [], []
        for j in range(i+1):
            a_name = names[i]
            b_name = names[j]
            pair_name = f'{a_name}-{b_name}'
            h_dist, s_dist = name_result[pair_name]
            h_dist_list.append(h_dist)
            s_dist_list.append(s_dist)
        h_dist_matrix.append(h_dist_list)
        s_dist_matrix.append(s_dist_list)
    h_dist_matrix_ = tril_to_matrix(h_dist_matrix)
    s_dist_matrix_ = tril_to_matrix(s_dist_matrix)
    return h_dist_matrix_, s_dist_matrix_


# def build_nj_tree(m_name, names: list, matrix: list, arg):
#     from Bio import Phylo
#     from Bio.Phylo.TreeConstruction import DistanceMatrix, \
#         DistanceTreeConstructor
#     out_tree = arg.input.parent / f'{arg.output}-{m_name}.nwk'
#     if m_name == 'h_dist' and not arg.h_dist:
#         log.warning('Skip h_dist tree')
#         return
#     if m_name == 's_dist' and not arg.s_dist:
#         log.warning('Skip s_dist tree')
#         return
#     if m_name == 'f_dist' and not arg.f_dist:
#         log.warning('Skip f_dist tree')
#         return
#     log.info('Start building NJ tree')
#     distance_matrix_obj = DistanceMatrix(names, matrix)
#     constructor = DistanceTreeConstructor()
#     # build NJ tree
#     tree = constructor.nj(distance_matrix_obj)
#     # s_dist may be negative
#     for t in tree.get_terminals():
#         if t.branch_length < 0:
#             log.debug(f'{t.branch_length:.2f}')
#         t.branch_length = abs(t.branch_length)
#     log.debug(tree)
#     # Phylo.draw(tree, branch_labels=lambda x: f'{x.branch_length:.2f}', do_show=True)
#     # plt.savefig(name + '.pdf')
#     Phylo.write(tree, out_tree, 'newick')
#     log.info(f'Output tree {out_tree}')
#     return tree
def tree_from_dist(input_file):
    # todo: edit
    a = np.loadtxt(input_file, delimiter=',', dtype=str)
    b = a[1:, 1:].astype(np.float64)
    b += b.T
    np.fill_diagonal(b, 0)
    a[1:, 1:] = b
    names = a[1:, 0]

    dis_matrix = DistanceMatrix2(b, names)
    tree = nj(dis_matrix)
    tree.write('tree.nwk', 'newick')
    np.savetxt('new.csv', a, delimiter=',', fmt='%s')
    return

def build_nj_tree2(m_name:str, names: np.array, matrix: np.ndarray[float],
                   out_path) -> Path:
    out_tree = out_path.parent / f'{out_path.name}-{m_name}.nwk'
    log.debug('Building NJ tree')
    # matrix = tril_to_matrix(data)
    dis_matrix = DistanceMatrix2(matrix, names)
    tree = nj(dis_matrix)
    tree.write(out_tree, 'newick')
    log.info(f'Output tree {out_tree}')
    return out_tree


def matrix_to_kinds(sample_names: list, matrix: np.ndarray[float],
                    name_kind: dict, kinds: list):
    kind_values = defaultdict(list)
    for x in range(len(sample_names)):
        for y in range(x+1):
            x_kind = name_kind.get(sample_names[x], 'Unknown')
            y_kind = name_kind.get(sample_names[y], 'Unknown')
            if x_kind == 'Unknown' or y_kind == 'Unknown':
                value = 0
            else:
                value = matrix[x][y]
            if x_kind < y_kind:
                x_kind, y_kind = y_kind, x_kind
            kind_values[f'{x_kind}-{y_kind}'].append(value)

    kind_mean_matrix = []
    kind_std_matrix = []
    for i in range(len(kinds)):
        mean_list, std_list = [], []
        for j in range(i+1):
            a_name = kinds[i]
            b_name = kinds[j]
            pair_name = f'{a_name}-{b_name}'
            values = kind_values[pair_name]
            mean_list.append(np.mean(values))
            std_list.append(np.std(values))
        kind_mean_matrix.append(mean_list)
        kind_std_matrix.append(std_list)
    kind_mean_matrix_ = tril_to_matrix(kind_mean_matrix)
    kind_std_matrix_ = tril_to_matrix(kind_std_matrix)
    return kind_mean_matrix_, kind_std_matrix_


def PCA(matrix, kind_list):
    matrix = matrix.astype(np.float64)
    # ref: ShyBoy233
    std = ((matrix - matrix.mean(axis=0)) / matrix.std(axis=0))
    cov = np.cov(matrix, rowvar=False)
    eigen_values, eigen_vectors = np.linalg.eigh(cov)

    order_of_importance = np.argsort(eigen_values)[::-1]
    eigenvalues_sorted = eigen_values[order_of_importance]
    eigenvectors_sorted = eigen_vectors[:, order_of_importance]

    projected = np.matmul(std, eigenvectors_sorted[:, :2])
    effect = np.cumsum(eigenvalues_sorted) / np.sum(
        eigenvalues_sorted)
    # explained_variance = np.concatenate([[0], explained_variance])
    plt.figure(figsize=(5, 5))
    ax = plt.gca()
    ax.axis('equal')
    if not kind_list:
        kind_list = list(range(len(matrix)))
    kind_array = np.array(kind_list)
    for kind in set(kind_list):
        dots = projected[kind_array == kind]
        ax.scatter(dots[:, 0], dots[:, 1], label=kind)
    # for dot, name, color, in zip(projected, kind_list, color_label):
    #     ax.scatter(dot[0], dot[1], label=name, c=color, marker='o')
    # ax.scatter(projected[:, 0], projected[:, 1], c=color_label, label=kind_list,
    #            cmap='Set1')
    ax.set_xlabel(f'PC1 {effect[0]:.2%}')
    ax.set_ylabel(f'PC2 {effect[1]-effect[0]:.2%}')
    ax.legend()
    plt.savefig('pca.pdf')
    return

from ellishape_cli.cli import get_curve_from_efd

def min_shape_distance(phi, A_dots, B_efd, n_dots):
    B_efd_rotated = rotate_efd(B_efd, phi.item())
    B_a2, B_b2, B_c2, B_d2 = np.hsplit(B_efd_rotated, 4)
    # print(B_a2[0], )
    B_dots = get_curve_from_efd(B_a2, B_b2, B_c2, B_d2, B_a2.shape[0], n_dots)
    data = np.vstack([A_dots.ravel(), B_dots.ravel()])
    diff = get_distance_matrix2(data)[0][1]
    return diff


def min_shape_distance2(phi, A_dots, B_dots, n_dots):
    B_dots_rotated = rotate_dots(B_dots, phi.item())
    # print(B_a2[0], )
    data = np.vstack([A_dots.ravel(), B_dots_rotated.ravel()])
    diff = get_distance_matrix2(data)[0][1]
    return diff


def calibrate(A_dots, B_efd, n_dots, method='Powell'):
    x0 = np.array([0])
    if method == 'Bounded':
        result2 = minimize_scalar(min_shape_distance2, args=(A_dots, B_efd, n_dots),
                                  method='Bounded',
                                  bounds=(0, np.pi*2))
    else:
        result2 = minimize(min_shape_distance2, x0, args=(A_dots, B_efd, n_dots),
                          method=method,
                          bounds=[(0, np.pi*2)])
    return result2


def rotate_efd2(efd: np.ndarray, angle: float):
    """
    Args:
        efd: n*4 matrix
        angle: rad
    Returns:
        new_efd: n*4 matrix
    """
    # [[a, b, c, d]]
    # print(efd.shape)
    rotate_matrix = np.array([[np.cos(angle), -1*np.sin(angle)],
                              [np.sin(angle), np.cos(angle)]])
    # [[a, b], [c, d]]
    efd_matrix = efd.reshape(-1, 2, 2)
    # print(efd[0])
    # print(efd_matrix[0])
    new_efd = np.dot(rotate_matrix, efd_matrix).reshape(-1, 4)
    return new_efd


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


def get_min_rotate(A, B, draw=True):
    A_efd = A.reshape(-1, 4).astype(np.float64)
    B_efd = B.reshape(-1, 4).astype(np.float64)
    n_dots = 256
    # B_efd = rotate(A_efd, np.pi/3)
    # print(A_efd.shape, A_efd.ndim)
    # for m in ('Bounded', 'Powell', 'Nelder-Mead', 'L-BFGS-B', 'TNC'):

    A_a, A_b, A_c, A_d = np.hsplit(A_efd, 4)
    A_dots = get_curve_from_efd(A_a, A_b, A_c, A_d, A_a.shape[0], n_dots)
    B_a, B_b, B_c, B_d = np.hsplit(B_efd, 4)
    B_dots = get_curve_from_efd(B_a, B_b, B_c, B_d, B_a.shape[0], n_dots)

    # brute force, 360 times
    # result = brute(min_shape_distance, args=(A_dots, B_efd, n_dots), Ns=360,
    #                ranges=[(0, np.pi * 2)], workers=8, full_output=True)
    result = brute(min_shape_distance2, args=(A_dots, B_dots, n_dots), Ns=360,
                   ranges=[(0, np.pi * 2)], workers=8, full_output=True)
    x_result, y_result, x_list, y_list = result
    log.info(f'Brute force result: '
             f'-{np.rad2deg(np.pi*2-x_result.item()):.6f}\u00b0 '
             f'with {y_result.item():.6f} distance')
    B_result = rotate_efd(B_efd, x_result.item())
    B_a2, B_b2, B_c2, B_d2 = np.hsplit(B_result, 4)
    B_dots2 = get_curve_from_efd(B_a2, B_b2, B_c2, B_d2, B_a2.shape[0],
                                 n_dots)

    for m in ('Bounded', 'Powell'):
        log.warning(m)
        try:
            # c_result = calibrate(A_dots, B_efd, n_dots, method=m)
            c_result = calibrate(A_dots, B_dots, n_dots, method=m)
        except Exception:
            log.error(f'{m} failed')
            continue
        # assert c_result is not None
        log.info(m)
        # log.info(c_result.message)
        log.info(f'{c_result.nit} iterations')
        log.info(f'Rotate B -{np.rad2deg(np.pi*2-c_result.x.item()):.6f}\u00b0')
        log.info(f'Minimize distance {c_result.fun:.6f}')
        # log.info(c_result)
    if draw:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.set_title('Distance')
        ax2.set_title('Shape')
        ax1.plot(x_list, y_list)
        ax1.plot(x_result, y_result, 'ro')
        ax1.set_xticks(np.arange(0,9)*np.pi/4,
                       [rf'$\frac{{{i}\pi}}{{4}}$' for i in range(9)])
        ax1.set_xlabel('Degree')
        ax1.set_ylabel('Euclidean distance')

        ax2.plot(A_dots[:, 0], A_dots[:, 1], 'b', label='A', linewidth=2, alpha=0.8)
        ax2.plot(B_dots[:, 0], B_dots[:, 1], 'g-.', label='B')
        ax2.plot(B_dots2[:, 0], B_dots2[:, 1], 'r-', label='B rotated', linewidth=2, alpha=0.8)
        ax2.legend()
        ax2.set_aspect('equal')
        plt.show()
        # plt.savefig('rotate_result.png')
    return


def get_tree():
    # init args
    log.info('Start')
    start = timer()
    arg = parse_args()
    arg.input = Path(arg.input).absolute().resolve()
    out_path = arg.input.parent / arg.output
    check_input_csv(arg.input)

    names, data = read_csv(arg.input)
    if len(names) == 0:
        log.error('Empty input')
        raise SystemExit(-1)

    # optimize?
    raw = data[0].reshape(-1, 4).astype(np.float64)
    raw2 = data[1].reshape(-1, 4).astype(np.float64)
    # raw2 = raw.copy()
    deg = 45.005
    rad = np.deg2rad(deg)
    log.info(f'Rotate {np.rad2deg(rad):.6f}\u00b0')
    new = rotate_efd(raw2, rad)
    get_min_rotate(raw, new)
    print(np.sum(new-rotate_efd(new, -deg)))
    raise SystemExit
    # test optimize


    if arg.kind is not None:
        arg.kind = Path(arg.kind).absolute().resolve()
        check_input_csv(arg.kind)
        name_kind, kinds, kind_list = read_kind_csv(arg.kind, names)
    else:
        name_kind, kinds, kind_list = dict(), [], []

    read_time = timer()
    if arg.pca:
        PCA(data, kind_list)
    pca_time = timer()

    e_dist_matrix = get_distance_matrix2(data, arg.no_factor)
    if arg.s_dist or arg.h_dist:
        h_dist_matrix, s_dist_matrix = get_distance_matrix(
            names, data, arg.s_dist, arg.h_dist)
    else:
        h_dist_matrix, s_dist_matrix = None, None

    matrix_time = timer()

    for m_name, matrix in zip(
            ['e_dist', 'h_dist', 's_dist'],
            [e_dist_matrix, h_dist_matrix, s_dist_matrix]):
        if matrix is None:
            continue
        matrix2csv(m_name, names, matrix, out_path)
        build_nj_tree2(m_name, names, matrix, out_path)
        if arg.kind is not None:
            kind_mean_matrix, kind_std_matrix = matrix_to_kinds(
                names, matrix, name_kind, kinds)
            m2_name = f'{m_name}-kind_mean'
            m3_name = f'{m_name}-kind_std'
            matrix2csv(m2_name, kinds, kind_mean_matrix, out_path)
            matrix2csv(m3_name, kinds, kind_std_matrix, out_path)
            build_nj_tree2(m2_name, kinds, kind_mean_matrix, out_path)
    end = timer()

    log.info(f'{len(names):<10} samples')
    log.info(f'{len(data)*(len(data)-1)//2:<10d} pairs')
    log.info(f'Total time elapsed: {end - start:.3f}')
    log.info(f'\t{"Read:":<15} {read_time - start:.3f}')
    log.info(f'\t{"PCA:":<15} {pca_time - read_time:.3f}')
    log.info(f'\t{"Matrix:":<15} {matrix_time - pca_time:.3f}')
    # log.info(f'Write: {write_time - matrix_time:.3f}')
    # diff = np.sum(p_result - e_dist_matrix)
    # log.info(f'difference: {diff:.12f}')
    log.info(f'\t{"Tree and write:":<15} {end - matrix_time:.3f}')
    log.info('Done')
    return

if __name__ == '__main__':
    get_tree()
