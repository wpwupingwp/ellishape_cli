import sys
import csv
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor
from timeit import default_timer as timer

import cv2
import numpy as np
from Bio import Phylo
from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor
from ellishape_cli.global_vars import log

# contour coordinates
h_calc = cv2.createHausdorffDistanceExtractor()
s_calc = cv2.createShapeContextDistanceExtractor()

pool = ProcessPoolExecutor()

def parse_args():
    arg = argparse.ArgumentParser()
    arg.add_argument('-i', '-input', dest='input', required=True,
                     help='input csv of value matrix, first column for names')
    arg.add_argument('-o', '-output', dest='output', default='out',
                     help='output prefix')
    arg.add_argument('-h_dist', action='store_true',
                     help='calculate Hausdorff distance')
    arg.add_argument('-s_dist', action='store_true',
                     help='calculate shape context distance')
    return arg.parse_args()


# df = pd.read_excel(distance_csv)
# print(df)
def read_csv(input_file: Path, simple_name=True):
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


def x(name, a, b):
    pass


def matrix2csv(m_name: str, names:list, matrix: list, arg) -> Path|None:
    out_file = arg.input.parent / f'{arg.output}-{m_name}.matrix.csv'
    if m_name == 'h_dist' and not arg.h_dist:
        log.warning('Skip h_dist matrix')
        return
    if m_name == 's_dist' and not arg.s_dist:
        log.warning('Skip s_dist matrix')
        return
    with open(out_file, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        header = ['Name', ]
        header.extend(names)
        writer.writerow(header)
        for name, row in zip(names, matrix):
            line = [name,]
            line.extend(row)
            writer.writerow(line)
    log.info(f'Output matrix {out_file}')
    return out_file


def get_distance(a_name: str, b_name: str, a_raw: np.array, b_raw: np.array,
                 get_h_dist=False, get_s_dist=False):
    pair_name = f'{a_name}-{b_name}'
    # log.info(f'Pair {pair_name}')
    # log.debug(f'{a.shape=} {b.shape=}')
    a = a_raw.reshape(-1, 1, 2).astype(np.float16)
    b = b_raw.reshape(-1, 1, 2).astype(np.float16)
    if a_name == b_name:
        return pair_name, 0, 0, 0
    # Euclidean distance
    e_dist = np.linalg.norm(a - b)
    # s_dist shows long branch between square and rotated square
    s_dist, h_dist = 0, 0
    if get_h_dist:
        h_dist = h_calc.computeDistance(a, b)
    if get_s_dist:
        s_dist = s_calc.computeDistance(a, b)
    # log.debug(f'{e_dist=:.2f} {h_dist=:.2f} {s_dist=:.2f}')
    log.info(f'{pair_name} done')
    return pair_name, e_dist, h_dist, s_dist




def get_distance_matrix(names, data, arg):
    get_s_dist = arg.s_dist
    get_h_dist = arg.h_dist
    e_dist_matrix, h_dist_matrix, s_dist_matrix = [], [], []
    name_result = dict()
    # parallel
    futures = []

    # log.info('Submit jobs')
    with ProcessPoolExecutor() as executor:
        for i in range(len(data)):
            for j in range(i+1):
                a_name = names[i]
                b_name = names[j]
                a = data[i].copy()
                b = data[j].copy()
                futures.append(executor.submit(
                    get_distance, a_name, b_name, a, b, get_h_dist, get_s_dist))
    # log.info('Submit done')
    for r in futures:
        result = r.result()
        pair_name, e_dist, h_dist, s_dist = result
        name_result[pair_name] = [e_dist, h_dist, s_dist]

    # read result
    for i in range(len(data)):
        e_dist_list, h_dist_list, s_dist_list = [], [], []
        for j in range(i+1):
            a_name = names[i]
            b_name = names[j]
            pair_name = f'{a_name}-{b_name}'
            # print(data[i][1:][:10])
            # a = np.array(data[i]).reshape(-1,1, 2).astype(float)
            # b = np.array(data[j]).reshape(-1,1, 2).astype(float)

            # todo: parallel
            e_dist, h_dist, s_dist = name_result[pair_name]
            # e_dist, h_dist, s_dist = get_distance(a_name, b_name, a, b)
            # log.info(f'{data[i][0]} {data[j][0]}')
            e_dist_list.append(e_dist)
            h_dist_list.append(h_dist)
            s_dist_list.append(s_dist)
        e_dist_matrix.append(e_dist_list)
        h_dist_matrix.append(h_dist_list)
        s_dist_matrix.append(s_dist_list)
        # data = np.array(list(reader))
        # distance_matrix_full = np.array(list(reader))
    # from itertools import chain
    # e_ = list(chain.from_iterable(e_dist_matrix))
    # h_ = list(chain.from_iterable(h_dist_matrix))
    # s_ = list(chain.from_iterable(s_dist_matrix))
    # log.debug('max', np.max(e_), np.max(h_), np.max(s_))
    # log.debug('min', np.min(e_), np.min(h_), np.min(s_))
    # log.debug('mean', np.mean(e_), np.mean(h_), np.mean(s_))
    # log.debug('std', np.std(e_), np.std(h_), np.std(s_))
    # log.debug(e_dist_matrix)
    return e_dist_matrix, h_dist_matrix, s_dist_matrix


def build_nj_tree(m_name, names: list, matrix: list, arg):
    out_tree = arg.input.parent / f'{arg.output}-{m_name}.nwk'
    if m_name == 'h_dist' and not arg.h_dist:
        log.warning('Skip h_dist tree')
        return
    if m_name == 's_dist' and not arg.s_dist:
        log.warning('Skip s_dist tree')
        return
    distance_matrix_obj = DistanceMatrix(names, matrix)
    constructor = DistanceTreeConstructor()
    # build NJ tree
    tree = constructor.nj(distance_matrix_obj)
    # s_dist may be negative
    for t in tree.get_terminals():
        if t.branch_length < 0:
            log.debug(f'{t.branch_length:.2f}')
        t.branch_length = abs(t.branch_length)
    # for t in tree.get_nonterminals():
    #     t.name = ''
    log.debug(tree)
    # Phylo.draw(tree, branch_labels=lambda x: f'{x.branch_length:.2f}', do_show=True)
    # plt.savefig(name + '.pdf')
    Phylo.write(tree, out_tree, 'newick')
    log.info(f'Output tree {out_tree}')
    return tree


def get_tree():
    # init args
    log.info('Start')
    start = timer()
    arg = parse_args()
    arg.input = Path(arg.input).absolute().resolve()
    assert arg.input.exists()

    names, data = read_csv(arg.input)
    read_time = timer()
    r_limit = sys.getrecursionlimit()
    new_limit = len(names) * 10
    sys.setrecursionlimit(new_limit)
    log.warning(f'Set recursion limit from {r_limit} to {new_limit}')
    if len(names) == 0:
        log.error('Empty input')
        raise SystemExit(-1)

    e_dist_matrix, h_dist_matrix, s_dist_matrix = get_distance_matrix(
        names, data, arg)
    matrix_time = timer()

    for m_name, matrix in zip(['e_dist', 'h_dist', 's_dist'],
                            [e_dist_matrix, h_dist_matrix, s_dist_matrix]):
        matrix2csv(m_name, names, matrix, arg)
    with ProcessPoolExecutor() as executor:
        for m_name, matrix in zip(['e_dist', 'h_dist', 's_dist'],
                                  [e_dist_matrix, h_dist_matrix, s_dist_matrix]):
            executor.submit(build_nj_tree,m_name, names, matrix, arg)
            # build_nj_tree(m_name, names, matrix, arg)
    log.info('Done')
    end = timer()
    log.info(f'{len(names)} samples')
    log.info(f'{len(data)*(len(data)-1)/2} pairs')
    log.info(f'Total time elapsed: {end - start:.2f}')
    log.info(f'Read: {read_time - start:.2f}')
    log.info(f'Matrix: {matrix_time - read_time:.2f}')
    log.info(f'Tree: {end - matrix_time:.2f}')
    sys.setrecursionlimit(r_limit)

if __name__ == '__main__':
    get_tree()
