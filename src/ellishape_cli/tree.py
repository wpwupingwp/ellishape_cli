import csv
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from Bio import Phylo
from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor
from ellishape_cli.global_vars import log

# contour coordinates
h_calc = cv2.createHausdorffDistanceExtractor()
s_calc = cv2.createShapeContextDistanceExtractor()


def parse_args():
    arg = argparse.ArgumentParser()
    arg.add_argument('-i', '-input', dest='input', required=True,
                     help='input csv of value matrix, first column for names')
    arg.add_argument('-o', '-output', dest='output', default='out',
                     help='output prefix')
    arg.add_argument('-s_dist', action='store_true',
                     help='Calculate shape context distance, slow')
    return arg.parse_args()


# df = pd.read_excel(distance_csv)
# print(df)
def read_csv(input_file: Path, simple_name=True):
    # convert input value table to left lower matrix
    with open(input_file, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        header = next(reader)
        lines = list(reader)
    raw_data = np.array(lines)
    names = raw_data[:, 0:1].flatten().tolist()
    if simple_name:
        simple_names = []
        s2 = set()
        for i in names:
            simple = Path(i).stem
            if simple in s2:
                log.warning(f'Found duplicate names: {i} -> {simple}')
                raise SystemExit(-2)
            s2.add(simple)
            simple_names.append(i)
    else:
        simple_names = names
    # names = np.array(data)[:, 0:1].flatten().tolist()
    # todo: use copy?
    data = raw_data[:, 1:].copy()
    return simple_names, data


def matrix2csv(out_file: Path, names:list, dist_dict: dict) -> Path:
    # output matrix to csv file
    # todo
    e_dist_matrix, h_dist_matrix, s_dist_matrix = [], [], []
    for i in range(len(names)):
        e_dist_list, h_dist_list, s_dist_list = [], [], []
        for j in range(i+1):
            pass
    pass


def get_distance(a: np.array, b: np.array, skip_s_dist=True):
    # Euclidean distance
    e_dist = np.linalg.norm(a - b)
    h_dist = h_calc.computeDistance(a, b)
    # s_dist shows long branch between square and rotated square
    if skip_s_dist:
        s_dist = 0
    else:
        s_dist = s_calc.computeDistance(a, b)
    print(e_dist, h_dist, s_dist)
    return e_dist, h_dist, s_dist

def get_distance_matrix(names, data):
    e_dist_matrix, h_dist_matrix, s_dist_matrix = [], [], []
    for i in range(len(data)):
        e_dist_list, h_dist_list, s_dist_list = [], [], []
        for j in range(i+1):
            a_name = names[i]
            b_name = names[j]
            # print(data[i][1:][:10])
            a = np.array(data[i]).reshape(-1,1, 2).astype(float)
            b = np.array(data[j]).reshape(-1,1, 2).astype(float)
            if i == j:
                e_dist, h_dist, s_dist = 0, 0, 0
            else:
                e_dist, h_dist, s_dist = get_distance(a, b)
                # todo: parallel
            # log.info(f'{data[i][0]} {data[j][0]}')
            log.info(f'{a_name} {b_name}')
            log.debug(f'{a.shape=} {b.shape=}')
            log.info(f'{e_dist=:.2f} {h_dist=:.2f} {s_dist=:.2f}')
            e_dist_list.append(e_dist)
            h_dist_list.append(h_dist)
            s_dist_list.append(s_dist)
        e_dist_matrix.append(e_dist_list)
        h_dist_matrix.append(h_dist_list)
        s_dist_matrix.append(s_dist_list)
        # data = np.array(list(reader))
        # distance_matrix_full = np.array(list(reader))
    print('e_dist, h_dist, s_dist')
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


def build_nj_tree(names: list, matrix: list) -> 'Phylo.Tree':
    distance_matrix_obj = DistanceMatrix(names, matrix)
    constructor = DistanceTreeConstructor()
    # build NJ tree
    tree = constructor.nj(distance_matrix_obj)
    # s_dist may be negative
    for t in tree.get_terminals():
        if t.branch_length < 0:
            log.debug(f'{t.branch_length:.2f}')
        t.branch_length = abs(t.branch_length)
    log.info(tree)

    # Phylo.draw(tree, branch_labels=lambda x: f'{x.branch_length:.2f}', do_show=True)
    # plt.savefig(name + '.pdf')
    return tree


def get_tree():
    # init args
    log.info('Start')
    arg = parse_args()
    arg.input = Path(arg.input).absolute()
    assert arg.input.exists()

    names, data = read_csv(arg.input)
    if len(names) == 0:
        log.error('Empty input')
        raise SystemExit(-1)

    e_dist_matrix, h_dist_matrix, s_dist_matrix = get_distance_matrix(names, data)
    for name, matrix in zip(['e_dist', 'h_dist', 's_dist'],
                            [e_dist_matrix, h_dist_matrix, s_dist_matrix]):
        out_tree = arg.input.parent / (arg.output+name+'.nwk')
        out_matrix = arg.input.parent / (arg.output+name+'.matrix.csv')
        tree = build_nj_tree(names, matrix)
        Phylo.write(tree, out_tree, 'newick')
        # matrix2csv(out_matrix, matrix)
    log.info('Done')


if __name__ == '__main__':
    get_tree()
