import csv
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from Bio import Phylo
from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor
from ellishape_cli.global_vars import log

# contour coordinates
xy_csv = Path('../data/out.2.csv')
h_calc = cv2.createHausdorffDistanceExtractor()
s_calc = cv2.createShapeContextDistanceExtractor()


def get_distance(a: np.array, b: np.array) :
    # Euclidean distance
    e_dist = np.linalg.norm(a-b)
    h_dist = h_calc.computeDistance(a, b)
    # s_dist shows long branch between square and rotated square
    s_dist = 0
    # s_dist = s_calc.computeDistance(a, b)
    print(e_dist, h_dist, s_dist)
    return e_dist, h_dist, s_dist


# df = pd.read_excel(distance_csv)
# print(df)
with open(xy_csv, 'r', newline='') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    header = next(reader)
    data = list(reader)

names = np.array(data)[:, 0:1].flatten().tolist()
e_dist_matrix, h_dist_matrix, s_dist_matrix = [], [], []
for i in range(len(data)):
    e_dist_list, h_dist_list, s_dist_list = [], [], []
    for j in range(i+1):
        a_name = names[i]
        b_name = names[j]
        # print(data[i][1:][:10])
        a = np.array(data[i][1:]).reshape(-1,1, 2).astype(float)
        b = np.array(data[j][1:]).reshape(-1,1, 2).astype(float)
        if i == j:
            e_dist, h_dist, s_dist = 0, 0, 0
        else:
            e_dist, h_dist, s_dist = get_distance(a, b)
        log.info(f'{data[i][0]} {data[j][0]}')
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
from itertools import chain
e_ = list(chain.from_iterable(e_dist_matrix))
h_ = list(chain.from_iterable(h_dist_matrix))
s_ = list(chain.from_iterable(s_dist_matrix))
print('max', np.max(e_), np.max(h_), np.max(s_))
print('min', np.min(e_), np.min(h_), np.min(s_))
print('mean', np.mean(e_), np.mean(h_), np.mean(s_))
print('std', np.std(e_), np.std(h_), np.std(s_))
print(e_dist_matrix)
# distance_matrix_full = None
# 将 DataFrame 转换为 Python 的二维列表
# distance_matrix_full = df.values.tolist()
# 将上三角转为下三角
# names = distance_matrix_full[:, 0].tolist()
# print(names)
# distance_matrix_full = np.triu(distance_matrix_full).T  # 转置上三角为下三角
# distance_matrix_full = distance_matrix_full.tolist()
# # 将矩阵转换为下三角格式
# # 下三角格式只保留主对角线及其下方的数据
# distance_matrix_lower_triangle = []
# for i in range(len(distance_matrix_full)):
#     distance_matrix_lower_triangle.append(distance_matrix_full[i][:i + 1])
# for i in distance_matrix_lower_triangle:
#     print(len(i))
# 样本名称列表，假设有 21 个样本
# 提取标题行作为样本名称并确保每个元素为字符串
# names = df.columns.astype(str).tolist()

# 创建 DistanceMatrix 对象
try:
    # distance_matrix_obj = DistanceMatrix(names, distance_matrix_lower_triangle)
    for name, matrix in zip(['e_dist', 'h_dist', 's_dist'],
                            [e_dist_matrix, h_dist_matrix, s_dist_matrix]):
        distance_matrix_obj = DistanceMatrix(names, matrix)
        # 使用邻接法构建树
        constructor = DistanceTreeConstructor()
        tree = constructor.nj(distance_matrix_obj)
        # s_dist may be negative
        for t in tree.get_terminals():
            if t.branch_length < 0:
                log.debug(f'{t.branch_length:.2f}')
            t.branch_length = abs(t.branch_length)
        print(tree)

        # 使用 Phylo 绘制树
        Phylo.draw(tree, branch_labels=lambda x: f'{x.branch_length:.2f}', do_show=True)
        plt.savefig(name+'.pdf')
        Phylo.write(tree, name+'.nwk', 'newick')

        # Phylo.draw_ascii(tree)

        # 显示绘制的树（如果需要）

        # # 可视化树结构
        # sch.dendrogram(tree)
        # plt.show()

except ValueError as e:
    print(f"Error: {e}")
