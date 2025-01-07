import csv
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from Bio import Phylo
from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor
# import scipy.cluster.hierarchy as sch
# from scipy.spatial.distance import squareform

# contour coordinates
xy_csv = Path('../data/out.2.csv')

def get_distance(a: np.array, b: np.array) :
    # Euclidean distance
    e_dist = np.linalg.norm(a-b)
    return e_dist, 0, 0
    h_dist = cv2.createHausdorffDistanceExtractor().computeDistance(a, b)
    s_dist = cv2.createShapeContextDistanceExtractor().computeDistance(a, b)
    return e_dist, h_dist, s_dist


# df = pd.read_excel(distance_csv)
# print(df)
with open(xy_csv, 'r', newline='') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    header = next(reader)
    data = list(reader)

e_dist_matrix, h_dist_matrix, s_dist_matrix = [], [], []
for i in range(len(data)):
    e_dist_list, h_dist_list, s_dist_list = [], [], []
    for j in range(i+1):
        a_name = data[i][0]
        b_name = data[i][1]
        # print(data[i][1:][:10])
        a = np.array(data[i][1:]).reshape(-1, 2).astype(float)
        b = np.array(data[j][1:]).reshape(-1, 2).astype(float)
        e_dist, h_dist, s_dist = get_distance(a, b)
        e_dist_list.append(e_dist)
        h_dist_list.append(h_dist)
        s_dist_list.append(s_dist)
    e_dist_matrix.append(e_dist_list)
    h_dist_matrix.append(h_dist_list)
    s_dist_matrix.append(s_dist_list)
    # data = np.array(list(reader))
    # distance_matrix_full = np.array(list(reader))
print('edist')
print(e_dist_matrix)
distance_matrix_full = None
# 将 DataFrame 转换为 Python 的二维列表
# distance_matrix_full = df.values.tolist()
# 将上三角转为下三角
names = distance_matrix_full[:, 0].tolist()
print(names)
distance_matrix_full = np.triu(distance_matrix_full).T  # 转置上三角为下三角
distance_matrix_full = distance_matrix_full.tolist()
# 将矩阵转换为下三角格式
# 下三角格式只保留主对角线及其下方的数据
distance_matrix_lower_triangle = []
for i in range(len(distance_matrix_full)):
    distance_matrix_lower_triangle.append(distance_matrix_full[i][:i + 1])
for i in distance_matrix_lower_triangle:
    print(len(i))
# 样本名称列表，假设有 21 个样本
# 提取标题行作为样本名称并确保每个元素为字符串
# names = df.columns.astype(str).tolist()

# 创建 DistanceMatrix 对象
try:
    distance_matrix_obj = DistanceMatrix(names, distance_matrix_lower_triangle)
    # 使用邻接法构建树
    constructor = DistanceTreeConstructor()
    tree = constructor.nj(distance_matrix_obj)
    print(tree)

    # 使用 Phylo 绘制树
    Phylo.draw(tree)
    Phylo.write(tree, 'tree.nwk', 'newick')

    # 如果你想要以 ASCII 形式显示
    Phylo.draw_ascii(tree)

    # 显示绘制的树（如果需要）
    plt.show()
    # # 可视化树结构
    # sch.dendrogram(tree)
    # plt.show()

except ValueError as e:
    print(f"Error: {e}")
