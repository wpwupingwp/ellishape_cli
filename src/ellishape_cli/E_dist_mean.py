import numpy as np
from collections import defaultdict
from pathlib import Path

def check_input_encode(input_file: Path, encode='utf-8'):
    try:
        line = ''
        with open(input_file, 'r', encoding=encode) as _:
            for n, line in enumerate(_):
                pass
    except UnicodeDecodeError:
        print(f'Encode error found in {n} line, please convert it to utf-8')
        print(line)
        raise SystemExit(-1)


def calculate_category_distances(distance_df, category_df, output_csv):
    """
    逐类别计算类内和类间距离，生成矩阵结果。

    Args:
        distance_df (pd.DataFrame): 距离矩阵，索引和列名为文件名。
        category_df (pd.DataFrame): 类别表，包含文件名和类别。
        output_csv (str): 输出矩阵的 CSV 文件路径。

    Returns:
        None
    """
        # 读取距离矩阵和类别表
        # 清理并读取距离矩阵
    check_input_encode(distance_csv)
    check_input_encode(category_csv)
    # 读取距离矩阵和类别表
    # distance_df = pd.read_csv('distance_csv_clean.csv', index_col=0)
    # category_df = pd.read_csv('category_csv_clean.csv', index_col=0)
    pass

def matrix_to_kinds(sample_names: list, matrix: list, category_csv: Path):
    check_input_encode(category_csv)
    name_kind = dict()
    with open(category_csv, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            name, kind = line.rstrip().split(',')
            name_kind[name] = kind

    kinds_set = set(name_kind.values())
    kinds = sorted(list(kinds_set))

    kind_values = dict()
    kind_values = defaultdict(list)
    # for i in range(len(kinds)):
    #     # mean_list, std_list = [], []
    #     for j in range(len(kinds)):
    #         a_name = kinds[i]
    #         b_name = kinds[j]
    #         pair_name = f'{a_name}-{b_name}'
    #         kind_values[pair_name] = []
            # kind_matrix.append(line_value)
    for x in range(len(sample_names)):
        for y in range(x+1):
            x_kind = name_kind[sample_names[x]]
            y_kind = name_kind[sample_names[y]]
            value = matrix[x][y]
            if x_kind > y_kind:
                x_kind, y_kind = y_kind, x_kind
            kind_values[f'{x_kind}-{y_kind}'].append(value)

    kind_mean_matrix = []
    kind_std_matrix = []
    for i in range(len(kinds)):
        mean_list, std_list = [], []
        for j in range(len(kinds)):
            a_name = kinds[i]
            b_name = kinds[j]
            pair_name = f'{a_name}-{b_name}'
            values = kind_values[pair_name]
            mean_list.append(np.mean(values))
            std_list.append(np.std(values))
        kind_mean_matrix.append(mean_list)
        kind_std_matrix.append(std_list)
    print(kind_mean_matrix)
    print(kind_std_matrix)


    kinds_pair = list(combinations_with_replacement(kinds, 2))

    name_pair_distance = dict()
    for p in kinds_pair:
        name_pair_distance[p] = list()

    with open(distance_csv, 'r') as f:
        header = next(f)
        colnames = header.rstrip().split(',')[1:]
        for line in f:
            name, *value = line.rstrip().split(',')
            if name not in name_kind:
                print(name)
                continue
            a_kind = name_kind[name]
            for idx, v in enumerate(value):
                if colnames[idx] not in name_kind:
                    print(colnames[idx])
                    continue
                b_kind = name_kind[colnames[idx]]
                kind_pair = tuple(sorted([a_kind, b_kind]))
                name_pair_distance[kind_pair].append(v)
    for k in name_pair_distance:
        ar = np.array(name_pair_distance[k], dtype=float)
        mean = np.mean(ar)
        std = np.std(ar)
        name_pair_distance[k] = mean
        print(k, mean, std)
    # print(name_pair_distance)
    with open(output_csv, 'w', encoding='utf-8') as f:
        f.write('Name,'+','.join(kinds)+'\n')
        for k in kinds:
            line = k + ','
            for k2 in kinds:
                pair_name = tuple(sorted([k, k2]))
                if pair_name in name_pair_distance:
                    line += (name_pair_distance[pair_name]).astype(str)
                else: print('error', pair_name)
            line += '\n'
                
    

distance_csv = Path(r"R:\out-e_dist.matrix.csv")  # 距离矩阵 CSV 文件
category_csv = Path(r"R:\xinxibiao.csv")  # 文件类别 CSV 文件
output_csv = Path(r"R:\out.csv")  # 输出结果 CSV 文件

calculate_category_distances(distance_csv, category_csv, output_csv)