import numpy as np
from collections import defaultdict
from pathlib import Path

from ellishape_cli.cli import check_input






    

distance_csv = Path(r"R:\out-e_dist.matrix.csv")  # 距离矩阵 CSV 文件
category_csv = Path(r"R:\xinxibiao.csv")  # 文件类别 CSV 文件
output_csv = Path(r"R:\out.csv")  # 输出结果 CSV 文件

calculate_category_distances(distance_csv, category_csv, output_csv)