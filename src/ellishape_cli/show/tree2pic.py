import argparse
import re
from scipy import stats
from pathlib import Path
import numpy as np
from Bio import Phylo
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib

from ellishape_cli.tree import read_csv

matplotlib.use('Agg')

def get_parent(tree, child_clade):
    node_path = tree.get_path(child_clade)
    return node_path[-2]


def load_and_preprocess_tree(newick_file):
    tree = Phylo.read(newick_file, 'newick')
    zero_terminal = list()
    last_parent = None
    for t in tree.get_terminals():
        if t.branch_length == 0:
            zero_terminal.append(t.name)
            parent = get_parent(tree, t)
            if parent == last_parent:
                print(f'Collapse zero branch: {t.name}')
                tree.collapse(t)
            last_parent = parent
    # Phylo.draw_ascii(tree)
    return tree, zero_terminal


def get_terminals(tree) -> tuple[set, list]:
    # get long and normal terminals
    # use scipy
    z_score_limit = 3
    name = list()
    length = list()
    long_terminal = set()
    normal_terminal_len = list()
    normal_terminal = list()
    for i in tree.get_terminals():
        name.append(i.name)
        length.append(i.branch_length)
    z_score = stats.zscore(length)
    for index, score in enumerate(z_score):
        if np.abs(score) >= z_score_limit:
            long_terminal.add(name[index])
        else:
            normal_terminal_len.append((name[index], length[index]))
    # sort by length and select represents
    normal_terminal_len.sort(key=lambda x:x[1])
    # if len(long_terminal) == 0:
    #     normal_terminal = normal_terminal_len[0][0]
    # else:
    #     for i in range(0, len(normal_terminal_len), len(length)//len(long_terminal)):
    #         normal_terminal.append(normal_terminal_len[i][0])
    normal_terminal = [i[0] for i in normal_terminal_len]
    return long_terminal, normal_terminal


def draw_figure(long_terminal, normal_terminal, output, outdir):
    normal_terminal = normal_terminal[:len(long_terminal)]
    n_cols = len(long_terminal)
    if n_cols == 0:
        print('Long terminals not found!')
        return
    figsize = (n_cols*5, 10)
    long_imgs = [(Path(outdir) / f"{i}.png") for i in long_terminal]
    normal_imgs = [(Path(outdir) / f"{i}.png") for i in normal_terminal]

    # Create figure and subplots
    fig, axes = plt.subplots(2, n_cols, figsize=figsize)

    # If there's only one column, axes will be 1D array
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    # Plot first row
    for i, img_path in enumerate(long_imgs):
        img = mpimg.imread(img_path)
        axes[0, i].imshow(img)
        axes[0, i].set_title(img_path.stem+' bad')
        axes[0, i].axis('off')

    # Plot second row
    for i, img_path in enumerate(normal_imgs):
        img = mpimg.imread(img_path)
        axes[1, i].imshow(img)
        axes[1, i].set_title(img_path.stem)
        axes[1, i].axis('off')

    print('Up: bad Down: good')
    plt.tight_layout()
    plt.savefig(output)
    return


def sort_tree_by_leaf_count(tree):
    """按叶子节点数量对树进行排序"""
    def sort_clades(clade):
        if clade.is_terminal():
            return 1
        else:
            return sum(sort_clades(child) for child in clade.clades)
    
    def apply_sorting(clade):
        if not clade.is_terminal():
            clade.clades.sort(key=sort_clades)
            for child in clade.clades:
                apply_sorting(child)
    
    apply_sorting(tree.root)
    return tree

def calculate_circular_layout(tree):
    """计算环形布局的节点位置"""
    terminals = list(tree.get_terminals())
    n_terminals = len(terminals)
    
    # 为每个叶子节点分配等分角度
    angles = np.linspace(0, 2 * np.pi, n_terminals, endpoint=False)
    
    # 创建角度映射
    angle_map = {}
    for i, leaf in enumerate(terminals):
        angle_map[leaf] = angles[i]
    
    # 为内部节点计算角度（子节点角度的平均值）
    def assign_angles(clade):
        if clade.is_terminal():
            return angle_map[clade], angle_map[clade]
        else:
            child_angles = [assign_angles(child) for child in clade.clades]
            min_angle = min(angle for angle, _ in child_angles)
            max_angle = max(angle for _, angle in child_angles)
            # 处理角度环绕情况
            if max_angle - min_angle > np.pi:
                # 调整角度，确保连续性
                adjusted_angles = []
                for angle, _ in child_angles:
                    if angle < np.pi:
                        adjusted_angles.append(angle + 2 * np.pi)
                    else:
                        adjusted_angles.append(angle)
                avg_angle = np.mean(adjusted_angles) % (2 * np.pi)
            else:
                avg_angle = (min_angle + max_angle) / 2
            angle_map[clade] = avg_angle
            return min_angle, max_angle
    
    assign_angles(tree.root)
    
    # 计算半径（基于节点深度）
    depths = {tree.root: 0}
    for clade in tree.find_clades(order='level'):
        for child in clade.clades:
            depths[child] = depths[clade] + (child.branch_length if child.branch_length else 1)
    
    max_depth = max(depths.values()) if depths else 1
    radius_map = {}
    for clade, depth in depths.items():
        if clade.is_terminal():
            radius = 1.0  # 叶子节点在最外层
        else:
            radius = 0.2 + 0.6 * (depth / max_depth)  # 内部节点在内层
        radius_map[clade] = radius
    
    # 转换为笛卡尔坐标
    all_clades = list(tree.find_clades())
    pos = {}
    for clade in all_clades:
        angle = angle_map.get(clade, 0)
        radius = radius_map.get(clade, 0.5)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        pos[clade] = (x, y)
    
    return pos


def draw_leaf(name, dots, terminals, tree, outdir, color='blue' ):
    for leaf in tree.get_terminals():
        if leaf.name in terminals:
            #image_path = Path(leaf.name)
            #image_path = image_path.with_suffix(image_path.suffix+'.png')
            safe_name = re.sub(r'[\\/]+', '_', str(leaf.name))
            image_path = Path(outdir) / f"{safe_name}.png"
            x = np.argwhere(name == leaf.name)[0]
            leaf_dot = dots[x].reshape(-1, 2)
            fig2, ax2 = plt.subplots(figsize=(8, 8))
            ax2.plot(leaf_dot[:, 0], leaf_dot[:, 1], color, linewidth=2)
            ax2.plot(leaf_dot[0, 0], leaf_dot[0, 1], color='black',
                     marker='o', markersize=5, linewidth=1, alpha=0.5)
            plt.savefig(image_path)
    plt.close()
    return


def draw_circular_tree(tree, positions, img_width, text_size, output_file,
                       long_terminal: set):
    """绘制环形树并添加文字和图片"""
    fig, ax = plt.subplots(figsize=(40, 40))
    ax.set_aspect('equal')
    ax.axis('off')
    
    # 定义颜色 - 使用原脚本的颜色方案
    # COLORS = ["#377eb8", "#ff7f00", "#4daf4a"]
    # branch_color = COLORS[2]  # 绿色分支
    
    # 绘制分支 - 使用简单线条，避免默认样式
    for clade in tree.find_clades():
        if clade != tree.root:
            # 找到父节点
            for parent in tree.find_clades():
                if clade in parent.clades:
                    break
            else:
                continue
                
            x1, y1 = positions[parent]
            x2, y2 = positions[clade]
            # 使用plot绘制简单线条，避免默认箭头样式
            ax.plot([x1, x2], [y1, y2], color='black', linewidth=3, solid_capstyle='round')
    
    # 添加文字和图片到叶子节点
    for leaf in tree.get_terminals():
        x, y = positions[leaf]
        node_name = leaf.name
        
        # 计算文字位置和方向
        angle = np.arctan2(y, x)
        text_radius = 1.08  # 文字在节点外侧
        
        # 根据角度决定文字对齐方式
        if -np.pi/2 <= angle <= np.pi/2:
            # 右侧半圆 - 文字左对齐
            text_x = text_radius * np.cos(angle)
            text_y = text_radius * np.sin(angle)
            ha = 'left'
            text_angle = angle
        else:
            # 左侧半圆 - 文字右对齐并翻转
            text_x = text_radius * np.cos(angle)
            text_y = text_radius * np.sin(angle)
            ha = 'right'
            text_angle = angle + np.pi
        
        # 添加文字标签 - 显式设置背景为透明
        # mark long as red
        if leaf.name in long_terminal:
            text_color = 'red'
        else:
            text_color = 'black'
        ax.text(text_x, text_y, node_name,
                ha=ha, va='center', fontsize=text_size, color=text_color,
                rotation=np.degrees(text_angle), rotation_mode='anchor',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7, 
                         edgecolor='none'))  # 添加白色半透明背景提高可读性
        
        # 添加图片 - 在文字更外侧
        # only add figure for long branch
        # use "and false" to skip
        if leaf.name in long_terminal and False:
            l = Path(leaf.name)
            image_path = l.with_suffix(l.suffix+'.png')
            # print(name.dtype, type(leaf.name), leaf.name, name[0])
            # x = np.argwhere(name==leaf.name)[0]
            # leaf_dot = dots[x].reshape(-1, 2)
            # fig2, ax2 = plt.subplots(figsize=(16, 16))
            # ax2.plot(leaf_dot[:, 0], leaf_dot[:, 1], 'c--', linewidth=2)
            # ax2.plot(leaf_dot[0, 0], leaf_dot[0, 1], 'bo', linewidth=1, alpha=0.5)
            # plt.savefig(image_path)
            #
            img = mpimg.imread(image_path)

            # 计算图片位置 - 在文字外侧
            # todo: set radius according to text length
            img_radius = 2  # 图片在更外侧
            img_x = img_radius * np.cos(angle)
            img_y = img_radius * np.sin(angle)

            # 计算合适的缩放因子
            fig_width_inches = 16  # 图形宽度（英寸）
            dpi = 300  # 输出DPI
            target_width_pixels = img_width  # 目标宽度（像素）
            target_width_inches = target_width_pixels / dpi
            # 估算数据坐标中的宽度（假设图形显示范围约为2.4单位）
            data_width = target_width_inches / fig_width_inches * 2.4
            # 根据图片原始尺寸计算缩放
            if img.shape[1] > 0:  # 确保有宽度
                zoom_factor = data_width / img.shape[1] * 2.0  # 调整因子
            else:
                zoom_factor = 0.05

            #imagebox = OffsetImage(img, zoom=zoom_factor, alpha=1.0)
            imagebox = OffsetImage(img, zoom=0.05)
            ab = AnnotationBbox(imagebox, (img_x, img_y),
                              frameon=False,  # 无边框
                              boxcoords="data",
                              pad=0)
            ax.add_artist(ab)
            print(f"成功添加图片: {node_name}")

    # 设置绘图范围，留出足够空间显示文字和图片
    margin = 0.5
    ax.set_xlim(-1.2 - margin, 1.2 + margin)
    ax.set_ylim(-1.2 - margin, 1.2 + margin)
    
    # 保存图片
    plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')  # 确保背景为白色
    plt.close()
    print(f"进化树已保存至：{output_file}")


def draw_new(tree, long_terminal, outdir):
    # —— 动态尺寸：分段线性插值穿过指定锚点，并对两端做稳健边界 ——
    n_leaves = max(1, len(tree.get_terminals()))
    # 锚点（分支数 → 画布英寸）
    # 左端安全下限（60 → 15），右端外推（2600 → 375）
    anchors_leaves = np.array([60, 200, 600, 800, 1600, 2600], dtype=float)
    anchors_size   = np.array([15,  25,  75,  150,  250,  375 ], dtype=float)
    # np.interp 会在端点外进行“夹取”（返回端点值），
    # 因为我们已加入两端锚点，所以这里的插值等价于“分段线性 + 两端稳健边界”
    size_inch = float(np.interp([n_leaves], anchors_leaves, anchors_size)[0])

    # 如需再加保险：限制极端值（可选）
    size_inch = max(12.0, min(size_inch, 380.0))

    fig, ax = plt.subplots(figsize=(size_inch, size_inch))
    matplotlib.rcParams['font.size'] = 8

    # 标红长分支、规范叶名（保持原逻辑）
    for t in tree.get_terminals():
        if t.name in long_terminal:
            t.color = 'red'
        #t.name = '_'.join(re.findall(r'\d{2,}', t.name))
    Phylo.draw(tree, axes=ax)
    plt.savefig(outdir /'result.pdf', bbox_inches='tight')
    plt.close(fig)
    pass


def parse_args():
    # independent function
    parser = argparse.ArgumentParser(description="生成带文字和图片的环形进化树")
    parser.add_argument("newick_file", help="输入Newick格式的进化树文件")
    parser.add_argument('-dot', help='.dot.csv文件')
    parser.add_argument("-o", "--output", default="tree.png",
                        help="输出图片文件名（默认：tree.png）")
    parser.add_argument("-w", "--img_width", type=int, default=100,
                        help="图片显示宽度（默认：100像素）")
    parser.add_argument("-t", "--text_size", type=int, default=10,
                        help="文字大小（默认：10）")
    parser.add_argument("-outdir", "--outdir", default="output_dir",
                        help="输出目录（若不存在则自动创建）")
    args = parser.parse_args()
    return args


def init_args(arg):
    # init paths
    arg.newick_file = Path(arg.newick_file).resolve()
    assert arg.newick_file.exists()
    arg.dot = Path(arg.dot).resolve()
    assert arg.dot.exists()
    arg.output = Path(arg.output).resolve()
    # 创建输出目录（若不存在）
    arg.outdir = Path(arg.outdir).resolve()
    arg.outdir.mkdir(parents=True, exist_ok=True)
    return arg


def main():
    """主函数：处理树数据并生成可视化"""
    # 加载和预处理树
    arg = parse_args()
    arg = init_args(arg)
    name, data = read_csv(arg.dot)
    # remove in windows
    name = [i.split('\\')[-1] for i in name]
    name = np.array(name, dtype=np.str_)
    a, b = data.shape

    dots = data.reshape(a, b//2, 2)

    tree, zero_terminal = load_and_preprocess_tree(arg.newick_file)
    long_terminal, normal_terminal = get_terminals(tree)
    # todo: filter less than 5-10
    # todo: output list
    # 排序树
    tree = sort_tree_by_leaf_count(tree)
    # 计算环形布局
    # positions = calculate_circular_layout(tree)
    # draw shape
    draw_leaf(name, dots, long_terminal, tree, arg.outdir, 'red')
    # only draw few enough shape
    draw_leaf(name, dots, normal_terminal, tree, arg.outdir, 'deepskyblue')
    # draw zero branch length's terminal
    draw_leaf(name, dots, zero_terminal, tree, arg.outdir, 'darkorange')
    draw_figure(long_terminal,
                normal_terminal, arg.outdir /'compare.png', arg.outdir)
    # 绘制树
    # todo: skip or draw rectangular?
    draw_new(tree, long_terminal, arg.outdir)
    # draw_circular_tree(tree, positions, arg.img_width, arg.text_size, arg.output,
    #                    long_terminal)
    csv_out = arg.outdir / (arg.dot.stem+'.result.csv')
    clean_dot_out = arg.outdir / (arg.dot.stem+'.clean.dot')
    with open(csv_out, 'w') as f:
        f.write('Type,Name\n')
        for i in long_terminal:
            f.write(f'Long,{i}\n')
        for i in zero_terminal:
            f.write(f'Zero,{i}\n')
        for i in normal_terminal:
            f.write(f'Normal,{i}\n')
    # write clean dot
    clean_dot = []
    normal_name_set = set(normal_terminal)
    with open(arg.dot) as f:
        clean_dot.append(next(f))
        for line in f:
            name = Path(line.split(',')[0]).stem
            if name in normal_name_set:
                clean_dot.append(line)
    with open(clean_dot_out, 'w') as f:
        for line in clean_dot:
            f.write(line)
    return


if __name__ == "__main__":
    main()
