import argparse
import os
import numpy as np
from Bio import Phylo
from Bio.Phylo import BaseTree
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import FancyArrowPatch
import matplotlib
from matplotlib import colors

# 使用非交互式后端
matplotlib.use('Agg')

def load_and_preprocess_tree(newick_file):
    """加载Newick文件并预处理树结构"""
    tree = Phylo.read(newick_file, 'newick')
    
    # 清理叶子节点名称
    for leaf in tree.get_terminals():
        leaf.name = leaf.name.strip('"').strip("'").replace("_", " ").rstrip()
    
    return tree

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

def draw_circular_tree(tree, positions, image_dir, img_width, text_size, output_file):
    """绘制环形树并添加文字和图片"""
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.set_aspect('equal')
    ax.axis('off')
    
    # 定义颜色 - 使用原脚本的颜色方案
    COLORS = ["#377eb8", "#ff7f00", "#4daf4a"]
    branch_color = COLORS[2]  # 绿色分支
    
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
            ax.plot([x1, x2], [y1, y2], color=branch_color, linewidth=3, solid_capstyle='round')
    
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
        ax.text(text_x, text_y, node_name, 
                ha=ha, va='center', fontsize=text_size, color='black',
                rotation=np.degrees(text_angle), rotation_mode='anchor',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7, 
                         edgecolor='none'))  # 添加白色半透明背景提高可读性
        
        # 添加图片 - 在文字更外侧
        image_path = os.path.join(image_dir, f"{node_name}.png")
        if os.path.exists(image_path):
            try:
                img = mpimg.imread(image_path)
                
                # 计算图片位置 - 在文字外侧
                img_radius = 1.25  # 图片在更外侧
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
                
                imagebox = OffsetImage(img, zoom=zoom_factor, alpha=1.0)
                ab = AnnotationBbox(imagebox, (img_x, img_y), 
                                  frameon=False,  # 无边框
                                  boxcoords="data",
                                  pad=0)
                ax.add_artist(ab)
                print(f"成功添加图片: {node_name}")
            except Exception as e:
                print(f"警告: 无法加载图片 {image_path}: {e}")
        else:
            print(f"警告: 未找到节点 {node_name} 的图片: {image_path}")
    
    # 设置绘图范围，留出足够空间显示文字和图片
    margin = 0.4
    ax.set_xlim(-1.2 - margin, 1.2 + margin)
    ax.set_ylim(-1.2 - margin, 1.2 + margin)
    
    # 保存图片
    plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')  # 确保背景为白色
    plt.close()
    print(f"进化树已保存至：{output_file}")

def main(newick_file, output_image="tree.png", image_dir=".", img_width=100, text_size=12):
    """主函数：处理树数据并生成可视化"""
    # 加载和预处理树
    tree = load_and_preprocess_tree(newick_file)
    
    # 排序树
    tree = sort_tree_by_leaf_count(tree)
    
    # 计算环形布局
    positions = calculate_circular_layout(tree)
    
    # 绘制树
    draw_circular_tree(tree, positions, image_dir, img_width, text_size, output_image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成带文字和图片的环形进化树")
    parser.add_argument("newick_file", help="输入Newick格式的进化树文件")
    parser.add_argument("-o", "--output", default="tree.png", help="输出图片文件名（默认：tree.png）")
    parser.add_argument("-d", "--image_dir", default=".", help="图片目录路径（默认：当前目录）")
    parser.add_argument("-w", "--img_width", type=int, default=100, help="图片显示宽度（默认：100像素）")
    parser.add_argument("-t", "--text_size", type=int, default=12, help="文字大小（默认：12）")
    
    args = parser.parse_args()
    
    main(args.newick_file, args.output, args.image_dir, args.img_width, args.text_size)
