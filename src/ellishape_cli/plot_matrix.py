from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sys import argv
from pathlib import Path
import numpy as np

from ellishape_cli.tree import read_csv, get_distance_matrix2

# dots edist matrix
a_file = Path(argv[1])
# dots min dist matrix
b_file = Path(argv[2])
# efd dist matrix
c_file = Path(argv[3])
_, a = read_csv(a_file, no_same=False)
_, b = read_csv(b_file, no_same=False)
_, c = read_csv(c_file, no_same=False)
efd_e_dist_matrix = get_distance_matrix2(c, False, _type='efd')

vmin = np.min(b)
vmax = np.max(a)
cmap = 'coolwarm'

fig, axes = plt.subplots(2, 2, figsize=(20, 20))
font_settings = {'legend.fontsize': 'large', 'axes.labelsize': 'x-large',
                 'xtick.labelsize': 'large', 'ytick.labelsize': 'large',
                 'axes.titlesize': 'xx-large'}
plt.rcParams.update(font_settings)

img0 = axes[0, 0].imshow(efd_e_dist_matrix, cmap=cmap, interpolation='nearest')
img1 = axes[0, 1].imshow(a, vmin=vmin, vmax=vmax, cmap=cmap,
                      interpolation='nearest')
img2 = axes[1, 0].imshow(b, vmin=vmin, vmax=vmax, cmap=cmap,
                      interpolation='nearest')
img3 = axes[1, 1].imshow(
    a - b, vmin = vmin, vmax = vmax, cmap=cmap,
    interpolation = 'nearest')
t1 = 'EFD distance'
t2 = 'Dots euclidean distance'
t3 = 'Dots minimum distance'
t4 = 'Dots euclidean distance minus minimum distance'
axes[0, 0].set_title(t1, pad=20)
axes[0, 1].set_title(t2, pad=20)
axes[1, 0].set_title(t3, pad=20)
axes[1, 1].set_title(t4, pad=10)
for img, ax in zip([img0, img1, img2, img3], axes.ravel()):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(img, cax=cax, label='Distance')
plt.tight_layout(pad=5)
plt.show()