import numpy as np
import matplotlib.pyplot as plt
from calc_traversal_dist import calc_traversal_dist

def plot_chain_code(chain_code, color='b', line_width=2):
    # Calculate traversal distance
    x_ = calc_traversal_dist(chain_code)
    
    # Starting point is assumed to be [0, 0]
    x = np.vstack(([0, 0], x_))
    plt.plot(x[:, 0], x[:, 1], color, linewidth=line_width)
    plt.show()

