import numpy as np
import matplotlib.pyplot as plt
from fourier_approx_modify import fourier_approx_modify

def plot_fourier_approx_modify(ai, n, m, normalized=False, color='b', line_width=2, mode=None):
    if mode is None:
        mode = 'fourier'
    # Do Fourier approximation
    x_ = fourier_approx_modify(ai, n, m, normalized, mode)

    # Make it closed contour
    x = np.vstack((x_, x_[0]))

    plt.plot(x[:, 0], x[:, 1], color, linewidth=line_width)
    plt.show()
