# from pyefd
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
from sys import argv

def plot_efd(name, coeffs, locus=(0.0, 0.0), image=None, contour=None, n=300):
    """Plot a ``[2 x (N / 2)]`` grid of successive truncations of the series.

    .. note::

        Requires `matplotlib <http://matplotlib.org/>`_!

    :param numpy.ndarray coeffs: ``[N x 4]`` Fourier coefficient array.
    :param list, tuple or numpy.ndarray locus:
        The :math:`A_0` and :math:`C_0` elliptic locus in [#a]_ and [#b]_.
    :param int n: Number of points to use for plotting of Fourier series.

    """
    N = coeffs.shape[0]
    N_half = int(np.ceil(N / 2))
    n_rows = 2

    t = np.linspace(0, 1.0, n)
    xt = np.ones((n,)) * locus[0]
    yt = np.ones((n,)) * locus[1]

    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(name)
    for n in range(coeffs.shape[0]):
        xt += (coeffs[n, 0] * np.cos(2 * (n + 1) * np.pi * t)) + (
            coeffs[n, 1] * np.sin(2 * (n + 1) * np.pi * t)
        )
        yt += (coeffs[n, 2] * np.cos(2 * (n + 1) * np.pi * t)) + (
            coeffs[n, 3] * np.sin(2 * (n + 1) * np.pi * t)
        )
        ax = plt.subplot2grid((n_rows, N_half), (n // N_half, n % N_half))
        ax.set_title(str(n + 1))

        if image is not None:
            # A background image of shape [rows, cols] gets transposed
            # by imshow so that the first dimension is vertical
            # and the second dimension is horizontal.
            # This implies swapping the x and y axes when plotting a curve.
            if contour is not None:
                ax.plot(contour[:, 1], contour[:, 0], "c--", linewidth=2)
            ax.plot(yt, xt, "r", linewidth=2)
            ax.imshow(image, plt.cm.gray)
        else:
            # Without a background image, no transpose is implied.
            # This case is useful when (x,y) point clouds
            # without relation to an image are to be handled.
            if contour is not None:
                ax.plot(contour[:, 0], contour[:, 1], "c--", linewidth=2)
            ax.plot(xt, yt, "r", linewidth=2)
            ax.axis("equal")
    out = Path(name).with_suffix('.reconstruct.png').absolute()
    plt.savefig(out)
    return out


def main():
    n_harmonic = 35
    data = np.loadtxt(argv[1], delimiter=',', dtype=str)
    for row in data:
        name = Path(row[0])
        values = np.array(row[1:]).astype(np.float32).reshape((n_harmonic, 4))
        out_img = plot_efd(name.stem, values, n=512)
        print(out_img)
    return


if __name__ == "__main__":
    main()