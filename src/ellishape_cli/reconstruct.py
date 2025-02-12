# from pyefd
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
from sys import argv

def plot_reconstruct(name, coeffs, locus=(0.0, 0.0), contour=None, n=300):
    """
    Args:
        name: fig title
        coeffs: N*4 matrix
        locus: A0, C0
        contour: [[x,y]]
        n: n_harmonic
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
        out_img = plot_reconstruct(name.stem, values, n=512)
        print(out_img)
    return


if __name__ == "__main__":
    main()