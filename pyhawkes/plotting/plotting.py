import numpy as np
import matplotlib.pyplot as plt

def plot_network(A, W, vmax=None):
    """
    Plot an image of the network
    :param A:
    :param W:
    :return:
    """
    plt.figure()

    # Compute vmax
    if vmax is None:
        vmax = np.amax(A*W)

    im = plt.imshow(A*W, interpolation="none", cmap='gray', vmin=0, vmax=vmax)
    plt.ylabel('k')
    plt.xlabel('k\'')
    plt.title('W_{k \\to k\'}')
    plt.colorbar()
    plt.show()

    return im

