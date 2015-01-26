import matplotlib.pyplot as plt

def plot_network(A, W):
    """
    Plot an image of the network
    :param A:
    :param W:
    :return:
    """
    plt.figure()
    plt.imshow(A*W, interpolation="none", cmap='gray')
    plt.ylabel('k')
    plt.xlabel('k\'')
    plt.title('W_{k \\to k\'}')
    plt.show()

