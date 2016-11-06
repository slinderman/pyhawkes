import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from pyhawkes.internals.network import StochasticBlockModel
from pyhawkes.models import DiscreteTimeNetworkHawkesModelSpikeAndSlab

from pybasicbayes.util.text import progprint_xrange

def test_gibbs_sbm(seed=None):
    """
    Create a discrete time Hawkes model and generate from it.

    :return:
    """
    if seed is None:
        seed = np.random.randint(2**32)

    print("Setting seed to ", seed)
    np.random.seed(seed)

    C = 2
    K = 100
    c = np.arange(C).repeat(np.ceil(K/float(C)))[:K]
    T = 1000
    dt = 1.0
    B = 3

    # Generate from a true model
    true_p = np.random.rand(C,C) * 0.25
    true_network = StochasticBlockModel(K, C, c=c, p=true_p, v=10.0)
    true_model = \
        DiscreteTimeNetworkHawkesModelSpikeAndSlab(
                K=K, dt=dt, B=B, network=true_network)

    S,R = true_model.generate(T)

    # Plot the true network
    plt.ion()
    true_im = true_model.plot_adjacency_matrix()
    plt.pause(0.001)


    # Make a new model for inference
    test_network = StochasticBlockModel(K, C, beta=1./K)
    test_model = \
        DiscreteTimeNetworkHawkesModelSpikeAndSlab(
                K=K, dt=dt, B=B, network=test_network)
    test_model.add_data(S)

    # Gibbs sample
    N_samples = 100
    c_samples = []
    lps = []
    for itr in progprint_xrange(N_samples):
        c_samples.append(test_network.c.copy())
        lps.append(test_model.log_probability())

        # Resample the network only
        test_model.network.resample((true_model.weight_model.A,
                                     true_model.weight_model.W))

    c_samples = np.array(c_samples)
    plt.ioff()

    # Compute sample statistics for second half of samples
    print("True c: ", true_model.network.c)
    print("Test c: ", c_samples[-10:, :])

    # Compute the adjusted mutual info score of the clusterings
    amis = []
    arss = []
    for c in c_samples:
        amis.append(adjusted_mutual_info_score(true_model.network.c, c))
        arss.append(adjusted_rand_score(true_model.network.c, c))

    plt.figure()
    plt.plot(np.arange(N_samples), amis, '-r')
    plt.plot(np.arange(N_samples), arss, '-b')
    plt.xlabel("Iteration")
    plt.ylabel("Clustering score")
    plt.show()

test_gibbs_sbm()