import numpy as np
import matplotlib.pyplot as plt

from pyhawkes.models import DiscreteTimeNetworkHawkesModelGibbs, DiscreteTimeStandardHawkesModel
from pyhawkes.plotting.plotting import plot_network

def sample_from_network_hawkes(C, K, T, dt, B):
    # Create a true model
    p = 0.8 * np.eye(C)
    v = 10.0 * np.eye(C) + 20.0 * (1-np.eye(C))
    c = (0.0 * (np.arange(K) < 10) + 1.0 * (np.arange(K)  >= 10)).astype(np.int)
    true_model = DiscreteTimeNetworkHawkesModelGibbs(C=C, K=K, dt=dt, B=B, c=c, p=p, v=v)

    # Plot the true network
    plt.ion()
    plot_network(true_model.weight_model.A,
                 true_model.weight_model.W,
                 vmax=0.5)

    # Sample from the true model
    S,R = true_model.generate(T=T)

    # Return the spike count matrix
    return S, true_model

def demo(seed=None):
    """
    Create a discrete time Hawkes model and generate from it.

    :return:
    """
    if seed is None:
        seed = np.random.randint(2**32)

    print "Setting seed to ", seed
    np.random.seed(seed)

    C = 1       # Number of clusters in the true data
    K = 10      # Number of nodes
    T = 1000    # Number of time bins to simulate
    dt = 1.0    # Time bin size
    B = 3       # Number of basis functions

    # Sample from a sparse network Hawkes model
    S, true_model = sample_from_network_hawkes(C, K, T, dt, B)

    # Make a new model for inference
    test_model = DiscreteTimeStandardHawkesModel(K=K, dt=dt, B=B,
                                                 l2_penalty=0,
                                                 l1_penalty=0)
    test_model.add_data(S)

    # DEBUG: Initialize with the true parameters of the network Hawkes model
    # test_model.initialize_with_gibbs_model(true_model)

    test_model.fit_with_bfgs()

    print "W true:        ", true_model.weight_model.A * true_model.weight_model.W
    print "lambda0 true:  ", true_model.bias_model.lambda0
    print "ll true:       ", true_model.log_likelihood()
    print ""
    print "W test:        ", test_model.W
    print "lambda0 test   ", test_model.bias
    print "ll test:       ", test_model.log_likelihood()

    plot_network(np.ones((K,K)), test_model.W, vmax=0.5)

    plt.ioff()
    plt.show()

demo()