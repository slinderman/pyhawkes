import copy

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from pyhawkes.models import DiscreteTimeNetworkHawkesModelGibbs
from pyhawkes.plotting.plotting import plot_network

def test_gibbs_sbm(seed=None):
    """
    Create a discrete time Hawkes model and generate from it.

    :return:
    """
    if seed is None:
        seed = np.random.randint(2**32)

    print "Setting seed to ", seed
    np.random.seed(seed)

    C = 10
    K = 100
    T = 1000
    dt = 1.0
    B = 3

    # Generate from a true model
    true_model = DiscreteTimeNetworkHawkesModelGibbs(C=C, K=K, dt=dt, B=B, beta=1.0/K)
    # S,R = true_model.generate(T=T)
    c = true_model.network.c
    perm = np.argsort(c)

    # Plot the true network
    plt.ion()
    plot_network(true_model.weight_model.A[np.ix_(perm, perm)],
                 true_model.weight_model.W[np.ix_(perm, perm)])
    plt.pause(0.001)


    # Make a new model for inference
    test_model = DiscreteTimeNetworkHawkesModelGibbs(C=C, K=K, dt=dt, B=B, beta=1.0/K)
    # test_model.add_data(S)

    # Gibbs sample
    N_samples = 10
    samples = []
    lps = []
    for itr in xrange(N_samples):
        if itr % 5 == 0:
            print "Iteration: ", itr
        samples.append(copy.deepcopy(test_model.get_parameters()))

        lps.append(test_model.log_probability())

        # Resample the network only
        test_model.network.resample((true_model.weight_model.A,
                                     true_model.weight_model.W))

    plt.ioff()

    # Compute sample statistics for second half of samples
    c_samples       = np.array([c for _,_,_,_,c,_,_,_ in samples])

    print "True c: ", true_model.network.c
    print "Test c: ", c_samples[-10:, :]

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