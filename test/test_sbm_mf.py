import copy

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from pyhawkes.models import DiscreteTimeNetworkHawkesModelSpikeAndSlab, \
                            DiscreteTimeNetworkHawkesModelGammaMixture
from pyhawkes.plotting.plotting import plot_network

def test_sbm_mf(seed=None):
    """
    Create a discrete time Hawkes model and generate from it.

    :return:
    """
    if seed is None:
        seed = np.random.randint(2**32)

    print "Setting seed to ", seed
    np.random.seed(seed)

    C = 5
    K = 50
    T = 1000
    dt = 1.0
    B = 3

    # Generate from a true model
    true_model = DiscreteTimeNetworkHawkesModelSpikeAndSlab(C=C, K=K, dt=dt, B=B, tau1=0.5, tau0=0.5, beta=1.0/K)
    c = true_model.network.c
    perm = np.argsort(c)
    #
    # Plot the true network
    plt.ion()
    plot_network(true_model.weight_model.A[np.ix_(perm, perm)],
                 true_model.weight_model.W[np.ix_(perm, perm)])
    plt.pause(0.001)

    # Make a new model for inference
    test_model = DiscreteTimeNetworkHawkesModelGammaMixture(C=C, K=K, dt=dt, B=B, tau1=0.5, tau0=0.5, beta=1.0/K)
    test_model.weight_model.initialize_from_gibbs(true_model.weight_model.A,
                                                  true_model.weight_model.W)

    # Plot the block probabilities
    plt.figure()
    im = plt.imshow(test_model.network.mf_m[perm,:],
                    interpolation="none", cmap="Greys",
                    aspect=float(C)/K)
    plt.xlabel('C')
    plt.ylabel('K')
    plt.show()
    plt.pause(0.001)

    # Run mean field updates for the SBM given a fixed network
    N_iters = 50
    c_samples = []
    vlbs = []
    for itr in xrange(N_iters):
        if itr % 5 == 0:
            print "Iteration: ", itr

        # Update the plot
        im.set_data(test_model.network.mf_m[perm,:])
        plt.pause(0.001)


        # Resample from meanfield distribution
        test_model.network.resample_from_mf()
        c_samples.append(copy.deepcopy(test_model.network.c))
        vlbs.append(test_model.network.get_vlb() + test_model.weight_model.get_vlb())

        if itr > 0:

            if vlbs[-1] - vlbs[-2] < -1e-3:
                print "VLBS are not increasing"
                print np.array(vlbs)
                # import pdb; pdb.set_trace()
                # raise Exception("VLBS are not increasing!")


        # Take a mean field step
        test_model.network.meanfieldupdate(test_model.weight_model)

    plt.ioff()

    # Compute sample statistics for second half of samples
    c_samples = np.array(c_samples)
    vlbs = np.array(vlbs)

    print "True c: ", true_model.network.c
    print "Test c: ", c_samples[-10:, :]

    # Compute the adjusted mutual info score of the clusterings
    amis = []
    arss = []
    for c in c_samples:
        amis.append(adjusted_mutual_info_score(true_model.network.c, c))
        arss.append(adjusted_rand_score(true_model.network.c, c))

    plt.figure()
    plt.plot(np.arange(N_iters), amis, '-r')
    plt.plot(np.arange(N_iters), arss, '-b')
    plt.xlabel("Iteration")
    plt.ylabel("Clustering score")

    plt.figure()
    plt.plot(np.arange(N_iters), vlbs)
    plt.xlabel("Iteration")
    plt.ylabel("VLB")

    plt.show()
    #
    # plt.close('all')

test_sbm_mf(3055650126)