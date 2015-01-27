import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, auc_score

from pyhawkes.models import DiscreteTimeNetworkHawkesModelGibbs
from pyhawkes.plotting.plotting import plot_network

def demo(seed=None):
    """
    Create a discrete time Hawkes model and generate from it.

    :return:
    """
    if seed is None:
        seed = np.random.randint(2**32)

    print "Setting seed to ", seed
    np.random.seed(seed)

    C = 2
    K = 50
    T = 1000
    dt = 1.0
    B = 3

    # Generate from a true model
    true_model = DiscreteTimeNetworkHawkesModelGibbs(C=C, K=K, dt=dt, B=B, beta=1.0/K)
    S,R = true_model.generate(T=T)
    c = true_model.network.c
    perm = np.argsort(c)

    # Plot the true network
    plt.ion()
    plot_network(true_model.weight_model.A[np.ix_(perm, perm)],
                 true_model.weight_model.W[np.ix_(perm, perm)])
    plt.pause(0.001)


    # Make a new model for inference
    test_model = DiscreteTimeNetworkHawkesModelGibbs(C=C, K=K, dt=dt, B=B, beta=1.0/K)
    test_model.add_data(S)

    # Plot the true and inferred firing rate
    plt.figure()
    plt.plot(np.arange(T), R[:,0], '-k', lw=2)
    plt.ion()
    ln = plt.plot(np.arange(T), test_model.compute_rate()[:,0], '-r')[0]
    plt.show()

    # Plot the block affiliations
    plt.figure()
    im = plt.imshow(test_model.network.c[perm,:],
                    interpolation="none", cmap="gray",
                    aspect=float(C)/K)
    plt.show()
    plt.pause(0.001)

    # Gibbs sample
    N_samples = 200
    samples = []
    lps = []
    for itr in xrange(N_samples):
        lps.append(test_model.log_probability())
        samples.append(test_model.resample_and_copy())

        # Update plot
        if itr % 5 == 0:
            ln.set_data(np.arange(T), test_model.compute_rate()[:,0])
            plt.title("Iteration %d" % itr)
            plt.pause(0.001)

    plt.ioff()

    # Compute sample statistics for second half of samples
    A_samples       = np.array([A for A,_,_,_,_,_,_,_ in samples])
    W_samples       = np.array([W for _,W,_,_,_,_,_,_ in samples])
    beta_samples    = np.array([b for _,_,b,_,_,_,_,_ in samples])
    lambda0_samples = np.array([l for _,_,_,l,_,_,_,_ in samples])
    c_samples       = np.array([c for _,_,_,_,c,_,_,_ in samples])
    lps             = np.array(lps)

    offset = N_samples // 2
    A_mean       = A_samples[offset:, ...].mean(axis=0)
    W_mean       = W_samples[offset:, ...].mean(axis=0)
    beta_mean    = beta_samples[offset:, ...].mean(axis=0)
    lambda0_mean = lambda0_samples[offset:, ...].mean(axis=0)

    print "A true:        ", true_model.weight_model.A
    print "W true:        ", true_model.weight_model.W
    print "beta true:     ", true_model.impulse_model.g
    print "lambda0 true:  ", true_model.bias_model.lambda0
    print ""
    print "A mean:        ", A_mean
    print "W mean:        ", W_mean
    print "beta mean:     ", beta_mean
    print "lambda0 mean:  ", lambda0_mean

    plt.figure()
    plt.plot(np.arange(N_samples), lps)
    plt.xlabel("Iteration")
    plt.ylabel("Log probability")
    plt.show()

    # Compute the link prediction accuracy curves
    aucs = []
    for A in A_samples:
        aucs.append(auc_score(true_model.weight_model.A.ravel(), A.ravel()))

    plt.figure()
    plt.plot(aucs, '-r')
    plt.xlabel("Iteration")
    plt.ylabel("Link prediction AUC")
    plt.show()

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

# demo(2203329564)
demo()

