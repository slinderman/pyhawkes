import numpy as np
np.random.seed(1234)
import matplotlib.pyplot as plt

from scipy.stats import gamma, beta, betaprime

from pyhawkes.models import DiscreteTimeNetworkHawkesModelSpikeAndSlab
from pybasicbayes.util.text import progprint_xrange

if __name__ == "__main__":
    """
    Create a discrete time Hawkes model and generate from it.

    :return:
    """
    K = 1
    T = 50
    dt = 1.0
    dt_max = 3.0
    # network_hypers = {'C': 1, 'p': 0.5, 'kappa': 3.0, 'alpha': 3.0, 'beta': 1.0/20.0}
    network_hypers = {'c': np.zeros(K, dtype=np.int), 'p': 1.0, 'kappa': 10.0, 'v': 10*3.0}
    bkgd_hypers = {"alpha": 1., "beta": 10.}
    model = DiscreteTimeNetworkHawkesModelSpikeAndSlab(K=K, dt=dt, dt_max=dt_max,
                                                       network_hypers=network_hypers)
    model.generate(T=T)

    # Gibbs sample and then generate new data
    N_samples = 50000
    samples = []
    lps = []
    for itr in progprint_xrange(N_samples, perline=50):
        # Resample the model
        model.resample_model()
        samples.append(model.copy_sample())
        lps.append(model.log_likelihood())

        # Geweke step
        model.data_list.pop()
        model.generate(T=T)


    # Compute sample statistics for second half of samples
    A_samples       = np.array([s.weight_model.A     for s in samples])
    W_samples       = np.array([s.weight_model.W     for s in samples])
    g_samples       = np.array([s.impulse_model.g    for s in samples])
    lambda0_samples = np.array([s.bias_model.lambda0 for s in samples])
    lps             = np.array(lps)


    offset = 0
    A_mean       = A_samples[offset:, ...].mean(axis=0)
    W_mean       = W_samples[offset:, ...].mean(axis=0)
    g_mean       = g_samples[offset:, ...].mean(axis=0)
    lambda0_mean = lambda0_samples[offset:, ...].mean(axis=0)

    print "A mean:        ", A_mean
    print "W mean:        ", W_mean
    print "g mean:        ", g_mean
    print "lambda0 mean:  ", lambda0_mean


    # Plot the log probability over iterations
    plt.figure()
    plt.plot(np.arange(N_samples), lps)
    plt.xlabel("Iteration")
    plt.ylabel("Log probability")

    # Plot the histogram of bias samples
    plt.figure()
    p_lmbda0 = gamma(model.bias_model.alpha, scale=1./model.bias_model.beta)
    _, bins, _ = plt.hist(lambda0_samples[:,0], bins=50, alpha=0.5, normed=True)
    bincenters = 0.5*(bins[1:]+bins[:-1])
    plt.plot(bincenters, p_lmbda0.pdf(bincenters), 'r--', linewidth=1)
    plt.xlabel('lam0')
    plt.ylabel('p(lam0)')

    print "Expected p(A):  ", model.network.P
    print "Empirical p(A): ", A_samples.mean(axis=0)

    # Plot the histogram of weight samples
    plt.figure()
    Aeq1 = A_samples[:,0,0] == 1
    # p_W1 = gamma(model.network.kappa, scale=1./model.network.v[0,0])

    # The marginal distribution of W under a gamma prior on the scale
    # is a beta prime distribution
    # p_W1 = betaprime(model.network.kappa, model.network.alpha, scale=model.network.beta)
    p_W1 = gamma(model.network.kappa, scale=1./model.network.v[0,0])

    _, bins, _ = plt.hist(W_samples[Aeq1,0,0], bins=50, alpha=0.5, normed=True)
    bincenters = 0.5*(bins[1:]+bins[:-1])
    plt.plot(bincenters, p_W1.pdf(bincenters), 'r--', linewidth=1)
    plt.xlabel('W')
    plt.ylabel('p(W | A=1)')

    # Plot the histogram of impulse samples
    plt.figure()
    for b in range(model.B):
        plt.subplot(1,model.B, b+1)
        a = model.impulse_model.gamma[b]
        b = model.impulse_model.gamma.sum() - a
        p_beta11b = beta(a, b)

        _, bins, _ = plt.hist(g_samples[:,0,0,b], bins=20, alpha=0.5, normed=True)
        bincenters = 0.5*(bins[1:]+bins[:-1])
        plt.plot(bincenters, p_beta11b.pdf(bincenters), 'r--', linewidth=1)
        plt.xlabel('g_%d' % b)
        plt.ylabel('p(g_%d)' % b)

    # Plot the histogram of weight scale
    # plt.figure()
    # for c1 in range(model.C):
    #     for c2 in range(model.C):
    #         plt.subplot(model.C, model.C, 1 + c1*model.C + c2)
    #         p_v = gamma(model.network.alpha, scale=1./model.network.beta)
    #
    #         _, bins, _ = plt.hist(v_samples[:,c1,c2], bins=20, alpha=0.5, normed=True)
    #         bincenters = 0.5*(bins[1:]+bins[:-1])
    #         plt.plot(bincenters, p_v.pdf(bincenters), 'r--', linewidth=1)
    #         plt.xlabel('v_{%d,%d}' % (c1,c2))
    #         plt.ylabel('p(v)')
    #
    plt.show()
