import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import gamma, beta

from pyhawkes.models import DiscreteTimeNetworkHawkesModelSpikeAndSlab

def geweke_test():
    """
    Create a discrete time Hawkes model and generate from it.

    :return:
    """
    T = 50
    dt = 1.0
    dt_max = 3.0
    network_hypers = {'c': np.array([0], dtype=np.int),
                      'p': 0.5, 'kappa': 3.0, 'v': 15.0}
    model = DiscreteTimeNetworkHawkesModelSpikeAndSlab(K=1, dt=dt, dt_max=dt_max,
                                                       networkhypers=network_hypers)
    model.generate(T=T)

    # Gibbs sample and then generate new data
    N_samples = 10000
    samples = []
    lps = []
    for itr in xrange(N_samples):
        if itr % 10 == 0:
            print "Iteration: ", itr
        # Resample the model
        model.resample_model()
        samples.append(model.copy_sample())
        lps.append(model.log_probability())

        # Geweke step
        model.data_list.pop()
        model.generate(T=T)


    # Compute sample statistics for second half of samples
    A_samples       = np.array([s.weight_model.A     for s in samples])
    W_samples       = np.array([s.weight_model.W     for s in samples])
    g_samples       = np.array([s.impulse_model.g    for s in samples])
    lambda0_samples = np.array([s.bias_model.lambda0 for s in samples])
    c_samples       = np.array([s.network.c          for s in samples])
    p_samples       = np.array([s.network.p          for s in samples])
    v_samples       = np.array([s.network.v          for s in samples])
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
    _, bins, _ = plt.hist(lambda0_samples[:,0], bins=20, alpha=0.5, normed=True)
    bincenters = 0.5*(bins[1:]+bins[:-1])
    plt.plot(bincenters, p_lmbda0.pdf(bincenters), 'r--', linewidth=1)
    plt.xlabel('lam0')
    plt.ylabel('p(lam0)')

    print "Expected p(A):  ", model.network.P
    print "Empirical p(A): ", A_samples.mean(axis=0)

    # Plot the histogram of weight samples
    plt.figure()
    Aeq1 = A_samples[:,0,0] == 1
    p_W1 = gamma(model.network.kappa, scale=1./model.network.v[0,0])
    _, bins, _ = plt.hist(W_samples[Aeq1,0,0], bins=20, alpha=0.5, normed=True)
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

    plt.show()


geweke_test()
