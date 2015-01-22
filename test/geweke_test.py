import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import gamma, beta

from pyhawkes.models import DiscreteTimeNetworkHawkesModel

def geweke_test():
    """
    Create a discrete time Hawkes model and generate from it.

    :return:
    """
    T = 50
    dt = 1.0
    dt_max = 3.0
    model = DiscreteTimeNetworkHawkesModel(K=1, dt=dt, dt_max=3.0)
    model.generate(T=T)

    # Gibbs sample and then generate new data
    N_samples = 10000
    samples = []
    lps = []
    for itr in xrange(N_samples):
        if itr % 10 == 0:
            print "Iteration: ", itr
        # Resample the model
        samples.append(model.resample_and_copy())
        lps.append(model.log_probability())

        # Geweke step
        model.data_list.pop()
        model.generate(T=T)


    # Compute sample statistics for second half of samples
    A_samples       = np.array([A for A,_,_,_ in samples])
    W_samples       = np.array([W for _,W,_,_ in samples])
    beta_samples    = np.array([b for _,_,b,_ in samples])
    lambda0_samples = np.array([l for _,_,_,l in samples])
    lps             = np.array(lps)

    offset = 0
    A_mean       = A_samples[offset:, ...].mean(axis=0)
    W_mean       = W_samples[offset:, ...].mean(axis=0)
    beta_mean    = beta_samples[offset:, ...].mean(axis=0)
    lambda0_mean = lambda0_samples[offset:, ...].mean(axis=0)

    print "A mean:        ", A_mean
    print "W mean:        ", W_mean
    print "beta mean:     ", beta_mean
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

    # Plot the histogram of weight samples
    plt.figure()
    p_W = gamma(model.weight_model.network.kappa, scale=1./model.weight_model.network.v)
    _, bins, _ = plt.hist(W_samples[:,0,0], bins=20, alpha=0.5, normed=True)
    bincenters = 0.5*(bins[1:]+bins[:-1])
    plt.plot(bincenters, p_W.pdf(bincenters), 'r--', linewidth=1)

    # Plot the histogram of impulse samples
    plt.figure()
    for b in range(model.B):
        plt.subplot(1,model.B, b+1)
        a = model.impulse_model.gamma[b]
        b = model.impulse_model.gamma.sum() - a
        p_beta11b = beta(a, b)

        _, bins, _ = plt.hist(beta_samples[:,0,0,b], bins=20, alpha=0.5, normed=True)
        bincenters = 0.5*(bins[1:]+bins[:-1])
        plt.plot(bincenters, p_beta11b.pdf(bincenters), 'r--', linewidth=1)


    plt.show()


geweke_test()
