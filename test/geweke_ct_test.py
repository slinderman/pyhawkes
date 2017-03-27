import numpy as np
np.random.seed(1234)
import matplotlib.pyplot as plt

from scipy.stats import gamma, t, probplot

from pyhawkes.models import ContinuousTimeNetworkHawkesModel
from pybasicbayes.util.text import progprint_xrange

def test_geweke():
    """
    Create a discrete time Hawkes model and generate from it.

    :return:
    """
    K = 1
    T = 50.0
    dt = 1.0
    dt_max = 3.0
    # network_hypers = {'C': 1, 'p': 0.5, 'kappa': 3.0, 'alpha': 3.0, 'beta': 1.0/20.0}
    network_hypers = {'c': np.zeros(K, dtype=np.int), 'p': 0.5, 'kappa': 10.0, 'v': 10*3.0}
    bkgd_hypers = {"alpha": 1., "beta": 10.}
    model = ContinuousTimeNetworkHawkesModel(K=K, dt_max=dt_max,
                                             network_hypers=network_hypers)

    model.generate(T=T)

    # Gibbs sample and then generate new data
    N_samples = 1000
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
    A_samples       = np.array([s.weight_model.A       for s in samples])
    W_samples       = np.array([s.weight_model.W       for s in samples])
    mu_samples       = np.array([s.impulse_model.mu    for s in samples])
    tau_samples       = np.array([s.impulse_model.tau  for s in samples])
    lambda0_samples = np.array([s.bias_model.lambda0   for s in samples])
    lps             = np.array(lps)


    offset = 0
    A_mean       = A_samples[offset:, ...].mean(axis=0)
    W_mean       = W_samples[offset:, ...].mean(axis=0)
    mu_mean      = mu_samples[offset:, ...].mean(axis=0)
    tau_mean     = tau_samples[offset:, ...].mean(axis=0)
    lambda0_mean = lambda0_samples[offset:, ...].mean(axis=0)

    print("A mean:        ", A_mean)
    print("W mean:        ", W_mean)
    print("mu mean:       ", mu_mean)
    print("tau mean:      ", tau_mean)
    print("lambda0 mean:  ", lambda0_mean)


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

    print("Expected p(A):  ", model.network.P)
    print("Empirical p(A): ", A_samples.mean(axis=0))

    # Plot the histogram of weight samples
    plt.figure()
    Aeq1 = A_samples[:,0,0] == 1
    # p_W1 = gamma(model.network.kappa, scale=1./model.network.v[0,0])

    # p_W1 = betaprime(model.network.kappa, model.network.alpha, scale=model.network.beta)
    p_W1 = gamma(model.network.kappa, scale=1./model.network.v[0,0])

    if np.sum(Aeq1) > 0:
        _, bins, _ = plt.hist(W_samples[Aeq1,0,0], bins=50, alpha=0.5, normed=True)
        bincenters = 0.5*(bins[1:]+bins[:-1])
        plt.plot(bincenters, p_W1.pdf(bincenters), 'r--', linewidth=1)
        plt.xlabel('W')
        plt.ylabel('p(W | A=1)')

    # Plot the histogram of impulse precisions
    plt.figure()
    p_tau = gamma(model.impulse_model.alpha_0, scale=1./model.impulse_model.beta_0)

    _, bins, _ = plt.hist(tau_samples[:,0,0], bins=50, alpha=0.5, normed=True)
    bincenters = 0.5*(bins[1:]+bins[:-1])
    plt.plot(bincenters, p_tau.pdf(bincenters), 'r--', linewidth=1)
    plt.xlabel('tau')
    plt.ylabel('p(tau)')

    # Plot the histogram of impulse means
    plt.figure()
    p_mu = t(df=2*model.impulse_model.alpha_0,
             loc=model.impulse_model.mu_0,
             scale=np.sqrt(model.impulse_model.beta_0/(model.impulse_model.alpha_0*model.impulse_model.lmbda_0)))

    _, bins, _ = plt.hist(mu_samples[:,0,0], bins=50, alpha=0.5, normed=True)
    bincenters = 0.5*(bins[1:]+bins[:-1])
    plt.plot(bincenters, p_mu.pdf(bincenters), 'r--', linewidth=1)
    plt.xlabel('mu')
    plt.ylabel('p(mu)')

    plt.show()

def test_sample_nig():
    mu_0 = 0.0
    lmbda_0 = 10.
    alpha_0 = 10.
    beta_0 = 10.

    # Directly sample nig and lookg at marginals
    from pyhawkes.utils.utils import sample_nig
    mu_samples = \
        np.array([sample_nig(mu_0, lmbda_0, alpha_0, beta_0)[0]
                  for _ in range(10000)])

    # Plot the histogram of impulse means
    plt.figure()
    p_mu = t(df=2*alpha_0,
             loc=mu_0,
             scale=np.sqrt(beta_0/(alpha_0*lmbda_0)))

    _, bins, _ = plt.hist(mu_samples, bins=50, alpha=0.5, normed=True)
    bincenters = 0.5*(bins[1:]+bins[:-1])
    plt.plot(bincenters, p_mu.pdf(bincenters), 'r--', linewidth=1)
    plt.xlabel('mu')
    plt.ylabel('p(mu)')

    plt.figure()
    probplot(mu_samples, dist=p_mu, plot=plt.gca())

    plt.show()


if __name__ == "__main__":
    # test_sample_nig()
    test_geweke()