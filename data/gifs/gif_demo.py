import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score

import pyhawkes.models
import imp
imp.reload(pyhawkes.models)
from pyhawkes.models import DiscreteTimeNetworkHawkesModelSpikeAndSlab
from pyhawkes.internals.network import ErdosRenyiFixedSparsity
from pyhawkes.plotting.plotting import plot_network

np.random.seed(0)

def demo(K=3, T=1000, dt_max=20, p=0.25):
    """

    :param K:       Number of nodes
    :param T:       Number of time bins to simulate
    :param dt_max:  Number of future time bins an event can influence
    :param p:       Sparsity of network
    :return:
    """
    ###########################################################
    # Generate synthetic data
    ###########################################################
    network = ErdosRenyiFixedSparsity(K, p, v=1., allow_self_connections=False)
    bkgd_hypers = {"alpha": 1.0, "beta": 20.0}
    true_model = DiscreteTimeNetworkHawkesModelSpikeAndSlab(
        K=K, dt_max=dt_max, bkgd_hypers=bkgd_hypers, network=network)
    A_true = np.zeros((K,K))
    A_true[0,1] = A_true[0,2] = 1
    W_true = np.zeros((K,K))
    W_true[0,1] = W_true[0,2] = 1.0
    true_model.weight_model.A = A_true
    true_model.weight_model.W = W_true
    true_model.bias_model.lambda0[0] = 0.2
    assert true_model.check_stability()

    # Sample from the true model
    S,R = true_model.generate(T=T, keep=True, print_interval=50)

    plt.ion()
    true_figure, _ = true_model.plot(color="#377eb8", T_slice=(0,100))

    # Save the true figure
    true_figure.savefig("gifs/true.gif")

    ###########################################################
    # Create a test spike and slab model
    ###########################################################
    test_model = DiscreteTimeNetworkHawkesModelSpikeAndSlab(
        K=K, dt_max=dt_max, network=network)

    test_model.add_data(S)

    # Initialize plots
    test_figure, test_handles = test_model.plot(color="#e41a1c", T_slice=(0,100))
    test_figure.savefig("gifs/test0.gif")

    ###########################################################
    # Fit the test model with Gibbs sampling
    ###########################################################
    N_samples = 100
    samples = []
    lps = []
    for itr in range(N_samples):
        print("Gibbs iteration ", itr)
        test_model.resample_model()
        lps.append(test_model.log_probability())
        samples.append(test_model.copy_sample())

        # Update plots
        test_model.plot(handles=test_handles)
        test_figure.savefig("gifs/test%d.gif" % (itr+1))

    # ###########################################################
    # # Analyze the samples
    # ###########################################################
    # analyze_samples(true_model, samples, lps)

def initialize_plots(true_model, test_model, S):
    K = true_model.K
    C = true_model.C
    R = true_model.compute_rate(S=S)
    T = S.shape[0]

    # Plot the true network
    plt.ion()
    plot_network(true_model.weight_model.A,
                 true_model.weight_model.W)
    plt.pause(0.001)


    # Plot the true and inferred firing rate
    plt.figure(2)
    plt.plot(np.arange(T), R[:,0], '-k', lw=2)
    plt.ion()
    ln = plt.plot(np.arange(T), test_model.compute_rate()[:,0], '-r')[0]
    plt.show()
    plt.pause(0.001)

    return ln, im_net

def update_plots(itr, test_model, S, ln, im_net):
    K = test_model.K
    C = test_model.C
    T = S.shape[0]
    plt.figure(2)
    ln.set_data(np.arange(T), test_model.compute_rate()[:,0])
    plt.title("\lambda_{%d}. Iteration %d" % (0, itr))
    plt.pause(0.001)

    plt.figure(4)
    plt.title("W: Iteration %d" % itr)
    im_net.set_data(test_model.weight_model.W_effective)
    plt.pause(0.001)

def analyze_samples(true_model, samples, lps):
    N_samples = len(samples)

    # Compute sample statistics for second half of samples
    A_samples       = np.array([s.weight_model.A     for s in samples])
    W_samples       = np.array([s.weight_model.W     for s in samples])
    lps             = np.array(lps)

    offset = N_samples // 2
    A_mean       = A_samples[offset:, ...].mean(axis=0)
    W_mean       = W_samples[offset:, ...].mean(axis=0)

    plt.figure()
    plt.plot(np.arange(N_samples), lps, 'k')
    plt.xlabel("Iteration")
    plt.ylabel("Log probability")
    plt.show()

    # Compute the link prediction accuracy curves
    auc_A_mean = roc_auc_score(true_model.weight_model.A.ravel(),
                               A_mean.ravel())
    auc_W_mean = roc_auc_score(true_model.weight_model.A.ravel(),
                               W_mean.ravel())

    aucs = []
    for A in A_samples:
        aucs.append(roc_auc_score(true_model.weight_model.A.ravel(), A.ravel()))

    plt.figure()
    plt.plot(aucs, '-r')
    plt.plot(auc_A_mean * np.ones_like(aucs), '--r')
    plt.plot(auc_W_mean * np.ones_like(aucs), '--b')
    plt.xlabel("Iteration")
    plt.ylabel("Link prediction AUC")
    plt.show()

    plt.ioff()
    plt.show()

demo()