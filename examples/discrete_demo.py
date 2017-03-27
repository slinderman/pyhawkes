import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score

import pyhawkes.models
import imp
imp.reload(pyhawkes.models)
from pyhawkes.models import DiscreteTimeNetworkHawkesModelSpikeAndSlab

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
    network_hypers = {"p": p, "allow_self_connections": False}
    true_model = DiscreteTimeNetworkHawkesModelSpikeAndSlab(
        K=K, dt_max=dt_max,
        network_hypers=network_hypers)
    assert true_model.check_stability()

    # Sample from the true model
    S,R = true_model.generate(T=T, keep=True, print_interval=50)

    plt.ion()
    true_figure, _ = true_model.plot(color="#377eb8", T_slice=(0,100))

    ###########################################################
    # Create a test spike and slab model
    ###########################################################
    test_model = DiscreteTimeNetworkHawkesModelSpikeAndSlab(
        K=K, dt_max=dt_max,
        network_hypers=network_hypers)

    test_model.add_data(S)

    # Initialize plots
    test_figure, test_handles = test_model.plot(color="#e41a1c", T_slice=(0,100))

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

    ###########################################################
    # Analyze the samples
    ###########################################################
    analyze_samples(true_model, samples, lps)

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