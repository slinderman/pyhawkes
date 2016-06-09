import numpy as np
import os
import cPickle
import gzip

import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score

from pybasicbayes.util.text import progprint_xrange

from pyhawkes.models import \
    DiscreteTimeNetworkHawkesModelGammaMixture, \
    DiscreteTimeStandardHawkesModel


if __name__ == "__main__":

    seed = 11223344
    print "Setting seed to ", seed
    np.random.seed(seed)

    ###########################################################
    # Load some example data.
    # See data/synthetic/generate.py to create more.
    ###########################################################
    data_path = os.path.join("data", "synthetic", "synthetic_K20_C4_T10000.pkl.gz")
    with gzip.open(data_path, 'r') as f:
        S, true_model = cPickle.load(f)

    T      = S.shape[0]
    K      = true_model.K
    B      = true_model.B
    dt     = true_model.dt
    dt_max = true_model.dt_max

    ###########################################################
    # Initialize with MAP estimation on a standard Hawkes model
    ###########################################################
    init_with_map = True
    if init_with_map:
        init_len   = T
        print "Initializing with BFGS on first ", init_len, " time bins."
        init_model = DiscreteTimeStandardHawkesModel(K=K, dt=dt, dt_max=dt_max, B=B,
                                                     alpha=1.0, beta=1.0)
        init_model.add_data(S[:init_len, :])

        init_model.initialize_to_background_rate()
        init_model.fit_with_bfgs()
    else:
        init_model = None

    ###########################################################
    # Create a test weak spike-and-slab model
    ###########################################################

    # Copy the network hypers.
    # Give the test model p, but not c, v, or m
    network_hypers = true_model.network_hypers.copy()
    network_hypers['v'] = None
    test_model = DiscreteTimeNetworkHawkesModelGammaMixture(K=K, dt=dt, dt_max=dt_max, B=B,
                                                            basis_hypers=true_model.basis_hypers,
                                                            bkgd_hypers=true_model.bkgd_hypers,
                                                            impulse_hypers=true_model.impulse_hypers,
                                                            weight_hypers=true_model.weight_hypers)
    test_model.add_data(S)

    # Initialize with the standard model parameters
    if init_model is not None:
        test_model.initialize_with_standard_model(init_model)


    ###########################################################
    # Fit the test model with Gibbs sampling
    ###########################################################
    N_samples = 500
    samples = []
    lps = []
    for itr in progprint_xrange(N_samples):
        lps.append(test_model.log_probability())
        samples.append(test_model.copy_sample())
        test_model.resample_model()

    ###########################################################
    # Analyze the samples
    ###########################################################
    N_samples = len(samples)
    A_samples       = np.array([s.weight_model.A     for s in samples])
    W_samples       = np.array([s.weight_model.W     for s in samples])
    g_samples       = np.array([s.impulse_model.g    for s in samples])
    lambda0_samples = np.array([s.bias_model.lambda0 for s in samples])
    lps             = np.array(lps)

    offset = N_samples // 2
    A_mean       = A_samples[offset:, ...].mean(axis=0)
    W_mean       = W_samples[offset:, ...].mean(axis=0)
    g_mean       = g_samples[offset:, ...].mean(axis=0)
    lambda0_mean = lambda0_samples[offset:, ...].mean(axis=0)

    plt.figure()
    plt.plot(np.arange(N_samples), lps, 'k')
    plt.xlabel("Iteration")
    plt.ylabel("Log probability")
    plt.show()

    # Compute the link prediction accuracy curves
    auc_init = roc_auc_score(true_model.weight_model.A.ravel(),
                             init_model.W.ravel())
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
    plt.plot(auc_init * np.ones_like(aucs), '--k')
    plt.xlabel("Iteration")
    plt.ylabel("Link prediction AUC")
    plt.show()


    plt.ioff()
    plt.show()

