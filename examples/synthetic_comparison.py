"""
Compare the various algorithms on a synthetic dataset.
"""
import time
import cPickle
import os
import numpy as np
import matplotlib.pyplot as plt

from pyhawkes.models import DiscreteTimeStandardHawkesModel, \
    DiscreteTimeNetworkHawkesModelGibbs, DiscreteTimeNetworkHawkesModelMeanField
from pyhawkes.plotting.plotting import plot_network

# np.seterr(over='raise', divide='raise')

def run_comparison(data_path, output_path, seed=None):
    """
    Run the comparison on the given data file
    :param data_path:
    :return:
    """
    if seed is None:
        seed = np.random.randint(2**32)

    print "Setting seed to ", seed
    np.random.seed(seed)

    with open(data_path, 'r') as f:
        S,true_model = cPickle.load(f)

    K      = true_model.K
    B      = true_model.B
    dt     = true_model.dt
    dt_max = true_model.dt_max

    # Fit a standard Hawkes model on subset of data with BFGS
    init_model, init_time = fit_standard_hawkes_model_bfgs(S, K, B, dt, dt_max)

    # Save the models
    with open(output_path + ".bfgs.pkl", 'w') as f:
        print "Saving BFGS initialization to ", (output_path + ".bfgs.pkl")
        cPickle.dump((init_model, init_time), f, protocol=-1)

    # Fit a standard Hawkes model with SGD
    # standard_models, timestamps = fit_standard_hawkes_model_sgd(S, K, B, dt, dt_max,
    #                                                         init_model=init_model)
    #
    # # Save the models
    # with open(output_path + ".sgd.pkl", 'w') as f:
    #     print "Saving SGD results to ", (output_path + ".sgd.pkl")
    #     cPickle.dump((standard_models, timestamps), f, protocol=-1)

    # Fit a network Hawkes model with Gibbs
    gibbs_samples, timestamps = fit_network_hawkes_gibbs(S, K, C, B, dt, dt_max,
                                             standard_model=init_model)

    with open(output_path + ".gibbs.pkl", 'w') as f:
        print "Saving Gibbs results to ", (output_path + ".gibbs.pkl")
        cPickle.dump((gibbs_samples, timestamps), f, protocol=-1)

    # Fit a network Hawkes model with Batch VB
    # vb_models, timestamps = fit_network_hawkes_vb(S, K, B, dt, dt_max,
    #                                          standard_model=standard_models[-1])
    #
    # with open(output_path + ".vb.pkl", 'w') as f:
    #     print "Saving VB results to ", (output_path + ".vb.pkl")
    #     cPickle.dump((vb_models, timestamps), f, protocol=-1)

    # Fit a network Hawkes model with SVI
    # svi_models, timestamps = fit_network_hawkes_svi(S, K, B, dt, dt_max,
    #                                          standard_model=standard_models[-1])
    #
    # with open(output_path + ".svi.pkl", 'w') as f:
    #     print "Saving SVI results to ", (output_path + ".svi.pkl")
    #     cPickle.dump((svi_models, timestamps), f, protocol=-1)


def fit_standard_hawkes_model_bfgs(S, K, B, dt, dt_max):
    """
    Fit
    :param S:
    :return:
    """
    print "Fitting the data with a standard Hawkes model"

    # Make a model to initialize the parameters
    init_len   = 1000
    init_model = DiscreteTimeStandardHawkesModel(K=K, dt=dt, B=B, dt_max=dt_max,
                                                 l2_penalty=0, l1_penalty=0)
    init_model.add_data(S[:init_len, :])

    # Initialize the background rates to their mean
    init_model.initialize_to_background_rate()

    print "Initializing with BFGS on first ", init_len, " time bins."
    start = time.clock()
    init_model.fit_with_bfgs()
    init_time = time.clock() - start


    return init_model, init_time

def fit_standard_hawkes_model_sgd(S, K, B, dt, dt_max, init_model=None):
    """
    Fit
    :param S:
    :return:
    """
    print "Fitting the data with a standard Hawkes model using SGD"

    # Make a new model for inference
    test_model = DiscreteTimeStandardHawkesModel(K=K, dt=dt, dt_max=dt_max, B=B,
                                                 l2_penalty=0, l1_penalty=0)
    test_model.add_data(S, minibatchsize=256)

    # Initialize the test model with the init model weights
    if init_model is not None:
        test_model.weights = init_model.weights

    plt.ion()
    im = plot_network(np.ones((K,K)), test_model.W, vmax=0.5)
    plt.pause(0.001)

    # Gradient descent
    N_steps = 1000
    samples = []
    lls = []
    timestamps = []

    learning_rate = 0.01 * np.ones(N_steps)
    momentum = 0.8 * np.ones(N_steps)
    prev_velocity = None
    for itr in xrange(N_steps):
        # W,ll,grad = test_model.gradient_descent_step(stepsz=0.001)
        W,ll,prev_velocity = test_model.sgd_step(prev_velocity, learning_rate[itr], momentum[itr])
        samples.append(test_model.copy_sample())
        lls.append(ll)
        timestamps.append(time.clock())

        if itr % 1 == 0:
            print "Iteration ", itr, "\t LL: ", ll
            im.set_data(np.ones((K,K)) * test_model.W)
            plt.pause(0.001)

    plt.ioff()
    plt.figure()
    plt.plot(np.arange(N_steps), lls)
    plt.xlabel("Iteration")
    plt.ylabel("Log likelihood")

    plot_network(np.ones((K,K)), test_model.W)
    plt.show()

    return samples, timestamps

def fit_network_hawkes_gibbs(S, K, C, B, dt, dt_max, standard_model=None):
    print "Fitting the data with a network Hawkes model using Gibbs sampling"

    # Make a new model for inference
    test_model = DiscreteTimeNetworkHawkesModelGibbs(C=C, K=K, dt=dt, dt_max=dt_max, B=B,
                                                     alpha=1.0, beta=1.0/20.0)
    test_model.add_data(S)

    # Initialize with the standard model parameters
    if standard_model is not None:
        test_model.initialize_with_standard_model(standard_model)

    plt.ion()
    im = plot_network(test_model.weight_model.A, test_model.weight_model.W, vmax=0.5)
    plt.pause(0.001)

    # Gibbs sample
    N_samples = 2
    samples = []
    lps = []
    timestamps = []
    for itr in xrange(N_samples):
        # lps.append(test_model.log_probability())
        lps.append(test_model.log_likelihood())
        samples.append(test_model.resample_and_copy())
        timestamps.append(time.clock())

        if itr % 1 == 0:
            print "Iteration ", itr, "\t LL: ", lps[-1]
            im.set_data(test_model.weight_model.A * \
                        test_model.weight_model.W)
            plt.pause(0.001)

    # Compute sample statistics for second half of samples
    A_samples       = np.array([s.weight_model.A     for s in samples])
    W_samples       = np.array([s.weight_model.W     for s in samples])
    g_samples       = np.array([s.impulse_model.g    for s in samples])
    lambda0_samples = np.array([s.bias_model.lambda0 for s in samples])
    c_samples       = np.array([s.network.c          for s in samples])
    lps             = np.array(lps)

    offset = N_samples // 2
    A_mean          = A_samples[offset:, ...].mean(axis=0)
    W_mean          = W_samples[offset:, ...].mean(axis=0)
    g_mean          = g_samples[offset:, ...].mean(axis=0)
    lambda0_mean    = lambda0_samples[offset:, ...].mean(axis=0)

    print "A mean:        ", A_mean
    print "W mean:        ", W_mean
    print "g mean:        ", g_mean
    print "lambda0 mean:  ", lambda0_mean

    plt.ioff()
    plt.figure()
    plt.plot(np.arange(N_samples), lps)
    plt.xlabel("Iteration")
    plt.ylabel("Log probability")
    plt.show()

    plot_network(test_model.weight_model.A, test_model.weight_model.W)
    plt.show()

    return samples, timestamps


def fit_network_hawkes_svi(S, K, C, B, dt, dt_max, standard_model=None):
    print "Fitting the data with a network Hawkes model using SVI"

    # Make a new model for inference
    test_model = DiscreteTimeNetworkHawkesModelMeanField(C=C, K=K, dt=dt, dt_max=dt_max, B=B,
                                                     alpha=1.0, beta=1.0/20.0)
    test_model.add_data(S)

    # Initialize with the standard model parameters
    if standard_model is not None:
        test_model.initialize_with_standard_model(standard_model)

    plt.ion()
    im = plot_network(test_model.weight_model.A, test_model.weight_model.W, vmax=0.5)
    plt.pause(0.001)

    # Gibbs sample
    N_samples = 200
    samples = []
    lps = []
    timestamps = []
    for itr in xrange(N_samples):
        lps.append(test_model.log_probability())
        samples.append(test_model.resample_and_copy())
        timestamps.append(time.clock())

        if itr % 1 == 0:
            print "Iteration ", itr, "\t LL: ", lps[-1]
            im.set_data(test_model.weight_model.A * \
                        test_model.weight_model.W)
            plt.pause(0.001)

    # Compute sample statistics for second half of samples
    A_samples       = np.array([s.weight_model.A     for s in samples])
    W_samples       = np.array([s.weight_model.W     for s in samples])
    g_samples       = np.array([s.impulse_model.g    for s in samples])
    lambda0_samples = np.array([s.bias_model.lamdba0 for s in samples])
    c_samples       = np.array([s.network.c          for s in samples])
    lps             = np.array(lps)

    offset = N_samples // 2
    A_mean          = A_samples[offset:, ...].mean(axis=0)
    W_mean          = W_samples[offset:, ...].mean(axis=0)
    g_mean          = g_samples[offset:, ...].mean(axis=0)
    lambda0_mean    = lambda0_samples[offset:, ...].mean(axis=0)

    print "A mean:        ", A_mean
    print "W mean:        ", W_mean
    print "g mean:        ", g_mean
    print "lambda0 mean:  ", lambda0_mean

    plt.figure()
    plt.plot(np.arange(N_samples), lps)
    plt.xlabel("Iteration")
    plt.ylabel("Log probability")
    plt.show()

    plot_network(test_model.weight_model.A, test_model.weight_model.W)
    plt.show()

    return samples, timestamps

def compute_auc():
    """
    Compute the AUC score
    :return:
    """
    raise NotImplementedError()

def compute_predictive_ll():
    """
    Compute the predictive log likelihood
    :return:
    """
    raise NotImplementedError()

def compute_clustering_score():
    """
    Compute a few clustering scores.
    :return:
    """
    raise NotImplementedError()




seed = 2650533028
# seed = None
K = 20
C = 4
T = 100000
data_path = os.path.join("data", "synthetic", "synthetic_K%d_C%d_T%d.pkl" % (K,C,T))
out_path = os.path.join("data", "synthetic", "results_K%d_C%d_T%d" % (K,C,T))
run_comparison(data_path, out_path, seed=seed)