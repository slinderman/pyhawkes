"""
Compare the various algorithms on a synthetic dataset.
"""
import cPickle
import os
import numpy as np
import matplotlib.pyplot as plt

from pyhawkes.models import DiscreteTimeStandardHawkesModel, DiscreteTimeNetworkHawkesModelGibbs
from pyhawkes.plotting.plotting import plot_network

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

    # Fit a standard Hawkes model with SGD
    # standard_model = fit_standard_hawkes_model(S, K, B, dt, dt_max)
    # # Pop the data list and save the output
    # standard_model.data_list.pop()
    # with open(output_path + ".sgd.pkl", 'w') as f:
    #     cPickle.dump(standard_model, f, protocol=-1)

    # Fit a network Hawkes model with Gibbs
    gibbs_model = fit_network_hawkes_gibbs(S, K, B, dt, dt_max)

    # Pop the data list and save the output
    gibbs_model.data_list.pop()
    with open(output_path + ".gibbs.pkl", 'w') as f:
        cPickle.dump(gibbs_model, f, protocol=-1)


def fit_standard_hawkes_model(S, K, B, dt, dt_max):
    """
    Fit
    :param S:
    :return:
    """
    print "Fitting the data with a standard Hawkes model"

    # Make a new model for inference
    test_model = DiscreteTimeStandardHawkesModel(K=K, dt=dt, dt_max=dt_max, B=B, l2_penalty=0, l1_penalty=0)
    test_model.add_data(S, minibatchsize=256)

    # Gradient descent
    N_steps = 1000
    lls = []
    # learning_rate = 10 * np.ones(N_steps)
    learning_rate = 10 * (np.arange(N_steps)+100.0)**(-0.5)
    decay = 0.8 * np.ones(N_steps)
    prev_grad = None
    for itr in xrange(N_steps):
        # W,ll,grad = test_model.gradient_descent_step(stepsz=0.001)
        W,ll,prev_grad = test_model.sgd_step(prev_grad, learning_rate[itr], decay[itr])
        lls.append(ll)

        if itr % 1 == 0:
            print "Iteration ", itr, "\t LL: ", ll

    plt.figure()
    plt.plot(np.arange(N_steps), lls)
    plt.xlabel("Iteration")
    plt.ylabel("Log likelihood")

    plot_network(np.ones((K,K)), test_model.W)
    plt.show()

    return test_model

def fit_network_hawkes_gibbs(S, K, B, dt, dt_max):
    print "Fitting the data with a network Hawkes model using Gibbs sampling"

    # Make a new model for inference
    test_model = DiscreteTimeNetworkHawkesModelGibbs(C=5, K=K, dt=dt, dt_max=dt_max, B=B)
    test_model.add_data(S)

    # Gibbs sample
    N_samples = 200
    samples = []
    lps = []
    for itr in xrange(N_samples):
        lps.append(test_model.log_probability())
        samples.append(test_model.resample_and_copy())

        if itr % 1 == 0:
            print "Iteration ", itr, "\t LL: ", lps[-1]

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

    print "A mean:        ", A_mean
    print "W mean:        ", W_mean
    print "beta mean:     ", beta_mean
    print "lambda0 mean:  ", lambda0_mean

    plt.figure()
    plt.plot(np.arange(N_samples), lps)
    plt.xlabel("Iteration")
    plt.ylabel("Log probability")
    plt.show()

    plot_network(test_model.weight_model.A, test_model.weight_model.W)
    plt.show()

    return test_model

# seed = 2650533028
seed = None
data_path = os.path.join("data","synthetic_K100_C5","synthetic_K100_C5_T10000.pkl")
out_path = os.path.join("data","synthetic_K100_C5","results")
run_comparison(data_path, out_path, seed=seed)