"""
Compare the various algorithms on a synthetic dataset.
"""
import time
import cPickle
import os
import gzip
import pprint
import numpy as np
from scipy.misc import logsumexp
import matplotlib.pyplot as plt

from pyhawkes.models import DiscreteTimeStandardHawkesModel, \
    DiscreteTimeNetworkHawkesModelSpikeAndSlab, \
    DiscreteTimeNetworkHawkesModelGammaMixture
from pyhawkes.plotting.plotting import plot_network

from sklearn.metrics import roc_auc_score

# np.seterr(over='raise', divide='raise')

def run_comparison(data_path, test_path, output_path, seed=None):
    """
    Run the comparison on the given data file
    :param data_path:
    :return:
    """
    if seed is None:
        seed = np.random.randint(2**32)

    print "Setting seed to ", seed
    np.random.seed(seed)

    if data_path.endswith(".gz"):
        with gzip.open(data_path, 'r') as f:
            S, true_model = cPickle.load(f)
    else:
        with open(data_path, 'r') as f:
            S, true_model = cPickle.load(f)

    if test_path.endswith(".gz"):
        with gzip.open(test_path, 'r') as f:
            S_test, test_model = cPickle.load(f)
    else:
        with open(test_path, 'r') as f:
            S_test, test_model = cPickle.load(f)

    K      = true_model.K
    C      = true_model.C
    B      = true_model.B
    dt     = true_model.dt
    dt_max = true_model.dt_max

    # Fit a standard Hawkes model on subset of data with BFGS
    init_model, init_time = fit_standard_hawkes_model_bfgs(S, K, B, dt, dt_max,
                                                           output_path=output_path)

    # Fit a standard Hawkes model with SGD
    # standard_models, timestamps = fit_standard_hawkes_model_sgd(S, K, B, dt, dt_max,
    #                                                         init_model=init_model)
    #
    # # Save the models
    # with open(output_path + ".sgd.pkl", 'w') as f:
    #     print "Saving SGD results to ", (output_path + ".sgd.pkl")
    #     cPickle.dump((standard_models, timestamps), f, protocol=-1)

    # Fit a network Hawkes model with Gibbs
    # gibbs_samples, timestamps = fit_network_hawkes_gibbs(S, K, C, B, dt, dt_max,
    #                                          output_path=output_path,
    #                                          standard_model=init_model)

    # Fit a network Hawkes model with Batch VB
    # vb_models, timestamps = fit_network_hawkes_vb(S, K, B, dt, dt_max,
    #                                          standard_model=standard_models[-1])
    #
    # with open(output_path + ".vb.pkl", 'w') as f:
    #     print "Saving VB results to ", (output_path + ".vb.pkl")
    #     cPickle.dump((vb_models, timestamps), f, protocol=-1)

    # Fit a network Hawkes model with SVI
    svi_models, timestamps = fit_network_hawkes_svi(S, K, C, B, dt, dt_max,
                                                    output_path,
                                                    standard_model=init_model)

    aucs = compute_auc(true_model, bfgs_model=init_model, svi_models=svi_models)
    pprint.pprint(aucs)

    plls = compute_predictive_ll(S_test, S, bfgs_model=init_model, svi_models=svi_models)
    N_iters = plls['svi'].size
    N_test  = S_test.sum()
    plt.figure()
    plt.plot(np.arange(N_iters), (plls['bfgs'] - plls['homog'])/N_test * np.ones(N_iters), '-k')
    plt.plot(np.arange(N_iters), (plls['svi'] - plls['homog'])/N_test, '-r')
    plt.xlabel('Iteration')
    plt.ylabel('Log Predictive Likelihood')
    plt.show()

def fit_standard_hawkes_model_bfgs(S, K, B, dt, dt_max, output_path):
    """
    Fit
    :param S:
    :return:
    """
    # Check for existing results
    if os.path.exists(out_path + ".bfgs.pkl"):
        print "Existing BFGS results found. Loading from file."
        with open(output_path + ".bfgs.pkl", 'r') as f:
            init_model, init_time = cPickle.load(f)

    else:
        print "Fitting the data with a standard Hawkes model"

        # Make a model to initialize the parameters
        init_len   = 10000
        init_model = DiscreteTimeStandardHawkesModel(K=K, dt=dt, B=B, dt_max=dt_max,
                                                     l2_penalty=0, l1_penalty=0)
        init_model.add_data(S[:init_len, :])

        # Initialize the background rates to their mean
        init_model.initialize_to_background_rate()

        print "Initializing with BFGS on first ", init_len, " time bins."
        start = time.clock()
        init_model.fit_with_bfgs()
        init_time = time.clock() - start

        # Save the model (sans data)
        init_model.data_list.pop()
        with open(output_path + ".bfgs.pkl", 'w') as f:
            print "Saving BFGS results to ", (output_path + ".bfgs.pkl")
            cPickle.dump((init_model, init_time), f, protocol=-1)

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

def fit_network_hawkes_gibbs(S, K, C, B, dt, dt_max,
                             output_path,
                             standard_model=None):

    # Check for existing Gibbs results
    if os.path.exists(output_path + ".gibbs.pkl"):
        with open(output_path + ".gibbs.pkl", 'r') as f:
            print "Loading Gibbs results from ", (output_path + ".gibbs.pkl")
            (samples, timestamps) = cPickle.load(f)

    else:
        print "Fitting the data with a network Hawkes model using Gibbs sampling"

        # Make a new model for inference
        test_model = DiscreteTimeNetworkHawkesModelGammaMixture(C=C, K=K, dt=dt, dt_max=dt_max, B=B,
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

            # Save this sample
            with open(output_path + ".gibbs.itr%04d.pkl" % itr, 'w') as f:
                cPickle.dump(samples[-1], f, protocol=-1)

        # Save the Gibbs samples
        with open(output_path + ".gibbs.pkl", 'w') as f:
            print "Saving Gibbs samples to ", (output_path + ".gibbs.pkl")
            cPickle.dump((samples, timestamps), f, protocol=-1)

    # Remove the temporary sample files
    # print "Cleaning up temporary sample files"
    # for itr in xrange(N_samples):
    #     if os.path.exists(output_path + ".gibbs.itr%04d.pkl" % itr):
    #         os.remove(output_path + ".gibbs.itr%04d.pkl" % itr)

    # TODO: Remove the plotting code for server runs
    # # Compute sample statistics for second half of samples
    # A_samples       = np.array([s.weight_model.A     for s in samples])
    # W_samples       = np.array([s.weight_model.W     for s in samples])
    # g_samples       = np.array([s.impulse_model.g    for s in samples])
    # lambda0_samples = np.array([s.bias_model.lambda0 for s in samples])
    # c_samples       = np.array([s.network.c          for s in samples])
    # lps             = np.array(lps)
    #
    # offset = len(samples) // 2
    # A_mean          = A_samples[offset:, ...].mean(axis=0)
    # W_mean          = W_samples[offset:, ...].mean(axis=0)
    # g_mean          = g_samples[offset:, ...].mean(axis=0)
    # lambda0_mean    = lambda0_samples[offset:, ...].mean(axis=0)
    #
    # print "A mean:        ", A_mean
    # print "W mean:        ", W_mean
    # print "g mean:        ", g_mean
    # print "lambda0 mean:  ", lambda0_mean
    #
    # plt.ioff()
    # plt.figure()
    # plt.plot(np.arange(N_samples), lps)
    # plt.xlabel("Iteration")
    # plt.ylabel("Log probability")
    # plt.show()
    #
    # plot_network(samples[-1].weight_model.A,
    #              samples[-1].weight_model.W)
    # plt.show()

    return samples, timestamps


def fit_network_hawkes_svi(S, K, C, B, dt, dt_max,
                           output_path,
                           standard_model=None):


    # Check for existing Gibbs results
    if os.path.exists(output_path + ".svi.pkl.gz"):
        with gzip.open(output_path + ".svi.pkl.gz", 'r') as f:
            print "Loading SVI results from ", (output_path + ".svi.pkl.gz")
            (samples, timestamps) = cPickle.load(f)

    else:
        print "Fitting the data with a network Hawkes model using SVI"

        # Make a new model for inference
        test_model = DiscreteTimeNetworkHawkesModelGammaMixture(C=C, K=K, dt=dt, dt_max=dt_max, B=B,
                                                                alpha=1.0, beta=1.0/20.0)
        # Initialize with the standard model parameters
        if standard_model is not None:
            test_model.initialize_with_standard_model(standard_model)

        plt.ion()
        im = plot_network(test_model.weight_model.A, test_model.weight_model.W, vmax=0.5)
        plt.pause(0.001)

        # TODO: Add the data in minibatches
        minibatchsize = 500
        test_model.add_data(S)


        # Stochastic variational inference
        N_iters = 1000
        samples = []
        delay = 1.0
        forgetting_rate = 0.5
        stepsize = (np.arange(N_iters) + delay)**(-forgetting_rate)
        timestamps = []
        for itr in xrange(N_iters):
            print "SVI Iter: ", itr, "\tStepsize: ", stepsize[itr]
            test_model.sgd_step(minibatchsize=minibatchsize, stepsize=stepsize[itr])
            test_model.resample_from_mf()
            samples.append(test_model.copy_sample())
            timestamps.append(time.clock())

            if itr % 1 == 0:
                im.set_data(test_model.weight_model.expected_W())
                plt.pause(0.001)

            # Save this sample
            with open(output_path + ".svi.itr%04d.pkl" % itr, 'w') as f:
                cPickle.dump(samples[-1], f, protocol=-1)

        # Save the Gibbs samples
        with gzip.open(output_path + ".svi.pkl.gz", 'w') as f:
            print "Saving SVI samples to ", (output_path + ".svi.pkl.gz")
            cPickle.dump((samples, timestamps), f, protocol=-1)

    return samples, timestamps

def compute_auc(true_model,
                bfgs_model=None,
                sgd_model=None,
                gibbs_samples=None,
                vb_models=None,
                svi_models=None):
    """
    Compute the AUC score for each of competing models
    :return:
    """
    aucs = {}

    # Get the true adjacency matrix
    A_true = true_model.weight_model.A.ravel()

    if bfgs_model is not None:
        assert isinstance(bfgs_model, DiscreteTimeStandardHawkesModel)
        aucs['bfgs'] = roc_auc_score(A_true,
                                     bfgs_model.W.ravel())

    if sgd_model is not None:
        assert isinstance(sgd_model, DiscreteTimeStandardHawkesModel)
        aucs['sgd'] = roc_auc_score(A_true,
                                     sgd_model.W.ravel())

    if gibbs_samples is not None:
        # Compute ROC based on mean value of W_effective in second half of samples
        Weff_samples = np.array([s.weight_model.W_effective for s in gibbs_samples])
        N_samples    = Weff_samples.shape[0]
        offset       = N_samples // 2
        Weff_mean    = Weff_samples[offset:,:,:].mean(axis=0)

        aucs['gibbs'] = roc_auc_score(A_true, Weff_mean)

    if vb_models is not None:
        # Compute ROC based on E[A] under variational posterior
        aucs['vb'] = roc_auc_score(A_true,
                                   vb_models[-1].weight_model.expected_A().ravel())

    if svi_models is not None:
        # Compute ROC based on E[A] under variational posterior
        aucs['svi'] = roc_auc_score(A_true,
                                    svi_models[-1].weight_model.expected_A().ravel())

    return aucs

def compute_predictive_ll(S_test, S_train,
                          bfgs_model=None,
                          sgd_models=None,
                          gibbs_samples=None,
                          vb_models=None,
                          svi_models=None):
    """
    Compute the predictive log likelihood
    :return:
    """
    plls = {}

    # Compute homogeneous pred ll
    T = S_train.shape[0]
    T_test = S_test.shape[0]
    lam_homog = S_train.sum(axis=0) / float(T)
    plls['homog'] = (-lam_homog * T_test + S_train.sum(axis=0) * np.log(lam_homog)).sum()

    if bfgs_model is not None:
        assert isinstance(bfgs_model, DiscreteTimeStandardHawkesModel)
        plls['bfgs'] = bfgs_model.heldout_log_likelihood(S_test)

    if sgd_models is not None:
        assert isinstance(sgd_models, list)

        plls['sgd'] = np.zeros(len(sgd_models))
        for i,sgd_model in enumerate(sgd_models):
            plls['sgd'] = sgd_model.heldout_log_likelihood(S_test)

    if gibbs_samples is not None:
        # Compute log(E[pred likelihood]) on second half of samplese
        offset       = len(gibbs_samples) // 2
        # Preconvolve with the Gibbs model's basis
        F_test = gibbs_samples[0].basis.convolve_with_basis(S_test)

        plls['gibbs'] = []
        for s in gibbs_samples[offset:]:
            plls['gibbs'].append(s.heldout_log_likelihood(S_test, F=F_test))

        # Convert to numpy array
        plls['gibbs'] = np.array(plls['gibbs'])

    if vb_models is not None:
        # Compute predictive likelihood over samples from VB model
        N_models  = len(vb_models)
        N_samples = 100
        # Preconvolve with the VB model's basis
        F_test = vb_models[0].basis.convolve_with_basis(S_test)

        vb_plls = np.zeros((N_models, N_samples))
        for i, vb_model in enumerate(vb_models):
            for j in xrange(N_samples):
                vb_model.resample_from_mf()
                vb_plls[i,j] = vb_model.heldout_log_likelihood(S_test, F=F_test)

        # Compute the log of the average predicted likelihood
        plls['vb'] = -np.log(N_samples) + logsumexp(vb_plls, axis=1)

    if svi_models is not None:
        # Compute predictive likelihood over samples from VB model
        N_models  = len(svi_models)
        N_samples = 1
        # Preconvolve with the VB model's basis
        F_test = svi_models[0].basis.convolve_with_basis(S_test)

        svi_plls = np.zeros((N_models, N_samples))
        for i, svi_model in enumerate(svi_models):
            print "Computing pred ll for SVI iteration ", i
            for j in xrange(N_samples):
                svi_model.resample_from_mf()
                svi_plls[i,j] = svi_model.heldout_log_likelihood(S_test, F=F_test)

        plls['svi'] = -np.log(N_samples) + logsumexp(svi_plls, axis=1)

    return plls

def compute_clustering_score():
    """
    Compute a few clustering scores.
    :return:
    """
    # TODO: Implement simple clustering
    raise NotImplementedError()

# seed = 2650533028
seed = None
K = 50
C = 5
T = 100000
T_test = 1000
data_path = os.path.join("data", "synthetic", "synthetic_K%d_C%d_T%d.pkl" % (K,C,T))
test_path = os.path.join("data", "synthetic", "synthetic_test_K%d_C%d_T%d.pkl" % (K,C,T_test))
out_path = os.path.join("data", "synthetic", "results_K%d_C%d_T%d" % (K,C,T))
run_comparison(data_path, test_path, out_path, seed=seed)