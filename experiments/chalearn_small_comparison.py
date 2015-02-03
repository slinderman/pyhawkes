"""
Compare the various algorithms on the small Chalearn network of 100 neurons
"""
import time
import cPickle
import os
import gzip
import pprint
import numpy as np
from scipy.misc import logsumexp
from scipy.special import gammaln

# Use the Agg backend in running on a server without the DISPLAY variable
if "DISPLAY" not in os.environ:
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pyhawkes.models import DiscreteTimeStandardHawkesModel, \
    DiscreteTimeNetworkHawkesModelGammaMixture
from pyhawkes.plotting.plotting import plot_network

from baselines.xcorr import infer_net_from_xcorr

from sklearn.metrics import roc_auc_score

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

    assert os.path.exists(os.path.dirname(output_path)), "Output directory does not exist!"

    if data_path.endswith(".gz"):
        with gzip.open(data_path, 'r') as f:
            S_full, F, bins, network, pos = cPickle.load(f)
    else:
        with open(data_path, 'r') as f:
            S_full, F, bins, network, pos = cPickle.load(f)

    # Cast to int
    S_full = S_full.astype(np.int)

    # Train on all but the last ten minutes (20ms time bins = 50Hz)
    T_test = 10 * 60 * 50
    S      = S_full[:-T_test, :]
    S_test = S_full[-T_test:, :]

    K      = S.shape[1]
    C      = 1
    B      = 3
    dt     = 0.02
    dt_max = 0.08

    # Compute the cross correlation to estimate the connectivity
    print "Estimating network via cross correlation"
    # W_xcorr = infer_net_from_xcorr(F[:10000,:], dtmax=3)

    # Compute the cross correlation to estimate the connectivity
    # print "Estimating network via cross correlation"
    W_xcorr = infer_net_from_xcorr(S[:10000], dtmax=dt_max // dt)

    # Fit a standard Hawkes model on subset of data with BFGS
    bfgs_model = None
    # bfgs_model, bfgs_time = fit_standard_hawkes_model_bfgs(S, K, B, dt, dt_max,
    #                                                        output_path=output_path)

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
    # svi_models, timestamps = fit_network_hawkes_svi(S, K, C, B, dt, dt_max,
    #                                                 output_path,
    #                                                 standard_model=bfgs_model)

    # Compute AUC of inferred network
    aucs = compute_auc(network,
                       W_xcorr=W_xcorr,
                       bfgs_model=bfgs_model)
    pprint.pprint(aucs)

    plls = compute_predictive_ll(S_test, S,
                                 bfgs_model=bfgs_model)

    # print "Log Predictive Likelihoods: "
    # pprint.pprint(plls)

    # Plot the predictive log likelihood
    import pdb; pdb.set_trace()
    # N_iters = plls['svi'].size
    N_iters = 100
    N_test  = S_test.size
    plt.ioff()
    plt.figure()
    plt.plot(np.arange(N_iters),
             (plls['bfgs'] - plls['homog'])/N_test * np.ones(N_iters),
             '-b', label='BFGS')
    # plt.plot(np.arange(N_iters),
    #          (plls['svi'] - plls['homog'])/N_test,
    #          '-r', label='SVI')
    plt.xlabel('Iteration')
    plt.ylabel('Log Predictive Likelihood')
    plt.legend()
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
        # betas = np.logspace(-1,1.3,num=1)
        betas = [0]

        init_models = []
        init_len    = 10000
        S_init      = S[:init_len,:]

        xv_len      = 1000
        xv_ll       = np.zeros(len(betas))
        S_xv        = S[init_len:init_len+xv_len, :]

        start = time.clock()
        for i,beta in enumerate(betas):
            print "Fitting with BFGS on first ", init_len, " time bins, beta = ", beta
            # Make a model to initialize the parameters
            init_model = DiscreteTimeStandardHawkesModel(K=K, dt=dt, B=B, dt_max=dt_max, beta=beta,
                                                         allow_instantaneous=True,
                                                         allow_self_connections=False)
            init_model.add_data(S_init)
            # Initialize the background rates to their mean
            init_model.initialize_to_background_rate()
            init_model.fit_with_bfgs()
            init_models.append(init_model)

            # Compute the heldout likelihood on the xv data
            xv_ll[i] = init_model.heldout_log_likelihood(S_xv)
            if not np.isfinite(xv_ll[i]):
                xv_ll[i] = -np.inf


        init_time = time.clock() - start

        # Take the best model
        print "XV predictive log likelihoods: "
        for beta, ll in zip(betas, xv_ll):
            print "Beta: %.2f\tLL: %.2f" % (beta, ll)
        best_ind = np.argmax(xv_ll)
        print "Best beta: ", betas[best_ind]
        init_model = init_models[best_ind]

        if best_ind == 0 or best_ind == len(betas) - 1:
            print "WARNING: Best BFGS model was for extreme value of beta. " \
                  "Consider expanding the beta range."

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

def compute_auc(A_true,
                W_xcorr=None,
                bfgs_model=None,
                sgd_model=None,
                gibbs_samples=None,
                vb_models=None,
                svi_models=None):
    """
    Compute the AUC score for each of competing models
    :return:
    """
    A_flat = A_true.ravel()
    aucs = {}

    if W_xcorr is not None:
        aucs['xcorr'] = roc_auc_score(A_flat,
                                      W_xcorr.ravel())

    if bfgs_model is not None:
        assert isinstance(bfgs_model, DiscreteTimeStandardHawkesModel)
        W_bfgs = bfgs_model.W.copy()
        W_bfgs -= np.diag(np.diag(W_bfgs))
        aucs['bfgs'] = roc_auc_score(A_flat,
                                     W_bfgs.ravel())

    if sgd_model is not None:
        assert isinstance(sgd_model, DiscreteTimeStandardHawkesModel)
        aucs['sgd'] = roc_auc_score(A_flat,
                                     sgd_model.W.ravel())

    if gibbs_samples is not None:
        # Compute ROC based on mean value of W_effective in second half of samples
        Weff_samples = np.array([s.weight_model.W_effective for s in gibbs_samples])
        N_samples    = Weff_samples.shape[0]
        offset       = N_samples // 2
        Weff_mean    = Weff_samples[offset:,:,:].mean(axis=0)

        aucs['gibbs'] = roc_auc_score(A_flat, Weff_mean)

    if vb_models is not None:
        # Compute ROC based on E[A] under variational posterior
        aucs['vb'] = roc_auc_score(A_flat,
                                   vb_models[-1].weight_model.expected_A().ravel())

    if svi_models is not None:
        # Compute ROC based on E[A] under variational posterior
        aucs['svi'] = roc_auc_score(A_flat,
                                    svi_models[-1].weight_model.expected_A().ravel())

    return aucs

def compute_predictive_ll(S_test, S_train,
                          true_model=None,
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
    plls['homog']  = 0
    plls['homog'] += -gammaln(S_test+1).sum()
    plls['homog'] += (-lam_homog * T_test).sum()
    plls['homog'] += (S_test.sum(axis=0) * np.log(lam_homog)).sum()

    if true_model is not None:
        plls['true'] = true_model.heldout_log_likelihood(S_test)

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
            # print "Computing pred ll for SVI iteration ", i
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
run = 4
data_path = os.path.join("data", "chalearn", "small", "network1a.pkl.gz")
out_path  = os.path.join("data", "chalearn", "small", "network1_run%03d" %run, "results" )
run_comparison(data_path, out_path, seed=seed)