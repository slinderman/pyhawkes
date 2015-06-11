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

from pyhawkes.utils.basis import IdentityBasis
from pyhawkes.utils.utils import convert_discrete_to_continuous
from pyhawkes.models import DiscreteTimeStandardHawkesModel, \
    DiscreteTimeNetworkHawkesModelGammaMixture, \
    DiscreteTimeNetworkHawkesModelGammaMixtureFixedSparsity, \
    DiscreteTimeNetworkHawkesModelSpikeAndSlab, \
    ContinuousTimeNetworkHawkesModel
from pyhawkes.plotting.plotting import plot_network


from baselines.xcorr import infer_net_from_xcorr

from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score

def run_comparison(data_path, output_path, seed=None, thresh=0.5):
    """
    Run the comparison on the given data file
    :param data_path:
    :return:
    """
    import ipdb; ipdb.set_trace()
    if seed is None:
        seed = np.random.randint(2**32)

    print "Setting seed to ", seed
    np.random.seed(seed)

    assert os.path.exists(os.path.dirname(output_path)), "Output directory does not exist!"

    if data_path.endswith("_oopsi.pkl.gz"):
        # The oopsi data has a probability of spike
        with gzip.open(data_path, 'r') as f:
            P, F, Cf, network, pos = cPickle.load(f)
            S_full = P > thresh
            # onespk = np.bitwise_and(P > thresh, Cf < 0.3)
            # twospk = np.bitwise_and(P > thresh, Cf >= 0.3)
            # S_full = np.zeros_like(P)
            # S_full[onespk] = 1
            # S_full[twospk] = 2

    elif data_path.endswith(".gz"):
        with gzip.open(data_path, 'r') as f:
            S_full, F, bins, network, pos = cPickle.load(f)
    else:
        with open(data_path, 'r') as f:
            S_full, F, bins, network, pos = cPickle.load(f)

    # Cast to int
    S_full = S_full.astype(np.int)

    # Train on all but the last ten minutes (20ms time bins = 50Hz)
    T_train = 10 * 60 * 50
    T_test = 10 * 60 * 50
    # S      = S_full[:-T_test, :]
    S      = S_full[:T_train, :]
    S_test = S_full[-T_test:, :]

    K      = S.shape[1]
    C      = 1
    dt     = 0.02
    dt_max = 0.08

    # Compute the cross correlation to estimate the connectivity
    print "Estimating network via cross correlation"
    W_xcorr = infer_net_from_xcorr(S, dtmax=dt_max // dt)

    # HACK! Select the threshold by looking at the data
    test_thresholds = False
    if test_thresholds:
        print "Estimating network via cross correlation"
        F_xcorr = infer_net_from_xcorr(F, dtmax=3)
        aucs, _, _ = compute_auc_roc(network, W_xcorr=F_xcorr)
        print "AUC F: ", aucs["xcorr"]
        for thresh in np.linspace(0.1, 0.95, 20):
            S_thr = (P > thresh).astype(np.int)
            S_train = S_thr[:T_train, :]

            W_tmp = infer_net_from_xcorr(S_train, dtmax=dt_max // dt)
            aucs, _, _ = compute_auc_roc(network, W_xcorr=W_tmp)
            print "AUC (", thresh, "): ", aucs["xcorr"]

        import pdb; pdb.set_trace()

    # Fit a standard Hawkes model on subset of data with BFGS
    bfgs_model, bfgs_time = \
        fit_standard_hawkes_model_bfgs_noxv(S, K, dt, dt_max,
                                            output_path=output_path,
                                            W_max=None)

    # Fit a network Hawkes model with Gibbs
    gibbs_samples = gibbs_timestamps = None
    gibbs_samples, gibbs_timestamps = \
        fit_network_hawkes_gibbs(S, K, C, dt, dt_max,
                                 output_path=output_path,
                                 standard_model=bfgs_model)

    # gibbs_samples, gibbs_timestamps = \
    #     fit_ct_network_hawkes_gibbs(S, K, C, dt, dt_max,
    #                              output_path=output_path,
    #                              standard_model=bfgs_model)

    # Fit a network Hawkes model with Batch VB
    # vb_models, vb_timestamps = fit_network_hawkes_vb(S, K, dt, dt_max,
    #                                          standard_model=standard_models[-1])
    #
    # with open(output_path + ".vb.pkl", 'w') as f:
    #     print "Saving VB results to ", (output_path + ".vb.pkl")
    #     cPickle.dump((vb_models, timestamps), f, protocol=-1)

    # Fit a network Hawkes model with SVI
    # svi_models = None
    svi_models, timestamps = fit_network_hawkes_svi(S, K, C, dt, dt_max,
                                                    output_path,
                                                    standard_model=bfgs_model,
                                                    true_network=network)

    # Plot the network and its uncertainty
    import ipdb; ipdb.set_trace()

    # Compute area under roc curve of inferred network
    auc_rocs, fprs, tprs = compute_auc_roc(network,
                               W_xcorr=W_xcorr,
                               bfgs_model=bfgs_model,
                               gibbs_samples=gibbs_samples,
                               svi_models=svi_models)
    print "AUC-ROC"
    pprint.pprint(auc_rocs)

    plot_roc_curves(fprs, tprs, fig_path=output_path)

    # Compute area under precisino recall curve of inferred network
    auc_prcs, precs, recalls = compute_auc_prc(network,
                               W_xcorr=W_xcorr,
                               bfgs_model=bfgs_model,
                               gibbs_samples=gibbs_samples,
                               svi_models=svi_models)
    print "AUC-PRC"
    pprint.pprint(auc_prcs)

    plot_prc_curves(precs, recalls, fig_path=output_path)


    # Compute the predictive log likelihoods
    plls = compute_predictive_ll(S_test, S,
                                 bfgs_model=bfgs_model,
                                 gibbs_samples=gibbs_samples,
                                 svi_models=svi_models)

    print "Log Predictive Likelihoods: "
    pprint.pprint(plls)

    # # Plot the predictive log likelihood
    # # N_iters = plls['svi'].size
    # N_iters = 100
    # N_test  = S_test.size
    # plt.ioff()
    # plt.figure()
    # plt.plot(np.arange(N_iters),
    #          (plls['bfgs'] - plls['homog'])/N_test * np.ones(N_iters),
    #          '-b', label='BFGS')
    # import pdb; pdb.set_trace()
    # plt.plot(np.arange(N_iters),
    #          (plls['svi'] - plls['homog'])/N_test * np.ones(N_iters),
    #          '-g', label='SVI')c
    #
    # # plt.plot(np.arange(N_iters),
    # #          (plls['svi'] - plls['homog'])/N_test,
    # #          '-r', label='SVI')
    # plt.xlabel('Iteration')
    # plt.ylabel('Log Predictive Likelihood')
    # plt.legend()
    # plt.show()



def fit_standard_hawkes_model_bfgs(S, K, dt, dt_max, output_path,
                                   W_max=None):
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
        # betas = [ 0.0 ]

        # We want the max W ~ -.025 and the mean to be around 0.01
        # W ~ Gamma(alpha, beta) => E[W] = alpha/beta, so beta ~100 * alpha
        alpha = 1.1
        betas =  [ alpha * 1.0 / 0.01 ]

        init_models = []
        xv_len      = 10000
        init_len    = S.shape[0] - 10000
        S_init      = S[:init_len,:]

        xv_ll       = np.zeros(len(betas))
        S_xv        = S[init_len:init_len+xv_len, :]

        # Make a model to initialize the parameters
        test_basis = IdentityBasis(dt, dt_max, allow_instantaneous=True)
        init_model = DiscreteTimeStandardHawkesModel(K=K, dt=dt, dt_max=dt_max,
                                                     alpha=alpha, beta=0.0,
                                                     basis=test_basis,
                                                     allow_self_connections=False,
                                                     W_max=W_max)
        init_model.add_data(S_init)
        # Initialize the background rates to their mean
        init_model.initialize_to_background_rate()

        start = time.clock()
        for i,beta in enumerate(betas):
            print "Fitting with BFGS on first ", init_len, " time bins, ", \
                "beta = ", beta, "W_max = ", W_max
            init_model.beta = beta
            init_model.fit_with_bfgs()
            init_models.append(init_model.copy_sample())

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
        with open(output_path + ".bfgs.pkl", 'w') as f:
            print "Saving BFGS results to ", (output_path + ".bfgs.pkl")
            cPickle.dump((init_model, init_time), f, protocol=-1)

    return init_model, init_time

def fit_standard_hawkes_model_bfgs_noxv(S, K, dt, dt_max, output_path,
                                        W_max=None):
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
        # We want the max W ~ -.025 and the mean to be around 0.01
        # W ~ Gamma(alpha, beta) => E[W] = alpha/beta, so beta ~100 * alpha
        alpha = 1.1
        beta = alpha * 1.0 / 0.01

        # Make a model to initialize the parameters
        test_basis = IdentityBasis(dt, dt_max, allow_instantaneous=True)
        init_model = DiscreteTimeStandardHawkesModel(K=K, dt=dt, dt_max=dt_max,
                                                     alpha=alpha, beta=beta,
                                                     basis=test_basis,
                                                     allow_self_connections=False,
                                                     W_max=W_max)
        init_model.add_data(S)

        # Initialize the background rates to their mean
        init_model.initialize_to_background_rate()

        start = time.clock()
        init_model.fit_with_bfgs()
        init_time = time.clock() - start

        # Save the model (sans data)
        with open(output_path + ".bfgs.pkl", 'w') as f:
            print "Saving BFGS results to ", (output_path + ".bfgs.pkl")
            cPickle.dump((init_model, init_time), f, protocol=-1)

    return init_model, init_time

def fit_network_hawkes_gibbs(S, K, C, dt, dt_max,
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
        # test_model = DiscreteTimeNetworkHawkesModelGammaMixture(C=C, K=K, dt=dt, dt_max=dt_max, B=B,
        #                                                         alpha=1.0, beta=1.0/20.0)
        test_basis = IdentityBasis(dt, dt_max, allow_instantaneous=True)

        # Set the network prior such that E[W] ~= 0.01
        # W ~ Gamma(kappa, v) for kappa = 1.25 => v ~ 125
        # v ~ Gamma(alpha, beta) for alpha = 10, beta = 10 / 125
        E_W = 0.01
        kappa = 10.
        E_v = kappa / E_W
        alpha = 10.
        beta = alpha / E_v
        network_hypers = {'C': 2,
                          'kappa': kappa, 'alpha': alpha, 'beta': beta,
                          'p': 0.8,
                          'allow_self_connections': False}
        test_model = DiscreteTimeNetworkHawkesModelSpikeAndSlab(K=K, dt=dt, dt_max=dt_max,
                                                                basis=test_basis,
                                                                network_hypers=network_hypers)
        test_model.add_data(S)

        # Initialize with the standard model parameters
        if standard_model is not None:
            test_model.initialize_with_standard_model(standard_model)

        plt.ion()
        im = plot_network(test_model.weight_model.A, test_model.weight_model.W, vmax=0.5)
        plt.pause(0.001)

        # Gibbs sample
        N_samples = 100
        samples = []
        lps = [test_model.log_probability()]
        timestamps = []
        for itr in xrange(N_samples):
            if itr % 1 == 0:
                print "Iteration ", itr, "\tLL: ", lps[-1]
                im.set_data(test_model.weight_model.W_effective)
                plt.pause(0.001)

            # lps.append(test_model.log_probability())
            lps.append(test_model.log_probability())
            samples.append(test_model.resample_and_copy())
            timestamps.append(time.clock())

            # Save this sample
            with open(output_path + ".gibbs.itr%04d.pkl" % itr, 'w') as f:
                cPickle.dump(samples[-1], f, protocol=-1)

        # Save the Gibbs samples
        with open(output_path + ".gibbs.pkl", 'w') as f:
            print "Saving Gibbs samples to ", (output_path + ".gibbs.pkl")
            cPickle.dump((samples, timestamps), f, protocol=-1)

    return samples, timestamps


def fit_ct_network_hawkes_gibbs(S, K, C, dt, dt_max,
                                output_path,
                                standard_model=None):

    # Check for existing Gibbs results
    if os.path.exists(output_path + ".gibbs.pkl"):
        with open(output_path + ".gibbs.pkl", 'r') as f:
            print "Loading Gibbs results from ", (output_path + ".gibbs.pkl")
            (samples, timestamps) = cPickle.load(f)

    else:
        print "Fitting the data with a network Hawkes model using Gibbs sampling"

        S_ct, C_ct, T = convert_discrete_to_continuous(S, dt)

        # Set the network prior such that E[W] ~= 0.01
        # W ~ Gamma(kappa, v) for kappa = 1.25 => v ~ 125
        # v ~ Gamma(alpha, beta) for alpha = 10, beta = 10 / 125
        E_W = 0.2
        kappa = 10.
        E_v = kappa / E_W
        alpha = 5.
        beta = alpha / E_v
        network_hypers = {'C': 1,
                          "c": np.zeros(K).astype(np.int),
                          "p": 0.25,
                          "v": E_v,
                          # 'kappa': kappa,
                          # 'alpha': alpha, 'beta': beta,
                          # 'p': 0.1,
                          'allow_self_connections': False}

        test_model = \
            ContinuousTimeNetworkHawkesModel(K, dt_max=dt_max,
                                             network_hypers=network_hypers)
        test_model.add_data(S_ct, C_ct, T)

        # Initialize with the standard model parameters
        if standard_model is not None:
            test_model.initialize_with_standard_model(standard_model)

        plt.ion()
        im = plot_network(test_model.weight_model.A, test_model.weight_model.W, vmax=0.025)
        plt.pause(0.001)

        # Gibbs sample
        N_samples = 100
        samples = []
        lps = [test_model.log_probability()]
        timestamps = []
        for itr in xrange(N_samples):
            if itr % 1 == 0:
                print "Iteration ", itr, "\tLL: ", lps[-1]
                im.set_data(test_model.weight_model.W_effective)
                plt.pause(0.001)

            # lps.append(test_model.log_probability())
            lps.append(test_model.log_probability())
            samples.append(test_model.resample_and_copy())
            timestamps.append(time.clock())

            print test_model.network.p

            # Save this sample
            with open(output_path + ".gibbs.itr%04d.pkl" % itr, 'w') as f:
                cPickle.dump(samples[-1], f, protocol=-1)

        # Save the Gibbs samples
        with open(output_path + ".gibbs.pkl", 'w') as f:
            print "Saving Gibbs samples to ", (output_path + ".gibbs.pkl")
            cPickle.dump((samples, timestamps), f, protocol=-1)

    return samples, timestamps

def fit_network_hawkes_svi(S, K, C, dt, dt_max,
                           output_path,
                           standard_model=None,
                           N_iters=500,
                           true_network=None):


    # Check for existing Gibbs results
    if os.path.exists(output_path + ".svi.pkl.gz"):
        with gzip.open(output_path + ".svi.pkl.gz", 'r') as f:
            print "Loading SVI results from ", (output_path + ".svi.pkl.gz")
            (samples, timestamps) = cPickle.load(f)
    elif os.path.exists(output_path + ".svi.itr%04d.pkl" % (N_iters-1)):
        with open(output_path + ".svi.itr%04d.pkl" % (N_iters-1), 'r') as f:
            print "Loading SVI results from ", (output_path + ".svi.itr%04d.pkl" % (N_iters-1))
            sample = cPickle.load(f)
            samples = [sample]
            timestamps = None
            # (samples, timestamps) = cPickle.load(f)

    else:
        print "Fitting the data with a network Hawkes model using SVI"

        # Make a new model for inference
        test_basis = IdentityBasis(dt, dt_max, allow_instantaneous=True)
        E_W = 0.01
        kappa = 10.
        E_v = kappa / E_W
        alpha = 10.
        beta = alpha / E_v
        # network_hypers = {'C': 2,
        #                   'kappa': kappa, 'alpha': alpha, 'beta': beta,
        #                   'p': 0.1, 'tau1': 1.0, 'tau0': 1.0,
        #                   'allow_self_connections': False}
        # test_model = DiscreteTimeNetworkHawkesModelGammaMixture(K=K, dt=dt, dt_max=dt_max,
        #                                                         basis=test_basis,
        #                                                         network_hypers=network_hypers)

        network_hypers = {'C': 2,
                          'kappa': kappa, 'alpha': alpha, 'beta': beta,
                          'p': 0.8,
                          'allow_self_connections': False}
        test_model = DiscreteTimeNetworkHawkesModelGammaMixtureFixedSparsity(K=K, dt=dt, dt_max=dt_max,
                                                                basis=test_basis,
                                                                network_hypers=network_hypers)
        # Initialize with the standard model parameters
        if standard_model is not None:
            test_model.initialize_with_standard_model(standard_model)
        #
        # plt.ion()
        # im = plot_network(test_model.weight_model.A, test_model.weight_model.W, vmax=0.03)
        # plt.pause(0.001)

        # TODO: Add the data in minibatches
        minibatchsize = 3000
        test_model.add_data(S)

        # Stochastic variational inference
        samples = []
        delay = 10.0
        forgetting_rate = 0.5
        stepsize = (np.arange(N_iters) + delay)**(-forgetting_rate)
        timestamps = []
        for itr in xrange(N_iters):
            if true_network is not None:
                # W_score = test_model.weight_model.expected_A()
                W_score = test_model.weight_model.expected_W()
                print "AUC: ", roc_auc_score(true_network.ravel(),
                                             W_score.ravel())

            print "SVI Iter: ", itr, "\tStepsize: ", stepsize[itr]
            test_model.sgd_step(minibatchsize=minibatchsize, stepsize=stepsize[itr])
            test_model.resample_from_mf()
            samples.append(test_model.copy_sample())
            timestamps.append(time.clock())
            #
            # if itr % 1 == 0:
            #     plt.figure(1)
            #     im.set_data(test_model.weight_model.expected_W())
            #     plt.pause(0.001)

            # Save this sample
            with open(output_path + ".svi.itr%04d.pkl" % itr, 'w') as f:
                cPickle.dump(samples[-1], f, protocol=-1)

        # Save the Gibbs samples
        with gzip.open(output_path + ".svi.pkl.gz", 'w') as f:
            print "Saving SVI samples to ", (output_path + ".svi.pkl.gz")
            cPickle.dump((samples, timestamps), f, protocol=-1)

    return samples, timestamps

def compute_auc_roc(A_true,
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
    fprs = {}
    tprs = {}

    if W_xcorr is not None:
        aucs['xcorr'] = roc_auc_score(A_flat,
                                      W_xcorr.ravel())
        fprs['xcorr'], tprs['xcorr'], _ = roc_curve(A_flat, W_xcorr.ravel())

    if bfgs_model is not None:
        assert isinstance(bfgs_model, DiscreteTimeStandardHawkesModel)
        W_bfgs = bfgs_model.W.copy()
        W_bfgs -= np.diag(np.diag(W_bfgs))
        aucs['bfgs'] = roc_auc_score(A_flat,
                                     W_bfgs.ravel())
        fprs['bfgs'], tprs['bfgs'], _ = roc_curve(A_flat, W_bfgs.ravel())

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

        aucs['gibbs'] = roc_auc_score(A_flat, Weff_mean.ravel())

    if vb_models is not None:
        # Compute ROC based on E[A] under variational posterior
        aucs['vb'] = roc_auc_score(A_flat,
                                   vb_models[-1].weight_model.expected_A().ravel())

    if svi_models is not None:
        # Compute ROC based on E[A] under variational posterior
        W_svi = svi_models[-1].weight_model.expected_W()
        aucs['svi'] = roc_auc_score(A_flat,
                                    W_svi.ravel())
        fprs['svi'], tprs['svi'], _ = roc_curve(A_flat, W_svi.ravel())


    return aucs, fprs, tprs

def compute_auc_prc(A_true,
                    W_xcorr=None,
                    bfgs_model=None,
                    sgd_model=None,
                    gibbs_samples=None,
                    vb_models=None,
                    svi_models=None,
                    average="macro"):
    """
    Compute the AUC of the precision recall curve
    :return:
    """
    A_flat = A_true.ravel()
    aucs = {}
    precs = {}
    recalls = {}

    if W_xcorr is not None:
        aucs['xcorr'] = average_precision_score(A_flat,
                                                W_xcorr.ravel(),
                                                average=average)
        precs['xcorr'], recalls['xcorr'], _ = precision_recall_curve(A_flat, W_xcorr.ravel())

    if bfgs_model is not None:
        assert isinstance(bfgs_model, DiscreteTimeStandardHawkesModel)
        W_bfgs = bfgs_model.W.copy()
        W_bfgs -= np.diag(np.diag(W_bfgs))
        aucs['bfgs'] = average_precision_score(A_flat,
                                               W_bfgs.ravel(),
                                               average=average)
        precs['bfgs'], recalls['bfgs'], _ = precision_recall_curve(A_flat, W_bfgs.ravel())

    if sgd_model is not None:
        assert isinstance(sgd_model, DiscreteTimeStandardHawkesModel)
        aucs['sgd'] = average_precision_score(A_flat,
                                              sgd_model.W.ravel(),
                                              average=average)
        # precs['sgd'], recalls['sgd'], _ = precision_recall_curve(A_flat, W_sgd.ravel())

    if gibbs_samples is not None:
        # Compute ROC based on mean value of W_effective in second half of samples
        Weff_samples = np.array([s.weight_model.W_effective for s in gibbs_samples])
        N_samples    = Weff_samples.shape[0]
        offset       = N_samples // 2
        Weff_mean    = Weff_samples[offset:,:,:].mean(axis=0)

        aucs['gibbs'] = average_precision_score(A_flat, Weff_mean.ravel(),
                                                average=average)

    if vb_models is not None:
        # Compute ROC based on E[A] under variational posterior
        aucs['vb'] = average_precision_score(A_flat,
                                             vb_models[-1].weight_model.expected_A().ravel(),
                                             average=average)

    if svi_models is not None:
        # Compute ROC based on E[A] under variational posterior
        W_svi = svi_models[-1].weight_model.expected_W()
        aucs['svi'] = average_precision_score(A_flat,
                                              W_svi.ravel(),
                                              average=average)
        precs['svi'], recalls['svi'], _ = precision_recall_curve(A_flat, W_svi.ravel())

    return aucs, precs, recalls


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
        print "Computing pred ll for Gibbs"
        # Compute log(E[pred likelihood]) on second half of samplese
        offset       = len(gibbs_samples) // 2
        # Preconvolve with the Gibbs model's basis
        # F_test = gibbs_samples[0].basis.convolve_with_basis(S_test)
        S_ct, C_ct, T = convert_discrete_to_continuous(S_test, 0.02)

        plls['gibbs'] = []
        for s in gibbs_samples[offset:]:
            # plls['gibbs'].append(s.heldout_log_likelihood(S_test, F=F_test))
            plls['gibbs'].append(s.heldout_log_likelihood(S_ct, C_ct, T))

        # Convert to numpy array
        plls['gibbs'] = np.array(plls['gibbs'])

    import ipdb; ipdb.set_trace()

    if vb_models is not None:
        print "Computing pred ll for VB"
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
        print "Computing predictive likelihood for SVI models"
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

def plot_roc_curves(fprs, tprs, fig_path="./"):
    from hips.plotting.layout import create_figure
    from hips.plotting.colormaps import harvard_colors
    col = harvard_colors()

    fig = create_figure((3,3))
    ax = fig.add_subplot(111)

    # Plot the ROC curves
    if "xcorr" in fprs:
        ax.plot(fprs['xcorr'], tprs['xcorr'], color=col[7], lw=1.5, label="xcorr")
    if "bfgs" in fprs:
        ax.plot(fprs['bfgs'], tprs['bfgs'], color=col[3], lw=1.5, label="MAP")
    if "svi" in fprs:
        ax.plot(fprs['svi'], tprs['svi'], color=col[0], lw=1.5, label="SVI")

    # Plot the diagonal
    ax.plot([0,1], [0,1], '-k', lw=0.5)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")

    # this is another inset axes over the main axes
    # parchment = np.array([243,243,241])/255.
    # inset = plt.axes([0.55, 0.275, .265, .265], axisbg=parchment)
    # inset.plot(fprs['xcorr'], tprs['xcorr'], color=col[7], lw=1.5,)
    # inset.plot(fprs['bfgs'], tprs['bfgs'], color=col[3], lw=1.5,)
    # inset.plot(fprs['svi'], tprs['svi'], color=col[0], lw=1.5, )
    # inset.plot([0,1], [0,1], '-k', lw=0.5)
    # plt.setp(inset, xlim=(0,.2), ylim=(0,.2), xticks=[0, 0.2], yticks=[0,0.2], aspect=1.0)
    # inset.yaxis.tick_right()

    plt.legend(loc=4)
    ax.set_title("ROC Curve")

    plt.subplots_adjust(bottom=0.2, left=0.2)

    plt.savefig(os.path.join(os.path.dirname(fig_path), "figure3c.pdf"))
    plt.show()

def plot_prc_curves(precs, recalls, fig_path="./"):
    from hips.plotting.layout import create_figure
    from hips.plotting.colormaps import harvard_colors
    col = harvard_colors()

    fig = create_figure((3,3))
    ax = fig.add_subplot(111)
    if "xcorr" in recalls:
        ax.plot(recalls['xcorr'], precs['xcorr'], color=col[7], lw=1.5, label="xcorr")
    if "bfgs" in recalls:
        ax.plot(recalls['bfgs'], precs['bfgs'], color=col[3], lw=1.5, label="MAP")
    if "svi" in recalls:
        ax.plot(recalls['svi'], precs['svi'], color=col[0], lw=1.5, label="SVI")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")

    plt.legend(loc=1)
    ax.set_title("Network %d" % net)
    plt.subplots_adjust(bottom=0.25, left=0.25)

    plt.savefig(os.path.join(os.path.dirname(fig_path), "figure3d.pdf"))
    plt.show()

def plot_networks(W_xcorr, bfgs_model, network):
    # Plot a few of the network
    plt.figure()
    plt.subplot(131)
    plt.imshow(W_xcorr, vmin=0, interpolation="none", cmap="Reds")
    plt.colorbar()
    plt.subplot(132)
    plt.title("xcorr")
    plt.imshow(bfgs_model.W, vmin=0, interpolation="none", cmap="Reds")
    plt.colorbar()
    plt.title("bfgs")
    plt.subplot(133)
    plt.imshow(network, vmin=0, interpolation="none", cmap="Reds")
    plt.colorbar()
    plt.title("true")
    plt.show()


def plot_confidence_network(A_true,
                            svi_models=None):
    """
    Compute the AUC of the precision recall curve
    :return:
    """
    A_flat = A_true.ravel()
    # Compute ROC based on E[A] under variational posterior
    # W_svi = svi_models[-1].weight_model.expected_W()
    mean_A = svi_models[-1].weight_model.expected_A()
    std_A = np.sqrt(svi_models[-1].weight_model.mf_p
                    * (1-svi_models[-1].weight_model.mf_p))

    score = mean_A / std_A

    plt.figure()
    plt.imshow(score, interpolation="none", cmap="Reds")
    plt.colorbar()
    plt.show()


# seed = 2650533028
seed = 11223344
net = 6
run = 2
thresh = 0.68
data_path = os.path.join("data", "chalearn", "small", "network%d_oopsi.pkl.gz" % net)
out_path  = os.path.join("data", "chalearn", "small", "network%d_run%03d" % (net,run), "results" )
run_comparison(data_path, out_path, seed=seed, thresh=thresh)
