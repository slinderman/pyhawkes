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
from scipy.special import gammaln

# Use the Agg backend in running on a server without the DISPLAY variable
if "DISPLAY" not in os.environ:
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pyhawkes.models import DiscreteTimeStandardHawkesModel, \
    DiscreteTimeNetworkHawkesModelGammaMixture, DiscreteTimeNetworkHawkesModelSpikeAndSlab
from pyhawkes.plotting.plotting import plot_network

from baselines.xcorr import infer_net_from_xcorr

from sklearn.metrics import roc_auc_score, average_precision_score


def run_comparison(data_path, test_path, output_path, T_train=None, seed=None):
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
            S, true_model = cPickle.load(f)
    else:
        with open(data_path, 'r') as f:
            S, true_model = cPickle.load(f)

    # If T_train is given, only use a fraction of the dataset
    if T_train is not None:
        S = S[:T_train,:]

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

    # Fix the sparsity for the spike and slab model
    p      = 0.4 * np.eye(C) + 0.01 * (1-np.eye(C))

    use_parse_results = False
    if use_parse_results and  os.path.exists(output_path + ".parsed_results.pkl"):
        with open(output_path + ".parsed_results.pkl") as f:
            auc_rocs, auc_prcs, plls, timestamps = cPickle.load(f)
            timestamps['svi'] = np.array(timestamps['svi'])
        import pdb; pdb.set_trace()

    else:
        # Compute the cross correlation to estimate the connectivity
        W_xcorr = infer_net_from_xcorr(S, dtmax=true_model.dt_max // true_model.dt)

        # Fit a standard Hawkes model on subset of data with BFGS
        bfgs_model, bfgs_time = fit_standard_hawkes_model_bfgs(S, K, B, dt, dt_max,
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
        gibbs_samples = gibbs_timestamps = None
        # gibbs_samples, gibbs_timestamps = fit_network_hawkes_gibbs(S, K, C, B, dt, dt_max,
        #                                                      output_path=output_path,
        #                                                      standard_model=bfgs_model)

        # Fit a spike and slab network Hawkes model with Gibbs
        gibbs_ss_samples, gibbs_ss_timestamps = fit_network_hawkes_gibbs_ss(S, K, C, B, dt, dt_max,
                                                                 output_path=output_path,
                                                                 p=p,
                                                                 standard_model=bfgs_model)

        # Fit a network Hawkes model with Batch VB
        vb_models, vb_timestamps = fit_network_hawkes_vb(S, K, C, B, dt, dt_max,
                                                      output_path=output_path,
                                                      standard_model=bfgs_model)

        # Fit a network Hawkes model with SVI
        # svi_models = svi_timestamps = None
        svi_models, svi_timestamps = fit_network_hawkes_svi(S, K, C, B, dt, dt_max,
                                                        output_path,
                                                        standard_model=bfgs_model)

        # Combine timestamps into a dict
        timestamps = {}
        timestamps['bfgs'] = bfgs_time
        timestamps['gibbs'] = gibbs_timestamps
        timestamps['gibbs_ss'] = gibbs_ss_timestamps
        timestamps['svi'] = svi_timestamps
        timestamps['vb'] = vb_timestamps

        auc_rocs = compute_auc(true_model,
                           W_xcorr=W_xcorr,
                           bfgs_model=bfgs_model,
                           gibbs_samples=gibbs_samples,
                           gibbs_ss_samples=gibbs_ss_samples,
                           svi_models=svi_models,
                           vb_models=vb_models)
        print "AUC-ROC"
        pprint.pprint(auc_rocs)

        # Compute area under precisino recall curve of inferred network
        auc_prcs = compute_auc_prc(true_model,
                                   W_xcorr=W_xcorr,
                                   bfgs_model=bfgs_model,
                                   gibbs_samples=gibbs_samples,
                                   gibbs_ss_samples=gibbs_ss_samples,
                                   svi_models=svi_models,
                                   vb_models=vb_models)
        print "AUC-PRC"
        pprint.pprint(auc_prcs)


        plls = compute_predictive_ll(S_test, S,
                                     true_model=true_model,
                                     bfgs_model=bfgs_model,
                                     gibbs_samples=gibbs_samples,
                                     gibbs_ss_samples=gibbs_ss_samples,
                                     svi_models=svi_models,
                                     vb_models=vb_models)

        with open(output_path + ".parsed_results.pkl", 'w') as f:
            print "Saving parsed results to ", output_path + ".parsed_results.pkl"
            cPickle.dump((auc_rocs, auc_prcs, plls, timestamps), f, protocol=-1)

    plot_pred_ll_vs_time(plls, timestamps, Z=float(S.size), T_train=T_train)


def fit_standard_hawkes_model_bfgs(S, K, B, dt, dt_max, output_path,
                                   init_len=10000, xv_len=1000):
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
        # betas = np.logspace(-3,-0.8,num=10)
        betas = np.array([0.01, 0.1, 1.0, 10.0, 20.0])
        # betas = np.concatenate(([0], betas))

        init_models = []
        S_init      = S[:init_len,:]
        xv_ll       = np.zeros(len(betas))
        S_xv        = S[init_len:init_len+xv_len, :]

        # Make a model to initialize the parameters
        init_model = DiscreteTimeStandardHawkesModel(K=K, dt=dt, B=B, dt_max=dt_max, beta=0.0)
        init_model.add_data(S_init)
        # Initialize the background rates to their mean
        init_model.initialize_to_background_rate()


        start = time.clock()
        for i,beta in enumerate(betas):
            print "Fitting with BFGS on first ", init_len, " time bins, beta = ", beta
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

def load_partial_results(output_path, typ="gibbs"):
    import glob
    # import pdb; pdb.set_trace()

    # Check for existing Gibbs results
    if os.path.exists(output_path + ".%s.pkl" % typ):
        with open(output_path + ".%s.pkl" % typ, 'r') as f:
            print "Loading %s results from " % typ, (output_path + ".%s.pkl" % typ)
            (samples, timestamps) = cPickle.load(f)
            return samples, timestamps
    else:
        if os.path.exists(os.path.join(os.path.dirname(output_path),
                                       "%s_timestamps.pkl" % typ)):
            with open(os.path.join(os.path.dirname(output_path),
                                       "%s_timestamps.pkl" % typ), 'r') as f:
                names_and_timestamps = dict(cPickle.load(f))

        # Look for individual iteration files instead
        files = glob.glob(output_path + ".%s.itr*.pkl" % typ)
        if len(files) > 0:
            full_samples = []
            for file in files:
                with open(file, 'r') as f:
                    print "Loading sample from ", file
                    try:
                        res = cPickle.load(f)
                        if isinstance(res, tuple):
                            sample, timestamp = res
                        else:
                            sample = res
                            timestamp = names_and_timestamps[os.path.basename(file)]
                        full_samples.append((file, sample, timestamp))
                    except:
                        print "Failed to load file ", file



            # Sort the samples by iteration name
            full_samples = sorted(full_samples, key=lambda x: x[0])
            names      = [n for (n,s,t) in full_samples]
            itrs       = np.array([int(n[-8:-4]) for n in names])     # Hack out the iteration number
            samples    = [s for (n,s,t) in full_samples]
            timestamps = np.array([t for (n,s,t) in full_samples])

            if np.all(timestamps > 1e8):
                timestamps = timestamps - timestamps[0]
                samples = samples

            assert np.all(np.diff(itrs) == 1), "Iterations are not sequential!"
            return samples, timestamps

def fit_network_hawkes_gibbs(S, K, C, B, dt, dt_max,
                             output_path,
                             standard_model=None):

    samples_and_timestamps = load_partial_results(output_path, typ="gibbs")
    if samples_and_timestamps is not None:
        samples, timestamps = samples_and_timestamps

    # # Check for existing Gibbs results
    # if os.path.exists(output_path + ".gibbs.pkl"):
    #     with open(output_path + ".gibbs.pkl", 'r') as f:
    #         print "Loading Gibbs results from ", (output_path + ".gibbs.pkl")
    #         (samples, timestamps) = cPickle.load(f)

    else:
        print "Fitting the data with a network Hawkes model using Gibbs sampling"

        # Make a new model for inference
        network_hypers = {'C': C, 'alpha': 1.0, 'beta': 1.0/20.0}
        test_model = DiscreteTimeNetworkHawkesModelGammaMixture(K=K, dt=dt, dt_max=dt_max, B=B,
                                                                network_hypers=network_hypers)
        test_model.add_data(S)

        # Initialize with the standard model parameters
        if standard_model is not None:
            test_model.initialize_with_standard_model(standard_model)

        plt.ion()
        im = plot_network(test_model.weight_model.A, test_model.weight_model.W, vmax=0.5)
        plt.pause(0.001)

        # Gibbs sample
        N_samples = 1000
        samples = []
        lps = []
        timestamps = [time.clock()]
        for itr in xrange(N_samples):
            lps.append(test_model.log_probability())
            # lps.append(test_model.log_likelihood())
            samples.append(test_model.resample_and_copy())
            timestamps.append(time.clock())

            if itr % 1 == 0:
                print "Iteration ", itr, "\t LL: ", lps[-1]
            #    im.set_data(test_model.weight_model.A * \
            #                test_model.weight_model.W)
            #    plt.pause(0.001)

            # Save this sample
            with open(output_path + ".gibbs.itr%04d.pkl" % itr, 'w') as f:
                cPickle.dump((samples[-1], timestamps[-1]-timestamps[0]), f, protocol=-1)

        # Save the Gibbs timestamps
        timestamps = np.array(timestamps)
        with open(output_path + ".gibbs.timestamps.pkl", 'w') as f:
            print "Saving Gibbs samples to ", (output_path + ".gibbs.timestamps.pkl")
            cPickle.dump(timestamps, f, protocol=-1)

        # Save the Gibbs samples
        with open(output_path + ".gibbs.pkl", 'w') as f:
            print "Saving Gibbs samples to ", (output_path + ".gibbs.pkl")
            cPickle.dump((samples, timestamps[1:] - timestamps[0]), f, protocol=-1)

    return samples, timestamps


def fit_network_hawkes_gibbs_ss(S, K, C, B, dt, dt_max,
                                output_path, p,
                                standard_model=None):

    samples_and_timestamps = load_partial_results(output_path, typ="gibbs_ss")
    if samples_and_timestamps is not None:
        samples, timestamps = samples_and_timestamps


    else:
        print "Fitting the data with a spike and slab network Hawkes model using Gibbs sampling"

        # Make a new model for inference
        network_hypers = {'C': C, 'alpha': 1.0, 'beta': 1.0/20.0, 'p': p,
                          'v': 5.0,'c': np.arange(C).repeat((K // C))}
        test_model = DiscreteTimeNetworkHawkesModelSpikeAndSlab(K=K, dt=dt, dt_max=dt_max, B=B,
                                                                network_hypers=network_hypers)
        test_model.add_data(S)

        # Initialize with the standard model parameters
        if standard_model is not None:
            test_model.initialize_with_standard_model(standard_model)

        # Gibbs sample
        N_samples = 1000
        samples = []
        lps = []
        timestamps = [time.clock()]
        for itr in xrange(N_samples):
            lps.append(test_model.log_probability())
            samples.append(test_model.resample_and_copy())
            timestamps.append(time.clock())

            print test_model.network.v

            if itr % 1 == 0:
                print "Iteration ", itr, "\t LP: ", lps[-1]

            # Save this sample
            with open(output_path + ".gibbs_ss.itr%04d.pkl" % itr, 'w') as f:
                cPickle.dump((samples[-1], timestamps[-1]-timestamps[0]), f, protocol=-1)

        # Save the Gibbs timestamps
        timestamps = np.array(timestamps)
        with open(output_path + ".gibbs_ss.timestamps.pkl", 'w') as f:
            print "Saving spike and slab Gibbs samples to ", (output_path + ".gibbs_ss.timestamps.pkl")
            cPickle.dump(timestamps, f, protocol=-1)

        # Save the Gibbs samples
        with open(output_path + ".gibbs_ss.pkl", 'w') as f:
            print "Saving Gibbs samples to ", (output_path + ".gibbs_ss.pkl")
            cPickle.dump((samples, timestamps[1:] - timestamps[0]), f, protocol=-1)

    return samples, timestamps

def fit_network_hawkes_vb(S, K, C, B, dt, dt_max,
                           output_path,
                           standard_model=None):

    samples_and_timestamps = load_partial_results(output_path, typ="vb")
    if samples_and_timestamps is not None:
        samples, timestamps = samples_and_timestamps

    # # Check for existing Gibbs results
    # if os.path.exists(output_path + ".vb.pkl.gz"):
    #     with gzip.open(output_path + ".vb.pkl.gz", 'r') as f:
    #         print "Loading vb results from ", (output_path + ".vb.pkl.gz")
    #         (samples, timestamps) = cPickle.load(f)
    #
    #         if isinstance(timestamps, list):
    #             timestamps = np.array(timestamps)

    else:
        print "Fitting the data with a network Hawkes model using Batch VB"

        # Make a new model for inference
        network_hypers = {'C': C, 'alpha': 1.0, 'beta': 1.0/20.0}
        test_model = DiscreteTimeNetworkHawkesModelGammaMixture(K=K, dt=dt, dt_max=dt_max, B=B,
                                                                network_hypers=network_hypers)
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
        vlbs = []
        samples = []
        start = time.clock()
        timestamps = []
        for itr in xrange(N_iters):
            vlbs.append(test_model.meanfield_coordinate_descent_step())
            print "Batch VB Iter: ", itr, "\tVLB: ", vlbs[-1]
            samples.append(test_model.copy_sample())
            timestamps.append(time.clock())

            if itr % 1 == 0:
                im.set_data(test_model.weight_model.expected_W())
                plt.pause(0.001)

            # Save this sample
            with open(output_path + ".vb.itr%04d.pkl" % itr, 'w') as f:
                cPickle.dump((samples[-1], timestamps[-1] - start), f, protocol=-1)

        # Save the Gibbs samples
        timestamps = np.array(timestamps)
        with gzip.open(output_path + ".vb.pkl.gz", 'w') as f:
            print "Saving VB samples to ", (output_path + ".vb.pkl.gz")
            cPickle.dump((samples, timestamps - start), f, protocol=-1)

    return samples, timestamps


def fit_network_hawkes_svi(S, K, C, B, dt, dt_max,
                           output_path,
                           standard_model=None):

    samples_and_timestamps = load_partial_results(output_path, typ="svi")
    if samples_and_timestamps is not None:
        samples, timestamps = samples_and_timestamps

    else:
        print "Fitting the data with a network Hawkes model using SVI"

        # Make a new model for inference
        network_hypers = {'C': C, 'alpha': 1.0, 'beta': 1.0/20.0}
        test_model = DiscreteTimeNetworkHawkesModelGammaMixture(K=K, dt=dt, dt_max=dt_max, B=B,
                                                                network_hypers=network_hypers)
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
        N_iters = 10000
        samples = []
        delay = 1.0
        forgetting_rate = 0.5
        stepsize = (np.arange(N_iters) + delay)**(-forgetting_rate)
        start = time.clock()
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
                cPickle.dump((samples[-1], timestamps[-1] -start), f, protocol=-1)

        # Save the Gibbs samples
        timestamps = np.array(timestamps)
        with gzip.open(output_path + ".svi.pkl.gz", 'w') as f:
            print "Saving SVI samples to ", (output_path + ".svi.pkl.gz")
            cPickle.dump((samples, timestamps - start), f, protocol=-1)

    return samples, timestamps

def compute_auc(true_model,
                W_xcorr=None,
                bfgs_model=None,
                sgd_model=None,
                gibbs_samples=None,
                gibbs_ss_samples=None,
                vb_models=None,
                svi_models=None):
    """
    Compute the AUC score for each of competing models
    :return:
    """
    aucs = {}

    # Get the true adjacency matrix
    A_true = true_model.weight_model.A.ravel()

    if W_xcorr is not None:
        aucs['xcorr'] = roc_auc_score(A_true,
                                     W_xcorr.ravel())

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

        aucs['gibbs'] = roc_auc_score(A_true, Weff_mean.ravel())

    if gibbs_ss_samples is not None:
        # Compute ROC based on mean value of W_effective in second half of samples
        Weff_samples = np.array([s.weight_model.W_effective for s in gibbs_ss_samples])
        N_samples    = Weff_samples.shape[0]
        offset       = N_samples // 2
        Weff_mean    = Weff_samples[offset:,:,:].mean(axis=0)

        aucs['gibbs_ss'] = roc_auc_score(A_true, Weff_mean.ravel())

    if vb_models is not None:
        # Compute ROC based on E[A] under variational posterior
        aucs['vb'] = roc_auc_score(A_true,
                                   vb_models[-1].weight_model.expected_A().ravel())

    if svi_models is not None:
        # Compute ROC based on E[A] under variational posterior
        aucs['svi'] = roc_auc_score(A_true,
                                    svi_models[-1].weight_model.expected_A().ravel())

    return aucs


def compute_auc_prc(true_model,
                    W_xcorr=None,
                    bfgs_model=None,
                    sgd_model=None,
                    gibbs_samples=None,
                    gibbs_ss_samples=None,
                    vb_models=None,
                    svi_models=None,
                    average="macro"):
    """
    Compute the AUC of the precision recall curve
    :return:
    """
    A_flat = true_model.weight_model.A.ravel()
    aucs = {}

    if W_xcorr is not None:
        aucs['xcorr'] = average_precision_score(A_flat,
                                                W_xcorr.ravel(),
                                                average=average)

    if bfgs_model is not None:
        assert isinstance(bfgs_model, DiscreteTimeStandardHawkesModel)
        W_bfgs = bfgs_model.W.copy()
        W_bfgs -= np.diag(np.diag(W_bfgs))
        aucs['bfgs'] = average_precision_score(A_flat,
                                               W_bfgs.ravel(),
                                               average=average)

    if sgd_model is not None:
        assert isinstance(sgd_model, DiscreteTimeStandardHawkesModel)
        aucs['sgd'] = average_precision_score(A_flat,
                                              sgd_model.W.ravel(),
                                              average=average)

    if gibbs_samples is not None:
        # Compute ROC based on mean value of W_effective in second half of samples
        Weff_samples = np.array([s.weight_model.W_effective for s in gibbs_samples])
        N_samples    = Weff_samples.shape[0]
        offset       = N_samples // 2
        Weff_mean    = Weff_samples[offset:,:,:].mean(axis=0)

        aucs['gibbs'] = average_precision_score(A_flat, Weff_mean.ravel(), average=average)

    if gibbs_ss_samples is not None:
        # Compute ROC based on mean value of W_effective in second half of samples
        Weff_samples = np.array([s.weight_model.W_effective for s in gibbs_ss_samples])
        N_samples    = Weff_samples.shape[0]
        offset       = N_samples // 2
        Weff_mean    = Weff_samples[offset:,:,:].mean(axis=0)

        aucs['gibbs_ss'] = average_precision_score(A_flat, Weff_mean.ravel(), average=average)


    if vb_models is not None:
        # Compute ROC based on E[A] under variational posterior
        aucs['vb'] = average_precision_score(A_flat,
                                             vb_models[-1].weight_model.expected_A().ravel(),
                                             average=average)

    if svi_models is not None:
        # Compute ROC based on E[A] under variational posterior
        aucs['svi'] = average_precision_score(A_flat,
                                              svi_models[-1].weight_model.expected_A().ravel(),
                                              average=average)

    return aucs

def compute_predictive_ll(S_test, S_train,
                          true_model=None,
                          bfgs_model=None,
                          sgd_models=None,
                          gibbs_samples=None,
                          gibbs_ss_samples=None,
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
        print "Computing predictive log likelihood for Gibbs samples"
        # Compute log(E[pred likelihood]) on second half of samplese
        offset       = 0
        # Preconvolve with the Gibbs model's basis
        F_test = gibbs_samples[0].basis.convolve_with_basis(S_test)

        plls['gibbs'] = []
        for s in gibbs_samples[offset:]:
            plls['gibbs'].append(s.heldout_log_likelihood(S_test, F=F_test))

        # Convert to numpy array
        plls['gibbs'] = np.array(plls['gibbs'])

    if gibbs_ss_samples is not None:
        print "Computing predictive log likelihood for spike and slab Gibbs samples"
        # Compute log(E[pred likelihood]) on second half of samplese
        offset       = 0
        # Preconvolve with the Gibbs model's basis
        F_test = gibbs_samples[0].basis.convolve_with_basis(S_test)

        plls['gibbs_ss'] = []
        for s in gibbs_ss_samples[offset:]:
            plls['gibbs_ss'].append(s.heldout_log_likelihood(S_test, F=F_test))

        # Convert to numpy array
        plls['gibbs_ss'] = np.array(plls['gibbs_ss'])


    if vb_models is not None:
        print "Computing predictive log likelihood for VB iterations"
        # Compute predictive likelihood over samples from VB model
        N_models  = len(vb_models)
        N_samples = 10
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
        print "Computing predictive log likelihood for SVI iterations"
        # Compute predictive likelihood over samples from VB model
        N_models  = len(svi_models)
        N_samples = 10
        # Preconvolve with the VB model's basis
        F_test = svi_models[0].basis.convolve_with_basis(S_test)

        svi_plls = np.zeros((N_models, N_samples))
        for i, svi_model in enumerate(svi_models):
            # print "Computing pred ll for SVI iteration ", i
            if i % 10 != 0:
                svi_plls[i,:] = np.nan
                continue

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

def plot_pred_ll_vs_time(plls, timestamps, Z=1.0, T_train=None, nbins=4):

    # import seaborn as sns
    # sns.set(style="whitegrid")

    from hips.plotting.layout import create_figure
    from hips.plotting.colormaps import harvard_colors

    # Make the ICML figure
    fig = create_figure((4,3))
    ax = fig.add_subplot(111)
    col = harvard_colors()
    plt.grid()

    # Compute the max and min time in seconds
    print "Homog PLL: ", plls['homog']

    # DEBUG
    plls['homog'] = 0.0
    Z = 1.0

    assert "bfgs" in plls and "bfgs" in timestamps
    # t_bfgs = timestamps["bfgs"]
    t_bfgs = 1.0
    t_start = 1.0
    t_stop = 0.0

    if 'svi' in plls and 'svi' in timestamps:
        # import pdb; pdb.set_trace()
        isreal = ~np.isnan(plls['svi'])
        svis = plls['svi'][isreal]
        t_svi = timestamps['svi'][isreal]
        t_svi = t_bfgs + t_svi - t_svi[0]
        t_stop = max(t_stop, t_svi[-1])
        ax.semilogx(t_svi, (svis - plls['homog'])/Z, color=col[0], label="SVI", lw=1.5)

    if 'vb' in plls and 'vb' in timestamps:
        t_vb = timestamps['vb']
        t_vb = t_bfgs + t_vb
        t_stop = max(t_stop, t_vb[-1])
        ax.semilogx(t_vb, (plls['vb'] - plls['homog'])/Z, color=col[1], label="VB", lw=1.5)

    if 'gibbs' in plls and 'gibbs' in timestamps:
        t_gibbs = timestamps['gibbs']
        t_gibbs = t_bfgs + t_gibbs
        t_stop = max(t_stop, t_gibbs[-1])
        ax.semilogx(t_gibbs, (plls['gibbs'] - plls['homog'])/Z, color=col[2], label="Gibbs", lw=1.5)

    # if 'gibbs_ss' in plls and 'gibbs_ss' in timestamps:
    #     t_gibbs = timestamps['gibbs_ss']
    #     t_gibbs = t_bfgs + t_gibbs
    #     t_stop = max(t_stop, t_gibbs[-1])
    #     ax.semilogx(t_gibbs, (plls['gibbs_ss'] - plls['homog'])/Z, color=col[8], label="Gibbs-SS", lw=1.5)

    # Extend lines to t_st
    if 'svi' in plls and 'svi' in timestamps:
        final_svi_pll = -np.log(4) + logsumexp(svis[-4:])
        ax.semilogx([t_svi[-1], t_stop],
                    [(final_svi_pll - plls['homog'])/Z,
                     (final_svi_pll - plls['homog'])/Z],
                    '--',
                    color=col[0], lw=1.5)

    if 'vb' in plls and 'vb' in timestamps:
        ax.semilogx([t_vb[-1], t_stop],
                    [(plls['vb'][-1] - plls['homog'])/Z,
                     (plls['vb'][-1] - plls['homog'])/Z],
                    '--',
                    color=col[1], lw=1.5)

    ax.semilogx([t_start, t_stop],
                [(plls['bfgs'] - plls['homog'])/Z, (plls['bfgs'] - plls['homog'])/Z],
                color=col[3], lw=1.5, label="Std." )

    # Put a legend above
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=5, mode="expand", borderaxespad=0.,
               prop={'size':9})

    ax.set_xlim(t_start, t_stop)

    # Format the ticks
    # plt.locator_params(nbins=nbins)

    import matplotlib.ticker as ticker
    logxscale = 3
    xticks = ticker.FuncFormatter(lambda x, pos: '{0:.1f}'.format(x/10.**logxscale))
    ax.xaxis.set_major_formatter(xticks)
    ax.set_xlabel('Time ($10^{%d}$ s)' % logxscale)

    logyscale = 4
    yticks = ticker.FuncFormatter(lambda y, pos: '{0:.3f}'.format(y/10.**logyscale))
    ax.yaxis.set_major_formatter(yticks)
    ax.set_ylabel('Pred. LL ($ \\times 10^{%d}$)' % logyscale)

    # ylim = ax.get_ylim()
    # ax.plot([t_bfgs, t_bfgs], ylim, '--k')
    # ax.set_ylim(ylim)


    # plt.tight_layout()
    plt.subplots_adjust(bottom=0.2, left=0.2)
    # plt.title("Predictive Log Likelihood ($T=%d$)" % T_train)
    plt.show()
    fig.savefig('figure2a.pdf')


# seed = 2650533028
seed = None
run = 3
K = 50
C = 5
T = 100000
T_train = 11000
T_test = 1000
data_path = os.path.join("data", "synthetic", "synthetic_K%d_C%d_T%d.pkl.gz" % (K,C,T))
test_path = os.path.join("data", "synthetic", "synthetic_test_K%d_C%d_T%d.pkl" % (K,C,T_test))
out_path = os.path.join("data", "synthetic", "results_K%d_C%d_T%d" % (K,C,T), "run%03d" %run, "results" )
run_comparison(data_path, test_path, out_path, T_train=T_train, seed=seed)
