"""
Compare the various algorithms on a synthetic dataset.
"""
import cPickle
import os
import copy
import gzip
import numpy as np

# Use the Agg backend in running on a server without the DISPLAY variable
if "DISPLAY" not in os.environ:
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import brewer2mpl
# colors = brewer2mpl.get_map("Set1", "Qualitative",  9).mpl_colors
# goodcolors = np.array([0,1,2,4,6,7,8])
# colors = np.array(colors)[goodcolors]

import harness



def load_data(data_path, test_path):
    with gzip.open(data_path, 'r') as f:
        S, true_model = cPickle.load(f)

    with gzip.open(test_path, 'r') as f:
        S_test, test_model = cPickle.load(f)

    return S, S_test, true_model

def plot_pred_ll_vs_time(models, results, burnin=0,
                         homog_ll=np.nan,
                         std_ll=np.nan,
                         nlin_ll=np.nan,
                         true_ll=np.nan):
    from hips.plotting.layout import create_figure
    from hips.plotting.colormaps import harvard_colors

    # Make the ICML figure
    fig = create_figure((4,3))
    ax = fig.add_subplot(111)
    col = harvard_colors()
    plt.grid()

    t_start = 0
    t_stop = 0

    for i, (model, result) in enumerate(zip(models, results)):
        plt.plot(result.timestamps[burnin:], result.test_lls[burnin:], lw=2, color=col[i], label=model)

        # Update time limits
        t_start = min(t_start, result.timestamps[burnin:].min())
        t_stop = max(t_stop, result.timestamps[burnin:].max())

    # plt.legend(loc="outside right")

    # Plot baselines
    plt.plot([t_start, t_stop], homog_ll*np.ones(2), lw=2, color='k', label="Homog")

    # Plot the standard Hawkes test ll
    plt.plot([t_start, t_stop], std_ll*np.ones(2), lw=2, color=col[len(models)], label="Std.")

    # Plot the Nonlinear Hawkes test ll
    plt.plot([t_start, t_stop], nlin_ll*np.ones(2), lw=2, color=col[len(models)+1], label="Nonlinear")

    # Plot the true ll
    plt.plot([t_start, t_stop], true_ll*np.ones(2), '--k',  lw=2,label="True")


    ax.set_xscale("log")
    ax.set_xlim(t_start, t_stop)
    ax.set_xlabel("time [sec]")
    ax.set_ylabel("Pred. Log Lkhd.")

    fig.savefig("pred_ll_vs_time.pdf")
    plt.show()



def plot_impulse_responses(models, results):
    from hips.plotting.layout import create_figure
    from hips.plotting.colormaps import harvard_colors

    # Make the ICML figure
    fig = create_figure((6,6))
    col = harvard_colors()
    plt.grid()

    y_max = 0

    for i, (model, result) in enumerate(zip(models, results)):
        smpl = result.samples[-1]
        W = smpl.W_effective
        if "continuous" in str(smpl.__class__).lower():
            t, irs = smpl.impulses

            for k1 in xrange(K):
                for k2 in xrange(K):
                    plt.subplot(K,K,k1*K + k2 + 1)
                    plt.plot(t, W[k1,k2] * irs[:,k1,k2], color=col[i], lw=2)
        else:
            irs = smpl.impulses
            for k1 in xrange(K):
                for k2 in xrange(K):
                    plt.subplot(K,K,k1*K + k2 + 1)
                    plt.plot(W[k1,k2] * irs[:,k1,k2], color=col[i], lw=2)

        y_max = max(y_max, (W*irs).max())

    for k1 in xrange(K):
        for k2 in xrange(K):
            plt.subplot(K,K,k1*K+k2+1)
            plt.ylim(0,y_max*1.05)
    plt.show()

# def run_comparison(data_path, test_path, output_dir, T_train=None, seed=None):
#     """
#     Run the comparison on the given data file
#     :param data_path:
#     :return:
#     """

if __name__ == "__main__":
    seed = None
    run = 4
    K = 50
    C = 1
    T = 100000
    T_train = 100000
    T_test = 1000
    data_path = os.path.join("data", "synthetic", "synthetic_K%d_C%d_T%d.pkl.gz" % (K,C,T))
    test_path = os.path.join("data", "synthetic", "synthetic_test_K%d_C%d_T%d.pkl.gz" % (K,C,T_test))
    output_dir = os.path.join("results", "synthetic_K%d_C%d_T%d" % (K,C,T_train), "run%03d" % run)
    # run_comparison(data_path, test_path, output_dir, T_train=T_train, seed=seed)

    if seed is None:
        seed = np.random.randint(2**32)
    print "Setting seed to ", seed
    np.random.seed(seed)

    assert os.path.exists(os.path.dirname(output_dir)), "Output directory does not exist!"

    S, S_test, true_model = load_data(data_path, test_path)
    # If T_train is given, only use a fraction of the dataset
    if T_train is not None:
        S = S[:T_train,:]

    # Use the true basis
    dt, dt_max = true_model.dt, true_model.dt_max
    basis = true_model.basis
    network = true_model.network

    # First fit the standard model
    results = []

    output_path = os.path.join(output_dir, "homog.pkl.gz")
    homog_results = \
        harness.fit_homogeneous_pp_model(S, S_test, dt, dt_max, output_path,
                      model_args={"basis": basis})
    homog_model = homog_results.samples[0]

    output_path = os.path.join(output_dir, "std.pkl.gz")
    std_results = \
        harness.fit_standard_hawkes_model_bfgs(S, S_test, dt, dt_max, output_path,
                      model_args={"basis": basis})
    std_model = std_results.samples[0]

    # Fit a nonlinear Hawkes (Poisson GLM) model
    output_path = os.path.join(output_dir, "nonlinear_hawkes.pkl.gz")
    nlin_results = \
        harness.fit_nonlinear_hawkes_model_bfgs(
            S, S_test, dt, dt_max, output_path,
            model_args={"basis": basis, "sigma": np.inf})
    nlin_model = nlin_results.samples[0]

    # Now fit the Bayesian models with MCMC or VB,
    # initializing with the standard model
    models = [
        # "SS-DTH (Gibbs)",
        "SS-CTH (Gibbs)",
        # "MoG-DTH (VB)",
        # "MoG-DTH (SVI)"
    ]
    methods = [
        # harness.fit_spikeslab_network_hawkes_gibbs,
        harness.fit_ct_network_hawkes_gibbs,
        # harness.fit_network_hawkes_vb,
        # harness.fit_network_hawkes_svi
    ]
    inf_args = [
        # {"N_samples": 10000, "standard_model": std_model, "time_limit": 20*60*60},
        {"N_samples": 100000, "standard_model": std_model, "time_limit": 20*60*60},
        # {"N_samples": 10000, "standard_model": std_model, "time_limit": 20*60*60},
        # {"N_samples": 200000, "standard_model": std_model, "time_limit": 20*60*60}
    ]
    model_args = [
        # {"basis": basis, "network": copy.deepcopy(network)},
        {"network": copy.deepcopy(network), "impulse_hypers" : {"mu_0": 0., "lmbda_0": 2.0, "alpha_0": 2.0, "beta_0" : 1.0}},
        # {"basis": basis, "network": copy.deepcopy(network)},
        # {"basis": basis, "network": copy.deepcopy(network)},
    ]

    assert len(models) == len(methods) == len(inf_args) == len(model_args)

    for model, method, iargs, margs in zip(models, methods, inf_args, model_args):
        output_path = os.path.join(output_dir, model.lower() + ".pkl.gz")
        results.append(method(S, S_test, dt, dt_max, output_path,
                              model_args=margs,
                              **iargs))

    # Plot the reuslts
    # plt.ion()
    plot_pred_ll_vs_time(models, results, burnin=1,
                         homog_ll=homog_model.heldout_log_likelihood(S_test),
                         std_ll=std_results.samples[-1].heldout_log_likelihood(S_test),
                         # nlin_ll=nlin_results.samples[-1].heldout_log_likelihood(S_test),
                         true_ll=true_model.heldout_log_likelihood(S_test))

    # Plot impulse responses
    # plot_impulse_responses(models, results)
