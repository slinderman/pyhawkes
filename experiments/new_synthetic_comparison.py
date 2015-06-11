"""
Compare the various algorithms on a synthetic dataset.
"""
import cPickle
import os
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

def plot_pred_ll_vs_time(models, results, S_test):

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
        plt.plot(result.timestamps, result.test_lls, lw=2, color=col[i], label=model)

        # Update time limits
        t_start = min(t_start, result.timestamps.min())
        t_stop = max(t_start, result.timestamps.max())

    plt.legend()

    ax.set_xlim(t_start, t_stop)
    ax.set_xlabel("time [sec]")
    ax.set_ylabel("Pred. Log Lkhd.")
    plt.show()


def run_comparison(data_path, test_path, output_dir, T_train=None, seed=None):
    """
    Run the comparison on the given data file
    :param data_path:
    :return:
    """
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

    models = ["Std",
              "SS-DTH (Gibbs)",
              #"SS-CTH (Gibbs)",
              "MoG-DTH (VB)",
              "MoG-DTH (SVI)"]
    methods = [harness.fit_standard_hawkes_model_bfgs,
               harness.fit_spikeslab_network_hawkes_gibbs,
               #harness.fit_ct_network_hawkes_gibbs,
               harness.fit_network_hawkes_vb,
               harness.fit_network_hawkes_svi]
    params = [{"basis": basis, "alpha": 1.0, "beta": 1.0},
              {"basis": basis, "network": network},
              #{},
              {"basis": basis, "network": network},
              {"basis": basis, "network": network},
              ]

    results = []
    for model, method, args in zip(models, methods, params):
        output_path = os.path.join(output_dir, model.lower() + ".pkl.gz")
        results.append(method(S, S_test, dt, dt_max, output_path, model_args=args))

    # Plot the reuslts
    plot_pred_ll_vs_time(models, results, S_test)

# seed = 2650533028
seed = None
run = 1
K = 4
C = 1
T = 1000
T_train = 1000
T_test = 1000
data_path = os.path.join("data", "synthetic", "synthetic_K%d_C%d_T%d.pkl.gz" % (K,C,T))
test_path = os.path.join("data", "synthetic", "synthetic_test_K%d_C%d_T%d.pkl.gz" % (K,C,T_test))
output_dir = os.path.join("results", "synthetic_K%d_C%d_T%d" % (K,C,T), "run%03d" % run)
run_comparison(data_path, test_path, output_dir, T_train=T_train, seed=seed)
