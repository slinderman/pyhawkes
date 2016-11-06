import time
import numpy as np
import imp
np.random.seed(1111)
np.seterr(over="raise")
import pickle, os

from hips.plotting.layout import create_figure
import matplotlib.pyplot as plt
import brewer2mpl
colors = brewer2mpl.get_map("Set1", "Qualitative",  9).mpl_colors
# goodcolors = np.array([0,1,2,4,6,7,8])
# colors = np.array(colors)[goodcolors]


from pybasicbayes.util.general import ibincount
from pybasicbayes.util.text import progprint_xrange

import pyhawkes.models
imp.reload(pyhawkes.models)

# Set globals
K = 10
B = 3
dt = 1
dt_max = 10.
T = 100.
network_hypers = {'C': 1, 'kappa': 1., 'c': np.zeros(K, dtype=np.int), 'p': 1*np.ones((1,1)), 'v': 10.}


def generate_dataset(bias=1.):
    # Create the model with these parameters
    network_hypers = {'C': 1, 'kappa': 1., 'c': np.zeros(K, dtype=np.int), 'p': 1*np.ones((1,1)), 'v': 100.}
    bkgd_hypers = {"alpha": 3., "beta": 3./bias}
    dt_model = pyhawkes.models.\
        DiscreteTimeNetworkHawkesModelSpikeAndSlab(K=K, dt=dt, dt_max=dt_max, B=B,
                                                   bkgd_hypers=bkgd_hypers,
                                                   network_hypers=network_hypers)
    # dt_model.bias_model.lambda0 = bias * np.ones(K)
    assert dt_model.check_stability()

    S_dt,_ = dt_model.generate(T=int(np.ceil(T/dt)), keep=False)

    print("sampled dataset with ", S_dt.sum(), "events")

    # Convert S_dt to continuous time
    S_ct = dt * np.concatenate([ibincount(S) for S in S_dt.T]).astype(float)
    S_ct += dt * np.random.rand(*S_ct.shape)
    assert np.all(S_ct < T)
    C_ct = np.concatenate([k*np.ones(S.sum()) for k,S in enumerate(S_dt.T)]).astype(int)

    # Sort the data
    perm = np.argsort(S_ct)
    S_ct = S_ct[perm]
    C_ct = C_ct[perm]

    return S_dt, S_ct, C_ct

def fit_discrete_time_model_gibbs(S_dt, N_samples=100):

    # Now fit a DT model
    dt_model_test = pyhawkes.models.\
        DiscreteTimeNetworkHawkesModelSpikeAndSlab(K=K, dt=dt, dt_max=dt_max, B=B,
                                                   network_hypers=network_hypers)
    dt_model_test.add_data(S_dt)

    tic = time.time()
    for iter in progprint_xrange(N_samples, perline=25):
        dt_model_test.resample_model()
    toc = time.time()

    return (toc-tic) / N_samples

def fit_continuous_time_model_gibbs(S_ct, C_ct, N_samples=100):

    # Now fit a DT model
    ct_model = pyhawkes.models.\
        ContinuousTimeNetworkHawkesModel(K, dt_max=dt_max,
                                         network_hypers=network_hypers)
    ct_model.add_data(S_ct, C_ct, T)

    tic = time.time()
    for iter in progprint_xrange(N_samples, perline=25):
        ct_model.resample_model()
    toc = time.time()

    return (toc-tic) / N_samples

# def run_time_vs_bias():
if __name__ == "__main__":
    # run_time_vs_bias()
    # biases = np.logspace(-1,1, num=10)
    res_file = os.path.join("results", "run_time_vs_rate_2.pkl")

    if os.path.exists(res_file):
        print("Loading results from ", res_file)
        with open(res_file, "r") as f:
            events_per_bin, dt_times, ct_times = pickle.load(f)
    else:
        biases = np.linspace(10**-1,3**1, num=5)
        N_runs_per_bias = 5
        N_samples = 100

        events_per_bin = []
        dt_times = []
        ct_times = []
        for bias in biases:
            for iter in range(N_runs_per_bias):
                print("Bias ", bias, " Run (%d/%d)" % (iter, N_runs_per_bias))
                S_dt, S_ct, C_ct = generate_dataset(bias)
                events_per_bin.append(S_dt.sum() / float(S_dt.size))
                dt_times.append(fit_discrete_time_model_gibbs(S_dt, N_samples))
                ct_times.append(fit_continuous_time_model_gibbs(S_ct, C_ct, N_samples))

        with open(res_file, "w") as f:
            pickle.dump((events_per_bin, dt_times, ct_times), f, protocol=-1)

    events_per_bin = np.array(events_per_bin)
    dt_times = np.array(dt_times)
    ct_times = np.array(ct_times)
    perm = np.argsort(events_per_bin)

    events_per_bin = events_per_bin[perm]
    dt_times = dt_times[perm]
    ct_times = ct_times[perm]

    # Plot the results
    fig = create_figure(figsize=(2.5,2.5))
    fig.set_tight_layout(True)
    ax = fig.add_subplot(111)

    # Plot DT data
    ax.plot(events_per_bin, dt_times, 'o', linestyle="none",
            markerfacecolor=colors[2], markeredgecolor=colors[2], markersize=4,
            label="Discrete")

    # Plot linear fit
    p_dt = np.poly1d(np.polyfit(events_per_bin, dt_times, deg=1))
    dt_pred = p_dt(events_per_bin)
    ax.plot(events_per_bin, dt_pred, ':', lw=2, color=colors[2])

    # Plot CT data
    ax.plot(events_per_bin, ct_times, 's', linestyle="none",
             markerfacecolor=colors[7], markeredgecolor=colors[7], markersize=4,
             label="Continuous")

    # Plot quadratic fit
    p_ct = np.poly1d(np.polyfit(events_per_bin, ct_times, deg=2))
    ct_pred = p_ct(sorted(events_per_bin))
    ax.plot(events_per_bin, ct_pred, ':', lw=2, color=colors[7])

    plt.xlabel("Events per bin")
    # plt.xlim(0, events_per_bin.max())
    plt.xlim(0, 6)
    plt.ylabel("time per iter [sec]")
    plt.ylim(0, 0.15)
    plt.legend(loc="upper left", prop={"size": 8})

    fig.savefig(os.path.join("results", "discrete_cont_comparison.pdf"))
    plt.show()
