
import pickle
import os
import gzip
import numpy as np
import matplotlib.pyplot as plt

from hips.plotting.layout import create_figure
from hips.plotting.colormaps import harvard_colors

def make_figure_a(S, F, C):
    """
    Plot fluorescence traces, filtered fluorescence, and spike times
    for three neurons
    """
    col = harvard_colors()
    dt = 0.02
    T_start = 0
    T_stop = 1 * 50 * 60
    t = dt * np.arange(T_start, T_stop)

    ks = [0,1]
    nk = len(ks)
    fig = create_figure((3,3))
    for ind,k in enumerate(ks):
        ax = fig.add_subplot(nk,1,ind+1)
        ax.plot(t, F[T_start:T_stop, k], color=col[1], label="$F$")    # Plot the raw flourescence in blue
        ax.plot(t, C[T_start:T_stop, k], color=col[0], lw=1.5, label="$\widehat{F}$")    # Plot the filtered flourescence in red
        spks  = np.where(S[T_start:T_stop, k])[0]
        ax.plot(t[spks], C[spks,k], 'ko', label="S")            # Plot the spike times in black

        # Make a legend
        if ind == 0:
            # Put a legend above
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                       ncol=3, mode="expand", borderaxespad=0.,
                       prop={'size':9})

        # Add labels
        ax.set_ylabel("$F_%d(t)$" % (k+1))
        if ind == nk-1:
            ax.set_xlabel("Time $t$ [sec]")

        # Format the ticks
        ax.set_ylim([-0.1,1.0])
        plt.locator_params(nbins=5, axis="y")


    plt.subplots_adjust(left=0.2, bottom=0.2)
    fig.savefig("figure3a.pdf")
    plt.show()



data_path = os.path.join("data", "chalearn", "small", "network1_oopsi.pkl.gz")

with gzip.open(data_path, 'r') as f:
    P, F, Cf, network, pos = pickle.load(f)
    S_full = (P > 0.1).astype(np.int)

make_figure_a(S_full, F, Cf)