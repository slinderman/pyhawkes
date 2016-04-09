import os
import cPickle

import numpy as np
# np.random.seed(1111)
np.seterr(over="raise")
from scipy.io import loadmat
from scipy.misc import logsumexp

import seaborn as sns
sns.set(style="white")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib
matplotlib.rcParams.update({'font.sans-serif' : 'Helvetica',
                            'axes.labelsize': 9,
                            'xtick.labelsize' : 7,
                            'ytick.labelsize' : 7,
                            'axes.titlesize' : 9})


from hips.plotting.layout import create_axis_at_location, create_figure, remove_plot_labels
from hips.plotting.colormaps import harvard_colors, gradient_cmap
colors = harvard_colors()

from graphistician.internals.utils import compute_optimal_rotation

from pybasicbayes.util.text import progprint_xrange

import pyhawkes.models
reload(pyhawkes.models)
from pyhawkes.models import ContinuousTimeNetworkHawkesModel

from pyhawkes.internals.network import LatentDistanceAdjacencyModel, ErdosRenyiFixedSparsity, ErdosRenyiModel

# Load the hippocampal data
with open("data/hippocampus.mat", "r") as f:
    matdata = loadmat(f)
    S = matdata['data']['S'][0,0].ravel().astype(np.float)
    C = matdata['data']['C'][0,0].ravel().astype(np.int)
    X = matdata['data']['X'][0,0].astype(np.float)
    M = S.shape[0]

# Sort the data
perm = np.argsort(S)
S = S[perm]
C = C[perm]
X = X[perm]

# Subtract off min spike time and co
S -= S.min()
T = np.ceil(S.max())

# Set first neuron to index 0
C -= 1
K = C.max() + 1

# Center locations
X -= X.mean(0)
rad = 61

# Get the place fields
pfs = np.zeros((K, 2))
pfs_cov = np.zeros((K,2,2))
for k in range(K):
    pfs[k] = X[C == k].mean(0)
    pfs_cov[k] = np.cov(X[C == k].T)

pf_size = np.array([np.linalg.eigvalsh(cc).max() for cc in pfs_cov])
pf_size -= pf_size.min()
pf_size /= pf_size.max()

# Sort the nodes by location
pfs_rad = np.sqrt(np.sum(pfs**2, 1))
pfs_th = np.arctan2(pfs[:,1], pfs[:,0])
node_perm = np.lexsort((pfs_rad, pfs_th))

# Set dynamics parameters
B = 1
dt_max = 1.

# Withold some data for testing
train_frac = 0.8
T_train = train_frac * T
T_test = T - T_train
train_inds = S < T_train
S_train, C_train, X_train = S[train_inds], C[train_inds], X[train_inds]
S_test, C_test, X_test = S[~train_inds], C[~train_inds], X[~train_inds]
S_test -= T_train

# Create four different models: empty, dense, ER, distance
dense_network = ErdosRenyiFixedSparsity(K=K, p=1.0, kappa=1, alpha=1.0, beta=1.0)
er_network    = ErdosRenyiModel(K=K, kappa=1, alpha=1.0, beta=1.0)
dist_network  = LatentDistanceAdjacencyModel(K=K, kappa=1, dim=2, alpha=1.0, beta=1.0)

networks = [dense_network, er_network, dist_network]
names = ["Dense", "ER", "Distance"]


### Fit the models
N_samples = 1000
results = []
results_dir = os.path.join("results", "hippocampus", "run002")
for network, name in zip(networks, names):
    results_file = os.path.join(results_dir, "%s.pkl" % name)
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            result = cPickle.load(f)
            results.append(result)
        continue

    print "Fitting model with ", name, " network."
    model = ContinuousTimeNetworkHawkesModel(
        K, dt_max=1.,
        network=network)

    model.add_data(S_train, C_train, T_train)
    model.resample_model()

    # Add the test data and then remove it. That way we can
    # efficiently compute its predictive log likelihood
    model.add_data(S_test, C_test, T - T_train)
    test_data = model.data_list.pop()

    ### Fit the model
    lls = [model.log_likelihood()]
    plls = [model.log_likelihood(test_data)]
    Weffs = []
    Ps = []
    Ls = []

    for iter in progprint_xrange(N_samples, perline=25):
        model.resample_model()

        lls.append(model.log_likelihood())
        plls.append(model.log_likelihood(test_data))
        Weffs.append(model.W_effective)
        Ps.append(model.network.P)
        if isinstance(network, LatentDistanceAdjacencyModel):
            Ls.append(model.network.L)

    result = (lls, plls, Weffs, Ps, Ls)
    results.append(result)

    # Save results
    with open(results_file, "w") as f:
        cPickle.dump(result, f, protocol=-1)

def plot_results(result):
    lls, plls, Weffs, Ps, Ls = result

    ### Colored locations
    wheel_cmap = gradient_cmap([colors[0], colors[3], colors[2], colors[1], colors[0]])
    fig = create_figure(figsize=(1.8, 1.8))
    # ax = create_axis_at_location(fig, .1, .1, 1.5, 1.5, box=False)
    ax = create_axis_at_location(fig, .6, .4, 1.1, 1.1)

    for i,k in enumerate(node_perm):
        color = wheel_cmap((np.pi+pfs_th[k])/(2*np.pi))
        # alpha = pfs_rad[k] / 47
        alpha = 0.7
        ax.add_patch(Circle((pfs[k,0], pfs[k,1]),
                            radius=3+4*pf_size[k],
                            color=color, ec="none",
                            alpha=alpha)
                            )

    plt.title("True place fields")
    # ax.text(0, 45, "True Place Fields",
    #         horizontalalignment="center",
    #         fontdict=dict(size=9))
    plt.xlim(-45,45)
    plt.xticks([-40, -20, 0, 20, 40])
    plt.xlabel("$x$ [cm]")
    plt.ylim(-45,45)
    plt.yticks([-40, -20, 0, 20, 40])
    plt.ylabel("$y$ [cm]")
    plt.savefig(os.path.join(results_dir, "hipp_colored_locations.pdf"))


    # Plot the inferred weighted adjacency matrix
    fig = create_figure(figsize=(1.8, 1.8))
    ax = create_axis_at_location(fig, .4, .4, 1.1, 1.1)

    Weff = np.array(Weffs[N_samples//2:]).mean(0)
    Weff = Weff[np.ix_(node_perm, node_perm)]
    lim = Weff[(1-np.eye(K)).astype(np.bool)].max()
    im = ax.imshow(np.kron(Weff, np.ones((20,20))),
                   interpolation="none", cmap="Greys", vmax=lim)
    ax.set_xticks([])
    ax.set_yticks([])

    # node_colors = wheel_cmap()
    node_values = ((np.pi+pfs_th[node_perm])/(2*np.pi))[:,None] *np.ones((K,2))
    yax = create_axis_at_location(fig, .2, .4, .3, 1.1)
    remove_plot_labels(yax)
    yax.imshow(node_values, interpolation="none",
               cmap=wheel_cmap)
    yax.set_xticks([])
    yax.set_yticks([])
    yax.set_ylabel("pre")

    xax = create_axis_at_location(fig, .4, .2, 1.1, .3)
    remove_plot_labels(xax)
    xax.imshow(node_values.T, interpolation="none",
               cmap=wheel_cmap)
    xax.set_xticks([])
    xax.set_yticks([])
    xax.set_xlabel("post")

    cbax = create_axis_at_location(fig, 1.55, .4, .04, 1.1)
    plt.colorbar(im, cax=cbax, ticks=[0, .1, .2,  .3])
    cbax.tick_params(labelsize=8, pad=1)
    cbax.set_ticklabels=["0", ".1", ".2",  ".3"]

    ax.set_title("Inferred Weights")
    plt.savefig(os.path.join(results_dir, "hipp_W.pdf"))

    # # Plot the inferred connection probability
    # plt.figure()
    # plt.imshow(P, interpolation="none", cmap="Greys", vmin=0)
    # plt.colorbar()

        # Plot the inferred weighted adjacency matrix
    fig = create_figure(figsize=(1.8, 1.8))
    ax = create_axis_at_location(fig, .4, .4, 1.1, 1.1)

    P = np.array(Ps[N_samples//2:]).mean(0)
    P = P[np.ix_(node_perm, node_perm)]
    im = ax.imshow(np.kron(P, np.ones((20,20))),
                   interpolation="none", cmap="Greys", vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])

    # node_colors = wheel_cmap()
    node_values = ((np.pi+pfs_th[node_perm])/(2*np.pi))[:,None] *np.ones((K,2))
    yax = create_axis_at_location(fig, .2, .4, .3, 1.1)
    remove_plot_labels(yax)
    yax.imshow(node_values, interpolation="none",
               cmap=wheel_cmap)
    yax.set_xticks([])
    yax.set_yticks([])
    yax.set_ylabel("pre")

    xax = create_axis_at_location(fig, .4, .2, 1.1, .3)
    remove_plot_labels(xax)
    xax.imshow(node_values.T, interpolation="none",
               cmap=wheel_cmap)
    xax.set_xticks([])
    xax.set_yticks([])
    xax.set_xlabel("post")

    cbax = create_axis_at_location(fig, 1.55, .4, .04, 1.1)
    plt.colorbar(im, cax=cbax, ticks=[0, .5, 1])
    cbax.tick_params(labelsize=8, pad=1)
    cbax.set_ticklabels=["0.0", "0.5",  "1.0"]

    ax.set_title("Inferred Probability")
    plt.savefig(os.path.join(results_dir, "hipp_P.pdf"))


    plt.show()

def plot_pred_lls(results):
    # Plot predictive log likelihoods
    homog_rates = np.zeros(K)
    for k in range(K):
        homog_rates[k] = (C_train==k).sum() / T_train

    homog_pll = 0
    for k in range(K):
        homog_pll += -T_test * homog_rates[k]
        homog_pll += (C_test==k).sum() * np.log(homog_rates[k])

    # Plot predictive log likelihoods relative to standard Poisson
    plt.figure()
    for i,result in enumerate(results):
        lls, plls, Weffs, Ps, Ls = result
        plls = plls[N_samples//2:]
        J = len(plls)
        avg_pll = -np.log(J) + logsumexp(plls)
        avg_pll = (avg_pll - homog_pll) / len(S_test)

        samples = np.random.choice(plls, size=(100, J), replace=True)
        pll_samples = logsumexp((samples - homog_pll)/len(S_test), axis=1) - np.log(J)
        std_hll = pll_samples.std()

        print "PLL: ", avg_pll, " +- ", std_hll
        plt.bar(i, avg_pll)
    plt.show()


def plot_locations(result, offset=0):
    ### Plot the sampled locations for a few neurons
    _, _, _, _, Ls = result
    Ls_rot = []
    for L in Ls:
        R = compute_optimal_rotation(L, pfs, scale=False)
        Ls_rot.append(L.dot(R))
    Ls_rot = np.array(Ls_rot)

    fig = create_figure(figsize=(1.4,2.9))
    ax = create_axis_at_location(fig, .3, 1.7, 1, 1)

    # toplot = np.random.choice(np.arange(K), size=4, replace=False)
    toplot = np.linspace(offset,K+offset, 4, endpoint=False).astype(np.int)
    print toplot
    wheel_cmap = gradient_cmap([colors[0], colors[3], colors[2], colors[1], colors[0]])
    plot_colors = [wheel_cmap((np.pi+pfs_th[node_perm[j]])/(2*np.pi)) for j in toplot]

    for i,k in enumerate(node_perm):
        # plt.text(pfs[k,0], pfs[k,1], "%d" % i)
        if i not in toplot:
            color = 0.8 * np.ones(3)

            plt.plot(pfs[k,0], pfs[k, 1], 'o',
                     markerfacecolor=color, markeredgecolor=color,
                     markersize=4 + 4 * pf_size[k],
                     alpha=1.0)

    for i,k in enumerate(node_perm):
        # plt.text(pfs[k,0], pfs[k,1], "%d" % i)
        if i in toplot:
            j = np.where(toplot==i)[0][0]
            color = plot_colors[j]

            plt.plot(pfs[k,0], pfs[k, 1], 'o',
                     markerfacecolor=color, markeredgecolor=color,
                     markersize=4 + 4 * pf_size[k])



    #     plt.gca().add_patch(Circle((0,0), radius=rad, ec='k', fc="none"))
    plt.title("True Place Fields")
    plt.xlim(-45,45)
    plt.xticks([-40, -20, 0, 20, 40])
    # plt.xlabel("$x$")
    plt.ylim(-45,45)
    plt.yticks([-40, -20, 0, 20, 40])
    # plt.ylabel("$y$")

    # Now plot the inferred locations
    # plt.subplot(212, aspect='equal')
    ax = create_axis_at_location(fig, .3, .2, 1, 1)
    for L in Ls_rot[::2]:
        for j in np.random.permutation(len(toplot)):
            k = node_perm[toplot][j]
            color = plot_colors[j]
            plt.plot(L[k,0], L[k,1], 'o',
                     markerfacecolor=color, markeredgecolor="none",
                     markersize=4, alpha=0.25)

    plt.title("Locations Samples")
    # plt.xlim(-30, 30)
    # plt.xticks([])
    # plt.ylim(-30, 30)
    # plt.yticks([])
    plt.xlim(-3, 3)
    plt.xticks([-2, 0, 2])
    plt.ylim(-3, 3)
    plt.yticks([-2, 0, 2])

    plt.savefig(os.path.join(results_dir, "locations_%d.pdf" % offset))
    # plt.show()



def plot_mean_locations():
    ### Plot the sampled locations for a few neurons
    _, _, _, _, Ls = result
    Ls_rot = []
    for L in Ls:
        R = compute_optimal_rotation(L, pfs)
        Ls_rot.append(L.dot(R))
    Ls_rot = np.array(Ls_rot)

    Ls_mean = np.mean(Ls_rot, 0)

    fig = create_figure(figsize=(1.4,2.5))
    plt.subplot(211, aspect='equal')

    wheel_cmap = gradient_cmap([colors[0], colors[3], colors[2], colors[1], colors[0]])

    for i,k in enumerate(node_perm):
        color = wheel_cmap((np.pi+pfs_th[k])/(2*np.pi))
        plt.plot(pfs[k,0], pfs[k, 1], 'o',
                 markerfacecolor=color, markeredgecolor=color,
                 markersize=4 + 4 * pf_size[k],
                 alpha=0.7)

    #     plt.gca().add_patch(Circle((0,0), radius=rad, ec='k', fc="none"))
    plt.title("True Place Fields")
    plt.xlim(-45, 45)
    # plt.xlabel("$x$")
    plt.xticks([-40, -20, 0, 20, 40], [])
    plt.ylim(-45, 45)
    # plt.ylabel("$y$")
    plt.yticks([-40, -20, 0, 20, 40], [])

    # Now plot the inferred locations


    plt.subplot(212, aspect='equal')

    for i,k in enumerate(node_perm):
        color = wheel_cmap((np.pi+pfs_th[k])/(2*np.pi))
        plt.plot(Ls_mean[k,0], Ls_mean[k, 1], 'o',
                 markerfacecolor=color, markeredgecolor=color,
                 markersize=4 + 4 * pf_size[k],
                 alpha=0.7)

    #     plt.gca().add_patch(Circle((0,0), radius=rad, ec='k', fc="none"))
    plt.title("Mean Locations")
    plt.xlim(-30, 30)
    plt.xticks([])
    plt.ylim(-30, 30)
    plt.yticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "hipp_mean_locations.pdf"))
    plt.show()


def plot_pca_locations():
    ### Plot the sampled locations for a few neurons

    # Bin the data
    from pyhawkes.utils.utils import convert_continuous_to_discrete
    S_dt = convert_continuous_to_discrete(S, C, 0.25, 0, T)

    # Smooth the data to get a firing rate
    from scipy.ndimage.filters import gaussian_filter1d
    S_smooth = np.array([gaussian_filter1d(s, 4) for s in S_dt.T]).T

    # Run pca to gte an embedding
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(S_smooth)
    Z = pca.components_.T

    # Rotate
    R = compute_optimal_rotation(Z, pfs)
    Z = Z.dot(R)

    fig = create_figure(figsize=(1.4,2.5))
    plt.subplot(211, aspect='equal')

    wheel_cmap = gradient_cmap([colors[0], colors[3], colors[2], colors[1], colors[0]])

    for i,k in enumerate(node_perm):
        color = wheel_cmap((np.pi+pfs_th[k])/(2*np.pi))
        plt.plot(pfs[k,0], pfs[k, 1], 'o',
                 markerfacecolor=color, markeredgecolor=color,
                 markersize=4 + 4 * pf_size[k],
                 alpha=0.7)

    #     plt.gca().add_patch(Circle((0,0), radius=rad, ec='k', fc="none"))
    plt.title("True Place Fields")
    plt.xlim(-45, 45)
    # plt.xlabel("$x$")
    plt.xticks([-40, -20, 0, 20, 40], [])
    plt.ylim(-45, 45)
    # plt.ylabel("$y$")
    plt.yticks([-40, -20, 0, 20, 40], [])

    # Now plot the inferred locations


    plt.subplot(212, aspect='equal')

    for i,k in enumerate(node_perm):
        color = wheel_cmap((np.pi+pfs_th[k])/(2*np.pi))
        plt.plot(Z[k,0], Z[k, 1], 'o',
                 markerfacecolor=color, markeredgecolor=color,
                 markersize=4 + 4 * pf_size[k],
                 alpha=0.7)

    #     plt.gca().add_patch(Circle((0,0), radius=rad, ec='k', fc="none"))
    plt.title("PCA Locations")
    plt.xlim(-25, 25)
    # plt.xlabel("$x$")
    plt.xticks([-20, 0, 20], [])
    plt.ylim(-25, 25)
    # plt.ylabel("$y$")
    plt.yticks([-20, 0, 20], [])

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "hipp_pca_locations.pdf"))
    plt.show()


def plot_mean_locations():
    ### Plot the sampled locations for a few neurons
    _, _, _, _, Ls = result
    Ls_rot = []
    for L in Ls:
        R = compute_optimal_rotation(L, pfs)
        Ls_rot.append(L.dot(R))
    Ls_rot = np.array(Ls_rot)

    Ls_mean = np.mean(Ls_rot, 0)

    fig = create_figure(figsize=(1.4,2.5))
    plt.subplot(211, aspect='equal')

    wheel_cmap = gradient_cmap([colors[0], colors[3], colors[2], colors[1], colors[0]])

    for i,k in enumerate(node_perm):
        color = wheel_cmap((np.pi+pfs_th[k])/(2*np.pi))
        plt.plot(pfs[k,0], pfs[k, 1], 'o',
                 markerfacecolor=color, markeredgecolor=color,
                 markersize=4 + 4 * pf_size[k],
                 alpha=0.7)

    #     plt.gca().add_patch(Circle((0,0), radius=rad, ec='k', fc="none"))
    plt.title("True Place Fields")
    plt.xlim(-45, 45)
    # plt.xlabel("$x$")
    plt.xticks([-40, -20, 0, 20, 40], [])
    plt.ylim(-45, 45)
    # plt.ylabel("$y$")
    plt.yticks([-40, -20, 0, 20, 40], [])

    # Now plot the inferred locations


    plt.subplot(212, aspect='equal')

    for i,k in enumerate(node_perm):
        color = wheel_cmap((np.pi+pfs_th[k])/(2*np.pi))
        plt.plot(Ls_mean[k,0], Ls_mean[k, 1], 'o',
                 markerfacecolor=color, markeredgecolor=color,
                 markersize=4 + 4 * pf_size[k],
                 alpha=0.7)

    #     plt.gca().add_patch(Circle((0,0), radius=rad, ec='k', fc="none"))
    plt.title("Mean Locations")
    plt.xlim(-30, 30)
    plt.xticks([])
    plt.ylim(-30, 30)
    plt.yticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "hipp_mean_locations.pdf"))
    plt.show()


def plot_mean_and_pca_locations(result):
    ### Plot the sampled locations for a few neurons
    _, _, _, _, Ls = result
    Ls_rot = []
    for L in Ls:
        R = compute_optimal_rotation(L, pfs, scale=False)
        Ls_rot.append(L.dot(R))
    Ls_rot = np.array(Ls_rot)
    Ls_mean = np.mean(Ls_rot, 0)

    # Bin the data
    from pyhawkes.utils.utils import convert_continuous_to_discrete
    S_dt = convert_continuous_to_discrete(S, C, 0.25, 0, T)

    # Smooth the data to get a firing rate
    from scipy.ndimage.filters import gaussian_filter1d
    S_smooth = np.array([gaussian_filter1d(s, 4) for s in S_dt.T]).T

    # Run pca to gte an embedding
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(S_smooth)
    Z = pca.components_.T

    # Rotate
    R = compute_optimal_rotation(Z, pfs, scale=False)
    Z = Z.dot(R)

    wheel_cmap = gradient_cmap([colors[0], colors[3], colors[2], colors[1], colors[0]])
    fig = create_figure(figsize=(1.4,2.9))
    # plt.subplot(211, aspect='equal')
    ax = create_axis_at_location(fig, .3, 1.7, 1, 1)


    for i,k in enumerate(node_perm):
        color = wheel_cmap((np.pi+pfs_th[k])/(2*np.pi))
        plt.plot(Ls_mean[k,0], Ls_mean[k, 1], 'o',
                 markerfacecolor=color, markeredgecolor=color,
                 markersize=4 + 4 * pf_size[k],
                 alpha=0.7)

    #     plt.gca().add_patch(Circle((0,0), radius=rad, ec='k', fc="none"))
    plt.title("Mean Locations")
    plt.xlim(-3, 3)
    plt.xticks([-2, 0, 2])
    plt.ylim(-3, 3)
    plt.yticks([-2, 0, 2])


    # plt.subplot(212, aspect='equal')
    ax = create_axis_at_location(fig, .3, .2, 1, 1)

    for i,k in enumerate(node_perm):
        color = wheel_cmap((np.pi+pfs_th[k])/(2*np.pi))
        plt.plot(Z[k,0], Z[k, 1], 'o',
                 markerfacecolor=color, markeredgecolor=color,
                 markersize=4 + 4 * pf_size[k],
                 alpha=0.7)

    #     plt.gca().add_patch(Circle((0,0), radius=rad, ec='k', fc="none"))
    plt.title("PCA Locations")
    plt.xlim(-.5, .5)
    # plt.xlabel("$x$")
    plt.xticks([-.4, 0, .4])
    plt.ylim(-.5, .5)
    # plt.ylabel("$y$")
    plt.yticks([-.4, 0, .4])

    # plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "hipp_mean_pca_locations.pdf"))
    plt.show()


# plot_pred_lls(results)
plot_results(results[-1])
# plot_locations(results[-1], offset=0)
# plot_locations(results[-1], offset=3)
# plot_locations(results[-1], offset=7)
# plot_locations(results[-1], offset=11)
# plot_pca_locations()
# plot_mean_locations()
# plot_mean_and_pca_locations(results[-1])