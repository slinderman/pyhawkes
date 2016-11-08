import signal
import numpy as np
import importlib
np.random.seed(123)
np.seterr(over="raise")
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    # sns.set_style("white")
    sns.set_context("talk")

    color_names = ["windows blue",
                   "red",
                   "amber",
                   "faded green",
                   "dusty purple",
                   "crimson",
                   "greyish"]
    colors = sns.xkcd_palette(color_names)
except:
    colors = ['b', 'r', 'y', 'g']

from pybasicbayes.util.text import progprint_xrange

import pyhawkes.models
importlib.reload(pyhawkes.models)
from pyhawkes.continuous_models import SpatioTemporalHawkesModel

# Parameters
K_obs = 3           # Number of observed nodes
H = 1               # Number of hidden nodes
b_obs = 0.05        # Background rate of observed nodes
b_hid = 0.1         # Background rate of hiddennodes
dt_max = 5.         # Duration of temporal impulse response
g_sigma = 0.5       # Std dev of spatial impulse response
T = 100.            # Length of simulation
w_hid = 2.0         # Influence of hidden nodes on observed nodes

# Initialize model without connections from obs to hidden
true_model = SpatioTemporalHawkesModel(K_obs+H, dt_max=dt_max)
true_model.bias_model.lambda0[:-1] = b_obs
true_model.bias_model.lambda0[-1] = b_hid
true_model.weight_model.A[:,:] = False
true_model.weight_model.A[-1,:-1] = True
true_model.weight_model.W *= 0
true_model.weight_model.W[-1,:-1] = w_hid
true_model.impulse_model.sigma[:,:] = g_sigma
assert true_model.check_stability()

# Sample from the model
S, X, C = true_model.generate(T)
true_ll = true_model.log_likelihood()
print("Sampled dataset with ", len(S), "events")
print("True Log likelihood: ", true_ll)

# Treat the last H processes as latent
i_latent = np.where(C >= K_obs)[0]
S_hid = S[i_latent]
C_hid = C[i_latent]
X_hid = X[i_latent]
S_obs = np.delete(S, i_latent)
C_obs = np.delete(C, i_latent)
X_obs = np.delete(X, i_latent)
assert np.all(C_obs <= K_obs)

# Create a latent Hawkes model for inferring the latent events
latent_model = SpatioTemporalHawkesModel(K_obs, H, dt_max=dt_max)

# Test: Give it the right parameters -- just learn the hidden events
latent_model.bias_model.lambda0 = true_model.bias_model.lambda0.copy()
latent_model.weight_model.A = true_model.weight_model.A.copy()
latent_model.weight_model.W = true_model.weight_model.W.copy()
latent_model.impulse_model.mu = true_model.impulse_model.mu.copy()
latent_model.impulse_model.tau = true_model.impulse_model.tau.copy()
latent_model.impulse_model.sigma = true_model.impulse_model.sigma.copy()

latent_model.add_data(S_obs, C_obs, T, X=X_obs)
assert np.all(np.isfinite(latent_model.data_list[0].X))

# Interactive plotting
plt.ion()
fig = plt.figure(figsize=(12, 8))

Nmax = 100
for k in range(H):
    ax = plt.subplot(K_obs + H, 1, k + 1)
    # Plot the true event times/locations
    Sk, Xk = S[C == K_obs + k], X[C == K_obs + k]
    plt.plot(Sk, Xk, 'ko')

    # Plot up to Nmax inferred hidden points
    S_latent = latent_model.data_list[0].S[latent_model.data_list[0].C >= K_obs]
    X_latent = latent_model.data_list[0].X[latent_model.data_list[0].C >= K_obs]
    N_latent = min(Nmax, len(S_latent))

    # Plot the hidden nodes
    ss = np.zeros(Nmax)
    ss[:N_latent] = S_latent[:N_latent]
    xx = -10 * np.ones(Nmax)
    xx[:N_latent] = X_latent[:N_latent]
    pts = plt.plot(ss, xx, 'o', color=colors[0], markersize=8)[0]

    plt.xlim(0,T)
    ax.set_xticklabels([])
    plt.ylabel("$x$")
    plt.ylim(-3,3)
    plt.yticks([-3, 0, 3])
    plt.title("Hidden Node {0}".format(k+1))

for k in range(K_obs):
    ax = plt.subplot(K_obs + H, 1, H + k + 1)
    # Plot the points
    Sk, Xk = S[C == k], X[C == k]
    plt.plot(Sk, Xk, 'ko')


    if k < K_obs-1:
        ax.set_xticklabels([])
    else:
        plt.xlabel("$t$")
    plt.xlim(0, T)
    plt.ylabel("$x$")
    plt.ylim(-3, 3)
    plt.yticks([-3, 0, 3])
    plt.title("Observed Node {0}".format(k+1))

h_title = fig.suptitle("Iteration 0")
plt.pause(0.001)

# Fit the latent Hawkes model
ctrlc_pressed = [False]
def ctrlc_handler(signal, frame):
    print("Halting due to Ctrl-C")
    ctrlc_pressed[0] = True
signal.signal(signal.SIGINT, ctrlc_handler)

def evaluate(model):
    ll = model.log_likelihood()
    data = model.data_list[0]
    S_latent = data.S[data.C >= K_obs]
    X_latent = data.X[data.C >= K_obs]
    return ll, S_latent, X_latent

plot_intvl = 10
def update_plot(model, itr):
    if itr % plot_intvl != 0:
        return

    data = model.data_list[0]
    S_latent = data.S[data.C >= K_obs]
    X_latent = data.X[data.C >= K_obs]
    N_latent = min(Nmax, len(S_latent))

    # Plot the hidden nodes
    ss = np.zeros(Nmax)
    ss[:N_latent] = S_latent[:N_latent]
    xx = -10 * np.ones(Nmax)
    xx[:N_latent] = X_latent[:N_latent]
    pts.set_data(ss, xx)

    h_title.set_text("Iteration {}".format(itr))
    plt.pause(0.001)

def update(model, itr):
    model.data_list[0].resample_latent_events_mh()
    # model.resample_model()
    update_plot(model, itr)
    return evaluate(model)

# Run the Gibbs sampler
input("Press any key to start sampler...\n")
N_samples = 10000
samples = [evaluate(latent_model)]
for itr in progprint_xrange(N_samples):
    samples.append(update(latent_model, itr))
    if ctrlc_pressed[0]:
        break

# Analyze the samples
N_samples = len(samples)
lls = np.array([s[0] for s in samples])
Ss = [s[1] for s in samples]
Xs = [s[2] for s in samples]
N_hidden = [len(s) for s in Ss]

# Find posterior of hidden event counts in discrete bins
tbins = np.linspace(0,T,T/4.0)
xbins = np.linspace(-3,3,11)
tcenters = 0.5 * (tbins[:-1] + tbins[1:])
discretize = lambda s, x: np.histogram2d(s, x, (tbins, xbins))[0]
counts = np.array([discretize(s,x) for s,x in zip(Ss, Xs)])
hidden_rate = np.mean(counts, axis=0)
hidden_rate_std = np.std(counts, axis=0)

### Plotting
plt.figure()
plt.plot([0, N_samples], len(S_hid) * np.ones(2), '--k', label="true")
plt.plot(N_hidden, 'r', label="inferred")
plt.xlabel("Iteration")
plt.ylabel("Num Hidden Events")

plt.figure()
plt.plot([0, N_samples], true_ll * np.ones(2), '--k', label="true")
plt.plot(lls, 'r', label="inferred")
plt.xlabel("Iteration")
plt.ylabel("Log Likelihood")

plt.figure()
plt.imshow(hidden_rate.T, cmap="Blues", vmin=0, vmax=1.0, aspect="auto", extent=(0,T,3,-3))
plt.gca().invert_yaxis()
plt.plot(S_hid, X_hid, 'ko')
plt.ylabel("location $x$")
plt.ylabel("time $t$")
plt.colorbar(label="Probability")
plt.show()
