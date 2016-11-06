
import numpy as np
import importlib
np.random.seed(1111)
np.seterr(over="raise")
import matplotlib.pyplot as plt

from pybasicbayes.util.general import ibincount
from pybasicbayes.util.text import progprint_xrange

import pyhawkes.models
importlib.reload(pyhawkes.models)
from pyhawkes.models import ContinuousTimeNetworkHawkesModel, ContinuousTimeLatentHawkesModel

# Create the model with these parameters
K_obs = 3
H = 1
dt = 1
dt_max = 2.
T = 100.

# Initialize network without connections from obs to hidden
p = np.ones((K_obs+H, K_obs+H))
p[:, -H:] = 0
network_hypers = {'kappa': 1., 'p': p, 'v': 10.}

# Sample from a continuous time model with all nodes
true_model = ContinuousTimeNetworkHawkesModel(
    K_obs+H, dt_max=dt_max, network_hypers=network_hypers)
assert true_model.check_stability()
true_model.bias_model.lambda0[-1] = 0.1
true_model.weight_model.W[-1,:] = 2.0

S, C = true_model.generate(T)
true_ll = true_model.log_likelihood()
print("Sampled dataset with ", len(S), "events")
print("True Log likelihood: ", true_ll)

# Treat the last H processes as latent
i_latent = np.where(C >= K_obs)[0]
S_hid = S[i_latent]
C_hid = C[i_latent]
S_obs = np.delete(S, i_latent)
C_obs = np.delete(C, i_latent)
assert np.all(C_obs <= K_obs)

# Create a latent Hawkes model for inferring the latent events
latent_model = ContinuousTimeLatentHawkesModel(
    K_obs, H, dt_max=dt_max, network_hypers=network_hypers)
latent_model.add_data(S_obs, C_obs, T)

# Hard code parameters
latent_model.bias_model.lambda0 = true_model.bias_model.lambda0.copy()
latent_model.weight_model.A = true_model.weight_model.A.copy()
latent_model.weight_model.W = true_model.weight_model.W.copy()
latent_model.impulse_model.mu = true_model.impulse_model.mu.copy()
latent_model.impulse_model.tau = true_model.impulse_model.tau.copy()

# DEBUG: Start with the right set of latent events
import copy
latent_model.data_list[0] = copy.deepcopy(true_model.data_list[0])
print("Initial heldout log lkhd: ", latent_model.heldout_log_likelihood(S_obs, C_obs, T))
# print("Initial latent log lkhd:  ", latent_model.log_likelihood())

# Fit the latent Hawkes model
def evaluate(model):
    ll = model.log_likelihood()
    data = model.data_list[0]
    S_latent = data.S[data.C >= K_obs]
    return ll, S_latent

def update(model):
    model.resample_latent_events_mh()
    return evaluate(model)

N_samples = 10000
init_sample = evaluate(latent_model)
samples = [init_sample] + [update(latent_model) for _ in progprint_xrange(N_samples)]

# Analyze the samples
lls = np.array([s[0] for s in samples])
Ss = [s[1] for s in samples]
N_hidden = [len(s) for s in Ss]

# Find posterior of hidden event counts in discrete bins
bins = np.arange(0,T,step=dt)
centers = 0.5 * (bins[:-1] + bins[1:])
discretize = lambda s: np.histogram(s, bins)[0]
Ss_dt = np.array([discretize(s) for s in Ss])
hidden_rate = np.mean(Ss_dt, axis=0)
hidden_rate_std = np.std(Ss_dt, axis=0)

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

from hips.plotting.sausage import sausage_plot
plt.figure()
sausage_plot(centers, hidden_rate, 2*hidden_rate_std, color='r', alpha=0.5)
plt.plot(centers, hidden_rate, '-r', lw=2 )
for s in S_hid:
    plt.plot([s,s], [0,1], '-k')
    plt.plot([s], [1], 'ko', markerfacecolor='k')
plt.plot([0,T], [0,0], ':k')

plt.show()
