"""
Perform some basic analyses on the preprocessed connectomics data.
"""
import pickle
import os
import gzip
import pprint
import numpy as np
from scipy.special import logsumexp
from scipy.special import gammaln
import matplotlib.pyplot as plt

from pyhawkes.models import DiscreteTimeStandardHawkesModel, \
    DiscreteTimeNetworkHawkesModelGammaMixture
from pyhawkes.plotting.plotting import plot_network

from baselines.xcorr import infer_net_from_xcorr

from sklearn.metrics import roc_auc_score

data_path = os.path.join("data", "chalearn", "small", "network1_oopsi.pkl.gz")

with gzip.open(data_path, 'r') as f:
    P, F, Cf, network, pos = pickle.load(f)
    S_full = (P > 0.1).astype(np.int)

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

print("Num conns: ", network.sum())
print("Sparsity: ", float(network.sum()) / network.size)


# Compute the cross correlation to estimate the connectivity
print("Estimating network via cross correlation")
# W_xcorr = infer_net_from_xcorr(F[:10000,:], dtmax=3)

# Compute the cross correlation to estimate the connectivity
# print "Estimating network via cross correlation"
# W_xcorr = infer_net_from_xcorr(S[:10000], dtmax=dt_max // dt)
#
# # # Look at the cross correlation for edges vs non edges
# plt.figure()
# plt.hist(W_xcorr.ravel(), bins=50, color='r', alpha=0.5, normed=True, label="all")
# plt.hist(W_xcorr[network>0], bins=50, color='b', alpha=0.5, normed=True, label="edges")
# plt.hist(W_xcorr[network<1], bins=50, color='g', alpha=0.5, normed=True, label="nonedges")
# plt.legend()
# plt.title('Cross correlation distribution')
# plt.show()
#
# # TODO: Reprocess the data and condition on the total fluorescence level exceeding
# # TODO: some threshold. This should improve the xcorr score
#
#
# # Plot the number of spikes per neuron
# plt.figure()
# plt.bar(np.arange(1,101), S_full.sum(0), color='r', alpha=0.5)
# plt.title('Spike counts')
# plt.show()
#
#
# # Look at times when the entire network spikes. What do the fluorescence traces look like
# # It seems that there are times when the entire network fires simultaneously
# # all_spiking = np.where(S.sum(axis=1) == K)[0]
# all_spiking = [30000]
# window = 50 * 30
# inds = np.unique(np.random.choice(K, size=10))
# plt.figure()
# for i,ind in enumerate(inds):
#     plt.subplot(len(inds),1,i+1)
#     plt.plot(F[all_spiking[0]-5*window:all_spiking[0]+5*window, ind])
#     spks  = np.where(S[all_spiking[0]-5*window:all_spiking[0]+5*window, ind])
#     plt.plot(spks, F[spks,ind], 'ro')
#     plt.xlim(4*window, 6*window)
# plt.show()


# Look at the fluorescence values at time of spike
plt.figure()
plt.hist(Cf[S_full > 0], bins=100)
plt.show()