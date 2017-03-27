"""
Perform some basic analyses on the preprocessed connectomics data.
"""
import pickle
import os
import gzip
import numpy as np

from pyhawkes.internals.network import StochasticBlockModel

import matplotlib.pyplot as plt

data_path = os.path.join("data", "chalearn", "small", "network6_oopsi.pkl.gz")

with gzip.open(data_path, 'r') as f:
    P, F, Cf, network, pos = pickle.load(f)
    S_full = (P > 0.1).astype(np.int)

sbm = StochasticBlockModel(K=100, C=5)

plt.ion()
plt.figure()
im = plt.imshow(network, interpolation="none", cmap="Greys")
plt.show()

N_iter = 500
for itr in range(N_iter):
    if itr % 10 == 0:
        print("Iteration ", itr)
        print(sbm.p)
    sbm.resample(data=(network, network))

    c = np.argsort(sbm.c)
    im.set_data(network[np.ix_(c,c)])
    plt.pause(0.001)

plt.ioff()
