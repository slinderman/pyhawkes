import cPickle
import os
import numpy as np
import matplotlib.pyplot as plt

from pyhawkes.models import DiscreteTimeNetworkHawkesModelGibbs
from pyhawkes.plotting.plotting import plot_network

def generate_synthetic_data(seed=None):
    """
    Create a discrete time Hawkes model and generate from it.

    :return:
    """
    if seed is None:
        seed = np.random.randint(2**32)

    print "Setting seed to ", seed
    np.random.seed(seed)

    C = 5
    K = 100
    assert K % C == 0, "C must evenly divide K"
    T = 10000
    dt = 1.0
    B = 3

    # Create a true model
    p = 0.5 * np.eye(C) + 0.05 * (1-np.eye(C))
    v = 15.0 * np.eye(C) + 40.0 * (1-np.eye(C))
    # m = 0.5 * np.ones(C)
    c = np.arange(C).repeat(K // C).astype(np.int)
    assert len(c) == K
    true_model = DiscreteTimeNetworkHawkesModelGibbs(C=C, K=K, dt=dt, B=B, c=c, p=p, v=v)

    # Check stability before generating data
    assert true_model.check_stability()

    # Plot the true network
    plt.ion()
    plot_network(true_model.weight_model.A,
                 true_model.weight_model.W)
    plt.pause(0.001)

    # Generate from the model
    S,_ = true_model.generate(T=T, keep=False)

    # Pickle and save the data
    out_dir  = os.path.join('data', "synthetic_K%d_C%d" % (K,C))
    out_name = 'synthetic_K%d_C%d_T%d.pkl' % (K,C,T)
    out_path = os.path.join(out_dir, out_name)
    with open(out_path, 'w') as f:
        print "Saving output to ", out_path
        cPickle.dump((S, true_model), f, protocol=-1)

# demo(2203329564)
generate_synthetic_data()

