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

    # Create a true model
    # Larger v (weight scale) implies smaller weights

    # Small network:
    # Seed: 1957629166
    # C = 4
    # K = 20
    # T = 100000
    # dt = 1.0
    # B = 3
    # kappa = 3.0
    # p = 0.9 * np.eye(C) + 0.05 * (1-np.eye(C))
    # v = kappa * (5.0 * np.eye(C) + 25.0 * (1-np.eye(C)))

    # Medium network:
    # Seed: 2723361959
    # C = 5
    # K = 50
    # T = 100000
    # dt = 1.0
    # B = 3
    # kappa = 3.0
    # p = 0.75 * np.eye(C) + 0.05 * (1-np.eye(C))
    # v = kappa * (9 * np.eye(C) + 25.0 * (1-np.eye(C)))

    # Large network:
    # Seed = 2467634490
    # C = 5
    # K = 100
    # T = 100000
    # dt = 1.0
    # B = 3
    # kappa = 3.0
    # p = 0.4 * np.eye(C) + 0.025 * (1-np.eye(C))
    # v = kappa * (10 * np.eye(C) + 25.0 * (1-np.eye(C)))

    # Large network 2:
    # Seed =
    # C = 10
    # K = 100
    # T = 100000
    # dt = 1.0
    # B = 3
    # kappa = 3.0
    # p = 0.75 * np.eye(C) + 0.05 * (1-np.eye(C))
    # v = kappa * (9 * np.eye(C) + 25.0 * (1-np.eye(C)))

    # Extra large network:
    # Seed: 2327447870
    C = 20
    K = 1000
    T = 100000
    dt = 1.0
    B = 3
    kappa = 3.0
    p = 0.25 * np.eye(C) + 0.0025 * (1-np.eye(C))
    v = kappa * (15 * np.eye(C) + 30.0 * (1-np.eye(C)))

    # Create the model with these parameters
    assert K % C == 0
    c = np.arange(C).repeat((K // C))
    true_model = DiscreteTimeNetworkHawkesModelGibbs(C=C, K=K, dt=dt, B=B, kappa=kappa, c=c, p=p, v=v)
    assert true_model.check_stability()

    # Plot the true network
    plt.ion()
    plot_network(true_model.weight_model.A,
                 true_model.weight_model.W)
    plt.pause(0.001)

    # Sample from the true model
    S,R = true_model.generate(T=T, keep=False, print_interval=100)

    # Pickle and save the data
    out_dir  = os.path.join('data', "synthetic")
    out_name = 'synthetic_K%d_C%d_T%d.pkl' % (K,C,T)
    out_path = os.path.join(out_dir, out_name)
    with open(out_path, 'w') as f:
        print "Saving output to ", out_path
        cPickle.dump((S, true_model), f, protocol=-1)

# demo(2203329564)
generate_synthetic_data(2327447870)

