import cPickle
import os
import gzip
import numpy as np
import matplotlib.pyplot as plt

from pyhawkes.models import DiscreteTimeNetworkHawkesModelSpikeAndSlab
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

    T_test=1000

    # Debugging network:
    # C = 1
    # K = 4
    # T = 1000
    # dt = 1.0
    # B = 3
    # p = 0.5
    # kappa = 3.0
    # v = kappa * 5.0
    # c = np.zeros(K, dtype=np.int)

    # Small network:
    # Seed: 1957629166
    # C = 4
    # K = 20
    # T = 10000
    # dt = 1.0
    # B = 3
    # kappa = 3.0
    # p = 0.9 * np.eye(C) + 0.05 * (1-np.eye(C))
    # v = kappa * (5.0 * np.eye(C) + 25.0 * (1-np.eye(C)))
    # c = np.arange(C).repeat((K // C))

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
    # c = np.arange(C).repeat((K // C))

    # Medium netowrk 2:
    # Seed = 3848328624
    # C = 5
    # K = 50
    # T = 100000
    # dt = 1.0
    # B = 3
    # kappa = 2.0
    # c = np.arange(C).repeat((K // C))
    # p = 0.4 * np.eye(C) + 0.01 * (1-np.eye(C))
    # v = kappa * (5 * np.eye(C) + 5.0 * (1-np.eye(C)))

    # Medium netowrk, one cluster
    # Seed: 3848328624
    C = 1
    K = 50
    T = 100000
    dt = 1.0
    B = 3
    p = 0.08
    kappa = 3.0
    v = kappa * 5.0
    c = np.zeros(K, dtype=np.int)

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
    # c = np.arange(C).repeat((K // C))

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
    # c = np.arange(C).repeat((K // C))

    # Extra large network:
    # Seed: 2327447870
    # C = 20
    # K = 1000
    # T = 100000
    # dt = 1.0
    # B = 3
    # kappa = 3.0
    # p = 0.25 * np.eye(C) + 0.0025 * (1-np.eye(C))
    # v = kappa * (15 * np.eye(C) + 30.0 * (1-np.eye(C)))
    # c = np.arange(C).repeat((K // C))


    # Create the model with these parameters
    network_hypers = {'C': C, 'kappa': kappa, 'c': c, 'p': p, 'v': v}

    # Create a simple network
    from pyhawkes.internals.network import ErdosRenyiFixedSparsity
    network = ErdosRenyiFixedSparsity(K, p, kappa, v=v)

    true_model = DiscreteTimeNetworkHawkesModelSpikeAndSlab(K=K, dt=dt, B=B,
                                                            network=network)

    assert true_model.check_stability()

    # Plot the true network
    plt.ion()
    plot_network(true_model.weight_model.A,
                 true_model.weight_model.W)
    plt.pause(0.001)

    # Sample from the true model
    S,R = true_model.generate(T=T, keep=False, print_interval=50)

    # Pickle and save the data
    out_dir  = os.path.join('data', "synthetic")
    out_name = 'synthetic_K%d_C%d_T%d.pkl.gz' % (K,C,T)
    out_path = os.path.join(out_dir, out_name)
    with gzip.open(out_path, 'w') as f:
        print "Saving output to ", out_path
        cPickle.dump((S, true_model), f, protocol=-1)

    # Sample test data
    S_test,_ = true_model.generate(T=T_test, keep=False)

    # Pickle and save the data
    out_dir  = os.path.join('data', "synthetic")
    out_name = 'synthetic_test_K%d_C%d_T%d.pkl.gz' % (K,C,T_test)
    out_path = os.path.join(out_dir, out_name)
    with gzip.open(out_path, 'w') as f:
        print "Saving output to ", out_path
        cPickle.dump((S_test, true_model), f, protocol=-1)


# demo(2203329564)
generate_synthetic_data(3848328624)

