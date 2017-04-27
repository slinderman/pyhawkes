import numpy as np
import matplotlib.pyplot as plt

from pyhawkes.models import DiscreteTimeNetworkHawkesModelSpikeAndSlab, DiscreteTimeStandardHawkesModel
from pyhawkes.utils.basis import IdentityBasis

def sample_from_network_hawkes(K, T, dt, dt_max, B):
    # Create a true model
    true_model = DiscreteTimeNetworkHawkesModelSpikeAndSlab(K=K, dt=dt, dt_max=dt_max, B=B,
                                                            network_hypers=dict(p=0.1))

    # Plot the true network
    plt.ion()
    true_model.plot_network()

    # Sample from the true model
    S,R = true_model.generate(T=T)

    # Return the spike count matrix
    return S, true_model

def demo(seed=None):
    """
    Create a discrete time Hawkes model and generate from it.

    :return:
    """
    if seed is None:
        seed = np.random.randint(2**32)

    print("Setting seed to ", seed)
    np.random.seed(seed)

    K = 5       # Number of nodes
    T = 10000     # Number of time bins to simulate
    dt = 1       # Time bin size
    dt_max = 50  # Impulse response length
    B = 1        # Number of basis functions

    # Sample from a sparse network Hawkes model
    S, true_model = sample_from_network_hawkes(K, T, dt, dt_max, B)

    # Make a new model for inference
    # test_basis = IdentityBasis(dt, dt_max, allow_instantaneous=False)
    test_basis = true_model.basis
    test_model = DiscreteTimeStandardHawkesModel(K=K, dt=dt, dt_max=dt_max+dt,
                                                 beta=1.0,
                                                 basis=test_basis,
                                                 allow_self_connections=True)
    test_model.add_data(S)

    # DEBUG: Initialize with the true parameters of the network Hawkes model
    # test_model.initialize_with_gibbs_model(true_model)

    test_model.fit_with_bfgs()

    print("lambda0 true:  ", true_model.bias_model.lambda0)
    print("lambda0 test   ", test_model.bias)

    print("")
    print("W true:        ", true_model.weight_model.A * true_model.weight_model.W)
    print("W test:        ", test_model.W)

    print("")
    print("ll true:       ", true_model.log_likelihood())
    print("ll test:       ", test_model.log_likelihood())

    # test_model.plot_network()

    # Plot the rates
    plt.figure()
    for k in range(3):
        plt.subplot(3,1,k+1)
        plt.plot(np.arange(T) * dt, true_model.compute_rate(proc=k), '-b')
        plt.plot(np.arange(T) * dt, test_model.compute_rate(ks=k), '-r')
        lim = plt.ylim()
        plt.ylim(0, 1.25*lim[1])

    plt.ioff()
    plt.show()

demo(11223344)