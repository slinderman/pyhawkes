import numpy as np
import matplotlib.pyplot as plt

from pyhawkes.models import DiscreteTimeNetworkHawkesModelSpikeAndSlab, DiscreteTimeStandardHawkesModel
from pyhawkes.plotting.plotting import plot_network


def sample_from_network_hawkes(C, K, T, dt, B):
    # Create a true model
    p = 0.8 * np.eye(C)
    v = 10.0 * np.eye(C) + 20.0 * (1-np.eye(C))
    c = (0.0 * (np.arange(K) < 10) + 1.0 * (np.arange(K)  >= 10)).astype(np.int)
    true_model = DiscreteTimeNetworkHawkesModelSpikeAndSlab(C=C, K=K, dt=dt, B=B, c=c, p=p, v=v)

    # Plot the true network
    plt.ion()
    plot_network(true_model.weight_model.A,
                 true_model.weight_model.W,
                 vmax=0.5)

    # Sample from the true model
    S,R = true_model.generate(T=T)

    # Return the spike count matrix
    return S, R, true_model

def demo(seed=None):
    """
    Suppose we have a very long recording such that computing gradients of
    the log likelihood is quite expensive. Here we explore the use of
    stochastic gradient descent to fit the standard Hawkes model, which has
    a convex log likelihood. We first initialize the parameters using BFGS
    on a manageable subset of the data. Then we use SGD to refine the parameters
    on the entire dataset.

    :return:
    """
    if seed is None:
        seed = np.random.randint(2**32)

    print "Setting seed to ", seed
    np.random.seed(seed)

    C = 1       # Number of clusters in the true data
    K = 10      # Number of nodes
    T = 10000   # Number of time bins to simulate
    dt = 1.0    # Time bin size
    B = 3       # Number of basis functions

    # Sample from the network Hawkes model
    S, R, true_model = sample_from_network_hawkes(C, K, T, dt, B)

    # Make a model to initialize the parameters
    init_len   = 256
    init_model = DiscreteTimeStandardHawkesModel(K=K, dt=dt, B=B, beta=1.0)
    init_model.add_data(S[:init_len, :])

    print "Initializing with BFGS on first ", init_len, " time bins."
    init_model.fit_with_bfgs()

    # Make another model for inference
    test_model = DiscreteTimeStandardHawkesModel(K=K, dt=dt, B=B, beta=1.0)
    # Initialize with the BFGS parameters
    test_model.weights = init_model.weights
    # Add the data in minibatches
    test_model.add_data(S, minibatchsize=256)

    # Plot the true and inferred firing rate
    kplt = 0
    plt.figure()
    plt.plot(np.arange(256), R[:256,kplt], '-k', lw=2)
    plt.ion()
    ln = plt.plot(np.arange(256), test_model.compute_rate(ks=kplt)[:256], '-r')[0]
    plt.show()

    # Gradient descent
    N_steps = 10000
    lls = []
    learning_rate = 0.01 * np.ones(N_steps)
    momentum = 0.8 * np.ones(N_steps)
    prev_velocity = None
    for itr in xrange(N_steps):
        W,ll,prev_velocity = test_model.sgd_step(prev_velocity, learning_rate[itr], momentum[itr])
        lls.append(ll)

        # Update plot
        if itr % 5 == 0:
            ln.set_data(np.arange(256), test_model.compute_rate(ks=kplt))
            plt.title("Iteration %d" % itr)
            plt.pause(0.001)

    plt.ioff()

    print "W true:        ", true_model.weight_model.A * true_model.weight_model.W
    print "lambda0 true:  ", true_model.bias_model.lambda0
    print ""
    print "W test:        ", test_model.W
    print "lambda0 test   ", test_model.bias

    plt.figure()
    plt.plot(np.arange(N_steps), lls)
    plt.xlabel("Iteration")
    plt.ylabel("Log likelihood")

    plot_network(np.ones((K,K)), test_model.W)
    plt.show()


# demo(2203329564)
# demo(1940839255)
# demo(288408413)
# demo(2074381354)
demo()