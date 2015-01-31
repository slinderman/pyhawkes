import numpy as np
import matplotlib.pyplot as plt

from pyhawkes.models import _DiscreteTimeNetworkHawkesModelGammaMixtureMF, DiscreteTimeNetworkHawkesModelGibbs
from pyhawkes.plotting.plotting import plot_network

def demo(seed=None):
    """
    Create a discrete time Hawkes model and generate from it.

    :return:
    """
    if seed is None:
        seed = np.random.randint(2**32)

    print "Setting seed to ", seed
    np.random.seed(seed)

    C = 2
    K = 20
    T = 10000
    dt = 1.0
    B = 3

    # Create a true model
    p = 0.8 * np.eye(C)
    v = 10.0 * np.eye(C) + 20.0 * (1-np.eye(C))
    # m = 0.5 * np.ones(C)
    c = (0.0 * (np.arange(K) < 10) + 1.0 * (np.arange(K)  >= 10)).astype(np.int)
    true_model = DiscreteTimeNetworkHawkesModelGibbs(C=C, K=K, dt=dt, B=B, c=c, p=p, v=v)
    c = true_model.network.c
    perm = np.argsort(c)

    # Plot the true network
    plt.ion()
    plot_network(true_model.weight_model.A[np.ix_(perm, perm)],
                 true_model.weight_model.W[np.ix_(perm, perm)])
    plt.pause(0.001)

    # Sample from the true model
    S,R = true_model.generate(T=T)


    # Make a new model for inference
    test_model = _DiscreteTimeNetworkHawkesModelGammaMixtureMF(C=C, K=K, dt=dt, B=B, tau1=1, tau0=5, beta=1.0/5.0)
    test_model.resample_from_mf()
    test_model.add_data(S)

    plt.figure(2)
    im1 = plt.imshow(test_model.weight_model.A * test_model.weight_model.W,
                     vmin=0.0, vmax=0.6,
                     interpolation="none", cmap="gray")

    # Plot the true and inferred firing rate
    plt.figure(3)
    plt.plot(np.arange(T), R[:,1], '-k', lw=2)
    ln = plt.plot(np.arange(T), test_model.compute_rate()[:,1], '-r')[0]

    # Plot the block probabilities
    plt.figure(4)
    im2 = plt.imshow(test_model.network.mf_m[perm,:],
                    interpolation="none", cmap="gray",
                    aspect=float(C)/K)
    plt.show()
    plt.pause(0.001)

    # VB coordinate descent
    N_iters = 1000
    vlbs = []
    for itr in xrange(N_iters):
        vlbs.append(test_model.meanfield_coordinate_descent_step())

        if itr > 0:
            if (vlbs[-2] - vlbs[-1]) > 1e-1:
                import pdb; pdb.set_trace()
                raise Exception("VLB is not increasing!")

        # Resample from variational distribution and plot
        test_model.resample_from_mf()

        # Update plot
        if itr % 5 == 0:
            plt.figure(2)
            im1.set_data(test_model.weight_model.A * test_model.weight_model.W)
            plt.title("Iteration %d" % itr)
            plt.pause(0.001)

            plt.figure(3)
            ln.set_data(np.arange(T), test_model.compute_rate()[:,1])
            plt.title("Iteration %d" % itr)
            plt.pause(0.001)

            plt.figure(4)
            im2.set_data(test_model.network.mf_m[perm,:])
            plt.title("Iteration %d" % itr)
            plt.pause(0.001)

    plt.ioff()

    print "A true:        ", true_model.weight_model.A
    print "W true:        ", true_model.weight_model.W
    print "g true:         ", true_model.impulse_model.g
    print "lambda0 true:  ", true_model.bias_model.lambda0
    print ""
    print "A mean:        ", test_model.weight_model.expected_A()
    print "W mean:        ", test_model.weight_model.expected_W()
    print "g mean:        ", test_model.impulse_model.expected_g()
    print "lambda0 mean:  ", test_model.bias_model.expected_lambda0()

    plt.figure()
    plt.plot(np.arange(N_iters), vlbs)
    plt.xlabel("Iteration")
    plt.ylabel("VLB")
    plt.show()

demo()
