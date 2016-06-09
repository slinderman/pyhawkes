import numpy as np
import matplotlib.pyplot as plt

from pyhawkes.models import DiscreteTimeNetworkHawkesModelSpikeAndSlab, DiscreteTimeStandardHawkesModel

def demo(seed=None):
    """
    Create a discrete time Hawkes model and generate from it.

    :return:
    """
    raise NotImplementedError("This example needs to be updated.")

    if seed is None:
        seed = np.random.randint(2**32)

    print "Setting seed to ", seed
    np.random.seed(seed)

    C = 1
    K = 10
    T = 1000
    dt = 1.0
    B = 3

    # Create a true model
    p = 0.8 * np.eye(C)
    v = 10.0 * np.eye(C) + 20.0 * (1-np.eye(C))
    # m = 0.5 * np.ones(C)
    c = (0.0 * (np.arange(K) < 10) + 1.0 * (np.arange(K)  >= 10)).astype(np.int)
    true_model = DiscreteTimeNetworkHawkesModelSpikeAndSlab(C=C, K=K, dt=dt, B=B, c=c, p=p, v=v)

    # Plot the true network
    plt.ion()
    plot_network(true_model.weight_model.A,
                 true_model.weight_model.W,
                 vmax=0.5)
    plt.pause(0.001)

    # Sample from the true model
    S,R = true_model.generate(T=T)


    # Make a new model for inference
    test_model = DiscreteTimeStandardHawkesModel(K=K, dt=dt, B=B, beta=1.0)
    test_model.add_data(S)

    # Plot the true and inferred firing rate
    kplt = 0
    plt.figure()
    plt.plot(np.arange(T), R[:,kplt], '-k', lw=2)
    plt.ion()
    ln = plt.plot(np.arange(T), test_model.compute_rate(ks=kplt), '-r')[0]
    plt.show()

    # Gradient descent
    N_steps = 10000
    lls = []
    for itr in xrange(N_steps):
        W,ll,grad = test_model.gradient_descent_step(stepsz=0.001)
        lls.append(ll)

        # Update plot
        if itr % 5 == 0:
            ln.set_data(np.arange(T), test_model.compute_rate(ks=kplt))
            plt.title("Iteration %d" % itr)
            plt.pause(0.001)

    plt.ioff()

    print "W true:        ", true_model.weight_model.A * true_model.weight_model.W
    print "lambda0 true:  ", true_model.bias_model.lambda0
    print "ll true:       ", true_model.log_likelihood()
    print ""
    print "W test:        ", test_model.W
    print "lambda0 test   ", test_model.bias
    print "ll test:       ", test_model.log_likelihood()


    plt.figure()
    plt.plot(np.arange(N_steps), lls)
    plt.xlabel("Iteration")
    plt.ylabel("Log likelihood")

    plot_network(np.ones((K,K)), test_model.W, vmax=0.5)
    plt.show()


# demo(2203329564)
# demo(1940839255)
# demo(288408413)
# demo(2074381354)
demo()