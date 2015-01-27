import numpy as np
import matplotlib.pyplot as plt

from pyhawkes.models import DiscreteTimeNetworkHawkesModelGibbs, DiscreteTimeStandardHawkesModel
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

    K = 5
    T = 1000
    dt = 1.0
    B = 3

    # Generate from a true model
    true_model = DiscreteTimeNetworkHawkesModelGibbs(K=K, dt=dt, B=B, c=np.zeros(K, dtype=np.int), p=0.0, v=K)
    S,R = true_model.generate(T=T)

    # Plot the true network
    plt.ion()
    plot_network(true_model.weight_model.A,
                 true_model.weight_model.W)
    plt.pause(0.001)


    # Make a new model for inference
    test_model = DiscreteTimeStandardHawkesModel(K=K, dt=dt, B=B, l2_penalty=10, l1_penalty=10)
    test_model.add_data(S)

    # Plot the true and inferred firing rate
    kplt = 0
    plt.figure()
    plt.plot(np.arange(T), R[:,kplt], '-k', lw=2)
    plt.ion()
    ln = plt.plot(np.arange(T), test_model.compute_rate(ks=kplt), '-r')[0]
    plt.show()

    # Gradient descent
    N_steps = 1000
    lls = []
    for itr in xrange(N_steps):
        W,ll,grad = test_model.gradient_descent_step(stepsz=0.001)
        lls.append(ll)

        # Update plot
        if itr % 1 == 0:
            ln.set_data(np.arange(T), test_model.compute_rate(ks=kplt))
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
demo(288408413)
