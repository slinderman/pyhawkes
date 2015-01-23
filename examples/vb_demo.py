import numpy as np
import matplotlib.pyplot as plt

from pyhawkes.models import DiscreteTimeNetworkHawkesModelMeanField, DiscreteTimeNetworkHawkesModelGibbs

def demo():
    """
    Create a discrete time Hawkes model and generate from it.

    :return:
    """
    K = 2
    T = 1000
    dt = 1.0
    B = 3

    # Generate from a true model
    true_model = DiscreteTimeNetworkHawkesModelGibbs(K=K, dt=dt, B=B, p=0.5)
    # true_model.resample_from_mf()
    S,R = true_model.generate(T=T)

    # Make a new model for inference
    model = DiscreteTimeNetworkHawkesModelMeanField(K=K, dt=dt, B=B, p=0.5)
    model.resample_from_mf()
    model.add_data(S)

    # Plot the true and inferred firing rate
    plt.figure()
    plt.plot(np.arange(T), R[:,0], '-k', lw=2)
    plt.ion()
    ln = plt.plot(np.arange(T), model.compute_rate()[:,0], '-r')[0]
    plt.show()

    # Gibbs sample
    N_iters = 100
    vlbs = []
    for itr in xrange(N_iters):
        vlbs.append(model.meanfield_coordinate_descent_step())
        print "VLB: ", vlbs[-1]

        # Resample from variational distribution and plot
        model.resample_from_mf()

        # Update plot
        if itr % 5 == 0:
            ln.set_data(np.arange(T), model.compute_rate()[:,0])
            plt.title("Iteration %d" % itr)
            plt.pause(0.001)

    plt.ioff()

    print "A true:        ", true_model.weight_model.A
    print "W true:        ", true_model.weight_model.W
    print "g true:         ", true_model.impulse_model.g
    print "lambda0 true:  ", true_model.bias_model.lambda0
    print ""
    print "A mean:        ", model.weight_model.expected_A()
    print "W mean:        ", model.weight_model.expected_W()
    print "g mean:        ", model.impulse_model.expected_g()
    print "lambda0 mean:  ", model.bias_model.expected_lambda0()

    plt.figure()
    plt.plot(np.arange(N_iters), vlbs)
    plt.xlabel("Iteration")
    plt.ylabel("VLB")
    plt.show()

demo()
