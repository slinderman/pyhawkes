import numpy as np
import matplotlib.pyplot as plt

from pyhawkes.models import DiscreteTimeNetworkHawkesModel

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
    true_model = DiscreteTimeNetworkHawkesModel(K=K, dt=dt, B=B)
    S,R = true_model.generate(T=T)

    # Make a new model for inference
    model = DiscreteTimeNetworkHawkesModel(K=K, dt=dt, B=B)
    model.add_data(S)

    # Plot the true and inferred firing rate
    plt.figure()
    plt.plot(np.arange(T), R[:,0], '-k', lw=2)
    plt.ion()
    ln = plt.plot(np.arange(T), model.compute_rate()[:,0], '-r')[0]
    plt.show()

    # Gibbs sample
    N_samples = 1000
    samples = []
    lps = []
    for itr in xrange(N_samples):
        lps.append(model.log_probability())
        samples.append(model.resample_and_copy())

        # print "Iteration ", itr
        # print "Lambda0:     ", model.bias_model.lambda0
        # print "W:           ", model.weight_model.W
        # print ""

        # Update plot
        if itr % 5 == 0:
            ln.set_data(np.arange(T), model.compute_rate()[:,0])
            plt.title("Iteration %d" % itr)
            plt.pause(0.001)

    plt.ioff()

    # Compute sample statistics for second half of samples
    A_samples       = np.array([A for A,_,_,_ in samples])
    W_samples       = np.array([W for _,W,_,_ in samples])
    beta_samples    = np.array([b for _,_,b,_ in samples])
    lambda0_samples = np.array([l for _,_,_,l in samples])
    lps             = np.array(lps)

    offset = N_samples // 2
    A_mean       = A_samples[offset:, ...].mean(axis=0)
    W_mean       = W_samples[offset:, ...].mean(axis=0)
    beta_mean    = beta_samples[offset:, ...].mean(axis=0)
    lambda0_mean = lambda0_samples[offset:, ...].mean(axis=0)

    print "A true:        ", true_model.weight_model.A
    print "W true:        ", true_model.weight_model.W
    print "beta true:     ", true_model.impulse_model.g
    print "lambda0 true:  ", true_model.bias_model.lambda0
    print ""
    print "A mean:        ", A_mean
    print "W mean:        ", W_mean
    print "beta mean:     ", beta_mean
    print "lambda0 mean:  ", lambda0_mean

    plt.figure()
    plt.plot(np.arange(N_samples), lps)
    plt.xlabel("Iteration")
    plt.ylabel("Log probability")
    plt.show()

demo()
