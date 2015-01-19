import numpy as np
import matplotlib.pyplot as plt

from pyhawkes.models import DiscreteTimeNetworkHawkesModel

def demo():
    """
    Create a discrete time Hawkes model and generate from it.

    :return:
    """
    T = 100
    dt = 1.0
    model = DiscreteTimeNetworkHawkesModel(K=2, dt=dt)
    S,R = model.generate(T=T)

    print "Expected number of events: ", np.trapz(R, dt * np.arange(T), axis=0)
    print "Actual number of events:   ", S.sum(axis=0)

    print "Lambda0:  ", model.bias_model.lambda0
    print "W:        ", model.weight_model.W
    print ""

    R_test = model.compute_rate()
    if not np.allclose(R, R_test):
        print "Generative rate does not match inference rate."
        plt.figure()
        plt.plot(R, 'b')
        plt.plot(R_test, 'r')
        plt.show()
        import pdb; pdb.set_trace()

    # Gibbs sample
    N_samples = 100
    samples = []
    for itr in xrange(N_samples):
        samples.append(model.resample_and_copy())
        # print "Iteration ", itr
        # print "Lambda0:     ", model.bias_model.lambda0
        # print "W:           ", model.weight_model.W
        # print ""

    # Compute sample statistics for second half of samples
    A_samples       = np.array([A for A,_,_,_ in samples])
    W_samples       = np.array([W for _,W,_,_ in samples])
    beta_samples    = np.array([b for _,_,b,_ in samples])
    lambda0_samples = np.array([l for _,_,_,l in samples])

    offset = N_samples // 2
    A_mean       = A_samples[offset:, ...].mean(axis=0)
    W_mean       = W_samples[offset:, ...].mean(axis=0)
    beta_mean    = beta_samples[offset:, ...].mean(axis=0)
    lambda0_mean = lambda0_samples[offset:, ...].mean(axis=0)

    print "A mean:        ", A_mean
    print "W mean:        ", W_mean
    print "beta mean:     ", beta_mean
    print "lambda0 mean:  ", lambda0_mean



demo()
