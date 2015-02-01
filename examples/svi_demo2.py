import numpy as np
# np.seterr(all='raise')

import matplotlib.pyplot as plt

from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, roc_auc_score

from pyhawkes.models import DiscreteTimeNetworkHawkesModelSpikeAndSlab, \
    DiscreteTimeStandardHawkesModel, DiscreteTimeNetworkHawkesModelGammaMixture
from pyhawkes.plotting.plotting import plot_network

def sample_from_network_hawkes(C, K, T, dt, B, kappa, c, p, v, T_test=1000):
    # Create a true model
    # K=20, C=2
    # p = 0.75 * np.eye(C)
    # v = kappa * (8.0 * np.eye(C) + 25.0 * (1-np.eye(C)))

    # K=20, C=5
    # p = 0.2 * np.eye(C) + 0.02 * (1-np.eye(C))
    # v = kappa * (3 * np.eye(C) + 5.0 * (1-np.eye(C)))

    # K=50, C=5
    # p = 0.75 * np.eye(C) + 0.05 * (1-np.eye(C))
    # v = kappa * (9 * np.eye(C) + 25.0 * (1-np.eye(C)))

    # assert K % C == 0
    # c = np.arange(C).repeat((K // C))
    # true_model = DiscreteTimeNetworkHawkesModelGammaMixture(C=C, K=K, dt=dt, B=B, kappa=kappa, c=c, p=p, v=v)
    true_model = DiscreteTimeNetworkHawkesModelSpikeAndSlab(C=C, K=K, dt=dt, B=B, kappa=kappa, c=c, p=p, v=v)

    assert true_model.check_stability()

    # Sample from the true model
    S,R = true_model.generate(T=T)

    # Sample test data
    S_test,_ = true_model.generate(T=T_test)

    # Return the spike count matrix
    return S, R, S_test, true_model

def demo(seed=None):
    """
    Create a discrete time Hawkes model and generate from it.

    :return:
    """
    if seed is None:
        seed = np.random.randint(2**32)

    print "Setting seed to ", seed
    np.random.seed(seed)

    # C = 5
    # K = 50
    # T = 1000
    # dt = 1.0
    # B = 3
    # kappa = 5.0
    # c = np.arange(C).repeat((K // C))
    # p = 0.5 * np.eye(C)
    # v = kappa * (10 * np.eye(C) + 25.0 * (1-np.eye(C)))

    C = 2
    K = 20
    T = 10000
    dt = 1.0
    B = 3
    kappa = 2.0
    c = np.arange(C).repeat((K // C))
    p = 0.4 * np.eye(C) + 0.02 * (1-np.eye(C))
    v = kappa * (5 * np.eye(C) + 5.0 * (1-np.eye(C)))

    S, R, S_test, true_model = sample_from_network_hawkes(C, K, T, dt, B, kappa, c, p, v)

    # Make a model to initialize the parameters
    init_len   = T
    init_model = DiscreteTimeStandardHawkesModel(K=K, dt=dt, B=B,
                                                 alpha=1.0, beta=1.0)
    init_model.add_data(S[:init_len, :])

    print "Initializing with BFGS on first ", init_len, " time bins."
    init_model.initialize_to_background_rate()
    init_model.fit_with_bfgs()

    # Make another new model for inference
    # test_model = DiscreteTimeNetworkHawkesModelGammaMixture(C=C, K=K, dt=dt, B=B,
    #                                                         kappa=kappa, beta=2.0/K,
    #                                                         tau0=5.0, tau1=1.0)
    test_model = DiscreteTimeNetworkHawkesModelGammaMixture(C=C, K=K, dt=dt, B=B,
                                                            kappa=kappa,
                                                            tau0=8.0, tau1=1.0,
                                                            alpha=2.0, beta=2.0/(2.0*5.0))
    test_model.add_data(S)
    F_test = test_model.basis.convolve_with_basis(S_test)


    # Initialize with the standard model parameters
    test_model.initialize_with_standard_model(init_model)
    test_model.resample_from_mf()


    # Plot the true network
    plt.ion()
    plot_network(true_model.weight_model.A,
                 true_model.weight_model.W)
    plt.pause(0.001)


    # Plot the true and inferred firing rate
    plt.figure(2)
    plt.plot(np.arange(T), R[:,0], '-k', lw=2)
    plt.ion()
    ln = plt.plot(np.arange(T), test_model.compute_rate()[:,0], '-r')[0]
    plt.show()

    # Plot the block affiliations
    plt.figure(3)
    KC = np.zeros((K,C))
    KC[np.arange(K), test_model.network.c] = 1.0
    im_clus = plt.imshow(KC,
                    interpolation="none", cmap="Greys",
                    aspect=float(C)/K)

    im_net = plot_network(np.ones((K,K)),
                          test_model.weight_model.W_effective,
                          vmax=np.amax(true_model.weight_model.W_effective))
    plt.pause(0.001)
    plt.show()

    # VB coordinate descent
    N_iters = 1000
    minibatchsize = 256
    delay = 1.0
    forgetting_rate = 0.5
    stepsize = (np.arange(N_iters) + delay)**(-forgetting_rate)
    vlbs = []
    plls = []
    for itr in xrange(N_iters):
        print "SVI Iter: ", itr, "\tStepsize: ", stepsize[itr]
        test_model.sgd_step(minibatchsize=minibatchsize, stepsize=stepsize[itr])
        # vlbs.append(test_model.get_vlb())

        # Resample from variational distribution and plot
        test_model.resample_from_mf()

        # Compute predictive log likelihood
        plls.append(test_model.heldout_log_likelihood(S_test, F=F_test))

        print "N_conns: ", test_model.weight_model.A.sum()
        print "W_max: ", test_model.weight_model.W.max()

        # Update plot
        if itr % 1 == 0:
            plt.figure(2)
            ln.set_data(np.arange(T), test_model.compute_rate()[:,0])
            plt.title("Iteration %d" % itr)
            plt.pause(0.001)

            plt.figure(3)
            im_clus.set_data(test_model.network.mf_m)
            plt.title("Iteration %d" % itr)
            plt.pause(0.001)

            plt.figure(4)
            im_net.set_data(test_model.weight_model.W_effective)
            plt.pause(0.001)

            # plt.figure(5)
            # im_p.set_data(test_model.network.p)
            # plt.pause(0.001)
            #
            # plt.figure(6)
            # im_v.set_data(test_model.network.v)
            # plt.pause(0.001)


    print "A true:        ", true_model.weight_model.A
    print "W true:        ", true_model.weight_model.W
    # print "g true:         ", true_model.impulse_model.g
    print "lambda0 true:  ", true_model.bias_model.lambda0
    print ""
    print "A mean:        ", test_model.weight_model.expected_A()
    print "W mean:        ", test_model.weight_model.expected_W()
    # print "g mean:        ", test_model.impulse_model.expected_g()
    print "lambda0 mean:  ", test_model.bias_model.expected_lambda0()

    print "E[p]:  ", test_model.network.mf_tau1 / (test_model.network.mf_tau0 + test_model.network.mf_tau1)
    print "E[v]:  ", test_model.network.mf_alpha / test_model.network.mf_beta

    # plt.figure()
    # plt.plot(np.arange(N_iters), vlbs)
    # plt.xlabel("Iteration")
    # plt.ylabel("VLB")

    # Predictive log likelihood
    pll_init = init_model.heldout_log_likelihood(S_test)
    plt.figure()
    plt.plot(np.arange(N_iters), pll_init * np.ones(N_iters), 'k')
    plt.plot(np.arange(N_iters), plls, 'r')
    plt.xlabel("Iteration")
    plt.ylabel("Predictive log probability")

    # Compute the link prediction accuracy curves
    auc_init = roc_auc_score(true_model.weight_model.A.ravel(),
                             init_model.W.ravel())
    auc_A_mean = roc_auc_score(true_model.weight_model.A.ravel(),
                               test_model.weight_model.expected_A().ravel())
    auc_W_mean = roc_auc_score(true_model.weight_model.A.ravel(),
                               (test_model.weight_model.expected_A() *
                                test_model.weight_model.expected_W()).ravel())

    print "AUC init: ", auc_init
    print "AUC E[A]: ", auc_A_mean
    print "AUC E[WA]: ", auc_W_mean

    print "Random seed was: ", seed

    plt.ioff()
    plt.show()

# demo(2203329564)
# demo(2728679796)

# demo(11223344)

demo(3848328624)