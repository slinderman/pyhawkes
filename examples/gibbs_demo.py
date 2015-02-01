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
    true_model = DiscreteTimeNetworkHawkesModelGammaMixture(C=C, K=K, dt=dt, B=B, kappa=kappa, c=c, p=p, v=v)

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

    C = 5
    K = 50
    T = 1000
    dt = 1.0
    B = 3
    kappa = 5.0
    c = np.arange(C).repeat((K // C))
    p = 0.5 * np.eye(C)
    v = kappa * (10 * np.eye(C) + 25.0 * (1-np.eye(C)))

    S, R, S_test, true_model = sample_from_network_hawkes(C, K, T, dt, B, kappa, c, p, v)

    # Make a model to initialize the parameters
    init_len   = T
    init_model = DiscreteTimeStandardHawkesModel(K=K, dt=dt, B=B,
                                                 l2_penalty=0, l1_penalty=0)
    init_model.add_data(S[:init_len, :])

    print "Initializing with BFGS on first ", init_len, " time bins."
    init_model.initialize_to_background_rate()
    init_model.fit_with_bfgs()

    # Make another new model for inference
    # test_model = DiscreteTimeNetworkHawkesModelGammaMixture(C=C, K=K, dt=dt, B=B,
    #                                                         kappa=kappa, beta=2.0/K,
    #                                                         tau0=5.0, tau1=1.0)
    test_model = DiscreteTimeNetworkHawkesModelGammaMixture(C=C, K=K, dt=dt, B=B,
                                                            kappa=kappa, alpha=1.0, beta=1.0/(kappa*20.0))
    test_model.add_data(S)
    F_test = test_model.basis.convolve_with_basis(S_test)


    # Initialize with the standard model parameters
    # test_model.initialize_with_standard_model(init_model)

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

    im_net = plot_network(np.ones((K,K)), test_model.weight_model.W_effective, vmax=0.5)
    plt.pause(0.001)

    # plt.figure(5)
    # im_p = plt.imshow(test_model.network.p, interpolation="none", cmap="Greys")
    # plt.xlabel('c\'')
    # plt.ylabel('c')
    # plt.title('P_{c to c\'')
    #
    # plt.figure(6)
    # im_v = plt.imshow(test_model.network.v, interpolation="none", cmap="Greys")
    # plt.xlabel('c\'')
    # plt.ylabel('c')
    # plt.title('V_{c to c\'')


    plt.show()
    plt.pause(0.001)

    # Gibbs sample
    N_samples = 100
    samples = []
    lps = []
    plls = []
    for itr in xrange(N_samples):
        lps.append(test_model.log_probability())
        plls.append(test_model.heldout_log_likelihood(S_test, F=F_test))
        samples.append(test_model.copy_sample())

        print ""
        print "Gibbs iteration ", itr
        print "LP: ", lps[-1]

        test_model.resample_model()

        # Update plot
        if itr % 1 == 0:
            plt.figure(2)
            ln.set_data(np.arange(T), test_model.compute_rate()[:,0])
            plt.title("Iteration %d" % itr)
            plt.pause(0.001)

            plt.figure(3)
            KC = np.zeros((K,C))
            KC[np.arange(K), test_model.network.c] = 1.0
            im_clus.set_data(KC)
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


    # Compute sample statistics for second half of samples
    A_samples       = np.array([s.weight_model.A     for s in samples])
    W_samples       = np.array([s.weight_model.W     for s in samples])
    g_samples       = np.array([s.impulse_model.g    for s in samples])
    lambda0_samples = np.array([s.bias_model.lambda0 for s in samples])
    c_samples       = np.array([s.network.c          for s in samples])
    p_samples       = np.array([s.network.p          for s in samples])
    v_samples       = np.array([s.network.v          for s in samples])
    lps             = np.array(lps)

    offset = N_samples // 2
    A_mean       = A_samples[offset:, ...].mean(axis=0)
    W_mean       = W_samples[offset:, ...].mean(axis=0)
    g_mean       = g_samples[offset:, ...].mean(axis=0)
    lambda0_mean = lambda0_samples[offset:, ...].mean(axis=0)
    p_mean       = p_samples[offset:, ...].mean(axis=0)
    v_mean       = v_samples[offset:, ...].mean(axis=0)


    print "A true:        ", true_model.weight_model.A
    print "W true:        ", true_model.weight_model.W
    print "g true:        ", true_model.impulse_model.g
    print "lambda0 true:  ", true_model.bias_model.lambda0
    print ""
    print "A mean:        ", A_mean
    print "W mean:        ", W_mean
    print "g mean:        ", g_mean
    print "lambda0 mean:  ", lambda0_mean
    print "v mean:        ", v_mean
    print "p mean:        ", p_mean

    plt.figure()
    plt.plot(np.arange(N_samples), lps, 'k')
    plt.xlabel("Iteration")
    plt.ylabel("Log probability")
    plt.show()

    # Predictive log likelihood
    pll_init = init_model.heldout_log_likelihood(S_test)
    plt.figure()
    plt.plot(np.arange(N_samples), pll_init * np.ones(N_samples), 'k')
    plt.plot(np.arange(N_samples), plls, 'r')
    plt.xlabel("Iteration")
    plt.ylabel("Predictive log probability")
    plt.show()

    # Compute the link prediction accuracy curves
    auc_init = roc_auc_score(true_model.weight_model.A.ravel(),
                             init_model.W.ravel())
    auc_A_mean = roc_auc_score(true_model.weight_model.A.ravel(),
                               A_mean.ravel())
    auc_W_mean = roc_auc_score(true_model.weight_model.A.ravel(),
                               W_mean.ravel())

    aucs = []
    for A in A_samples:
        aucs.append(roc_auc_score(true_model.weight_model.A.ravel(), A.ravel()))

    plt.figure()
    plt.plot(aucs, '-r')
    plt.plot(auc_A_mean * np.ones_like(aucs), '--r')
    plt.plot(auc_W_mean * np.ones_like(aucs), '--b')
    plt.plot(auc_init * np.ones_like(aucs), '--k')
    plt.xlabel("Iteration")
    plt.ylabel("Link prediction AUC")
    plt.show()


    # Compute the adjusted mutual info score of the clusterings
    amis = []
    arss = []
    for c in c_samples:
        amis.append(adjusted_mutual_info_score(true_model.network.c, c))
        arss.append(adjusted_rand_score(true_model.network.c, c))

    plt.figure()
    plt.plot(np.arange(N_samples), amis, '-r')
    plt.plot(np.arange(N_samples), arss, '-b')
    plt.xlabel("Iteration")
    plt.ylabel("Clustering score")


    plt.ioff()
    plt.show()

# demo(2203329564)
# demo(2728679796)

demo(11223344)
