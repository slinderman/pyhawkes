import sys

import numpy as np
from scipy.special import gammaln, psi
from scipy.misc import logsumexp

from pybasicbayes.distributions import GibbsSampling, MeanField, MeanFieldSVI
from pyhawkes.internals.distributions import Bernoulli, Gamma
from pyhawkes.utils.utils import logistic, logit

from pyhawkes.utils.profiling import line_profiled
PROFILING = True


class SpikeAndSlabGammaWeights(GibbsSampling):
    """
    Encapsulates the KxK Bernoulli adjacency matrix and the
    KxK gamma weight matrix. Implements Gibbs sampling given
    the parent variables.
    """
    def __init__(self, K, network):
        """
        Initialize the spike-and-slab gamma weight model with either a
        network object containing the prior or rho, alpha, and beta to
        define an independent model.

        :param K:       Number of processes
        :param network: Pointer to a network object exposing rho, alpha, and beta
        :param rho:     Sparsity level
        :param alpha:   Gamma shape parameter
        :param beta:    Gamma scale parameter
        """
        self.K = K
        # assert isinstance(network, GibbsNetwork), "network must be a GibbsNetwork object"
        self.network = network


        # Initialize parameters A and W
        self.A = np.ones((self.K, self.K))
        self.W = np.zeros((self.K, self.K))
        self.resample()

    @property
    def W_effective(self):
        return self.A * self.W

    def log_likelihood(self, x):
        """
        Compute the log likelihood of the given A and W
        :param x:  an (A,W) tuple
        :return:
        """
        A,W = x
        assert isinstance(A, np.ndarray) and A.shape == (self.K,self.K), \
            "A must be a KxK adjacency matrix"
        assert isinstance(W, np.ndarray) and W.shape == (self.K,self.K), \
            "W must be a KxK weight matrix"

        # LL of A
        rho = np.clip(self.network.P, 1e-32, 1-1e-32)
        ll = (A * np.log(rho) + (1-A) * np.log(1-rho)).sum()
        ll = np.nan_to_num(ll)

        # Get the shape and scale parameters from the network model
        kappa = self.network.kappa
        v = self.network.V

        # Add the LL of the gamma weights
        lp_W = kappa * np.log(v) - gammaln(kappa) + \
               (kappa-1) * np.log(W) - v * W
        ll += (A*lp_W).sum()

        print "LL(AW):\t", ll

        return ll

    def log_probability(self):
        return self.log_likelihood((self.A, self.W))

    def rvs(self,size=[]):
        A = np.random.rand(self.K, self.K) < self.network.P
        W = np.random.gamma(self.network.kappa, 1.0/self.network.V,
                            size(self.K, self.K))

        return A,W

    def _joint_resample_A_W(self):
        """
        Not sure how to do this yet, but it would be nice to resample A
        from its marginal distribution after integrating out W, and then
        sample W | A.
        :return:
        """
        raise NotImplementedError()


    def _resample_A_given_W(self, model):
        """
        Resample A given W. This must be immediately followed by an
        update of z | A, W.
        :return:
        """
        # TODO: Write a Cython function to sample this more efficiently
        p = self.network.P
        for k1 in xrange(self.K):
            sys.stdout.write('.')
            sys.stdout.flush()
            for k2 in xrange(self.K):
                if model is None:
                    ll0 = 0
                    ll1 = 0
                else:
                    # Compute the log likelihood of the events given W and A=0
                    self.A[k1,k2] = 0
                    ll0 = model._log_likelihood_single_process(k2)

                    # Compute the log likelihood of the events given W and A=1
                    self.A[k1,k2] = 1
                    ll1 = model._log_likelihood_single_process(k2)

                # Sample A given conditional probability
                lp0 = ll0 + np.log(1.0 - p[k1,k2])
                lp1 = ll1 + np.log(p[k1,k2])
                Z   = logsumexp([lp0, lp1])

                # ln p(A=1) = ln (exp(lp1) / (exp(lp0) + exp(lp1)))
                #           = lp1 - ln(exp(lp0) + exp(lp1))
                #           = lp1 - Z
                self.A[k1,k2] = np.log(np.random.rand()) < lp1 - Z
        sys.stdout.write('\n')
        sys.stdout.flush()

    def _get_suff_statistics(self, N, Z):
        """
        Compute the sufficient statistics from the data set.
        :param data: a TxK array of event counts assigned to the background process
        :return:
        """
        ss = np.zeros((2, self.K, self.K))

        if N is not None and Z is not None:
            # ss[0,k1,k2] = \sum_t \sum_b Z[t,k1,k2,b]
            ss[0,:,:] = Z.sum(axis=(0,3))
            # ss[1,k1,k2] = N[k1] * A[k1,k2]
            ss[1,:,:] = N[:,None] * self.A

        return ss

    def _get_exact_suff_statistics(self, Z, F, beta):
        """
        For comparison, compute the exact sufficient statistics for ss[1,:,:]
        :param data: a TxK array of event counts assigned to the background process
        :return:
        """
        ss = np.zeros((2, self.K, self.K))

        if F is not None and beta is not None:
            # ss[0,k1,k2] = \sum_t \sum_b Z[t,k1,k2,b]
            ss[0,:,:] = Z.sum(axis=(0,3))

            # ss[1,k1,k2] = A_k1,k2 * \sum_t \sum_b F[t,k1,b] * beta[k1,k2,b]
            for k1 in range(self.K):
                for k2 in range(self.K):
                    ss[1,k1,k2] = self.A[k1,k2] * (F[:,k1,:].dot(beta[k1,k2,:])).sum()

        return ss

    def resample_W_given_A_and_z(self, N, Z, F, beta):
        """
        Resample the weights given A and z.
        :return:
        """
        assert (N is None and Z is None) \
               or (isinstance(Z, np.ndarray)
                   and Z.ndim == 4
                   and Z.shape[1] == self.K
                   and Z.shape[2] == self.K
                   and isinstance(N, np.ndarray)
                   and N.shape == (self.K,)), \
            "N must be a K-vector and Z must be a TxKxKxB array of parent counts"

        ss = self._get_suff_statistics(N, Z)
        kappa_post = self.network.kappa + ss[0,:]
        v_post  = self.network.V + ss[1,:]

        self.W = np.array(np.random.gamma(kappa_post, 1.0/v_post)).reshape((self.K, self.K))

    def resample(self, model=None, N=None, Z=None, F=None, beta=None):
        """
        Resample A and W given the parents
        :param N:   A length-K vector specifying how many events occurred
                    on each of the K processes
        :param Z:   A TxKxKxB array of parent assignment counts
        """
        # Resample W | A
        self.resample_W_given_A_and_z(N, Z, F, beta)

        # Resample A given W
        self._resample_A_given_W(model)

class GammaMixtureWeights(GibbsSampling, MeanField, MeanFieldSVI):
    """
    For variational inference we approximate the spike at zero with a smooth
    Gamma distribution that has infinite density at zero.
    """
    def __init__(self, K, network, kappa_0=0.1, nu_0=10.0):
        """
        Initialize the spike-and-slab gamma weight model with either a
        network object containing the prior or rho, alpha, and beta to
        define an independent model.

        :param K:           Number of processes
        :param network:     Pointer to a network object exposing rho, alpha, and beta
        :param kappa_1:     Shape for weight distribution
        :param kappa_0:     Shape for gamma spike (small)
        :param nu_0:        Scale for gamma spike (large)
        """
        self.K = K
        assert network is not None, "A network object must be given"

        self.network = network

        # Save gamma parameters
        self.kappa_0 = kappa_0
        self.nu_0    = nu_0

        # Initialize the variational parameters to the prior mean
        # Variational probability of edge
        # self.mf_p = network.P * np.ones((self.K, self.K))
        self.mf_p = np.ones((self.K, self.K)) - 1e-3
        # Variational weight distribution given that there is no edge
        self.mf_kappa_0 = self.kappa_0 * np.ones((self.K, self.K))
        self.mf_v_0 = self.nu_0 * np.ones((self.K, self.K))
        # Variational weight distribution given that there is an edge
        self.mf_kappa_1 = self.network.kappa * np.ones((self.K, self.K))
        # self.mf_v_1 = network.alpha / network.beta * np.ones((self.K, self.K))
        self.mf_v_1 = network.V * np.ones((self.K, self.K))

        # Initialize parameters A and W
        self.A = np.ones((self.K, self.K))
        self.W = np.zeros((self.K, self.K))
        self.resample()

    @property
    def W_effective(self):
        return self.W

    def log_likelihood(self, x):
        """
        Compute the log likelihood of the given A and W
        :param x:  an (A,W) tuple
        :return:
        """
        A,W = x
        assert isinstance(A, np.ndarray) and A.shape == (self.K,self.K), \
            "A must be a KxK adjacency matrix"
        assert isinstance(W, np.ndarray) and W.shape == (self.K,self.K), \
            "W must be a KxK weight matrix"

        # LL of A
        rho = self.network.P
        lp_A = (A * np.log(rho) + (1-A) * np.log(1-rho))

        # Get the shape and scale parameters from the network model
        kappa = self.network.kappa
        v = self.network.V

        # Add the LL of the gamma weights
        # lp_W = np.zeros((self.K, self.K))
        # lp_W = A * (kappa * np.log(v) - gammaln(kappa)
        #             + (kappa-1) * np.log(W) - v * W)

        lp_W0 = (self.kappa_0 * np.log(self.nu_0) - gammaln(self.kappa_0)
                    + (self.kappa_0-1) * np.log(W) - self.nu_0 * W)[A==0]

        lp_W1 = (kappa * np.log(v) - gammaln(kappa)
                    + (kappa-1) * np.log(W) - v * W)[A==1]

        # lp_W = A * (kappa * np.log(v) - gammaln(kappa)
        #             + (kappa-1) * np.log(W) - v * W) + \
        #        (1-A) * (self.kappa_0 * np.log(self.nu_0) - gammaln(self.kappa_0)
        #                 + (self.kappa_0-1) * np.log(W) - self.nu_0 * W)
        ll = lp_A.sum() + lp_W0.sum() + lp_W1.sum()

        return ll

    def log_probability(self):
        return self.log_likelihood((self.A, self.W))

    def rvs(self,size=[]):
        raise NotImplementedError()

    def expected_A(self):
        return self.mf_p

    def expected_W(self):
        """
        Compute the expected W under the variational approximation
        """
        p_A = self.expected_A()
        E_W =  p_A * self.expected_W_given_A(1.0) + (1-p_A) * self.expected_W_given_A(0.0)

        if not self.network.allow_self_connections:
            np.fill_diagonal(E_W, 0.0)

        return E_W

    def expected_W_given_A(self, A):
        """
        Compute the expected W given A under the variational approximation
        :param A:   Either zero or 1
        """
        return A * (self.mf_kappa_1 / self.mf_v_1) + \
               (1.0 - A) * (self.mf_kappa_0 / self.mf_v_0)

    def expected_log_W(self):
        """
        Compute the expected log W under the variational approximation
        """
        p_A = self.expected_A()
        E_ln_W =  p_A * self.expected_log_W_given_A(1.0) + \
               (1-p_A) * self.expected_log_W_given_A(0.0)

        if not self.network.allow_self_connections:
            np.fill_diagonal(E_ln_W, -np.inf)

        return E_ln_W

    def expected_log_W_given_A(self, A):
        """
        Compute the expected log W given A under the variational approximation
        """
        return A * (psi(self.mf_kappa_1) - np.log(self.mf_v_1)) + \
               (1.0 - A) * (psi(self.mf_kappa_0) - np.log(self.mf_v_0))

    def expected_log_likelihood(self,x):
        raise NotImplementedError()

    def meanfieldupdate_p(self, stepsize=1.0):
        """
        Update p given the network parameters and the current variational
        parameters of the weight distributions.
        :return:
        """
        logit_p = self.network.expected_log_p() - self.network.expected_log_notp()
        logit_p += self.network.kappa * self.network.expected_log_v() - gammaln(self.network.kappa)
        logit_p += gammaln(self.mf_kappa_1) - self.mf_kappa_1 * np.log(self.mf_v_1)
        logit_p += gammaln(self.kappa_0) - self.kappa_0 * np.log(self.nu_0)
        logit_p += self.mf_kappa_0 * np.log(self.mf_v_0) - gammaln(self.mf_kappa_0)

        # p_hat = logistic(logit_p)
        # self.mf_p = (1.0 - stepsize) * self.mf_p + stepsize * p_hat

        logit_p_hat = (1-stepsize) * logit(self.mf_p) + \
                       stepsize * logit_p
        self.mf_p = logistic(logit_p_hat)

    def meanfieldupdate_kappa_v(self, EZ, N, minibatchfrac=1.0, stepsize=1.0):
        """
        Update the variational weight distributions
        :return:
        """
        EZ_sum = EZ.sum(axis=(0,3))
        # kappa' = kappa + \sum_t \sum_b z[t,k,k',b]
        kappa0_hat = self.kappa_0 + EZ_sum / minibatchfrac
        kappa1_hat = self.network.kappa + EZ_sum / minibatchfrac
        self.mf_kappa_0 = (1.0 - stepsize) * self.mf_kappa_0 + stepsize * kappa0_hat
        self.mf_kappa_1 = (1.0 - stepsize) * self.mf_kappa_1 + stepsize * kappa1_hat

        # v_0'[k,k'] = self.nu_0 + N[k]
        v0_hat = self.nu_0 * np.ones((self.K, self.K)) + N[:,None] / minibatchfrac
        self.mf_v_0 = (1.0 - stepsize) * self.mf_v_0 + stepsize * v0_hat

        # v_1'[k,k'] = E[v[k,k']] + N[k]
        v1_hat = self.network.expected_v() + N[:,None] / minibatchfrac
        self.mf_v_1 = (1.0 - stepsize) * self.mf_v_1 + stepsize * v1_hat

    def meanfieldupdate(self, EZ, N):
        self.meanfieldupdate_kappa_v(EZ, N)
        self.meanfieldupdate_p()

    def meanfield_sgdstep(self, EZ, N, minibatchfrac,stepsize):
        self.meanfieldupdate_kappa_v(EZ, N, minibatchfrac=minibatchfrac, stepsize=stepsize)
        self.meanfieldupdate_p(stepsize=stepsize)

    def get_vlb(self):
        """
        Variational lower bound for A_kk' and W_kk'
        E[LN p(A_kk', W_kk' | p, kappa, v)] -
        E[LN q(A_kk', W_kk' | mf_p, mf_kappa, mf_v)]
        :return:
        """
        vlb = 0

        # First term:
        # E[LN p(A | p)]
        E_A       = self.expected_A()
        E_notA    = 1.0 - E_A
        E_ln_p    = self.network.expected_log_p()
        E_ln_notp = self.network.expected_log_notp()
        vlb += Bernoulli().negentropy(E_x=E_A, E_notx=E_notA,
                                      E_ln_p=E_ln_p, E_ln_notp=E_ln_notp).sum()

        # E[LN p(W | A=1, kappa, v)]
        kappa     = self.network.kappa
        E_v       = self.network.expected_v()
        E_ln_v    = self.network.expected_log_v()
        E_W1      = self.expected_W_given_A(A=1)
        E_ln_W1   = self.expected_log_W_given_A(A=1)
        vlb += (E_A * Gamma(kappa).negentropy(E_beta=E_v, E_ln_beta=E_ln_v,
                                              E_lambda=E_W1, E_ln_lambda=E_ln_W1)).sum()

        # E[LN p(W | A=0, kappa0, v0)]
        kappa0    = self.kappa_0
        v0        = self.nu_0
        E_W0      = self.expected_W_given_A(A=0)
        E_ln_W0   = self.expected_log_W_given_A(A=0)
        vlb += (E_notA * Gamma(kappa0, v0).negentropy(E_lambda=E_W0, E_ln_lambda=E_ln_W0)).sum()

        # Second term
        # E[LN q(A)]
        vlb -= Bernoulli(self.mf_p).negentropy().sum()

        # E[LN q(W | A=1)]
        vlb -= (E_A    * Gamma(self.mf_kappa_1, self.mf_v_1).negentropy()).sum()
        vlb -= (E_notA * Gamma(self.mf_kappa_0, self.mf_v_0).negentropy()).sum()

        return vlb

    def resample_from_mf(self):
        """
        Resample from the mean field distribution
        :return:
        """
        self.A = np.random.rand(self.K, self.K) < self.mf_p
        self.W = (1-self.A) * np.random.gamma(self.mf_kappa_0, 1.0/self.mf_v_0)
        self.W += self.A * np.random.gamma(self.mf_kappa_1, 1.0/self.mf_v_1)

    # Gibbs sampling
    def _get_suff_statistics(self, N, Z):
        """
        Compute the sufficient statistics from the data set.
        :param data: a TxK array of event counts assigned to the background process
        :return:
        """
        ss = np.zeros((2, self.K, self.K))

        if N is not None and Z is not None:
            # ss[0,k1,k2] = \sum_t \sum_b Z[t,k1,k2,b]
            ss[0,:,:] = Z.sum(axis=(0,3))
            # ss[1,k1,k2] = N[k1]
            ss[1,:,:] = N[:,None]

        return ss

    def resample(self, N=None, Z=None):
        ss = self._get_suff_statistics(N,Z)

        # First resample A from its marginal distribution after integrating out W
        self._resample_A(ss)

        # Then resample W given A
        self._resample_W_given_A(ss)

    def _resample_A(self, ss):
        """
        Resample A from the marginal distribution after integrating out W
        :param ss:
        :return:
        """
        p = self.network.P
        v = self.network.V

        kappa0_post = self.kappa_0 + ss[0,:,:]
        v0_post     = self.nu_0 + ss[1,:,:]

        kappa1_post = self.network.kappa + ss[0,:,:]
        v1_post     = v + ss[1,:,:]

        # Compute the marginal likelihood of A=1 and of A=0
        # The result of the integral is a ratio of gamma distribution normalizing constants
        lp0  = self.kappa_0 * np.log(self.nu_0) - gammaln(self.kappa_0)
        lp0 += gammaln(kappa0_post) - kappa0_post * np.log(v0_post)

        lp1  = self.network.kappa * np.log(v) - gammaln(self.network.kappa)
        lp1 += gammaln(kappa1_post) - kappa1_post * np.log(v1_post)

        # Add the prior and normalize
        lp0 = lp0 + np.log(1.0 - p)
        lp1 = lp1 + np.log(p)
        Z   = logsumexp(np.concatenate((lp0[:,:,None], lp1[:,:,None]),
                                       axis=2),
                        axis=2)

        # ln p(A=1) = ln (exp(lp1) / (exp(lp0) + exp(lp1)))
        #           = lp1 - ln(exp(lp0) + exp(lp1))
        #           = lp1 - Z
        self.A = np.log(np.random.rand(self.K, self.K)) < lp1 - Z

    def _resample_W_given_A(self, ss):
        # import pdb; pdb.set_trace()
        kappa_prior = self.kappa_0 * (1-self.A) + self.network.kappa * self.A
        kappa_cond  = kappa_prior + ss[0,:,:]

        v_prior     = self.nu_0 * (1-self.A) + self.network.V * self.A
        v_cond      = v_prior + ss[1,:,:]

        # Resample W from its gamma conditional
        self.W = np.array(np.random.gamma(kappa_cond, 1.0/v_cond)).\
                        reshape((self.K, self.K))

        self.W = np.clip(self.W, 1e-32, np.inf)

    def initialize_from_gibbs(self, A, W, scale=100):
        """
        Initialize from a Gibbs sample
        :param A: Given adjacency matrix
        :param W: Given weight matrix
        :return:
        """
        # Set mean field probability of connection to conf if A==1
        # and (1-conf) if A == 0
        conf = 0.95
        self.mf_p = conf * A + (1-conf) * (1-A)

        # Set variational weight distribution
        self.mf_kappa_0 = self.kappa_0
        self.mf_v_0     = self.nu_0

        self.mf_kappa_1 = scale * W
        self.mf_v_1     = scale


class SpikeAndSlabContinuousTimeGammaWeights(GibbsSampling):
    """
    Implementation of spike and slab gamma weights from L&A 2014
    """
    def __init__(self, model):
        self.model = model
        self.network = model.network
        self.K = self.model.K

        # Initialize parameters A and W
        self.A = np.ones((self.K, self.K))
        self.W = np.zeros((self.K, self.K))
        self.resample()

    def log_likelihood(self,x):
        raise NotImplementedError

    def rvs(self,size=[]):
        raise NotImplementedError

    @property
    def W_effective(self):
        return self.A * self.W

    def _compute_weighted_impulses_at_events_manual(self, data):
        # Compute the instantaneous rate at the individual events
        # Sum over potential parents.

        # TODO: Call cython function to evaluate instantaneous rate
        N, S, C, Z, dt_max = data.N, data.S, data.C, data.Z, self.model.dt_max
        W = self.W

        # Initialize matrix of weighted impulses from each process
        lmbda = np.zeros((self.K, N))
        for n in xrange(N):
            # First parent is just the background rate of this process
            # lmbda[self.K, n] += lambda0[C[n]]

            # Iterate backward from the most recent to compute probabilities of each parent spike
            for par in xrange(n-1, -1, -1):
                dt = S[n] - S[par]
                if dt == 0:
                    continue

                # Since the spikes are sorted, we can stop if we reach a potential
                # parent that occurred greater than dt_max in the past
                if dt >= dt_max:
                    break

                lmbda[C[par], n] += W[C[par], C[n]] * self.model.impulse_model.impulse(dt, C[par], C[n])

        return lmbda

    def _compute_weighted_impulses_at_events(self, data):
        from pyhawkes.internals.continuous_time_helpers import \
            compute_weighted_impulses_at_events

        N, S, C, Z, dt_max = data.N, data.S, data.C, data.Z, self.model.dt_max
        W = self.W
        mu, tau = self.model.impulse_model.mu, self.model.impulse_model.tau
        lmbda = np.zeros((N, self.K))
        compute_weighted_impulses_at_events(S, C, Z, dt_max, W, mu, tau, lmbda)
        return lmbda

    def _resample_A_given_W(self, data):
        """
        Resample A given W. This must be immediately followed by an
        update of z | A, W.
        :return:
        """
        # Precompute weightedi impulse responses for each event
        lmbda_irs = [self._compute_weighted_impulses_at_events(d) for d in data]

        # lmbda_irs_manual = [self._compute_weighted_impulses_at_events_manual(d) for d in data]
        # for l1,l2 in zip(lmbda_irs_manual, lmbda_irs):
        #     assert np.allclose(l1,l2)

        lmbda0 = self.model.lambda0

        def _log_likelihood_single_process(k):
            ll = 0
            for lmbda_ir, d in zip(lmbda_irs, data):
                Ns, C, T = d.Ns, d.C, d.T

                # - \int lambda_k(t) dt
                ll -= lmbda0[k] * T
                ll -= self.W_effective[:,k].dot(Ns)

                # + \sum_n log(lambda(s_n))
                ll += np.log(lmbda0[k] + np.sum(self.A[:,k][:,None] * lmbda_ir[C==k,:], axis=0)).sum()
            return ll

        # TODO: Write a Cython function to sample this more efficiently
        p = self.network.P
        for k1 in xrange(self.K):
            # sys.stdout.write('.')
            # sys.stdout.flush()
            for k2 in xrange(self.K):
                # Handle deterministic cases
                if p[k1,k2] == 0.:
                    self.A[k1,k2] = 0
                    continue

                if p[k1,k2] == 1.:
                    self.A[k1,k2] = 1
                    continue

                # Compute the log likelihood of the events given W and A=0
                self.A[k1,k2] = 0
                ll0 = _log_likelihood_single_process(k2)

                # Compute the log likelihood of the events given W and A=1
                self.A[k1,k2] = 1
                ll1 = _log_likelihood_single_process(k2)

                # Sample A given conditional probability
                lp0 = ll0 + np.log(1.0 - p[k1,k2])
                lp1 = ll1 + np.log(p[k1,k2])
                Z   = logsumexp([lp0, lp1])

                self.A[k1,k2] = np.log(np.random.rand()) < lp1 - Z

        # sys.stdout.write('\n')
        # sys.stdout.flush()

    def resample_W_given_A_and_z(self, N, Zsum):
        """
        Resample the weights given A and z.
        :return:
        """
        kappa_post = self.network.kappa + Zsum
        v_post  = self.network.V + N[:,None] * self.A

        self.W = np.array(np.random.gamma(kappa_post, 1.0/v_post)).reshape((self.K, self.K))

    def resample(self, data=[]):
        """
        Resample A and W given the parents
        :param N:   A length-K vector specifying how many events occurred
                    on each of the K processes
        :param Z:   A TxKxKxB array of parent assignment counts
        """
        assert isinstance(data, list)

        # Compute sufficient statistics
        N = np.zeros((self.K,))
        Zsum = np.zeros((self.K, self.K))
        for d in data:
            Zsum += d.weight_ss
            N += d.Ns

        # Resample W | A, Z
        self.resample_W_given_A_and_z(N, Zsum)

        # Resample A | W
        self._resample_A_given_W(data)