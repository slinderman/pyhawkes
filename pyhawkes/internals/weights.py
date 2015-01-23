import numpy as np
from scipy.special import gammaln, psi
from scipy.misc import logsumexp

from pyhawkes.deps.pybasicbayes.distributions import GibbsSampling, MeanField
from pyhawkes.internals.distributions import Bernoulli, Gamma
from pyhawkes.utils.utils import logistic

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
        ll = (A * np.log(rho) + (1-A) * np.log(1-rho)).sum()

        # Get the shape and scale parameters from the network model
        kappa = self.network.kappa
        v = self.network.V

        # Add the LL of the gamma weights
        ll += (kappa * np.log(v) - gammaln(kappa) + \
              (kappa-1) * np.log(W) - v * W).sum()

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

class GammaMixtureWeights(MeanField):
    """
    For variational inference we approximate the spike at zero with a smooth
    Gamma distribution that has infinite density at zero.
    """
    def __init__(self, K, network, kappa_0=0.1, nu_0=10):
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

    def log_likelihood(self, x):
        raise NotImplementedError()

    def rvs(self,size=[]):
        raise NotImplementedError()

    def expected_A(self):
        return self.mf_p

    def expected_W(self):
        """
        Compute the expected W under the variational approximation
        """
        p_A = self.expected_A()
        return p_A * self.expected_W_given_A(1.0) + (1-p_A) * self.expected_W_given_A(0.0)

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
        return p_A * self.expected_log_W_given_A(1.0) + \
               (1-p_A) * self.expected_log_W_given_A(0.0)

    def expected_log_W_given_A(self, A):
        """
        Compute the expected log W given A under the variational approximation
        """
        return A * (psi(self.mf_kappa_1) - np.log(self.mf_v_1)) + \
               (1.0 - A) * (psi(self.mf_kappa_0) - np.log(self.mf_v_0))

    def expected_log_likelihood(self,x):
        raise NotImplementedError()

    def meanfieldupdate(self, EZ, N):
        self.meanfieldupdate_kappa_v(EZ, N)
        self.meanfieldupdate_p()

    def meanfieldupdate_p(self):
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

        self.mf_p = logistic(logit_p)

    def meanfieldupdate_kappa_v(self, EZ, N):
        """
        Update the variational weight distributions
        :return:
        """
        # kappa' = kappa + \sum_t \sum_b z[t,k,k',b]
        dkappa = EZ.sum(axis=(0,3))
        self.mf_kappa_0 = self.kappa_0 + dkappa
        self.mf_kappa_1 = self.network.kappa + dkappa

        # v_0'[k,k'] = self.nu_0 + N[k]
        self.mf_v_0 = self.nu_0 * np.ones((self.K, self.K)) + N[:,None]

        # v_1'[k,k'] = E[v[k,k']] + N[k]
        self.mf_v_1 = self.network.expected_v() + N[:,None]


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
        vlb += Bernoulli().entropy(E_x=E_A, E_notx=E_notA,
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
        vlb -= Bernoulli(self.mf_p).entropy().sum()

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