"""
Network models expose a probability of connection and a scale of the weights
"""
import abc

import numpy as np
from scipy.special import gammaln, psi

from pyhawkes.deps.pybasicbayes.abstractions import \
    BayesianDistribution, GibbsSampling, MeanField

from pyhawkes.internals.distributions import Bernoulli, Gamma

# TODO: Make a base class for networks
# class Network(BayesianDistribution):
#
#     __metaclass__ = abc.ABCMeta
#
#     @abc.abstractproperty
#     def p(self):
#         """
#         Return a KxK matrix of probability of connection
#         """
#         pass
#
#     @abc.abstractproperty
#     def kappa(self):
#         """
#         Return a KxK matrix of gamma weight shape parameters
#         """
#         pass
#
#     @abc.abstractproperty
#     def v(self):
#         """
#         Return a KxK matrix of gamma weight scale parameters
#         """
#         pass

class _StochasticBlockModelBase(BayesianDistribution):
    """
    A stochastic block model is a clustered network model with
    K:          Number of nodes in the network
    C:          Number of blocks
    m[c]:       Probability that a node belongs block c
    p[c,c']:    Probability of connection from node in block c to node in block c'
    v[c,c']:    Scale of the gamma weight distribution from node in block c to node in block c'

    It is parameterized by:
    pi:         Parameter of Dirichlet prior over m
    tau0, tau1: Parameters of beta prior over p
    alpha:      Shape parameter of gamma prior over v
    beta:       Scale parameter of gamma prior over v
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, K, C,
                 c=None, pi=1.0,
                 p=None, tau0=0.1, tau1=0.1,
                 v=None, alpha=1.0, beta=1.0,
                 kappa=1.0):
        """
        Initialize SBM with parameters defined above.
        """
        assert isinstance(K, int) and C >= 1, "K must be a positive integer number of nodes"
        self.K = K

        assert isinstance(C, int) and C >= 1, "C must be a positive integer number of blocks"
        self.C = C

        # If m, p, and v are specified, then the model is fixed and the prior parameters
        # are ignored
        if None not in (c, p, v):
            self.fixed = True
            assert isinstance(c, np.ndarray) and c.shape == (K,) and c.dtype == np.int \
                   and np.amin(c) >= 0 and np.amax(c) <= self.C-1, \
                "c must be a length K-vector of block assignments"
            self.c = c

            if np.isscalar(p):
                assert p >= 0 and p <= 1, "p must be a probability"
                self.p = p * np.ones((C,C))

            else:
                assert isinstance(p, np.ndarray) and p.shape == (C,C) \
                       and np.amin(p) >= 0 and np.amax(p) <= 1.0, \
                    "p must be a CxC matrix of probabilities"
                self.p = p

            if np.isscalar(v):
                assert v >= 0, "v must be a probability"
                self.v = v * np.ones((C,C))

            else:
                assert isinstance(v, np.ndarray) and v.shape == (C,C) \
                       and np.amin(v) >= 0, \
                    "v must be a CxC matrix of nonnegative gamma scales"
                self.v = v

        else:
            self.fixed = False


        if isinstance(pi, (int, float)):
            self.pi = pi * np.ones(C)
        else:
            assert isinstance(pi, np.ndarray) and pi.shape == (C,), "pi must be a sclar or a C-vector"
            self.pi = pi

        self.tau0  = tau0
        self.tau1  = tau1
        self.kappa = kappa
        self.alpha = alpha
        self.beta  = beta

    @property
    def P(self):
        """
        Get the KxK matrix of probabilities
        :return:
        """
        return self.p[np.ix_(self.c, self.c)]

    @property
    def V(self):
        """
        Get the KxK matrix of scales
        :return:
        """
        return self.v[np.ix_(self.c, self.c)]

    @property
    def Kappa(self):
        return self.kappa * np.ones((self.K, self.K))

    def log_likelihood(self, x):
        """
        Compute the log likelihood of a weighted adjacency matrix

        :param x:    (m,p,v) tuple
        :return:
        """
        raise NotImplementedError()

    def rvs(self,size=[]):
        raise NotImplementedError()


class GibbsSBM(_StochasticBlockModelBase, GibbsSampling):
    """
    Implement Gibbs sampling for SBM
    """
    def __init__(self, K, C,
                 c=None, pi=1.0,
                 p=None, tau0=0.1, tau1=0.1,
                 v=None, alpha=1.0, beta=1.0,
                 kappa=1.0):
        super(GibbsSBM, self).__init__(K, C, c, pi, p, tau0, tau1, v, alpha, beta, kappa)

        # Initialize parameter estimates
        if not self.fixed:
            self.c = np.random.choice(self.C, size=(self.K))
            self.m = 1.0/C * np.ones(self.C)
            # self.p = self.tau1 / (self.tau0 + self.tau1) * np.ones((self.C, self.C))
            self.p = np.random.beta(self.tau1, self.tau0, size=(self.C, self.C))
            # self.v = self.alpha / self.beta * np.ones((self.C, self.C))
            self.v = np.random.gamma(self.alpha, 1.0/self.beta, size=(self.C, self.C))

    def resample_p(self, A):
        """
        Resample p given observations of the weights
        """
        for c1 in xrange(self.C):
            for c2 in xrange(self.C):
                Ac1c2 = A[self.c==c1, self.c==c2]
                tau1 = self.tau1 + Ac1c2.sum()
                tau0 = self.tau0 + (1-Ac1c2).sum()
                self.p[c1,c2] = np.random.beta(tau1, tau0)

    def resample_v(self, A, W):
        """
        Resample v given observations of the weights
        """
        for c1 in xrange(self.C):
            for c2 in xrange(self.C):
                Ac1c2 = A[self.c==c1, self.c==c2]
                Wc1c2 = W[self.c==c1, self.c==c2]
                alpha = self.alpha + Ac1c2.sum() * self.kappa
                beta  = self.beta + Wc1c2[Ac1c2 > 0].sum()
                self.v[c1,c2] = np.random.gamma(alpha, 1.0/beta)

    def resample_c(self, A, W):
        """
        Resample block assignments given the weighted adjacency matrix
        and the impulse response fits (if used)
        """
        # Sample each assignment in order
        for k in xrange(self.K):
            # Compute unnormalized log probs of each connection
            lp = np.zeros(self.C)

            # Prior from m
            lp += np.log(self.m)

            # Likelihood from network
            for ck in xrange(self.C):
                c_temp = self.c.copy().astype(np.int)
                c_temp[k] = ck

                # p(A[k,k'] | c)
                lp[ck] += Bernoulli(self.p[np.ix_([ck], c_temp)])\
                                .log_probability(A[k,:]).sum()
                # p(W[k,k'] | c)
                lp[ck] += A[k,k] * Gamma(self.kappa, self.v[np.ix_([ck], c_temp)])\
                                .log_probability(W[k,:]).sum()

                # TODO: Get probability of impulse responses g

    def resample_m(self):
        """
        Resample m given c and pi
        """
        pi = self.pi + np.bincount(self.c, minlength=self.C)
        self.m = np.random.dirichlet(pi)

    def resample(self, data=[]):
        A,W = data
        self.resample_p(A)
        self.resample_v(A, W)
        self.resample_c(A, W)
        self.resample_m()

class MeanFieldSBM(_StochasticBlockModelBase, MeanField):
    """
    Implement Gibbs sampling for SBM
    """
    def __init__(self, K, C,
                 c=None, pi=1.0,
                 p=None, tau0=0.1, tau1=0.1,
                 v=None, alpha=1.0, beta=1.0,
                 kappa=1.0):
        super(MeanFieldSBM, self).__init__(K, C, c, pi, p, tau0, tau1, v, alpha, beta, kappa)

        # Initialize mean field parameters
        self.mf_pi    = 1.0/self.C * np.ones(self.C)
        self.mf_m     = 1.0/self.C * np.ones((self.K, self.C))
        self.mf_tau0  = self.tau0  * np.ones((self.C, self.C))
        self.mf_tau1  = self.tau1  * np.ones((self.C, self.C))
        self.mf_alpha = self.alpha * np.ones((self.C, self.C))
        self.mf_beta  = self.beta  * np.ones((self.C, self.C))

    def expected_p(self):
        """
        Compute the expected probability of a connection, averaging over c
        :return:
        """
        if self.fixed:
            return self.P

        E_p = np.zeros((self.K, self.K))
        for c1 in xrange(self.C):
            for c2 in xrange(self.C):
                # Get the KxK matrix of joint class assignment probabilities
                pc1c2 = self.mf_m[:,c1][:, None] * self.mf_m[:,c2][None, :]

                # Get the probability of a connection for this pair of classes
                E_p += pc1c2 * self.mf_tau1 / (self.mf_tau0 + self.mf_tau1)
        return E_p

    def expected_notp(self):
        """
        Compute the expected probability of NO connection, averaging over c
        :return:
        """
        return 1.0 - self.expected_p()

    def expected_log_p(self):
        """
        Compute the expected log probability of a connection, averaging over c
        :return:
        """
        if self.fixed:
            return np.log(self.P)

        E_ln_p = np.zeros((self.K, self.K))
        for c1 in xrange(self.C):
            for c2 in xrange(self.C):
                # Get the KxK matrix of joint class assignment probabilities
                pc1c2 = self.mf_m[:,c1][:, None] * self.mf_m[:,c2][None, :]

                # Get the probability of a connection for this pair of classes
                E_ln_p += pc1c2 * (psi(self.mf_tau1) - psi(self.mf_tau0 + self.mf_tau1))

        return E_ln_p

    def expected_log_notp(self):
        """
        Compute the expected log probability of NO connection, averaging over c
        :return:
        """
        if self.fixed:
            return np.log(1.0 - self.P)

        E_ln_notp = np.zeros((self.K, self.K))
        for c1 in xrange(self.C):
            for c2 in xrange(self.C):
                # Get the KxK matrix of joint class assignment probabilities
                pc1c2 = self.mf_m[:,c1][:, None] * self.mf_m[:,c2][None, :]

                # Get the probability of a connection for this pair of classes
                E_ln_notp += pc1c2 * (psi(self.mf_tau0) - psi(self.mf_tau0 + self.mf_tau1))

        return E_ln_notp

    def expected_v(self):
        """
        Compute the expected scale of a connection, averaging over c
        :return:
        """
        if self.fixed:
            return self.V

        E_v = np.zeros((self.K, self.K))
        for c1 in xrange(self.C):
            for c2 in xrange(self.C):
                # Get the KxK matrix of joint class assignment probabilities
                pc1c2 = self.mf_m[:,c1][:, None] * self.mf_m[:,c2][None, :]

                # Get the probability of a connection for this pair of classes
                E_v += pc1c2 * self.mf_alpha / self.mf_beta
        return E_v

    def expected_log_v(self):
        """
        Compute the expected log scale of a connection, averaging over c
        :return:
        """
        if self.fixed:
            return np.log(self.V)

        E_log_v = np.zeros((self.K, self.K))
        for c1 in xrange(self.C):
            for c2 in xrange(self.C):
                # Get the KxK matrix of joint class assignment probabilities
                pc1c2 = self.mf_m[:,c1][:, None] * self.mf_m[:,c2][None, :]

                # Get the probability of a connection for this pair of classes
                E_log_v += pc1c2 * (psi(self.mf_alpha) - np.log(self.mf_beta))
        return E_log_v


    def expected_log_likelihood(self,x):
        pass

    def meanfieldupdate(self,data,weights):
        pass

    def get_vlb(self):
        raise NotImplementedError

    def resample_from_mf(self):
        """
        Resample from the mean field distribution
        :return:
        """
        raise NotImplementedError()

class StochasticBlockModel(GibbsSBM, MeanFieldSBM):
    pass

class ErdosRenyiModel(StochasticBlockModel):
    """
    The Erdos-Renyi network model is a special case of the SBM with one block.
    """
    def __init__(self, K,
                 p=None, tau0=0.1, tau1=0.1,
                 v=None, alpha=1.0, beta=1.0,
                 kappa=1.0):
        C = 1
        c = np.zeros(K, dtype=np.int)
        super(ErdosRenyiModel, self).__init__(K, C, c=c,
                                              p=p, tau0=tau0, tau1=tau1,
                                              v=v, alpha=alpha, beta=beta,
                                              kappa=kappa)
