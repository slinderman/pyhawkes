"""
Network models expose a probability of connection and a scale of the weights
"""
import abc

import numpy as np
from scipy.special import gammaln, psi

from pyhawkes.deps.pybasicbayes.abstractions import \
    BayesianDistribution, GibbsSampling, MeanField

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

    def __init__(self, K, C, pi=1.0, tau0=0.1, tau1=0.1, alpha=1.0, beta=1.0):
        """
        Initialize SBM with parameters defined above.
        """
        assert isinstance(K, int) and C >= 1, "K must be a positive integer number of nodes"
        self.K = K

        assert isinstance(C, int) and C >= 1, "C must be a positive integer number of blocks"
        self.C = C

        if isinstance(pi, (int, float)):
            self.pi = pi * np.ones(C)
        else:
            assert isinstance(pi, np.ndarray) and pi.shape == (C,), "pi must be a sclar or a C-vector"
            self.pi = pi

        self.tau0 = tau0
        self.tau1 = tau1
        self.alpha = alpha
        self.beta = beta

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
    def __init__(self, K, C, pi=1.0, tau0=0.1, tau1=0.1, alpha=1.0, beta=1.0):
        super(GibbsSBM, self).__init__(K, C, pi, tau0, tau1, alpha, beta)

        # Initialize parameter estimates
        self.c = np.random.choice(self.C, size=(self.K))
        self.m = 1.0/C * np.ones(self.C)
        self.p = self.tau1 / (self.tau0 + self.tau1) * np.ones((self.C, self.C))
        self.v = self.alpha / self.beta * np.ones((self.C, self.C))

    def resample(self,data=[]):
        raise NotImplementedError()

class MeanFieldSBM(_StochasticBlockModelBase, MeanField):
    """
    Implement Gibbs sampling for SBM
    """
    def __init__(self, K, C, pi=1.0, tau0=0.1, tau1=0.1, alpha=1.0, beta=1.0):
        super(MeanFieldSBM, self).__init__(K, C, pi, tau0, tau1, alpha, beta)

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
        E_ln_p = np.zeros((self.K, self.K))
        for c1 in xrange(self.C):
            for c2 in xrange(self.C):
                # Get the KxK matrix of joint class assignment probabilities
                pc1c2 = self.mf_m[:,c1][:, None] * self.mf_m[:,c2][None, :]

                # Get the probability of a connection for this pair of classes
                E_ln_p += pc1c2 * psi(self.mf_tau1) - psi(self.mf_tau0 + self.mf_tau1)

        return E_ln_p

    def expected_log_notp(self):
        """
        Compute the expected log probability of NO connection, averaging over c
        :return:
        """
        E_ln_notp = np.zeros((self.K, self.K))
        for c1 in xrange(self.C):
            for c2 in xrange(self.C):
                # Get the KxK matrix of joint class assignment probabilities
                pc1c2 = self.mf_m[:,c1][:, None] * self.mf_m[:,c2][None, :]

                # Get the probability of a connection for this pair of classes
                E_ln_notp += pc1c2 * psi(self.mf_tau0) - psi(self.mf_tau0 + self.mf_tau1)

        return E_ln_notp

    def expected_log_likelihood(self,x):
        pass

    def meanfieldupdate(self,data,weights):
        pass

    def get_vlb(self):
        raise NotImplementedError

class StochasticBlockModel(GibbsSBM, MeanFieldSBM):
    pass

class ErdosRenyiModel(StochasticBlockModel):
    """
    The Erdos-Renyi network model is a special case of the SBM with one block.
    """
    def __init__(self, K, tau0=0.1, tau1=0.1, alpha=1.0, beta=1.0):
        C = 1
        pi = 1.0
        super(ErdosRenyiModel, self).__init__(K, C, pi,
                                              tau0=tau0, tau1=tau1,
                                              alpha=alpha, beta=beta)
