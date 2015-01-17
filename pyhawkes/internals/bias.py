import numpy as np
from scipy.special import gammaln

from pyhawkes.deps.pybasicbayes.distributions import GibbsSampling

class GammaBias(GibbsSampling):
    """
    Encapsulates the vector of K gamma-distributed bias variables.
    """
    def __init__(self, K, dt, alpha, beta):
        """
        Initialize a bias vector for each of the K processes.

        :param K:       Number of processes
        :param dt:      Bin size. This is required to put lambda0 in the right units.
        :param alpha:   Shape parameter of the gamma prior
        :param beta:    Scale parameter of the gamma prior
        """
        self.K = K
        self.dt = dt
        self.alpha = alpha
        self.beta = beta

        # Initialize lambda0
        self.lambda0 = np.empty(self.K)
        self.resample()

    def log_likelihood(self, x):
        assert isinstance(x, np.ndarray) and x.shape == (self.K,), \
            "x must be a K-vector of background rates"

        return self.K * (self.alpha * np.log(self.beta) - gammaln(self.alpha)) + \
               ((self.alpha-1) * np.log(x) - self.beta * x).sum()

    def _get_suff_statistics(self, data):
        """
        Compute the sufficient statistics from the data set.
        :param data: a TxK array of event counts assigned to the background process
        :return:
        """
        ss = np.zeros((2, self.K))

        if data:
            # ss[0,k] = sum_t Z0[t,k]
            ss[0,:] = data.sum(axis=0)
            # ss[1,k] = T * dt
            T = data.shape[0]
            ss[1,:] = T * self.dt

        return ss

    def resample(self,data=[]):
        """
        Resample the background rate from its gamma conditional distribution.

        :param data: Z0, a TxK matrix of events assigned to the background.
        """

        assert len(data) == 0 or (isinstance(data, np.ndarray)
                                  and data.ndim == 2
                                  and data.shape[1] == self.K), \
            "Data must be a TxK array of event counts assigned to the background"

        ss = self._get_suff_statistics(data)
        alpha_post = self.alpha + ss[0,:]
        beta_post  = self.beta + ss[1,:]

        self.lambda0 = np.random.gamma(alpha_post, 1.0/beta_post)
