import numpy as np
from scipy.special import gammaln, psi

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

        # Initialize mean field parameters
        self.mf_alpha = self.alpha * np.ones(self.K)
        self.mf_beta  = self.beta  * np.ones(self.K)

    def log_likelihood(self, x):
        assert isinstance(x, np.ndarray) and x.shape == (self.K,), \
            "x must be a K-vector of background rates"

        return self.K * (self.alpha * np.log(self.beta) - gammaln(self.alpha)) + \
               ((self.alpha-1) * np.log(x) - self.beta * x).sum()

    def log_probability(self):
        return self.log_likelihood(self.lambda0)

    def rvs(self,size=[]):
        return np.random.gamma(self.alpha, 1.0/self.beta, size=(self.K,))

    ### Gibbs Sampling
    def _get_suff_statistics(self, Z0):
        """
        Compute the sufficient statistics from the data set.
        :param Z0: a TxK array of event counts assigned to the background process
        :return:
        """
        ss = np.zeros((2, self.K))

        if len(Z0) > 0:
            # ss[0,k] = sum_t Z0[t,k]
            ss[0,:] = Z0.sum(axis=0)
            # ss[1,k] = T * dt
            T = Z0.shape[0]
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

        self.lambda0 = np.array(np.random.gamma(alpha_post,
                                                1.0/beta_post)).reshape((self.K, ))

    ### Mean Field
    def expected_lambda0(self):
        return self.mf_alpha / self.mf_beta

    def expected_log_lambda0(self):
        return psi(self.mf_alpha) + np.log(self.mf_beta)

    def expected_log_likelihood(self,x):
        pass

    def meanfieldupdate(self,data,weights):
        pass

    def get_vlb(self):
        raise NotImplementedError
