import numpy as np
from scipy.special import gammaln, psi

from pybasicbayes.abstractions import GibbsSampling, MeanField, MeanFieldSVI
from pyhawkes.internals.distributions import Gamma

class GammaBias(GibbsSampling, MeanField, MeanFieldSVI):
    """
    Encapsulates the vector of K gamma-distributed bias variables.
    """
    def __init__(self, model, alpha, beta):
        """
        Initialize a bias vector for each of the K processes.

        :param K:       Number of processes
        :param dt:      Bin size. This is required to put lambda0 in the right units.
        :param alpha:   Shape parameter of the gamma prior
        :param beta:    Scale parameter of the gamma prior
        """
        self.model = model
        self.K = model.K
        self.dt = model.dt
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

    def resample(self, data=[]):
        """
        Resample the background rate from its gamma conditional distribution.

        :param data: Z0, a TxK matrix of events assigned to the background.
        """
        ss = np.zeros((2, self.K)) + \
             sum([d.compute_bkgd_ss() for d in data])

        alpha_post = self.alpha + ss[0,:]
        beta_post  = self.beta + ss[1,:]

        self.lambda0 = np.array(np.random.gamma(alpha_post,
                                                1.0/beta_post)).reshape((self.K, ))

    ### Mean Field
    def expected_lambda0(self):
        return self.mf_alpha / self.mf_beta

    def expected_log_lambda0(self):
        return psi(self.mf_alpha) - np.log(self.mf_beta)

    def expected_log_likelihood(self,x):
        pass

    def mf_update_lambda0(self, data=[], minibatchfrac=1.0, stepsize=1.0):
        """
        Update background rates given expected parent assignments.
        :return:
        """
        exp_ss = sum([d.compute_exp_bkgd_ss() for d in data])
        alpha_hat = self.alpha + exp_ss[0] / minibatchfrac
        self.mf_alpha = (1-stepsize) * self.mf_alpha + stepsize * alpha_hat

        beta_hat = self.beta + exp_ss[1] / minibatchfrac
        self.mf_beta  = (1-stepsize) * self.mf_beta + stepsize * beta_hat

    def meanfieldupdate(self, data=[]):
        self.mf_update_lambda0(data)

    def meanfield_sgdstep(self, data, minibatchfrac, stepsize):
        self.mf_update_lambda0(data, minibatchfrac=minibatchfrac, stepsize=stepsize)

    def get_vlb(self):
        """
        Variational lower bound for \lambda_k^0
        E[LN p(\lambda_k^0 | \alpha, \beta)] -
        E[LN q(\lambda_k^0 | \tilde{\alpha}, \tilde{\beta})]
        :return:
        """
        vlb = 0

        # First term
        # E[LN p(\lambda_k^0 | \alpha, \beta)]
        E_ln_lambda = self.expected_log_lambda0()
        E_lambda = self.expected_lambda0()

        vlb += Gamma(self.alpha, self.beta).negentropy(E_lambda=E_lambda,
                                                       E_ln_lambda=E_ln_lambda).sum()

        # Second term
        # E[LN q(\lambda_k^0 | \alpha, \beta)]
        vlb -= Gamma(self.mf_alpha, self.mf_beta).negentropy().sum()

        return vlb


    def resample_from_mf(self):
        """
        Resample from the mean field distribution
        :return:
        """
        self.lambda0 = np.random.gamma(self.mf_alpha, 1.0/self.mf_beta)



class ContinuousTimeGammaBias(GibbsSampling):
    """
    Encapsulates the vector of K gamma-distributed bias variables.
    """
    def __init__(self, model, K, alpha, beta):
        """
        Initialize a bias vector for each of the K processes.

        :param K:       Number of processes
        :param dt:      Bin size. This is required to put lambda0 in the right units.
        :param alpha:   Shape parameter of the gamma prior
        :param beta:    Scale parameter of the gamma prior
        """
        self.model = model
        self.K = K
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

    def log_probability(self):
        return self.log_likelihood(self.lambda0)

    def rvs(self,size=[]):
        return np.random.gamma(self.alpha, 1.0/self.beta, size=(self.K,))

    ### Gibbs Sampling
    def _get_suff_statistics(self, data):
        """
        Compute the sufficient statistics from the data set.
        :param Z0: a TxK array of event counts assigned to the background process
        :return:
        """
        ss = np.zeros((2, self.K))

        for d in data:
            # ss[0,k] = sum_t Z0[t,k]
            ss[0,:] = d.bkgd_ss
            # ss[1,k] = T
            ss[1,:] = d.T

        return ss

    def resample(self,data=[]):
        """
        Resample the background rate from its gamma conditional distribution.

        :param data: Z0, a TxK matrix of events assigned to the background.
        """
        assert isinstance(data, list)
        # assert len(data) == 0 or (isinstance(data, np.ndarray)
        #                           and data.ndim == 2
        #                           and data.shape[1] == self.K), \
        #     "Data must be a TxK array of event counts assigned to the background"

        ss = self._get_suff_statistics(data)
        alpha_post = self.alpha + ss[0,:]
        beta_post  = self.beta + ss[1,:]

        self.lambda0 = np.array(np.random.gamma(alpha_post,
                                                1.0/beta_post)).reshape((self.K, ))

