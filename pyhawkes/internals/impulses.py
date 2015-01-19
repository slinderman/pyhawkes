import numpy as np
from scipy.special import gammaln

from pyhawkes.deps.pybasicbayes.distributions import GibbsSampling

class DirichletImpulseResponses(GibbsSampling):
    """
    Encapsulates the impulse response vector distribution. In the
    discrete time Hawkes model this is a set of Dirichlet-distributed
    vectors of length B for each pair of processes, k and k', which
    we denote $\bbeta^{(k,k')}. This class contains all K^2 vectors.
    """
    def __init__(self, K, B, gamma=None):
        """
        Initialize a set of Dirichlet weight vectors.
        :param K:     The number of processes in the model.
        :param B:     The number of basis functions in the model.
        :param gamma: The Dirichlet prior parameter. If none it will be set
                      to a symmetric prior with parameter 1.
        """
        # assert isinstance(model, DiscreteTimeNetworkHawkesModel), \
        #        "model must be a DiscreteTimeNetworkHawkesModel"
        # self.model = model
        self.K = K
        self.B = B

        if gamma is not None:
            assert np.isscalar(gamma) or \
                   (isinstance(gamma, np.ndarray) and
                    gamma.shape == (B,)), \
                "gamma must be a scalar or a length B vector"

            if np.isscalar(gamma):
                self.gamma = gamma * np.ones(B)
            else:
                self.gamma = gamma
        else:
            self.gamma = np.ones(self.B)

        # Initialize with a draw from the prior
        self.beta = np.empty((self.K, self.K, self.B))
        self.resample()

    def rvs(self, size=[]):
        """
        Sample random variables from the Dirichlet impulse response distribution.
        :param size:
        :return:
        """
        pass

    def log_likelihood(self, x):
        '''
        log likelihood (either log probability mass function or log probability
        density function) of x, which has the same type as the output of rvs()
        '''
        assert isinstance(x, np.ndarray) and x.shape == (self.K,self.K,self.B), \
            "x must be a KxKxB array of impulse responses"

        gamma = self.gamma
        # Compute the normalization constant
        Z = gammaln(gamma).sum() - gammaln(gamma.sum())
        # Add the likelihood of x
        return self.K**2 * Z + ((gamma-1.0)[None,None,:] * np.log(x)).sum()

    def log_probability(self):
        return self.log_likelihood(self.beta)

    def _get_suff_statistics(self, data):
        """
        Compute the sufficient statistics from the data set.
        :param data: a TxK array of event counts assigned to the background process
        :return:
        """
        # The only sufficient statistic is the KxKxB array of event counts assigned
        # to each of the basis functions
        if data is not None:
            ss = data.sum(axis=0)
        else:
            ss = np.zeros((self.K, self.K, self.B))

        return ss

    def resample(self, data=None):
        """
        Resample the
        :param data: a TxKxKxB array of parents. T time bins, K processes,
                     K parent processes, and B bases for each parent process.
        """
        assert data is None or \
               (isinstance(data, np.ndarray) and
                data.ndim == 4 and
                data.shape[1] == data.shape[2] == self.K
                and data.shape[3] == self.B), \
            "Data must be a TxKxKxB array of parents"


        ss = self._get_suff_statistics(data)
        for k1 in xrange(self.K):
            for k2 in xrange(self.K):
                alpha_post = self.gamma + ss[k1, k2, :]
                self.beta[k1,k2,:] = np.random.dirichlet(alpha_post)