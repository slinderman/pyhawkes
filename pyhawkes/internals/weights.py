import numpy as np
from scipy.special import gammaln

from pyhawkes.deps.pybasicbayes.distributions import GibbsSampling

class SpikeAndSlabGammaWeights(GibbsSampling):
    """
    Encapsulates the KxK Bernoulli adjacency matrix and the
    KxK gamma weight matrix. Implements Gibbs sampling given
    the parent variables.
    """
    def __init__(self, K, network=None, rho=None, alpha=None, beta=None):
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
        assert network is not None or None not in (rho, alpha, beta), \
            "Either the network or (rho, alpha, beta) must be specified."

        if network is not None:
            self.network = network

        else:
            # Create a network
            # TODO: Instantiate an ErdosRenyi object instead
            class _default_network:
                def __init__(self, r,a,b):
                    self.rho = r
                    self.alpha = a
                    self.beta = b

            self.network = _default_network(rho, alpha, beta)

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
        rho = self.network.rho
        ll = (A * np.log(rho) + (1-A) * np.log(1-rho)).sum()

        # TODO: For now, assume alpha and beta are fixed for all entries in W
        alpha = self.network.alpha
        beta = self.network.beta

        # Add the LL of the gamma weights
        ll += self.K**2 * (alpha * np.log(beta) - gammaln(alpha)) + \
              ((alpha-1) * np.log(W) - beta * W).sum()

        return ll

    def rvs(self,size=[]):
        A = np.random.rand(self.K, self.K) < self.network.rho
        W = np.random.gamma(self.network.alpha, 1.0/self.network.beta,
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


    def _resample_A_given_W(self):
        """
        Resample A given W. This must be immediately followed by an
        update of z | A, W.
        :return:
        """
        pass

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
            ss[1,:,:] = np.repeat(N[:,None], self.K, axis=1) * self.A

        return ss

    def resample_W_given_A_and_z(self, N, Z):
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

        ss = self._get_suff_statistics(N,Z)
        alpha_post = self.network.alpha + ss[0,:]
        beta_post  = self.network.beta + ss[1,:]

        self.W = np.array(np.random.gamma(alpha_post,
                                          1.0/beta_post)).reshape((self.K, self.K))

    def resample(self, N=None, Z=None):
        """
        Resample A and W given the parents
        :param N:   A length-K vector specifying how many events occurred
                    on each of the K processes
        :param Z:   A TxKxKxB array of parent assignment counts
        """
        self.resample_W_given_A_and_z(N, Z)

        # TODO: Resample A given W