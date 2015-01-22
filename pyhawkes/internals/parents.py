import numpy as np

from pyhawkes.deps.pybasicbayes.distributions import BayesianDistribution, GibbsSampling, MeanField

class _ParentsBase(BayesianDistribution):
    """
    Encapsulates the TxKxKxB array of parent multinomial distributed
    parent variables.
    """
    def __init__(self, T, K, B, S, F):
        """
        Initialize a parent array Z of size TxKxKxB to model the
        event parents for data matrix S (TxK) which has been filtered
        to create filtered data array F (TxKxB).

        Also create a background parent array of size TxK to specify
        how many events are attributed to the background process.

        :param T: Number of time bins
        :param K: Number of processes
        :param B: Number of basis functions
        :param S: Data matrix (TxK)
        :param F: Filtered data matrix (TxKxB)
        """
        self.T = T
        self.K = K
        self.B = B
        self.S = S
        self.F = F

    def log_likelihood(self, x):
        pass

    def rvs(self, data=[]):
        raise NotImplementedError("No prior for parents to sample from.")


class GibbsParents(_ParentsBase, GibbsSampling):
    """
    Parent distribution with Gibbs sampling
    """
    def __init__(self, T, K, B, S, F):
        super(GibbsParents, self).__init__(T, K, B, S, F)

        # Initialize parent arrays for Gibbs sampling
        # Attribute all events to the background.
        self.Z  = np.zeros((T,K,K,B), dtype=np.int32)
        self.Z0 = np.copy(self.S).astype(np.int32)

    def _check_Z(self):
        """
        Check that Z adds up to the correct amount
        :return:
        """
        Zsum = self.Z0 + self.Z.sum(axis=(1,3))
        # assert np.allclose(self.S, Zsum), "_check_Z failed. Zsum does not add up to S!"
        if not np.allclose(self.S, Zsum):
            print "_check_Z failed. Zsum does not add up to S!"
            import pdb; pdb.set_trace()

    def _resample_Z_python(self, bias_model, weight_model, impulse_model):
        """
        Resample the parents in python.
        """
        for t in xrange(self.T):
            for k2 in xrange(self.K):
                # If there are no events then there's nothing to do
                if self.S[t, k2] == 0:
                    continue

                # Compute the normalized probability vector for the background rate and
                # each of the basis functions for every other process
                p0  = np.atleast_1d(bias_model.lambda0[k2])         # (1,)
                Ak2 = weight_model.A[:,k2]                          # (K,)
                Wk2 = weight_model.W[:,k2]                          # (K,)
                Bk2 = impulse_model.beta[:,k2,:]                    # (K,B)
                Ft  =  self.F[t,:,:]                                # (K,B)
                pkb = Ft * Ak2[:,None] * Wk2[:,None] * Bk2

                assert pkb.shape == (self.K, self.B)

                # Combine the probabilities into a normalized vector of length KB+1
                p = np.concatenate([p0, pkb.reshape((self.K*self.B,))])
                p = p / p.sum()

                # Sample a multinomial distribution to assign events to parents
                parents = np.random.multinomial(self.S[t, k2], p)

                # Copy the parents back into Z
                self.Z0[t,k2] = parents[0]
                self.Z[t,:,k2,:] = parents[1:].reshape((self.K, self.B))

        # DEBUG
        self._check_Z()

    def resample(self, bias_model, weight_model, impulse_model):
        """
        Resample the parents given the bias_model, weight_model, and impulse_model.

        :param bias_model:
        :param weight_model:
        :param impulse_model:
        :return:
        """

        # TODO: Write a Cython function to resample each of the TxKxKxB entries in Z
        self._resample_Z_python(bias_model, weight_model, impulse_model)


class MeanFieldParents(_ParentsBase, MeanField):
    """
    Parent distribution with Gibbs sampling
    """
    def __init__(self, T, K, B, S, F):
        super(MeanFieldParents, self).__init__(T, K, B, S, F)

        # Initialize arrays for mean field parameters
        self.u  = np.zeros((T,K,K,B))
        self.u0 = np.ones_like(self.S)

    def expected_Z(self):
        """
        E[z] = u[t,k,k',b] * s[t,k']
        :return:
        """
        # TODO: Just store E[Z] directly, we never just need u
        return self.u * self.S[:,None,:,None]

    def expected_Z0(self):
        """
        E[z0] = u0[t,k'] * s[t,k']
        :return:
        """
        # TODO: Just store E[Z0] directly, we never just need u
        return self.u0 * self.S

    def expected_log_likelihood(self,x):
        pass

    def meanfieldupdate(self,data,weights):
        pass

    def get_vlb(self):
        raise NotImplementedError

