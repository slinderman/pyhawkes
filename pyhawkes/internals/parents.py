import numpy as np

from pyhawkes.deps.pybasicbayes.distributions import BayesianDistribution, GibbsSampling, MeanField
from pyhawkes.internals.parent_updates import mf_update_Z, resample_Z

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
                Gk2 = impulse_model.g[:,k2,:]                    # (K,B)
                Ft  =  self.F[t,:,:]                                # (K,B)
                pkb = Ft * Ak2[:,None] * Wk2[:,None] * Gk2

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

        # Resample the parents in python
        # self._resample_Z_python(bias_model, weight_model, impulse_model)

        # Call cython function to resample parents
        lambda0 = bias_model.lambda0
        W = weight_model.A * weight_model.W
        g = impulse_model.g
        F = self.F
        resample_Z(self.Z0, self.Z, self.S, lambda0, W, g, F)

        self._check_Z()

class MeanFieldParents(_ParentsBase, MeanField):
    """
    Parent distribution with Gibbs sampling
    """
    def __init__(self, T, K, B, S, F):
        super(MeanFieldParents, self).__init__(T, K, B, S, F)

        # Initialize arrays for mean field parameters
        self.EZ  = np.zeros((T,K,K,B))
        self.EZ0 = np.copy(self.S).astype(np.float)

    def _check_EZ(self):
        """
        Check that Z adds up to the correct amount
        :return:
        """
        EZsum = self.EZ0 + self.EZ.sum(axis=(1,3))
        # assert np.allclose(self.S, Zsum), "_check_Z failed. Zsum does not add up to S!"
        if not np.allclose(self.S, EZsum):
            print "_check_Z failed. Zsum does not add up to S!"
            import pdb; pdb.set_trace()

    def expected_Z(self):
        """
        E[z] = u[t,k,k',b] * s[t,k']
        :return:
        """
        # We store E[Z] directly, we never just need u
        return self.EZ

    def expected_Z0(self):
        """
        E[z0] = u0[t,k'] * s[t,k']
        :return:
        """
        # We store E[Z0] directly, we never just need u
        return self.EZ0

    def expected_log_likelihood(self,x):
        pass

    def _mf_update_Z_python(self, bias_model, weight_model, impulse_model):
        """
        Update the mean field parameters for the latent parents
        :return:
        """
        for t in xrange(self.T):
            for k2 in xrange(self.K):
                # If there are no events then there's nothing to do
                if self.S[t, k2] == 0:
                    continue

                # Compute the normalized probability vector for the background rate and
                # each of the basis functions for every other process
                p0  = np.exp(bias_model.expected_log_lambda0()[k2])     # scalar
                Wk2 = np.exp(weight_model.expected_log_W()[:,k2])       # (K,)
                Gk2 = np.exp(impulse_model.expected_log_g()[:,k2,:])    # (K,B)
                Ft  =  self.F[t,:,:]                                    # (K,B)
                pkb = Ft * Wk2[:,None] * Gk2

                assert pkb.shape == (self.K, self.B)

                # Combine the probabilities into a normalized vector of length KB+1
                Z = p0 + pkb.sum()
                self.EZ0[t,k2] = p0 / Z * self.S[t,k2].astype(float)
                self.EZ[t,:,k2,:] = pkb.reshape((self.K,self.B)) / Z * self.S[t,k2].astype(float)

    def meanfieldupdate(self, bias_model, weight_model, impulse_model):
        """
        Perform the mean field update.

        :param bias_model:
        :param weight_model:
        :param impulse_model:
        :return:
        """

        # Uncomment this line to use python
        # self._mf_update_Z_python(bias_model, weight_model, impulse_model)

        # Use Cython
        exp_E_log_lambda0 = np.exp(bias_model.expected_log_lambda0())
        exp_E_log_W       = np.exp(weight_model.expected_log_W())
        exp_E_log_g       = np.exp(impulse_model.expected_log_g())
        mf_update_Z(self.EZ0, self.EZ, self.S,
                    exp_E_log_lambda0,
                    exp_E_log_W,
                    exp_E_log_g,
                    self.F)

        # self._check_EZ()

    def get_vlb(self, bias_model, weight_model, impulse_model):
        """
        E_q[\ln p(z | \lambda)] - E_q[\ln q(z)]
        :return:
        """
        vlb = 0
        # First term
        # E[LN p(z_tk^0 | \lambda_0)] = - LN z_tk^0! + z_tk^0 * LN \lambda_0 - \lambda_0
        # The factorial cancels with the second term
        E_ln_lam = bias_model.expected_log_lambda0()
        E_lam = bias_model.expected_lambda0()
        vlb += (self.EZ0 * E_ln_lam[None, :]).sum()
        vlb += (-self.T * E_lam).sum()

        # Second term
        # -E[LN q(z_tk^0)] = -LN s_tk! + LN z_tk^0! - z_tk^0 LN u_tk^0
        # The factorial of z cancels with the first term
        # The factorial of s is const wrt z
        ln_u0 = np.log(self.EZ0 / self.S.astype(np.float))
        ln_u0 = np.nan_to_num(ln_u0)
        vlb += (-self.EZ0 * ln_u0).sum()

        # Now do the same for the weighted impulse responses
        # First term
        E_ln_Wg = np.log(self.F[:,:,None,:]) + \
                  weight_model.expected_log_W()[None,:,:,None] + \
                  impulse_model.expected_log_g()[None,:,:,:]
        E_ln_Wg = np.nan_to_num(E_ln_Wg)

        E_Wg    = self.F[:,:,None,:] * \
                  weight_model.expected_W()[None,:,:,None] * \
                  impulse_model.expected_g()[None,:,:,:]

        assert E_ln_Wg.shape == (self.T, self.K, self.K, self.B)
        assert E_Wg.shape == (self.T, self.K, self.K, self.B)
        vlb += (self.EZ * E_ln_Wg[None,:,:,:]).sum()
        vlb += -E_Wg.sum()

        # Second term
        ln_u = np.log(self.EZ / self.S[:,None,:,None].astype(np.float))
        ln_u = np.nan_to_num(ln_u)
        vlb += (-self.EZ * ln_u).sum()

        return vlb



class Parents(GibbsParents, MeanFieldParents):
    pass
