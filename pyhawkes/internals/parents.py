import numpy as np

from pybasicbayes.distributions import GibbsSampling, MeanField
from gslrandom import multinomial_par, multinomial

from pyhawkes.utils.utils import initialize_pyrngs
from pyhawkes.internals.parent_updates import mf_update_Z
from pyhawkes.internals.continuous_time_helpers import ct_resample_Z_logistic_normal, ct_compute_suff_stats

from pyhawkes.utils.profiling import line_profiled
PROFILING = True


class DiscreteTimeParents(GibbsSampling, MeanField):
    """
    Encapsulates the TxKxKxB array of parent multinomial distributed
    parent variables.
    """
    def __init__(self, model, T, S, F):
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
        self.model = model
        self.dt = model.dt
        self.K = model.K
        self.B = model.B

        # TODO: Remove dependencies on S and F
        self.T = T
        self.S = S
        self.F = F

        # Save sparse versions of S and F
        self.ts = []
        self.Ts = []
        self.Ns = []
        self.Ss = []
        self.Fs = []
        for k in xrange(self.K):
            # Find the rows where S[:,k] is nonzero
                tk = np.where(S[:,k])[0]
                self.ts.append(tk)
                self.Ts.append(len(tk))
                self.Ss.append(S[tk,k].astype(np.uint32))
                self.Ns.append(S[tk,k].sum())
                self.Fs.append(F[tk])


        # The base class handles the parent variables
        # We use a sparse representation that only considers times (rows)
        # where there is a spike
        self._Z = None
        self._EZ = None

        # Initialize GSL RNGs for resampling Z
        self.pyrngs = initialize_pyrngs()

    @property
    def Z(self):
        if self._Z is None:
            self._Z = []
            for Tk in self.Ts:
                self._Z.append(np.zeros((Tk, 1+self.K*self.B), dtype=np.uint32))

        return self._Z

    @property
    def EZ(self):
        if self._EZ is None:
            self._EZ = []
            for Tk in self.Ts:
                self._EZ.append(np.zeros((Tk, 1+self.K*self.B)))

        return self._EZ

    # Debugging helper functions
    def _check_Z(self):
        """
        Check that Z adds up to the correct amount
        :return:
        """
        for Sk, Zk in zip(self.Ss, self.Z):
            assert (Sk == Zk.sum(1)).all()

    def _check_EZ(self):
        """
        Check that Z adds up to the correct amount
        :return:
        """
        for Sk, Zk in zip(self.Ss, self.Z):
            assert np.allclose(Sk, Zk.sum(1))

    def log_likelihood(self):
        """
        Compute the *marginal* log likelihood by summing over
        parent assignments. In practice, this just means compute
        the total area under the rate function (an easy sum) and
        the instantaneous rate at the time of spikes.
        """
        ll = 0
        for k in xrange(self.K):
            ll += self.log_likelihood_single_process(k)
        return ll

    def log_likelihood_single_process(self, k2):
        """
        Compute the *marginal* log likelihood by summing over
        parent assignments. In practice, this just means compute
        the total area under the rate function (an easy sum) and
        the instantaneous rate at the time of spikes.
        """
        lambda0 = self.model.bias_model.lambda0
        W = self.model.weight_model.W_effective
        g = self.model.impulse_model.g

        T, K, B, dt = self.T, self.K, self.B, self.dt
        Sk, Fk, Tk = self.Ss[k2], self.Fs[k2], self.Ts[k2]

        ll = 0
        # Compute the integrated rate
        # Each event induces a weighted impulse response
        ll += -lambda0[k2] * T * dt
        ll += -(W[:,k2] * self.Ns).sum()

        # Compute the instantaneous log rate
        Wk2 = W[:,k2][None,:,None] #(1,K,1)
        Gk2 = g[:,k2,:][None,:,:] # (1,K,B)
        lam = lambda0[k2] + (Wk2 * Gk2 * Fk).sum(axis=(1,2))

        ll += (Sk * np.log(lam)).sum()
        return ll


    def rvs(self, data=[]):
        raise NotImplementedError("No prior for parents to sample from.")


    # Compute sufficient statistics
    def compute_bkgd_ss(self):
        # \sum_{t} z_{t,k}^{0} and T * dt
        ss = np.zeros((2, self.K))

        ss[1,:] = self.T * self.dt
        for k,Zk in enumerate(self.Z):
            ss[0,k] = Zk[:,0].sum()
        return ss

    def compute_weight_ss(self):
        K, B = self.K, self.B
        ss = np.zeros((2, self.K, self.K))
        for k2, Zk in enumerate(self.Z):
            # ss[0,k1,k2] = \sum_t \sum_b Z_{t,k2}^{k1,b}
            ss[0,:,k2] = Zk[:,1:].sum(0).reshape((K,B)).sum(1)
            # ss[1,k1,k2] = N[k1] (to be multiplied by A)
            ss[1,:,k2] = self.Ns
        return ss

    def compute_ir_ss(self):
        """
        Compute the sufficient statistics for the impulse responses
        """
        K, B = self.K, self.B
        # ss[k1,k2,b] = \sum_t z_{t,k2}^{k1,b}
        ss = np.zeros((self.K, self.K, self.B))
        for k2, Zk in enumerate(self.Z):
            ss[:,k2,:] = Zk[:,1:].sum(0).reshape((K,B))
        return ss

    def _resample_Z_python(self):
        """
        Resample the parents in python.
        """
        bias_model, weight_model, impulse_model = \
            self.model.bias_model, self.model.weight_model, self.model.impulse_model

        # TODO: This can all be done with multinomial_par!
        for k2, (Sk, Fk, Zk) in enumerate(zip(self.Ss, self.Fs, self.Z)):
            for st,ft,zt in zip(Sk, Fk, Zk):
                assert st > 0

                # Compute the normalized probability vector for the background rate and
                # each of the basis functions for every other process
                p = np.zeros(1 + self.K * self.B)
                p[0]  = bias_model.lambda0[k2]                  # Background
                Wk2 = weight_model.W_effective[:,k2]            # (K,)
                Gk2 = impulse_model.g[:,k2,:]                   # (K,B)
                p[1:] = (ft *  Wk2[:,None] * Gk2).ravel()

                # Normalize
                p = p / p.sum()

                # Sample a multinomial distribution to assign events to parents
                zt[:] = np.random.multinomial(st, p)

        self._check_Z()

    def _resample_Z_gsl(self, data=[]):
        """
        Resample the parents given the bias_model, weight_model, and impulse_model.

        :param bias_model:
        :param weight_model:
        :param impulse_model:
        :return:
        """
        bias_model, weight_model, impulse_model = \
            self.model.bias_model, self.model.weight_model, self.model.impulse_model
        K, B = self.K, self.B
        # Make a big matrix of size T x (K*B + 1)
        for k2, (Sk, Fk, Tk, Zk) in enumerate(zip(self.Ss, self.Fs, self.Ts, self.Z)):
            P = np.zeros((Tk, 1+self.K * self.B))
            P[:,0] = bias_model.lambda0[k2]

            Wk2 = np.repeat(weight_model.W_effective[:,k2], B)
            Gk2 = impulse_model.g[:,k2,:].reshape((K*B,), order="C")
            P[:,1:] = Wk2 * Gk2 * Fk.reshape((Tk, K*B))

            # Normalize the rows
            P = P / P.sum(1)[:,None]

            # Sample parents from P with counts S[:,k2]
            # multinomial_par(self.pyrngs, Sk, P, Zk)
            multinomial(self.pyrngs[0], Sk, P, out=Zk)

        # DEBUG
        # self._check_Z()

    def resample(self,data=[]):
        self._resample_Z_python()
        # self._resample_Z_gsl()

    ### Mean Field
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
        # If necessary, initialize arrays for mean field parameters
        if self.EZ is None:
            self.EZ  = np.zeros((self.T,self.K,self.K,self.B))
        if self.EZ0 is None:
            self.EZ0 = np.copy(self.S).astype(np.float)

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


class ContinuousTimeParents(GibbsSampling):
    """
    Implementation of spike and slab categorical parents
    (one for each event). We only need to keep track of
    which process the parent spike comes from and the time
    elapsed between parent. For completeness, we also
    keep track of the index of the parent.
    """
    def __init__(self, model, S, C, T, K, dt_max):
        """
        :param S: Length N array of event times
        :param C: Length N array of process indices
        :param K: The number of processes
        """
        self.model = model

        assert S.ndim == 1 and S.shape == C.shape
        assert C.dtype == np.int and C.min() >= 0 and C.max() < K
        self.S = S
        self.C = C
        self.T = T
        self.K = K
        self.N = S.size
        self.Ns = np.bincount(C, minlength=self.K)
        self.dt_max = dt_max

        # Initialize parent arrays for Gibbs sampling
        self.Z = -1 * np.ones((self.N,), dtype=np.int)
        self.bkgd_ss = self.Ns.copy()
        self.weight_ss = np.zeros((self.K, self.K))
        self.imp_ss = np.zeros((self.K, self.K))

    def log_likelihood(self, x):
        pass

    def rvs(self, data=[]):
        raise NotImplementedError("No prior for parents to sample from.")

    def resample(self):
        # self.resample_Z_python()

        S, C, Z, dt_max = self.S, self.C, self.Z, self.dt_max
        lambda0 = self.model.bias_model.lambda0
        W = self.model.weight_model.W
        mu, tau = self.model.impulse_model.mu, self.model.impulse_model.tau

        ct_resample_Z_logistic_normal(
            S, C, Z, dt_max,
            lambda0, W, mu, tau)

        assert (Z > -2).all()
        assert (Z < np.arange(Z.shape[0])).all()

        # Update sufficient statistics
        self.bkgd_ss = np.zeros(self.K)
        self.weight_ss = np.zeros((self.K, self.K))
        self.imp_ss = np.zeros((3, self.K, self.K))
        ct_compute_suff_stats(S, C, Z, dt_max,
                              self.bkgd_ss, self.weight_ss, self.imp_ss)

        assert (self.bkgd_ss + self.weight_ss.sum(0) == self.Ns).all()

    def resample_Z_python(self):
        from pybasicbayes.util.stats import sample_discrete

        # TODO: Call cython function to resample parents
        S, C, Z, dt_max = self.S, self.C, self.Z, self.dt_max
        lambda0 = self.model.bias_model.lambda0
        W = self.model.weight_model.W
        impulse = self.model.impulse_model.impulse

        # Also compute number of parents assigned to background rate and
        # to specific connections
        self.bkgd_ss = np.zeros(self.K)
        self.weight_ss = np.zeros((self.K, self.K))
        self.imp_ss = np.zeros((self.K, self.K))

        # Resample parents
        for n in xrange(self.N):

            if n == 0:
                Z[n] = -1
                self.bkgd_ss[C[n]] += 1
                continue

            # Compute the probability of each parent spike
            p_par = np.zeros(n)
            denom = 0

            # First parent is just the background rate of this process
            p_bkgd = lambda0[C[n]]
            denom += p_bkgd

            # Iterate backward from the most recent to compute probabilities of each parent spike
            for par in xrange(n-1, -1, -1):
                dt = S[n] - S[par]

                # Since the spikes are sorted, we can stop if we reach a potential
                # parent that occurred greater than dt_max in the past
                if dt > dt_max:
                    p_par[par] = 0
                    break

                p_par[par] = W[C[par], C[n]] * impulse(dt, C[par], C[n])
                denom += p_par[par]

            # Now sample forward, starting from the minimum viable parent
            min_par = par
            p_par = np.concatenate([[p_bkgd], p_par[min_par:n]])

            # Sample from the discrete distribution p_par
            i_par = sample_discrete(p_par)

            if i_par == 0:
                # Sampled the background rate
                Z[n] = -1
                self.bkgd_ss[C[n]] += 1

            else:
                # Sampled one of the preceding spikes
                Z[n] = (i_par - 1) + min_par
                Cp = C[Z[n]]
                dt = S[n] - S[Z[n]]

                self.weight_ss[Cp, C[n]] += 1
                self.imp_ss[Cp, C[n]] += np.log(dt) - np.log(dt_max - dt)

    def compute_imp_suff_stats(self):
        S, C, Z, dt_max = self.S, self.C, self.Z, self.dt_max

        # 0: count, # 1: Sum of scaled dt, #2: Sum of sq scaled dt

        # Compute manually
        # imp_ss_manual = np.zeros((3, self.K, self.K))
        # for n in xrange(self.N):
        #     par = Z[n]
        #     if par > -1:
        #         dt = S[n] - S[par]
        #         imp_ss_manual[0, C[par], C[n]] += 1
        #         imp_ss_manual[1, C[par], C[n]] += np.log(dt) - np.log(dt_max - dt)
        #
        # # In a second pass, compute the sum of squares
        # mu = imp_ss_manual[1] / (imp_ss_manual[0] + 1e-64)
        # for n in xrange(self.N):
        #     par = Z[n]
        #     if par > -1:
        #         dt = S[n] - S[par]
        #         sdt = np.log(dt) - np.log(dt_max - dt)
        #         imp_ss_manual[2, C[par], C[n]] += (sdt - mu[C[par], C[n]])**2
        #
        # assert np.isfinite(imp_ss_manual).all()

        # Compute with cython
        # imp_ss = np.zeros((3, self.K, self.K))
        # from pyhawkes.internals.continuous_time_helpers import ct_compute_imp_suff_stats
        # ct_compute_imp_suff_stats(S, C, Z, dt_max, imp_ss)

        # assert np.allclose(imp_ss_manual, imp_ss2)


        return self.imp_ss
