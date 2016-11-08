import numpy as np
from scipy.special import gammaln, psi

from pybasicbayes.abstractions import GibbsSampling, MeanField, MeanFieldSVI
from pyhawkes.internals.distributions import Dirichlet
from pyhawkes.utils.utils import logistic

class DirichletImpulseResponses(GibbsSampling, MeanField, MeanFieldSVI):
    """
    Encapsulates the impulse response vector distribution. In the
    discrete time Hawkes model this is a set of Dirichlet-distributed
    vectors of length B for each pair of processes, k and k', which
    we denote $\bbeta^{(k,k')}. This class contains all K^2 vectors.
    """
    def __init__(self, model, gamma=None):
        """
        Initialize a set of Dirichlet weight vectors.
        :param K:     The number of processes in the model.
        :param B:     The number of basis functions in the model.
        :param gamma: The Dirichlet prior parameter. If none it will be set
                      to a symmetric prior with parameter 1.
        """
        # assert isinstance(model, DiscreteTimeNetworkHawkesModel), \
        #        "model must be a DiscreteTimeNetworkHawkesModel"
        self.model = model
        self.K = model.K
        self.B = model.B

        if gamma is not None:
            assert np.isscalar(gamma) or \
                   (isinstance(gamma, np.ndarray) and
                    gamma.shape == (self.B,)), \
                "gamma must be a scalar or a length B vector"

            if np.isscalar(gamma):
                self.gamma = gamma * np.ones(self.B)
            else:
                self.gamma = gamma
        else:
            self.gamma = np.ones(self.B)

        # Initialize with a draw from the prior
        self.g = np.empty((self.K, self.K, self.B))
        self.resample()

        # Initialize mean field parameters
        self.mf_gamma = self.gamma[None, None, :] * np.ones((self.K, self.K, self.B))

    @property
    def impulses(self):
        basis = self.model.basis.basis
        return np.tensordot(basis, self.g, axes=[1,2])

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
        return self.log_likelihood(self.g)

    def resample(self, data=[]):
        """
        Resample the
        """
        ss = np.zeros((self.K, self.K, self.B)) + \
             sum([d.compute_ir_ss() for d in data])

        for k1 in range(self.K):
            for k2 in range(self.K):
                alpha_post = self.gamma + ss[k1, k2, :]
                self.g[k1,k2,:] = np.random.dirichlet(alpha_post)

    def expected_g(self):
        # \sum_{b} \gamma_b
        trm2 = self.mf_gamma.sum(axis=2)
        E_g = self.mf_gamma / trm2[:,:,None]
        return E_g

    def expected_log_g(self):
        E_lng = np.zeros_like(self.mf_gamma)

        # \psi(\sum_{b} \gamma_b)
        trm2 = psi(self.mf_gamma.sum(axis=2))
        for b in range(self.B):
            E_lng[:,:,b] = psi(self.mf_gamma[:,:,b]) - trm2

        return E_lng

    def mf_update_gamma(self, data, minibatchfrac=1.0, stepsize=1.0):
        """
        Update gamma given E[Z]
        :return:
        """
        exp_ss = sum([d.compute_exp_ir_ss() for d in data])
        gamma_hat = self.gamma + exp_ss / minibatchfrac
        self.mf_gamma = (1.0 - stepsize) * self.mf_gamma + stepsize * gamma_hat

    def expected_log_likelihood(self,x):
        pass

    def meanfieldupdate(self, data=[]):
        self.mf_update_gamma(data)

    def meanfield_sgdstep(self, data, minibatchfrac, stepsize):
        self.mf_update_gamma(data, minibatchfrac=minibatchfrac, stepsize=stepsize)

    def get_vlb(self):
        """
        Variational lower bound for \lambda_k^0
        E[LN p(g | \gamma)] -
        E[LN q(g | \tilde{\gamma})]
        :return:
        """
        vlb = 0

        # First term
        # E[LN p(g | \gamma)]
        E_ln_g = self.expected_log_g()
        vlb += Dirichlet(self.gamma[None, None, :]).negentropy(E_ln_g=E_ln_g).sum()

        # Second term
        # E[LN q(g | \tilde{gamma})]
        vlb -= Dirichlet(self.mf_gamma).negentropy().sum()

        return vlb

    def resample_from_mf(self):
        """
        Resample from the mean field distribution
        :return:
        """
        self.g = np.zeros((self.K, self.K, self.B))
        for k1 in range(self.K):
            for k2 in range(self.K):
                self.g[k1,k2,:] = np.random.dirichlet(self.mf_gamma[k1,k2,:])


class SBMDirichletImpulseResponses(GibbsSampling):
    """
    A impulse response vector model with a set of Dirichlet-distributed
    vectors of length B for each pair of blocks, c and c', which
    we denote $\bbeta^{(c,c')}. This class contains all C^2 vectors.
    """
    def __init__(self, C, K, B, gamma=None):
        """
        Initialize a set of Dirichlet weight vectors.
        :param K:     The number of processes in the model.
        :param B:     The number of basis functions in the model.
        :param gamma: The Dirichlet prior parameter. If none it will be set
                      to a symmetric prior with parameter 1.
        """
        self.C = C
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
        self.blockg = np.empty((self.C, self.C, self.B))
        self.resample()

        # Initialize mean field parameters
        self.mf_gamma = self.gamma[None, None, :] * np.ones((self.C, self.C, self.B))

    def rvs(self, size=[]):
        """
        Sample random variables from the Dirichlet impulse response distribution.
        :param size:
        :return:
        """
        raise NotImplementedError()

    def log_likelihood(self, x):
        '''
        log likelihood (either log probability mass function or log probability
        density function) of x, which has the same type as the output of rvs()
        '''
        raise NotImplementedError()
        assert isinstance(x, np.ndarray) and x.shape == (self.K,self.K,self.B), \
            "x must be a KxKxB array of impulse responses"

        gamma = self.gamma
        # Compute the normalization constant
        Z = gammaln(gamma).sum() - gammaln(gamma.sum())
        # Add the likelihood of x
        return self.K**2 * Z + ((gamma-1.0)[None,None,:] * np.log(x)).sum()

    def log_probability(self):
        return self.log_likelihood(self.g)

    def _get_suff_statistics(self, data):
        """
        Compute the sufficient statistics from the data set.
        :param data: a TxK array of event counts assigned to the background process
        :return:
        """
        raise NotImplementedError()
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
        raise NotImplementedError()
        assert data is None or \
               (isinstance(data, np.ndarray) and
                data.ndim == 4 and
                data.shape[1] == data.shape[2] == self.K
                and data.shape[3] == self.B), \
            "Data must be a TxKxKxB array of parents"


        ss = self._get_suff_statistics(data)
        for k1 in range(self.K):
            for k2 in range(self.K):
                alpha_post = self.gamma + ss[k1, k2, :]
                self.g[k1,k2,:] = np.random.dirichlet(alpha_post)

    def expected_g(self):
        """
        Compute the expected impulse response vector wrt c and mf_gamma
        :return:
        """
        raise NotImplementedError()
        # \sum_{b} \gamma_b
        trm2 = self.mf_gamma.sum(axis=2)
        E_g = self.mf_gamma / trm2[:,:,None]
        return E_g

    def expected_log_g(self):
        """
        Compute the expected log impulse response vector wrt c and mf_gamma
        :return:
        """
        raise NotImplementedError()
        E_lng = np.zeros_like(self.mf_gamma)

        # \psi(\sum_{b} \gamma_b)
        trm2 = psi(self.mf_gamma.sum(axis=2))
        for b in range(self.B):
            E_lng[:,:,b] = psi(self.mf_gamma[:,:,b]) - trm2

        return E_lng

    def mf_update_gamma(self, EZ):
        """
        Update gamma given E[Z]
        :return:
        """
        raise NotImplementedError()
        self.mf_gamma = self.gamma + EZ.sum(axis=0)

    def expected_log_likelihood(self,x):
        raise NotImplementedError()
        pass

    def meanfieldupdate(self, EZ):
        raise NotImplementedError()
        self.mf_update_gamma(EZ)

    def get_vlb(self):
        """
        Variational lower bound for \lambda_k^0
        E[LN p(g | \gamma)] -
        E[LN q(g | \tilde{\gamma})]
        :return:
        """
        raise NotImplementedError()
        vlb = 0

        # First term
        # E[LN p(g | \gamma)]
        E_ln_g = self.expected_log_g()
        vlb += Dirichlet(self.gamma[None, None, :]).negentropy(E_ln_g=E_ln_g).sum()

        # Second term
        # E[LN q(g | \tilde{gamma})]
        vlb -= Dirichlet(self.mf_gamma).negentropy().sum()

        return vlb

    def resample_from_mf(self):
        """
        Resample from the mean field distribution
        :return:
        """
        raise NotImplementedError()
        self.g = np.zeros((self.K, self.K, self.B))
        for k1 in range(self.K):
            for k2 in range(self.K):
                self.g[k1,k2,:] = np.random.dirichlet(self.mf_gamma[k1,k2,:])


class ContinuousTimeImpulseResponses(GibbsSampling):
    """
    Continuous time impulse response model with logistic normal
    impulse response functions.
    """
    def __init__(self, model, mu_0=0., lmbda_0=1., alpha_0=1., beta_0=1.):
        self.model = model
        self.K = model.K
        self.dt_max = model.dt_max

        self.mu_0 = mu_0
        self.lmbda_0 = lmbda_0
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0

        from pyhawkes.utils.utils import sample_nig
        self.mu, self.tau = \
            sample_nig(self.mu_0 * np.ones((self.K, self.K)),
                       self.lmbda_0 * np.ones((self.K, self.K)),
                       self.alpha_0 * np.ones((self.K, self.K)),
                       self.beta_0 * np.ones((self.K, self.K)))

    @property
    def impulses(self):
        N_pts = 50
        t = np.linspace(0, self.dt_max, N_pts)
        ir = np.zeros((N_pts, self.K, self.K))
        for k1 in range(self.K):
            for k2 in range(self.K):
                ir[:,k1,k2] = self.impulse(t, k1, k2)
        return t, ir

    # TODO: Rename this
    def impulse(self, dt, k1, k2):
        """
        Impulse response induced by an event on process k1 on
        the rate of process k2 at lag dt
        """
        from pyhawkes.utils.utils import logit
        mu, tau, dt_max = self.mu[k1,k2], self.tau[k1,k2], self.dt_max
        Z = dt * (dt_max - dt)/dt_max * np.sqrt(2*np.pi/tau)
        return 1./Z * np.exp(-tau/2. * (logit(dt/dt_max) - mu)**2)

    def rvs(self, size, s_pa, x_pa, c_pa, c_ch):
        """
        Sample random events
        :param size:
        :return:
        """
        mu, tau, dt_max = self.mu, self.tau, self.dt_max

        # Sample normal RVs and take the logistic of them. This is equivalent
        # to sampling uniformly from the inverse CDF
        v_ch = mu[c_pa, c_ch] + np.sqrt(1. / tau[c_pa, c_ch]) * np.random.randn(size)

        # Event times are logistic transformation of x
        s_ch = s_pa + dt_max * logistic(v_ch)

        # No event marks here
        x_ch = np.array([None] * size)

        return s_ch, x_ch

    def log_likelihood(self, x):
        '''
        log likelihood (either log probability mass function or log probability
        density function) of x, which has the same type as the output of rvs()
        '''
        return 0

    def log_probability(self):
        return self.log_likelihood((self.mu, self.tau))

    def resample(self, data=[]):
        """
        Resample the
        :param data: a TxKxKxB array of parents. T time bins, K processes,
                     K parent processes, and B bases for each parent process.
        """
        mu_0, lmbda_0, alpha_0, beta_0 = self.mu_0, self.lmbda_0, self.alpha_0, self.beta_0
        assert data is None or isinstance(data, list)

        # 0: count, # 1: Sum of scaled dt, #2: Sum of sq scaled dt
        ss = np.zeros((3, self.K, self.K))
        for d in data:
            ss += d.compute_imp_suff_stats()

        n = ss[0]
        xbar = np.nan_to_num(ss[1] / n)
        xvar = ss[2]

        alpha_post = alpha_0 + n / 2.
        # beta_post = beta_0 + 0.5 * xvar
        # beta_post += 0.5 * lmbda_0 * n / (lmbda_0 + n) * (xbar-mu_0)**2
        beta_post = beta_0 + 0.5 * xvar + 0.5 * lmbda_0 * n * (xbar-mu_0)**2 / (lmbda_0+n)

        lmbda_post = lmbda_0 + n
        mu_post = (lmbda_0 * mu_0 + n * xbar) / (lmbda_0 + n)

        from pyhawkes.utils.utils import sample_nig
        self.mu, self.tau = \
            sample_nig(mu_post, lmbda_post, alpha_post, beta_post)

        assert np.isfinite(self.mu).all()
        assert np.isfinite(self.tau).all()

class SpatioTemporalImpulseResponses(ContinuousTimeImpulseResponses):
    """
    Extend the simple temporal impulse response with a Gaussian kernel
    in space. I.e.

        h(t, x | t', x') = h(t - t') g(x - x')

    where h(t-t') is the standard temporal kernel, and g(x-x') is a
    stationary Gaussian kernel. That is,

       g(x - x') = N(x - x' | 0, sigma)

    """

    def __init__(self, model, sigma=0.5, **kwargs):
        super(SpatioTemporalImpulseResponses, self).__init__(model, **kwargs)
        self.sigma = sigma * np.ones((self.K, self.K))

    def impulse(self, dt, dx, k1, k2):
        """
        Impulse response induced by an event on process k1 on
        the rate of process k2 at lag dt
        """
        from pyhawkes.utils.utils import logit
        mu, tau, dt_max, sigma = self.mu[k1, k2], self.tau[k1, k2], self.dt_max, self.sigma[k1,k2]
        Z = dt * (dt_max - dt) / dt_max * np.sqrt(2 * np.pi / tau)
        p_dt = 1. / Z * np.exp(-tau / 2. * (logit(dt / dt_max) - mu) ** 2)

        p_dx = 1./np.sqrt(2 * np.pi * sigma**2) * np.exp(-0.5 * dx**2 / sigma**2)

        return p_dt * p_dx

    def rvs(self, size, s_pa, x_pa, c_pa, c_ch):
        """
        Sample random events
        :param size:
        :return:
        """
        mu, tau, dt_max = self.mu, self.tau, self.dt_max

        # Sample normal RVs and take the logistic of them. This is equivalent
        # to sampling uniformly from the inverse CDF
        v_ch = mu[c_pa, c_ch] + np.sqrt(1. / tau[c_pa, c_ch]) * np.random.randn(size)

        # Event times are logistic transformation of x
        s_ch = s_pa + dt_max * logistic(v_ch)

        # No event marks here
        x_ch = x_pa + self.sigma[c_pa, c_ch] * np.random.randn(size)

        return s_ch, x_ch

    # TODO: Resample kernel parameters