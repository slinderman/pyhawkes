import numpy as np
from scipy.special import gammaln, psi

# TODO: define distribution base class


class Bernoulli:
    def __init__(self, p=0.5):
        assert np.all(p >= 0) and np.all(p <= 1.0)
        self.p = p

    def log_probability(self, x):
        """
        Log probability of x given p

        :param x:
        :return:
        """
        lp = x * np.log(self.p) + (1-x) * np.log(1.0-self.p)
        lp = np.nan_to_num(lp)
        return lp

    def expected_x(self):
        return self.p

    def expected_notx(self):
        return 1 - self.p

    def negentropy(self, E_x=None, E_notx=None, E_ln_p=None, E_ln_notp=None):
        """
        Compute the entropy of the gamma distribution.

        :param E_x:         If given, use this in place of expectation wrt p
        :param E_notx:      If given, use this in place of expectation wrt p
        :param E_ln_p:      If given, use this in place of expectation wrt p
        :param E_ln_notp:   If given, use this in place of expectation wrt p

        :return: E[ ln p(x | p)]
        """
        if E_x is None:
            E_x = self.expected_x()

        if E_notx is None:
            E_notx = self.expected_notx()

        if E_ln_p is None:
            E_ln_p = np.log(self.p)

        if E_ln_notp is None:
            E_ln_notp = np.log(1.0 - self.p)

        H = E_x * E_ln_p + E_notx * E_ln_notp
        return H


class Gamma:

    def __init__(self, alpha, beta=1.0):
        assert np.all(alpha) >= 0
        assert np.all(beta) >= 0
        self.alpha = alpha
        self.beta = beta

    def log_probability(self, lmbda):
        """
        Log probability of x given p

        :param x:
        :return:
        """
        lp = self.alpha * np.log(self.beta) - gammaln(self.alpha) \
             + (self.alpha-1) * np.log(lmbda) - self.beta * lmbda
        lp = np.nan_to_num(lp)
        return lp

    def expected_lambda(self):
        return self.alpha / self.beta

    def expected_log_lambda(self):
        return psi(self.alpha) - np.log(self.beta)

    def negentropy(self, E_ln_lambda=None, E_lambda=None, E_beta=None, E_ln_beta=None):
        """
        Compute the entropy of the gamma distribution.

        :param E_ln_lambda: If given, use this in place of expectation wrt alpha and beta
        :param E_lambda:    If given, use this in place of expectation wrt alpha and beta
        :param E_ln_beta:   If given, use this in place of expectation wrt alpha and beta
        :param E_beta:      If given, use this in place of expectation wrt alpha and beta
        :return: E[ ln p(\lambda | \alpha, \beta)]
        """
        if E_ln_lambda is None:
            E_ln_lambda = self.expected_log_lambda()

        if E_lambda is None:
            E_lambda = self.expected_lambda()

        if E_ln_beta is None:
            E_ln_beta = np.log(self.beta)

        if E_beta is None:
            E_beta = self.beta

        H =  self.alpha * E_ln_beta
        H += -gammaln(self.alpha)
        H += (self.alpha - 1) * E_ln_lambda
        H += -E_beta * E_lambda

        return H


class Dirichlet(object):
    def __init__(self, gamma):
        assert np.all(gamma) >= 0 and gamma.shape[-1] >= 1
        self.gamma = gamma

    def log_probability(self, x):
        assert np.allclose(x.sum(axis=-1), 1.0) and np.amin(x) >= 0.0
        return gammaln(self.gamma.sum()) - gammaln(self.gamma).sum() \
               + ((self.gamma-1) * np.log(x)).sum(axis=-1)

    def expected_g(self):
        return self.gamma / self.gamma.sum(axis=-1, keepdims=True)

    def expected_log_g(self):
        return psi(self.gamma) - psi(self.gamma.sum(axis=-1, keepdims=True))

    def negentropy(self, E_ln_g=None):
        """
        Compute the entropy of the gamma distribution.

        :param E_ln_g:    If given, use this in place of expectation wrt tau1 and tau0
        :return: E[ ln p(g | gamma)]
        """
        if E_ln_g is None:
            E_ln_g = self.expected_log_g()

        H =  gammaln(self.gamma.sum(axis=-1, keepdims=True)).sum()
        H += -gammaln(self.gamma).sum()
        H += ((self.gamma - 1) * E_ln_g).sum()
        return H


class Beta(Dirichlet):
    def __init__(self, tau1, tau0):
        tau1 = np.atleast_1d(tau1)
        tau0 = np.atleast_1d(tau0)
        gamma = np.concatenate((tau1[...,None], tau0[...,None]), axis=-1)
        super(Beta, self).__init__(gamma)

    def log_probability(self, p):
        x = np.concatenate((p[...,None], 1-p[...,None]), axis=-1)
        return super(Beta, self).log_probability(x)

    def expected_p(self):
        E_g = self.expected_g()
        return E_g[...,0]

    def expected_log_p(self):
        E_logg = self.expected_log_g()
        return E_logg[...,0]

    def expected_log_notp(self):
        E_logg = self.expected_log_g()
        return E_logg[...,1]

    def negentropy(self, E_ln_p=None, E_ln_notp=None):
        if E_ln_p is not None and E_ln_notp is not None:
            E_ln_g = np.concatenate((E_ln_p[...,None], E_ln_notp[...,None]), axis=-1)
        else:
            E_ln_g = None

        return super(Beta, self).negentropy(E_ln_g=E_ln_g)
