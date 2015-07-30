"""
This is a dummy module to facilitate joblib Parallel
 resampling of the adjacency matrix. It just keeps
 pointers to the global variables so that the parallel
 processes can reference them. No copying needed!
"""
import numpy as np
from scipy.misc import logsumexp

# Set these as module level global variables
model = None
data = None
K = 1

def _log_likelihood_single_process(k2, T, dt,
                                   lambda0, Wk2, g,
                                   Sk, Fk, Ns):
    """
    Compute the *marginal* log likelihood by summing over
    parent assignments. In practice, this just means compute
    the total area under the rate function (an easy sum) and
    the instantaneous rate at the time of spikes.
    """
    ll = 0
    # Compute the integrated rate
    # Each event induces a weighted impulse response
    ll += -lambda0[k2] * T * dt
    ll += -(Wk2 * Ns).sum()

    # Compute the instantaneous log rate
    Wk2 = Wk2[None,:,None] #(1,K,1)
    Gk2 = g[:,k2,:][None,:,:] # (1,K,B)
    lam = lambda0[k2] + (Wk2 * Gk2 * Fk).sum(axis=(1,2))

    ll += (Sk * np.log(lam)).sum()
    return ll

def _resample_column_of_A(k2):
    p = model.network.P
    K = model.K
    dt = model.dt
    lambda0 = model.lambda0
    W = model.W
    g = model.impulse_model.g

    A_col = model.A[:,k2].copy()
    W_col = W[:,k2]
    for k1 in xrange(K):
        # Compute the log likelihood of the events given W and A=0
        A_col[k1] = 0
        # ll0 = sum([d.log_likelihood_single_process(k2) for d in data])
        ll0 = 0
        for d in data:
            ll0 += _log_likelihood_single_process(k2, d.T, dt, lambda0, A_col*W_col, g, d.Ss[k2], d.Fs[k2], d.Ns)

        # Compute the log likelihood of the events given W and A=1
        A_col[k1] = 1
        # ll1 = sum([d.log_likelihood_single_process(k2) for d in data])
        ll1 = 0
        for d in data:
            ll1 += _log_likelihood_single_process(k2, d.T, dt, lambda0, A_col*W_col, g, d.Ss[k2], d.Fs[k2], d.Ns)

        # Sample A given conditional probability
        lp0 = ll0 + np.log(1.0 - p[k1,k2])
        lp1 = ll1 + np.log(p[k1,k2])
        Z   = logsumexp([lp0, lp1])

        # ln p(A=1) = ln (exp(lp1) / (exp(lp0) + exp(lp1)))
        #           = lp1 - ln(exp(lp0) + exp(lp1))
        #           = lp1 - Z
        A_col[k1] = np.log(np.random.rand()) < lp1 - Z

    return A_col


### Now do the same thing for continuous time
def _compute_weighted_impulses_at_events(data):
    from pyhawkes.internals.continuous_time_helpers import \
        compute_weighted_impulses_at_events

    N, S, C, Z, dt_max = data.N, data.S, data.C, data.Z, model.dt_max
    W = model.W
    mu, tau = model.impulse_model.mu, model.impulse_model.tau
    lmbda = np.zeros((N, model.K))
    compute_weighted_impulses_at_events(S, C, Z, dt_max, W, mu, tau, lmbda)
    return lmbda

# Precompute the weighted impulse responses
# lambda_irs = [_compute_weighted_impulses_at_events(d) for d in data]
lambda_irs = None

def _ct_log_likelihood_single_process(k, A_col):
    ll = 0
    lmbda0 = model.lambda0
    W = model.W

    for lmbda_ir, d in zip(lambda_irs, data):
        Ns, C, T = d.Ns, d.C, d.T

        # - \int lambda_k(t) dt
        ll -= lmbda0[k] * T
        ll -= (A_col * W[:,k]).dot(Ns)

        # + \sum_n log(lambda(s_n))
        ll += np.log(lmbda0[k] + np.sum(A_col * lmbda_ir[C==k,:], axis=1)).sum()
    return ll


def _ct_resample_column_of_A(k2):
    p = model.network.P
    K = model.K
    A_col = model.A[:,k2].copy()

    for k1 in xrange(K):
        if p[k1,k2] == 0:
            A_col[k1] = 0
            continue

        if p[k1,k2] == 1:
            A_col[k1] = 1
            continue

        # Compute the log likelihood of the events given W and A=0
        A_col[k1] = 0
        # ll0 = sum([d.log_likelihood_single_process(k2) for d in data])
        ll0 = _ct_log_likelihood_single_process(k2, A_col)

        # Compute the log likelihood of the events given W and A=1
        A_col[k1] = 1
        # ll1 = sum([d.log_likelihood_single_process(k2) for d in data])
        ll1 = _ct_log_likelihood_single_process(k2, A_col)

        # Sample A given conditional probability
        lp0 = ll0 + np.log(1.0 - p[k1,k2])
        lp1 = ll1 + np.log(p[k1,k2])
        Z   = logsumexp([lp0, lp1])

        # ln p(A=1) = ln (exp(lp1) / (exp(lp0) + exp(lp1)))
        #           = lp1 - ln(exp(lp0) + exp(lp1))
        #           = lp1 - Z
        A_col[k1] = np.log(np.random.rand()) < lp1 - Z

    return A_col
