# Cythonized updates for the parent variables
#
# distutils: extra_compile_args = -O3
# cython: wraparound=False
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=False

import numpy as np
cimport numpy as np

from cython.parallel import prange

from libc.math cimport log, exp, sqrt

cdef double SQRT_2PI = 2.5066282746310002


### Continuous time helper functions with logistic normal impulse responses
cdef inline double logit(double p) nogil:
    return log(p / (1-p))

cdef inline double ln_impulse(double dt, double mu, double tau, double dt_max) nogil:
    """
    Impulse response induced by an event on process k1 on
    the rate of process k2 at lag dt
    """
    cdef double Z = dt * (dt_max - dt)/dt_max * SQRT_2PI / sqrt(tau)
    return exp(-tau/2. * (logit(dt/dt_max) - mu)**2) / Z

cpdef void ct_resample_Z_logistic_normal_serial(
    double[::1] S, long[::1] C, long[::1] Z, double dt_max,
    double[::1] lambda0, double[:,::1] W, double[:,::1] mu, double[:,::1] tau):


    cdef int N = S.shape[0]
    cdef double[::1] p = np.zeros(N)
    cdef double p_bkgd = 0.0
    cdef double denom
    cdef double dt
    cdef int par, n, min_par

    # Precompute randomness
    cdef double[::1] u = np.random.rand(N)
    cdef double acc

    # Resample parents
    for n in range(N):
        Z[n] = -2
        if n == 0:
            Z[n] = -1
            continue

        # First potential parent is just the background rate of this process
        p_bkgd = lambda0[C[n]]
        denom = p_bkgd

        # Iterate backward from the most recent to compute probabilities of each parent spike
        for par in range(n-1, -1, -1):
            dt = S[n] - S[par]

            # Since the spikes are sorted, we can stop if we reach a potential
            # parent that occurred greater than dt_max in the past
            if dt > dt_max:
                p[par] = 0
                break

            p[par] = W[C[par], C[n]] * ln_impulse(dt, mu[C[par], C[n]], tau[C[par], C[n]], dt_max)
            denom += p[par]

        # Now sample forward, starting from the minimum viable parent
        min_par = par
        acc = p_bkgd / denom
        if u[n] < acc:
            # Sampled the background rate
            Z[n] = -1
        else:
            for par in range(min_par, n):
                acc += p[par] / denom
                if u[n] < acc:
                    Z[n] = par
                    break

        if Z[n] == -2:
            print "Failed!"
            print acc

cpdef void ct_resample_Z_logistic_normal(
    double[::1] S, long[::1] C, long[::1] Z, double dt_max,
    double[::1] lambda0, double[:,::1] W, double[:,::1] mu, double[:,::1] tau):


    cdef int N = S.shape[0]
    #cdef double[::1] p = np.zeros(N)
    #cdef double p_bkgd = 0.0
    cdef double p
    cdef double denom
    cdef double dt
    cdef int par, n, min_par

    # Precompute randomness
    cdef double[::1] u = np.random.rand(N)
    cdef double acc

    # Resample parents
    # TODO: prange!
    for n in prange(N, nogil=True):
        Z[n] = -2
        if n == 0:
            Z[n] = -1
            continue

        # First potential parent is just the background rate of this process
        denom = lambda0[C[n]]

        # Iterate backward from the most recent to compute probabilities of each parent spike
        for par in range(n-1, -1, -1):
            dt = S[n] - S[par]

            # Since the spikes are sorted, we can stop if we reach a potential
            # parent that occurred greater than dt_max in the past
            if dt > dt_max:
                break

            p = W[C[par], C[n]] * ln_impulse(dt, mu[C[par], C[n]], tau[C[par], C[n]], dt_max)
            denom = denom + p

        # Now sample forward, starting from the minimum viable parent
        min_par = par
        acc = lambda0[C[n]] / denom
        if u[n] < acc:
            # Sampled the background rate
            Z[n] = -1
        else:
            for par in range(min_par, n):
                dt = S[n] - S[par]
                if dt <= dt_max:
                    p = W[C[par], C[n]] * ln_impulse(dt, mu[C[par], C[n]], tau[C[par], C[n]], dt_max)
                    acc = acc + p / denom
                    if u[n] < acc:
                        Z[n] = par
                        break

cpdef ct_compute_suff_stats(
    double[::1] S, long[::1] C, long[::1] Z, double dt_max,
    double[::1] bkgd_ss,
    double[:,::1] weight_ss,
    double[:,:,::1] imp_ss
    ):

    cdef int N = S.shape[0]
    cdef int n, par
    cdef double dt, sdt


    for n in range(N):
        par = Z[n]
        if par == -1:
            bkgd_ss[C[n]] += 1
        else:
            weight_ss[C[par], C[n]] += 1

            dt = S[n] - S[par]
            imp_ss[0, C[par], C[n]] += 1
            imp_ss[1, C[par], C[n]] += np.log(dt) - np.log(dt_max - dt)

    # In a second pass, compute the sum of squares for the impulse responses
    cdef double[:,::1] mu = np.divide(imp_ss[1], imp_ss[0])
    for n in range(N):
        par = Z[n]
        if par > -1:
            dt = S[n] - S[par]
            sdt = np.log(dt) - np.log(dt_max - dt)
            imp_ss[2, C[par], C[n]] += (sdt - mu[C[par], C[n]])**2

    assert np.isfinite(imp_ss).all()


cpdef void compute_rate_at_events(
    double[::1] S, long[::1] C, double dt_max,
    double[::1] lambda0, double[:,::1] W,
    double[:,::1] mu, double[:,::1] tau,
    double[::1] lmbda):

    # Compute the instantaneous rate at the individual events
    # Sum over potential parents.

    cdef int N = S.shape[0]

    cdef int par, n
    cdef double dt

    # Compute rate at each event!
    for n in prange(N, nogil=True):
        # First parent is just the background rate of this process
        lmbda[n] += lambda0[C[n]]

        # Iterate backward from the most recent to compute probabilities of each parent spike
        for par in range(n-1, -1, -1):
            dt = S[n] - S[par]

            # Since the spikes are sorted, we can stop if we reach a potential
            # parent that occurred greater than dt_max in the past
            if dt > dt_max:
                break

            if W[C[par], C[n]] > 0:
                lmbda[n] += W[C[par], C[n]] * ln_impulse(dt, mu[C[par], C[n]], tau[C[par], C[n]], dt_max)


cpdef void compute_weighted_impulses_at_events(
    double[::1] S, long[::1] C, long[::1] Z, double dt_max,
    double[:,::1] W, double[:,::1] mu, double[:,::1] tau,
    double[:,::1] lmbda
    ):
    # Compute the instantaneous rate at the individual events
    # Sum over potential parents.

    cdef int N = S.shape[0]
    cdef int n, par, cn, cp
    cdef double dt

    for n in prange(N, nogil=True):
        cn = C[n]
        # Iterate backward from the most recent to compute probabilities of each parent spike
        for par in range(n-1, -1, -1):
            cp = C[par]
            dt = S[n] - S[par]
            if dt == 0:
                continue

            # Since the spikes are sorted, we can stop if we reach a potential
            # parent that occurred greater than dt_max in the past
            if dt >= dt_max:
                break

            lmbda[n, cp] += W[cp, cn] * ln_impulse(dt, mu[cp, cn], tau[cp, cn], dt_max)
