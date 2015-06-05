# Cythonized updates for the parent variables
#
# distutils: extra_compile_args = -O3
# cython: wraparound=False
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=False

import numpy as np
cimport numpy as np

from numpy.random import multinomial

from cython.parallel import prange

cpdef resample_Z(int[:,::1] Z0, int[:,:,:,::1] Z, long[:,::1] S,
                 double[::1] lambda0,
                 double[:,::1] W,
                 double[:,:,::1] g,
                 double[:,:,::1] F):

    cdef int T, K, B
    T = Z.shape[0]
    K = Z.shape[1]
    B = Z.shape[3]

    cdef int t, k1, k2, b, off
    cdef double[::1] p = np.zeros(1+K*B)
    cdef double psum
    cdef long[::1] parents = np.zeros(1+K*B, dtype=np.int)

    # Iterate over each event count, t and k2, in parallel
    for t in range(T):
        for k2 in range(K):
            off = 0
            psum = 0

            # TODO: If S[t,k2] is zero then we should be able to skip this
            if S[t,k2] == 0:
                continue

            # First compute the normalizer of the multinomial probability vector
            # TODO: Check that we are not reusing p
            # Compute the background rate
            p[off] = lambda0[k2]
            psum  += lambda0[k2]

            # Compute the rate from each other proc and basis function
            for k1 in range(K):
                for b in range(B):
                    off += 1
                    p[off] = W[k1,k2] * g[k1,k2,b] * F[t,k1,b]
                    psum += p[off]

            # Normalize p
            off = 0
            p[off] /= psum
            for k1 in range(K):
                for b in range(B):
                    off += 1
                    p[off] /= psum

            # Sample from p
            parents = multinomial(S[t,k2], p)

            # Update Z0 and Z with new samples
            off = 0
            Z0[t,k2] = parents[off]
            # TODO: Should we try to avoid recomputing the multiplications?
            for k1 in range(K):
                for b in range(B):
                    off += 1
                    Z[t,k1,k2,b] = parents[off]

cpdef mf_update_Z(double[:,::1] EZ0, double[:,:,:,::1] EZ, long[:,::1] S,
                  double[::1] exp_E_log_lambda0,
                  double[:,::1] exp_E_log_W,
                  double[:,:,::1] exp_E_log_g,
                  double[:,:,::1] F):

    cdef int t, k1, k2, b, i

    cdef int T, K, B
    T = EZ.shape[0]
    K = EZ.shape[1]
    B = EZ.shape[3]

    cdef double Z

    with nogil:
        # Iterate over each event count, t and k2, in parallel
        for t in prange(T):
            for k2 in range(K):
                # Zero out the Z buffer
                Z = 0.0

                # TODO: If S[t,k2] is zero then we should be able to skip this
                if S[t,k2] == 0:
                    continue

                # First compute the normalizer of the multinomial probability vector
                # TODO: Check that we are not reusing p
                # Compute the background rate
                Z = Z + exp_E_log_lambda0[k2]

                # Compute the rate from each other proc and basis function
                for k1 in range(K):
                    for b in range(B):
                        Z = Z + exp_E_log_W[k1, k2] * exp_E_log_g[k1,k2,b] * F[t, k1, b]


                # Now compute the expected counts
                EZ0[t,k2] = exp_E_log_lambda0[k2] / Z * S[t,k2]

                # TODO: Should we try to avoid recomputing the multiplications?
                for k1 in range(K):
                    for b in range(B):
                        EZ[t,k1,k2,b] = exp_E_log_W[k1, k2] * exp_E_log_g[k1,k2,b] * F[t, k1, b] / Z * S[t,k2]
