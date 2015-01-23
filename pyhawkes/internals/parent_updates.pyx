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

cpdef mf_update_Z(double[:,::1] EZ0, double[:,:,:,::1] EZ, long[:,::1] S,
                  double[::1] exp_E_log_lambda0,
                  double[:,::1] exp_E_log_W,
                  double[:,:,::1] exp_E_log_g,
                  double[:,:,::1] F):

    cdef int t, k1, k2, b
    cdef double p, Z

    cdef int T, K, B
    T = EZ.shape[0]
    K = EZ.shape[1]
    B = EZ.shape[3]

    with nogil:
        # Iterate over each event count, t and k2, in parallel
        for t in prange(T):
            for k2 in prange(K):

                # TODO: If S[t,k2] is zero then we should be able to skip this
                if S[t,k2] == 0:
                    continue

                # First compute the normalizer of the multinomial probability vector
                # TODO: Check that we are not reusing p
                # Compute the background rate
                Z = exp_E_log_lambda0[k2]

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

