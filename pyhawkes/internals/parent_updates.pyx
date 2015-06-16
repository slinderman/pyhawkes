# Cythonized updates for the parent variables
#
# distutils: extra_compile_args = -O3
# cython: wraparound=False
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=False

import numpy as np
cimport numpy as np

from libc.math cimport log

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
#
# cpdef mf_update_Z(double[:,::1] EZ0, double[:,:,:,::1] EZ, long[:,::1] S,
#                   double[::1] exp_E_log_lambda0,
#                   double[:,::1] exp_E_log_W,
#                   double[:,:,::1] exp_E_log_g,
#                   double[:,:,::1] F):
#
#     cdef int t, k1, k2, b, i
#
#     cdef int T, K, B
#     T = EZ.shape[0]
#     K = EZ.shape[1]
#     B = EZ.shape[3]
#
#     cdef double Z
#
#     with nogil:
#         # Iterate over each event count, t and k2, in parallel
#         for t in prange(T):
#             for k2 in range(K):
#                 # Zero out the Z buffer
#                 Z = 0.0
#
#                 # TODO: If S[t,k2] is zero then we should be able to skip this
#                 if S[t,k2] == 0:
#                     continue
#
#                 # First compute the normalizer of the multinomial probability vector
#                 # TODO: Check that we are not reusing p
#                 # Compute the background rate
#                 Z = Z + exp_E_log_lambda0[k2]
#
#                 # Compute the rate from each other proc and basis function
#                 for k1 in range(K):
#                     for b in range(B):
#                         Z = Z + exp_E_log_W[k1, k2] * exp_E_log_g[k1,k2,b] * F[t, k1, b]
#
#
#                 # Now compute the expected counts
#                 EZ0[t,k2] = exp_E_log_lambda0[k2] / Z * S[t,k2]
#
#                 # TODO: Should we try to avoid recomputing the multiplications?
#                 for k1 in range(K):
#                     for b in range(B):
#                         EZ[t,k1,k2,b] = exp_E_log_W[k1, k2] * exp_E_log_g[k1,k2,b] * F[t, k1, b] / Z * S[t,k2]



cpdef mf_update_Z(int k2,
                  double[:,::1] EZ,
                  long[::1] Sk,
                  double[::1] exp_E_log_lambda0,
                  double[:,::1] exp_E_log_W,
                  double[:,:,::1] exp_E_log_g,
                  double[:,:,::1] F):

    cdef int t, k1, b, i

    cdef int T, K, B
    T = EZ.shape[0]
    K = exp_E_log_lambda0.shape[0]
    B = exp_E_log_g.shape[2]

    cdef double Z

    with nogil:
        # Iterate over each event count, t and k2, in parallel
        for t in prange(T):
            # Zero out the Z buffer
            Z = 0.0

            # TODO: We should have removed the zeros during preprocessing
            if Sk[t] == 0:
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
            EZ[t,0] = exp_E_log_lambda0[k2] / Z * Sk[t]

            # TODO: Should we try to avoid recomputing the multiplications?
            for k1 in range(K):
                for b in range(B):
                    EZ[t,1+k1*B+b] = exp_E_log_W[k1, k2] * exp_E_log_g[k1, k2, b] * F[t, k1, b] / Z * Sk[t]



cpdef mf_vlb(int k2,
             int T_total,
             double[:,::1] EZ,
             unsigned int[::1] Sk,
             long[::1] Ns,
             double[::1] E_log_lambda0,
             double[::1] E_lambda0,
             double[:,::1] E_log_W,
             double[:,::1] E_W,
             double[:,:,::1] E_log_g,
             double[:,:,::1] F):

    cdef int t, k1, b, i

    cdef int T, K, B
    T = EZ.shape[0]
    K = E_log_lambda0.shape[0]
    B = E_log_g.shape[2]
    cdef double vlb = 0

    # Precompute the static part of the VLB
    # vlb += (-self.T * E_lam[k2]).sum()
    # vlb += -(self.Ns * E_W[:,k2]).sum()
    vlb += -T_total * E_lambda0[k2]
    for k1 in range(K):
        vlb += -Ns[k1] * E_W[k1,k2]

    # Iterate over each event count, t, in parallel
    cdef double[::1] vlbs = np.zeros(T)
    with nogil:
        for t in prange(T):
            # vlb += (EZk[:,0] * E_ln_lam[k]).sum()
            vlbs[t] += EZ[t,0] * E_log_lambda0[k2]

            # ln_u0 = log(EZk[:,0] / Sk.astype(np.float))
            # vlb += (-EZk[:,0] * ln_u0).sum()
            vlbs[t] += -EZ[t,0] * log(EZ[t,0] / Sk[t] + 1e-32)

            # Impulse terms
            for k1 in range(K):
                for b in range(B):
                    # E_ln_Wg = (log(Fk) +
                    #            weight_model.expected_log_W()[:,k][None,:,None] +
                    #            impulse_model.expected_log_g()[:,k,:][None,:,:])
                    # E_ln_Wg = np.nan_to_num(E_ln_Wg)
                    # vlb += (EZk[:,1:] * E_ln_Wg.reshape((Tk, K*B))).sum()
                    vlbs[t] += EZ[t,1+k1*B+b] * (log(F[t,k1,b]+1e-32) + E_log_W[k1,k2] + E_log_g[k1,k2,b])

                    # Second term
                    # ln_u = log(EZk[:,1:] / Sk[:,None].astype(np.float))
                    # vlb += (-EZk[:,1:] * ln_u).sum()
                    vlbs[t] += -EZ[t,1+k1*B+b] * log(EZ[t,1+k1*B+b] / Sk[t] + 1e-32)


    # Now sum up the vlbs serially
    vlb += np.sum(vlbs)
    return vlb