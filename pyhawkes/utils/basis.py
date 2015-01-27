import os
import abc

import numpy as np
import scipy.linalg
import scipy.signal as sig

class Basis(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, B, dt, dt_max,
                 orth=False,
                 norm=False):
        self.B = B
        self.dt = dt
        self.dt_max = dt_max
        self.orth = orth
        self.norm = norm

        self.basis = self.interpolate_basis(self.create_basis(), self.dt, self.dt_max, self.norm)
        self.L = self.basis.shape[0]

    @abc.abstractmethod
    def create_basis(self):
        raise NotImplementedError()

    def convolve_with_basis(self, S):
        """
        Convolve each column of the event count matrix with this basis
        :param S:     TxK matrix of inputs.
                      T is the number of time bins
                      K is the number of input dimensions.
        :return: TxKxB tensor of inputs convolved with bases
        """
        (T,K) = S.shape
        (R,B) = self.basis.shape

        # First, by convention, the impulse responses are apply to times
        # (t-R:t-1). That means we need to prepend a row of zeros to make
        # sure the basis remains causal
        basis = np.vstack((np.zeros((1,B)), self.basis.copy()))

        # Initialize array for filtered stimulus
        F = np.empty((T,K,B))

        # Compute convolutions fo each basis vector, one at a time
        for b in np.arange(B):
            F[:,:,b] = sig.fftconvolve(S,
                                       np.reshape(basis[:,b],(R+1,1)),
                                       'full')[:T,:]

        # Check for positivity
        if np.amin(self.basis) >= 0 and np.amin(S) >= 0:
            np.clip(F, 0, np.inf, out=F)
            assert np.amin(F) >= 0, "convolution should be >= 0"

        return F

    def interpolate_basis(self, basis, dt, dt_max, norm=True):
        # Interpolate basis at the resolution of the data
        L,B = basis.shape
        t_int = np.arange(0.0, dt_max, step=dt)
        t_bas = np.linspace(0.0, dt_max, L)

        ibasis = np.zeros((len(t_int), B))
        for b in np.arange(B):
            ibasis[:,b] = np.interp(t_int, t_bas, basis[:,b])

        # Normalize so that the interpolated basis has volume 1
        if norm:
            # ibasis /= np.trapz(ibasis,t_int,axis=0)
            ibasis /= (dt * np.sum(ibasis, axis=0))

        return ibasis

    def create_basis(self):
        raise NotImplementedError("Override this in base class")

class CosineBasis(Basis):
    """
    Create a basis of raised cosine tuning curves
    """
    def __init__(self,
                 B, dt, dt_max,
                 orth=False,
                 norm=True,
                 n_eye=0,
                 a=1.0/120,
                 b=0.5,
                 L=100):

        self.n_eye = n_eye
        self.a = a
        self.b = b
        self.L = L

        super(CosineBasis, self).__init__(B, dt, dt_max, orth, norm)

    def create_basis(self):
        n_pts = self.L              # Number of points at which to evaluate the basis
        n_cos = self.B - self.n_eye # Number of cosine basis functions'
        n_eye = self.n_eye          # Number of identity basis functions
        assert n_cos >= 0 and n_eye >= 0

        n_bas = n_eye + n_cos
        basis = np.zeros((n_pts,n_bas))

        # The first n_eye basis elements are identity vectors in the first time bins
        basis[:n_eye,:n_eye] = np.eye(n_eye)

        # The remaining basis elements are raised cosine functions with peaks
        # logarithmically warped between [n_eye*dt:dt_max].

        a = self.a                          # Scaling in log time
        b = self.b                          # Offset in log time
        nlin = lambda t: np.log(a*t+b)      # Nonlinearity
        u_ir = nlin(np.arange(n_pts))       # Time in log time
        ctrs = u_ir[np.floor(np.linspace(n_eye,(n_pts/2.0),n_cos)).astype(np.int)]
        if len(ctrs) == 1:
            w = ctrs/2
        else:
            w = (ctrs[-1]-ctrs[0])/(n_cos-1)    # Width of the cosine tuning curves

        # Basis function is a raised cosine centered at c with width w
        basis_fn = lambda u,c,w: (np.cos(np.maximum(-np.pi,np.minimum(np.pi,(u-c)*np.pi/w/2.0)))+1)/2.0
        for i in np.arange(n_cos):
            basis[:,n_eye+i] = basis_fn(u_ir,ctrs[i],w)


        # Orthonormalize basis (this may decrease the number of effective basis vectors)
        if self.orth:
            basis = scipy.linalg.orth(basis)
        if self.norm:
            # We can only normalize nonnegative bases
            if np.any(basis<0):
                raise Exception("We can only normalize nonnegative impulse responses!")

            # Normalize such that \int_0^1 b(t) dt = 1
            basis = basis / np.tile(np.sum(basis,axis=0), [n_pts,1]) / (1.0/n_pts)

        return basis

