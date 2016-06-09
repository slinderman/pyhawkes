import copy
import abc

from scipy.optimize import minimize

from pyhawkes.utils.basis import CosineBasis

import autograd.numpy as np
from autograd import grad


### Nodes
class _NonlinearHawkesNodeBase(object):
    """
    A single node of a nonlinear Hawkes process.
    """
    __metaclass__ = abc.ABCMeta

    constrained = False

    def bias_bnds(self):
        return None

    def weight_bnds(self):
        return None

    @abc.abstractmethod
    def link(self, psi):
        raise NotImplementedError

    @abc.abstractmethod
    def invlink(self, lam):
        raise NotImplementedError

    def __init__(self, K, B, dt=1.0, sigma=np.inf, lmbda=np.inf):
        self.K, self.B, self.dt, self.sigma, self.lmbda = K, B, dt, sigma, lmbda

        # Initialize weights
        self.w = np.zeros(1+self.K*self.B)

        # List of event counts and filtered inputs
        self.data_list = []

    def add_data(self, F, S):
        T = F.shape[0]
        assert S.shape == (T,) and S.dtype in (np.int, np.uint, np.uint32)

        if F.shape[1] == self.K * self.B:
            F = np.hstack([np.ones((T,)),  F])
        else:
            assert F.shape[1] == 1 + self.K * self.B

        self.data_list.append((F, S))

    def initialize_to_background_rate(self):
        # self.w = abs(1e-6 * np.random.randn(*self.w.shape))
        self.w = 1e-6 * np.ones_like(self.w)
        if len(self.data_list) > 0:
            N = 0
            T = 0
            for F,S in self.data_list:
                N += S.sum(axis=0)
                T += S.shape[0] * self.dt

            lambda0 = self.invlink(N / float(T))
            self.w[0] = lambda0

    def log_likelihood(self, index=None):
        if index is None:
            data_list = self.data_list
        else:
            data_list = [self.data_list[index]]

        ll = 0
        for F,S in data_list:
            psi = F.dot(self.w)
            lam = self.link(psi)
            ll += (S * np.log(lam) -lam*self.dt).sum()

        return ll

    def objective(self, w):
        obj = 0
        N = float(sum([np.sum(d[1]) for d in self.data_list]))
        for F,S in self.data_list:
            psi = np.dot(F, w)
            lam = self.link(psi)
            obj -= np.sum(S * np.log(lam) -lam*self.dt) / N
            # assert np.isfinite(ll)

        # Add penalties
        obj += (0.5 * np.sum(w[1:]**2) / self.sigma**2) / N
        obj += np.sum(np.abs(w[1:]) * self.lmbda) / N

        # assert np.isfinite(obj)

        return obj


    def fit_with_bfgs(self):
        """
        Fit the model with BFGS
        """
        # If W_max is specified, set this as a bound
        # if self.W_max is not None:
        bnds = self.bias_bnds + self.weight_bnds * (self.K * self.B) \
            if self.constrained else None

        # else:
        # bnds = [(None, None)] * (1 + self.K * self.B)

        itr = [0]
        def callback(w):
            if itr[0] % 10 == 0:
                print "Iteration: %03d\t LP: %.5f" % (itr[0], self.objective(w))
            itr[0] = itr[0] + 1

        itr[0] = 0
        x0 = self.w
        res = minimize(self.objective,           # Objective function
                       x0,                       # Initial value
                       jac=grad(self.objective), # Gradient of the objective
                       bounds=bnds,              # Bounds on x
                       callback=callback)
        self.w = res.x


    def copy_node(self):
        """
        Return a copy of the parameters of the node
        """
        # Shallow copy the data
        data_list = copy.copy(self.data_list)
        self.data_list = []

        # Make a deep copy without the data
        node_copy = copy.deepcopy(self)

        # Reset the data and return the data-less copy
        self.data_list = data_list
        return node_copy


class LinearHawkesNode(_NonlinearHawkesNodeBase):
    constrained = True

    @property
    def bias_bnds(self):
        return [(1e-16, None)]

    @property
    def weight_bnds(self):
        return [(1e-16, None)]

    def link(self, psi):
        return psi

    def invlink(self, lam):
        return lam


class RectLinearHawkesNode(_NonlinearHawkesNodeBase):
    def link(self, psi):
        return 1e-16 + np.log(1.+np.exp(psi))

    def invlink(self, lam):
        return np.log(np.exp(lam) - 1.)


class ExpNonlinearHawkesNode(_NonlinearHawkesNodeBase):
    def link(self, psi):
        return np.exp(psi)

    def invlink(self, lam):
        return np.log(lam)

# Dummy class for a homogeneous Hawkes node
class HomogeneousPoissonNode(_NonlinearHawkesNodeBase):
    def link(self, psi):
        return psi

    def invlink(self, lam):
        return lam

    def fit_with_bfgs(self):
        # Rather than fitting, just initialize to background rate
        self.initialize_to_background_rate()
        self.w[1:] = 0


class _NonlinearHawkesProcessBase(object):
    """
    Discrete time nonlinear Hawkes process, i.e. Poisson GLM
    """
    __metaclass__ = abc.ABCMeta

    _node_class = None

    def __init__(self, K, dt=1.0, dt_max=10.0,
                 B=5, basis=None,
                 sigma=np.inf,
                 lmbda=0):
        """
        Initialize a discrete time network Hawkes model with K processes.

        :param K:       Number of processes
        :param dt:      Time bin size
        :param dt_max:
        """
        self.K = K
        self.dt = dt
        self.dt_max = dt_max
        self.sigma = sigma
        self.lmbda = lmbda

        # Initialize the basis
        if basis is None:
            self.B = B
            self.basis = CosineBasis(self.B, self.dt, self.dt_max, norm=True,
                                     allow_instantaneous=False)
        else:
            self.basis = basis
            self.B = basis.B

        # Initialize nodes
        self.nodes = \
            [self._node_class(self.K, self.B, dt=self.dt,
                              sigma=self.sigma, lmbda=self.lmbda)
             for _ in xrange(self.K)]

    def initialize_to_background_rate(self):
        for node in self.nodes:
            node.initialize_to_background_rate()

    @property
    def W(self):
        full_W = np.array([node.w for node in self.nodes])
        WB = full_W[:,1:].reshape((self.K,self.K, self.B))

        # Weight matrix is summed over impulse response functions
        WT = WB.sum(axis=2)

        # Then we transpose so that the weight matrix is (outgoing x incoming)
        W = WT.T
        return W

    @property
    def G(self):
        full_W = np.array([node.w for node in self.nodes])
        WB = full_W[:,1:].reshape((self.K,self.K, self.B))

        # Weight matrix is summed over impulse response functions
        WT = WB.sum(axis=2)

        # Impulse response weights are normalized weights
        GT = WB / WT[:,:,None]

        # Then we transpose so that the impuolse matrix is (outgoing x incoming x basis)
        G = np.transpose(GT, [1,0,2])

        # TODO: Decide if this is still necessary
        for k1 in xrange(self.K):
            for k2 in xrange(self.K):
                if G[k1,k2,:].sum() < 1e-2:
                    G[k1,k2,:] = 1.0/self.B
        return G

    @property
    def bias(self):
        full_W = np.array([node.w for node in self.nodes])
        return full_W[:,0]

    def add_data(self, S, F=None):
        """
        Add a data set to the list of observations.
        First, filter the data with the impulse response basis,
        then instantiate a set of parents for this data set.

        :param S: a TxK matrix of of event counts for each time bin
                  and each process.
        """
        assert isinstance(S, np.ndarray) and S.ndim == 2 and S.shape[1] == self.K \
               and np.amin(S) >= 0 and S.dtype == np.int, \
               "Data must be a TxK array of event counts"

        T = S.shape[0]

        if F is None:
            # Filter the data into a TxKxB array
            Ftens = self.basis.convolve_with_basis(S)

            # Flatten this into a T x (KxB) matrix
            # [F00, F01, F02, F10, F11, ... F(K-1)0, F(K-1)(B-1)]
            F = Ftens.reshape((T, self.K * self.B))
            assert np.allclose(F[:,0], Ftens[:,0,0])
            if self.B > 1:
                assert np.allclose(F[:,1], Ftens[:,0,1])
            if self.K > 1:
                assert np.allclose(F[:,self.B], Ftens[:,1,0])

            # Prepend a column of ones
            F = np.hstack((np.ones((T,1)), F))

        for k,node in enumerate(self.nodes):
            node.add_data(F, S[:,k])

    def remove_data(self, index):
        for node in self.nodes:
            del node.data_list[index]

    def log_likelihood(self, index=None):
        ll = np.sum([node.log_likelihood(index=index) for node in self.nodes])
        return ll

    def heldout_log_likelihood(self, S, F=None):
        self.add_data(S, F=F)
        hll = self.log_likelihood(index=-1)
        self.remove_data(-1)
        return hll

    def copy_sample(self):
        """
        Return a copy of the parameters of the model
        """
        # Shallow copy the data
        nodes_original = copy.copy(self.nodes)

        # Make a deep copy without the data
        self.nodes = [n.copy_node() for n in nodes_original]
        model_copy = copy.deepcopy(self)

        # Reset the data and return the data-less copy
        self.nodes = nodes_original
        return model_copy

    def fit_with_bfgs(self):
        # TODO: This can be parallelized
        for k, node in enumerate(self.nodes):
            print ""
            print "Fitting Node ", k
            node.fit_with_bfgs()


class StandardHawkesProcess(_NonlinearHawkesProcessBase):
    _node_class = LinearHawkesNode

class ReluNonlinearHawkesProcess(_NonlinearHawkesProcessBase):
    _node_class = RectLinearHawkesNode

class ExpNonlinearHawkesProcess(_NonlinearHawkesProcessBase):
    _node_class = ExpNonlinearHawkesNode

class HomogeneousPoissonProcess(_NonlinearHawkesProcessBase):
    _node_class = HomogeneousPoissonNode
