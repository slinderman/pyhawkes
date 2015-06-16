import copy

from scipy.optimize import minimize

from pyhawkes.utils.basis import CosineBasis

import autograd.numpy as np
from autograd import grad


class NonlinearHawkesProcess(object):
    """
    Discrete time Poisson GLM
    """
    def __init__(self, K, dt=1.0, dt_max=10.0,
                 B=5, basis=None,
                 link="exp",
                 sigma=np.inf):
        """
        Initialize a discrete time network Hawkes model with K processes.

        :param K:       Number of processes
        :param dt:      Time bin size
        :param dt_max:
        """
        self.K = K
        self.dt = dt
        self.dt_max = dt_max

        # Initialize the basis
        if basis is None:
            self.B = B
            self.basis = CosineBasis(self.B, self.dt, self.dt_max, norm=True,
                                     allow_instantaneous=False)
        else:
            self.basis = basis
            self.B = basis.B

        # Initialize nodes
        self.nodes = [NonlinearHawkesNode(K, B, dt=dt, link=link, sigma=sigma)]

        # Initialize the data list to empty
        self.data_list = []

    def initialize_to_background_rate(self):
        if len(self.data_list) > 0:
            N = 0
            T = 0
            for S,_ in self.data_list:
                N += S.sum(axis=0)
                T += S.shape[0] * self.dt

            lambda0 = N / float(T)
            self.weights[:,0] = lambda0

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


    def log_likelihood(self):
        ll = np.sum([node.log_likelihood() for node in self.nodes])
        return ll

    def heldout_log_likelihood(self, S):
        data = self.add_data(S)
        hll = self.log_likelihood(data=[data])
        self.data_list.pop()
        return hll

    def copy_sample(self):
        """
        Return a copy of the parameters of the model
        :return: The parameters of the model (A,W,\lambda_0, \beta)
        """
        # return copy.deepcopy(self.get_parameters())

        # Shallow copy the data
        data_list = copy.copy(self.data_list)
        self.data_list = []

        # Make a deep copy without the data
        model_copy = copy.deepcopy(self)

        # Reset the data and return the data-less copy
        self.data_list = data_list
        return model_copy


class NonlinearHawkesNode(object):
    """
    A single node of a nonlinear Hawkes process.
    """
    def __init__(self, K, B, link="exp", dt=1.0, sigma=np.inf):
        self.K, self.B, self.dt, self.sigma = K, B, dt, sigma

        if link.lower() == "exp":
            self.link = np.exp
            self.invlink = np.log
        else:
            raise NotImplementedError()

        # Initialize weights
        self.w = np.zeros(1+self.K*self.B)

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
        self.w *= 0
        if len(self.data_list) > 0:
            N = 0
            T = 0
            for F,S in self.data_list:
                N += S.sum(axis=0)
                T += S.shape[0] * self.dt

            lambda0 = N / float(T)
            self.w[0] = self.invlink(lambda0)

    def log_likelihood(self, w=None):
        if w is None:
            w = self.w

        ll = 0
        for F,S in self.data_list:
            psi = F.dot(w)
            lam = self.link(psi)
            ll += (S * np.log(lam) -lam*self.dt).sum()

        return ll

    def objective(self, w):
        return -self.log_likelihood(w) + 0.5 * np.sum(w**2) / self.sigma**2


    def fit_with_bfgs(self):
        """
        Fit the model with BFGS
        """
        # If W_max is specified, set this as a bound
        # if self.W_max is not None:
        #     bnds = [(None, None)] + [(-self.W_max, self.W_max)] * (self.K * self.B)
        # else:
        #     bnds = [(None, None)] * (1 + self.K * self.B)
        itr = [0]
        def callback(w):
            if itr[0] % 10 == 0:
                print "Iteration: %03d\t LP: %.1f" % (itr[0], self.objective(w))
            itr[0] = itr[0] + 1

        for k in xrange(self.K):
            print "Optimizing process ", k
            itr[0] = 0
            x0 = self.w
            res = minimize(self.objective,           # Objective function
                           x0,                       # Initial value
                           jac=grad(self.objective), # Gradient of the objective
                           # bounds=bnds,              # Bounds on x
                           callback=callback)
            self.w = res.x
