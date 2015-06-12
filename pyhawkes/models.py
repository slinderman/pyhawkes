"""
Top level classes for the Hawkes process model.
"""
import abc
import copy

import numpy as np
from scipy.special import gammaln
from scipy.optimize import minimize

from pybasicbayes.models import ModelGibbsSampling, ModelMeanField

from pyhawkes.internals.bias import GammaBias
from pyhawkes.internals.weights import SpikeAndSlabGammaWeights, GammaMixtureWeights
from pyhawkes.internals.impulses import DirichletImpulseResponses
from pyhawkes.internals.parents import DiscreteTimeParents
from pyhawkes.internals.network import StochasticBlockModel, StochasticBlockModelFixedSparsity
from pyhawkes.utils.basis import CosineBasis


from pyhawkes.utils.profiling import line_profiled
PROFILING = True

# TODO: Add a simple HomogeneousPoissonProcessModel

class DiscreteTimeStandardHawkesModel(object):
    """
    Discrete time standard Hawkes process model with support for
    regularized (stochastic) gradient descent.
    """
    def __init__(self, K, dt=1.0, dt_max=10.0,
                 B=5, basis=None,
                 alpha=1.0, beta=1.0,
                 allow_instantaneous=False,
                 W_max=None,
                 allow_self_connections=True):
        """
        Initialize a discrete time network Hawkes model with K processes.

        :param K:       Number of processes
        :param dt:      Time bin size
        :param dt_max:
        """
        self.K = K
        self.dt = dt
        self.dt_max = dt_max
        self.allow_self_connections = allow_self_connections
        self.W_max = W_max

        # Initialize the basis
        if basis is None:
            self.B = B
            self.allow_instantaneous = allow_instantaneous
            self.basis = CosineBasis(self.B, self.dt, self.dt_max, norm=True,
                                     allow_instantaneous=allow_instantaneous)
        else:
            self.basis = basis
            self.allow_instantaneous = basis.allow_instantaneous
            self.B = basis.B

        assert not (self.allow_instantaneous and self.allow_self_connections), \
            "Cannot allow instantaneous self connections"

        # Save the gamma prior
        assert alpha >= 1.0, "Alpha must be greater than 1.0 to ensure log concavity"
        self.alpha = alpha
        self.beta = beta

        # Initialize with sample from Gamma(alpha, beta)
        # self.weights = np.random.gamma(self.alpha, 1.0/self.beta, size=(self.K, 1 + self.K*self.B))
        # self.weights = self.alpha/self.beta * np.ones((self.K, 1 + self.K*self.B))
        self.weights = 1e-3 * np.ones((self.K, 1 + self.K*self.B))
        if not self.allow_self_connections:
            self._remove_self_weights()

        # Initialize the data list to empty
        self.data_list = []

    def _remove_self_weights(self):
        for k in xrange(self.K):
                self.weights[k,1+(k*self.B):1+(k+1)*self.B] = 1e-32

    def initialize_with_gibbs_model(self, gibbs_model):
        """
        Initialize with a sample from the network Hawkes model
        :param W:
        :param g:
        :return:
        """
        assert isinstance(gibbs_model, _DiscreteTimeNetworkHawkesModelBase)
        assert gibbs_model.K == self.K
        assert gibbs_model.B == self.B

        lambda0 = gibbs_model.bias_model.lambda0,
        Weff = gibbs_model.weight_model.W_effective
        g = gibbs_model.impulse_model.g

        for k in xrange(self.K):
            self.weights[k,0]  = lambda0[k]
            self.weights[k,1:] = (Weff[:,k][:,None] * g[:,k,:]).ravel()

        if not self.allow_self_connections:
            self._remove_self_weights()

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
        WB = self.weights[:,1:].reshape((self.K,self.K, self.B))

        # DEBUG
        assert WB[0,0,self.B-1] == self.weights[0,1+self.B-1]
        assert WB[0,self.K-1,0] == self.weights[0,1+(self.K-1)*self.B]

        if self.B > 2:
            assert WB[self.K-1,self.K-1,self.B-2] == self.weights[self.K-1,-2]

        # Weight matrix is summed over impulse response functions
        WT = WB.sum(axis=2)
        # Then we transpose so that the weight matrix is (outgoing x incoming)
        W = WT.T

        return W

    @property
    def bias(self):
        return self.weights[:,0]

    def add_data(self, S, F=None, minibatchsize=None):
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
            F = np.concatenate((np.ones((T,1)), F), axis=1)

        # If minibatchsize is not None, add minibatches of data
        if minibatchsize is not None:
            for offset in np.arange(T, step=minibatchsize):
                end = min(offset+minibatchsize, T)
                S_mb = S[offset:end,:]
                F_mb = F[offset:end,:]

                # Add minibatch to the data list
                self.data_list.append((S_mb, F_mb))

        else:
            self.data_list.append((S,F))

    def check_stability(self):
        """
        Check that the weight matrix is stable

        :return:
        """
        # Compute the effective weight matrix
        W_eff = self.weights.sum(axis=2)
        eigs = np.linalg.eigvals(W_eff)
        maxeig = np.amax(np.real(eigs))
        # print "Max eigenvalue: ", maxeig
        if maxeig < 1.0:
            return True
        else:
            return False

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

    def compute_rate(self, index=None, ks=None):
        """
        Compute the rate of the k-th process.

        :param index:   Which dataset to comput the rate of
        :param k:       Which process to compute the rate of
        :return:
        """
        if index is None:
            index = 0
        _,F = self.data_list[index]

        if ks is None:
            ks = np.arange(self.K)

        if isinstance(ks, int):
            R = F.dot(self.weights[ks,:])
            return R

        elif isinstance(ks, np.ndarray):
            Rs = []
            for k in ks:
                Rs.append(F.dot(self.weights[k,:])[:,None])
            return np.concatenate(Rs, axis=1)

        else:
            raise Exception("ks must be int or array of indices in 0..K-1")

    def log_prior(self, ks=None):
        """
        Compute the log prior probability of log W
        :param ks:
        :return:
        """
        lp = 0
        for k in ks:
            # lp += (self.alpha * np.log(self.weights[k,1:])).sum()
            # lp += (-self.beta * self.weights[k,1:]).sum()
            if self.alpha > 1:
                lp += (self.alpha -1) * np.log(self.weights[k,1:]).sum()
            lp += (-self.beta * self.weights[k,1:]).sum()
        return lp

    def log_likelihood(self, indices=None, ks=None):
        """
        Compute the log likelihood
        :return:
        """
        ll = 0

        if indices is None:
            indices = np.arange(len(self.data_list))
        if isinstance(indices, int):
            indices = [indices]

        for index in indices:
            S,F = self.data_list[index]
            R = self.compute_rate(index, ks=ks)

            if ks is not None:
                ll += (-gammaln(S[:,ks]+1) + S[:,ks] * np.log(R) -R*self.dt).sum()
            else:
                ll += (-gammaln(S+1) + S * np.log(R) -R*self.dt).sum()

        return ll

    def log_posterior(self, indices=None, ks=None):
        if ks is None:
            ks = np.arange(self.K)

        lp = self.log_likelihood(indices, ks)
        lp += self.log_prior(ks)
        return lp

    def heldout_log_likelihood(self, S):
        self.add_data(S)
        hll = self.log_likelihood(indices=-1)
        self.data_list.pop()
        return hll

    def compute_gradient(self, k, indices=None):
        """
        Compute the gradient of the log likelihood with respect
        to the log biases and log weights

        :param k:   Which process to compute gradients for.
                    If none, return a list of gradients for each process.
        """
        grad = np.zeros(1 + self.K * self.B)

        if indices is None:
            indices = np.arange(len(self.data_list))

        # d_W_d_log_W = self._d_W_d_logW(k)
        for index in indices:
            d_rate_d_W = self._d_rate_d_W(index, k)
            # d_rate_d_log_W = d_rate_d_W.dot(d_W_d_log_W)
            d_ll_d_rate = self._d_ll_d_rate(index, k)
            # d_ll_d_log_W = d_ll_d_rate.dot(d_rate_d_log_W)
            d_ll_d_W = d_ll_d_rate.dot(d_rate_d_W)

            # grad += d_ll_d_log_W
            grad += d_ll_d_W

        # Add the prior
        # d_log_prior_d_log_W = self._d_log_prior_d_log_W(k)
        # grad += d_log_prior_d_log_W

        d_log_prior_d_W = self._d_log_prior_d_W(k)
        assert np.allclose(d_log_prior_d_W[0], 0.0)
        # grad += d_log_prior_d_W.dot(d_W_d_log_W)
        grad += d_log_prior_d_W

        # Zero out the gradient if
        if not self.allow_self_connections:
            assert np.allclose(self.weights[k,1+k*self.B:1+(k+1)*self.B], 0.0)
            grad[1+k*self.B:1+(k+1)*self.B] = 0

        return grad

    def _d_ll_d_rate(self, index, k):
        S,_ = self.data_list[index]
        T = S.shape[0]

        rate = self.compute_rate(index, k)
        # d/dR  S*ln(R) -R*dt
        grad = S[:,k] / rate  - self.dt * np.ones(T)
        return grad

    def _d_rate_d_W(self, index, k):
        _,F = self.data_list[index]
        grad = F
        return grad

    def _d_W_d_logW(self, k):
        """
        Let u = logW
        d{e^u}/du = e^u
                  = W
        """
        return np.diag(self.weights[k,:])

    def _d_log_prior_d_log_W(self, k):
        """
        Use a gamma prior on W (it is log concave for alpha >= 1)
        By change of variables this implies that
        LN p(LN W) = const + \alpha LN W - \beta W
        and
        d/d (LN W) (LN p(LN W)) = \alpha - \beta W

        TODO: Is this still concave? It is a concave function of W,
        but what about of LN W? As a function of u=LN(W) it is
        linear plus a -\beta e^u which is concave for beta > 0,
        so yes, it is still concave.

        So why does BFGS not converge monotonically?

        """
        d_log_prior_d_log_W = np.zeros_like(self.weights[k,:])
        d_log_prior_d_log_W[1:] = self.alpha  - self.beta * self.weights[k,1:]
        return d_log_prior_d_log_W

    def _d_log_prior_d_W(self, k):
        """
        Use a gamma prior on W (it is log concave for alpha >= 1)

        and
        LN p(W)       = (\alpha-1)LN W - \beta W
        d/dW LN p(W)) = (\alpha -1)/W  - \beta
        """
        d_log_prior_d_W = np.zeros_like(self.weights[k,:])
        if self.alpha > 1.0:
            d_log_prior_d_W[1:] += (self.alpha-1) / self.weights[k,1:]

        d_log_prior_d_W[1:] += -self.beta
        return d_log_prior_d_W

    def fit_with_bfgs_logspace(self):
        """
        Fit the model with BFGS
        """
        # If W_max is specified, set this as a bound
        if self.W_max is not None:
            bnds = [(None, None)] + [(None, np.log(self.W_max))] * (self.K * self.B)
        else:
            bnds = None

        def objective(x, k):
            self.weights[k,:] = np.exp(x)
            self.weights[k,:] = np.nan_to_num(self.weights[k,:])
            return np.nan_to_num(-self.log_posterior(ks=np.array([k])))

        def gradient(x, k):
            self.weights[k,:] = np.exp(x)
            self.weights[k,:] = np.nan_to_num(self.weights[k,:])
            dll_dW =  -self.compute_gradient(k)
            d_W_d_log_W = self._d_W_d_logW(k)
            return np.nan_to_num(dll_dW.dot(d_W_d_log_W))

        itr = [0]
        def callback(x):
            if itr[0] % 10 == 0:
                print "Iteration: %03d\t LP: %.1f" % (itr[0], self.log_posterior())
            itr[0] = itr[0] + 1

        for k in xrange(self.K):
            print "Optimizing process ", k
            itr[0] = 0
            x0 = np.log(self.weights[k,:])
            res = minimize(objective,           # Objective function
                           x0,                  # Initial value
                           jac=gradient,        # Gradient of the objective
                           args=(k,),           # Arguments to the objective and gradient fns
                           bounds=bnds,         # Bounds on x
                           callback=callback)
            self.weights[k,:] = np.exp(res.x)

    def fit_with_bfgs(self):
        """
        Fit the model with BFGS
        """
        # If W_max is specified, set this as a bound
        if self.W_max is not None:
            bnds = [(1e-16, None)] + [(1e-16, self.W_max)] * (self.K * self.B)
        else:
            bnds = [(1e-16, None)] * (1 + self.K * self.B)

        def objective(x, k):
            self.weights[k,:] = x
            return np.nan_to_num(-self.log_posterior(ks=np.array([k])))

        def gradient(x, k):
            self.weights[k,:] = x
            return np.nan_to_num(-self.compute_gradient(k))

        itr = [0]
        def callback(x):
            if itr[0] % 10 == 0:
                print "Iteration: %03d\t LP: %.1f" % (itr[0], self.log_posterior())
            itr[0] = itr[0] + 1

        for k in xrange(self.K):
            print "Optimizing process ", k
            itr[0] = 0
            x0 = self.weights[k,:]
            res = minimize(objective,           # Objective function
                           x0,                  # Initial value
                           jac=gradient,        # Gradient of the objective
                           args=(k,),           # Arguments to the objective and gradient fns
                           bounds=bnds,         # Bounds on x
                           callback=callback)
            self.weights[k,:] = res.x

    def gradient_descent_step(self, stepsz=0.01):
        grad = np.zeros((self.K, 1+self.K*self.B))

        # Compute gradient and take a step for each process
        for k in xrange(self.K):
            d_W_d_log_W = self._d_W_d_logW(k)
            grad[k,:] = self.compute_gradient(k).dot(d_W_d_log_W)
            self.weights[k,:] = np.exp(np.log(self.weights[k,:]) + stepsz * grad[k,:])

        # Compute the current objective
        ll = self.log_likelihood()

        return self.weights, ll, grad

    def sgd_step(self, prev_velocity, learning_rate, momentum):
        """
        Take a step of the stochastic gradient descent algorithm
        """
        if prev_velocity is None:
            prev_velocity = np.zeros((self.K, 1+self.K*self.B))

        # Compute this gradient row by row
        grad = np.zeros((self.K, 1+self.K*self.B))
        velocity = np.zeros((self.K, 1+self.K*self.B))

        # Get a minibatch
        mb = np.random.choice(len(self.data_list))
        T = self.data_list[mb][0].shape[0]

        # Compute gradient and take a step for each process
        for k in xrange(self.K):
            d_W_d_log_W = self._d_W_d_logW(k)
            grad[k,:] = self.compute_gradient(k, indices=[mb]).dot(d_W_d_log_W) / T
            velocity[k,:] = momentum * prev_velocity[k,:] + learning_rate * grad[k,:]

            # Gradient steps are taken in log weight space
            log_weightsk = np.log(self.weights[k,:]) + velocity[k,:]

            # The true weights are stored
            self.weights[k,:] = np.exp(log_weightsk)

        # Compute the current objective
        ll = self.log_likelihood()

        return self.weights, ll, velocity


class _DiscreteTimeNetworkHawkesModelBase(object):
    """
    Discrete time network Hawkes process model with support for
    Gibbs sampling inference, variational inference (TODO), and
    stochastic variational inference (TODO).
    """

    __metaclass__ = abc.ABCMeta

    # Define the model components and their default hyperparameters
    _basis_class            = CosineBasis
    _default_basis_hypers   = {'norm': True, 'allow_instantaneous': False}

    _bkgd_class             = GammaBias
    _default_bkgd_hypers    = {'alpha': 1.0, 'beta': 1.0}

    _impulse_class          = DirichletImpulseResponses
    _default_impulse_hypers = {'gamma' : 1.0}

    # Weight, parent, and network class must be specified by subclasses
    _weight_class           = None
    _default_weight_hypers  = {}

    _parent_class           = DiscreteTimeParents

    _network_class          = None
    _default_network_hypers = {}

    def __init__(self, K, dt=1.0, dt_max=10.0, B=5,
                 basis=None, basis_hypers={},
                 bkgd=None, bkgd_hypers={},
                 impulse=None, impulse_hypers={},
                 weights=None, weight_hypers={},
                 network=None, network_hypers={}):
        """
        Initialize a discrete time network Hawkes model with K processes.

        :param K:  Number of processes
        """
        self.K      = K
        self.dt     = dt
        self.dt_max = dt_max
        self.B      = B

        # Initialize the data list to empty
        self.data_list = []

        # Initialize the basis
        if basis is not None:
            # assert basis.B == B
            self.basis = basis
            self.B     = basis.B
        else:
            # Use the given basis hyperparameters
            self.basis_hypers = copy.deepcopy(self._default_basis_hypers)
            self.basis_hypers.update(basis_hypers)
            self.basis = self._basis_class(self.B, self.dt, self.dt_max,
                                           **self.basis_hypers)

        # Initialize the bias
        if bkgd is not None:
            self.bias_model = bkgd
        else:
            # Use the given basis hyperparameters
            self.bkgd_hypers = copy.deepcopy(self._default_bkgd_hypers)
            self.bkgd_hypers.update(bkgd_hypers)
            self.bias_model = self._bkgd_class(self, **self.bkgd_hypers)

        # Initialize the impulse response model
        if impulse is not None:
            assert impulse.B == self.B
            assert impulse.K == self.K
            self.impulse_model = impulse
        else:
            # Use the given basis hyperparameters
            self.impulse_hypers = copy.deepcopy(self._default_impulse_hypers)
            self.impulse_hypers.update(impulse_hypers)
            self.impulse_model = self._impulse_class(self, **self.impulse_hypers)


        # Initialize the network model
        if network is not None:
            assert network.K == self.K
            self.network = network
        else:
            # Use the given network hyperparameters
            self.network_hypers = copy.deepcopy(self._default_network_hypers)
            self.network_hypers.update(network_hypers)
            self.network = self._network_class(K=self.K,
                                               **self.network_hypers)

        # TODO: Remove this hack. Should C be a model parameter?
        self.C = self.network.C

        # Check that the model doesn't allow instantaneous self connections
        assert not (self.basis.allow_instantaneous and
                    self.network.allow_self_connections), \
            "Cannot allow instantaneous self connections"

        # Initialize the weight model
        if weights is not None:
            assert weights.K == self.K
            self.weight_model = weights
        else:
            self.weight_hypers = copy.deepcopy(self._default_weight_hypers)
            self.weight_hypers.update(weight_hypers)
            self.weight_model = self._weight_class(self, **self.weight_hypers)


    def initialize_with_standard_model(self, standard_model):
        """
        Initialize with a standard Hawkes model. Typically this will have
        been fit by gradient descent or BFGS, and we just want to copy
        over the parameters to get a good starting point for MCMC or VB.
        :param W:
        :param g:
        :return:
        """
        assert isinstance(standard_model, DiscreteTimeStandardHawkesModel)
        assert standard_model.K == self.K
        assert standard_model.B == self.B

        lambda0 = standard_model.weights[:,0]

        # Get the connection weights
        Wg = standard_model.weights[:,1:].reshape((self.K, self.K, self.B))
        # Permute to out x in x basis
        Wg = np.transpose(Wg, [1,0,2])
        # Sum to get the total weight
        W = Wg.sum(axis=2) + 1e-6

        # The impulse responses are normalized weights
        g = Wg / W[:,:,None]
        for k1 in xrange(self.K):
            for k2 in xrange(self.K):
                if g[k1,k2,:].sum() < 1e-2:
                    g[k1,k2,:] = 1.0/self.B

        # Clip g to make sure it is stable for MF updates
        g = np.clip(g, 1e-2, np.inf)

        # Make sure g is normalized
        g = g / g.sum(axis=2)[:,:,None]


        # We need to decide how to set A.
        # The simplest is to initialize it to all ones, but
        # A = np.ones((self.K, self.K))
        # Alternatively, we can start with a sparse matrix
        # of only strong connections. What sparsity? How about the
        # mean under the network model
        sparsity = self.network.tau1 / (self.network.tau0 + self.network.tau1)
        A = W > np.percentile(W, (1.0 - sparsity) * 100)

        # Set the model parameters
        self.bias_model.lambda0 = lambda0.copy('C')
        self.weight_model.A     = A.copy('C')
        self.weight_model.W     = W.copy('C')
        self.impulse_model.g    = g.copy('C')


        if not self.network.fixed:
            # Cluster the standard model with kmeans in order to initialize the network
            from sklearn.cluster import KMeans

            features = []
            for k in xrange(self.K):
                features.append(np.concatenate((W[:,k], W[k,:])))

            self.network.c = KMeans(n_clusters=self.C).fit(np.array(features)).labels_

            # print "DEBUG: Do not set p and v in init from standard model"
            self.network.resample_p(self.weight_model.A)
            self.network.resample_v(self.weight_model.A, self.weight_model.W)
            self.network.resample_m()

    def add_data(self, S, F=None, minibatchsize=None):
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

        # Filter the data into a TxKxB array
        if F is not None:
            assert isinstance(F, np.ndarray) and F.shape == (T, self.K, self.B), \
                "F must be a filtered event count matrix"
        else:
            F = self.basis.convolve_with_basis(S)

        # If minibatchsize is not None, add minibatches of data
        if minibatchsize is not None:
            for offset in np.arange(T, step=minibatchsize):
                end = min(offset+minibatchsize, T)
                T_mb = end - offset
                S_mb = S[offset:end,:]
                F_mb = F[offset:end,:]

                # Instantiate parent object for this minibatch
                parents = self._parent_class(self, T_mb, S_mb, F_mb)

                # Add minibatch to the data list
                self.data_list.append(parents)

        else:
            # Instantiate corresponding parent object
            parents = self._parent_class(self, T, S, F)

            # Add to the data list
            self.data_list.append(parents)

    def check_stability(self, verbose=False):
        """
        Check that the weight matrix is stable

        :return:
        """
        if self.K < 100:
            eigs = np.linalg.eigvals(self.weight_model.W_effective)
            maxeig = np.amax(np.real(eigs))
        else:
            from scipy.sparse.linalg import eigs
            maxeig = eigs(self.weight_model.W_effective, k=1)[0]

        if verbose:
            print "Max eigenvalue: ", maxeig

        return maxeig < 1.0

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

    def generate(self, keep=True, T=100, print_interval=None):
        """
        Generate a new data set with the sampled parameters

        :param keep: If True, add the generated data to the data list.
        :param T:    Number of time bins to simulate.
        :return: A TxK
        """
        assert isinstance(T, int), "T must be an integer number of time bins"

        # Test stability
        self.check_stability()

        # Initialize the output
        S = np.zeros((T, self.K))

        # Precompute the impulse responses (LxKxK array)
        G = np.tensordot(self.basis.basis, self.impulse_model.g, axes=([1], [2]))
        L = self.basis.L
        assert G.shape == (L,self.K, self.K)
        H = self.weight_model.W_effective[None,:,:] * G

        # Transpose H so that it is faster for tensor mult
        H = np.transpose(H, axes=[0,2,1])

        # Compute the rate matrix R
        R = np.zeros((T+L, self.K))

        # Add the background rate
        R += self.bias_model.lambda0[None,:]

        # Iterate over time bins
        for t in xrange(T):
            if print_interval is not None and t % print_interval == 0:
                print "Iteration ", t
            # Sample a Poisson number of events for each process
            S[t,:] = np.random.poisson(R[t,:] * self.dt)

            # Compute change in rate via tensor product
            dR = np.tensordot( H, S[t,:], axes=([2, 0]))
            R[t:t+L,:] += dR

            # For each sampled event, add a weighted impulse response to the rate
            # for k in xrange(self.K):
            #     if S[t,k] > 0:
            #         R[t+1:t+L+1,:] += S[t,k] * H[:,k,:]

            # Check Spike limit
            if np.any(S[t,:] >= 1000):
                print "More than 1000 events in one time bin!"
                import pdb; pdb.set_trace()

        # Only keep the first T time bins
        S = S[:T,:].astype(np.int)
        R = R[:T,:]

        if keep:
            # Xs = [X[:T,:] for X in Xs]
            # data = np.hstack(Xs + [S])
            self.add_data(S)


        return S, R

    def get_parameters(self):
        """
        Get a copy of the parameters of the model
        :return:
        """
        return self.weight_model.A, \
               self.weight_model.W, \
               self.impulse_model.g, \
               self.bias_model.lambda0, \
               self.network.c, \
               self.network.p, \
               self.network.v, \
               self.network.m \

    def set_parameters(self, params):
        """
        Set the parameters of the model
        :param params:
        :return:
        """
        A, W, beta, lambda0, c, p, v, m = params
        K, B, C = self.K, self.basis.B, self.C

        assert isinstance(A, np.ndarray) and A.shape == (K,K), \
            "A must be a KxK adjacency matrix"

        assert isinstance(W, np.ndarray) and W.shape == (K,K) \
               and np.amin(W) >= 0, \
            "W must be a KxK weight matrix"

        assert isinstance(beta, np.ndarray) and beta.shape == (K,K,B) and \
               np.allclose(beta.sum(axis=2), 1.0), \
            "beta must be a KxKxB impulse response array"

        assert isinstance(lambda0, np.ndarray) and lambda0.shape == (K,) \
               and np.amin(lambda0) >=0, \
            "lambda0 must be a K-vector of background rates"

        assert isinstance(c, np.ndarray) and c.shape == (K,) \
                and np.amin(c) >= 0 and np.amax(c) < self.C, \
            "c must be a K-vector of block assignments"

        assert isinstance(p, np.ndarray) and p.shape == (C,C) \
                and np.amin(p) >= 0 and np.amax(p) <= 1.0, \
            "p must be a CxC matrix block connection probabilities"

        assert isinstance(v, np.ndarray) and v.shape == (C,C) \
                and np.amin(v) >= 0, \
            "v must be a CxC matrix block weight scales"

        assert isinstance(m, np.ndarray) and m.shape == (C,) \
                and np.amin(m) >= 0 and np.allclose(m.sum(), 1.0), \
            "m must be a C vector of block probabilities"

        self.weight_model.A = A
        self.weight_model.W = W
        self.impulse_model.g = beta
        self.bias_model.lambda0 = lambda0
        self.network.c = c
        self.network.p = p
        self.network.v = v
        self.network.m = m

    def compute_rate(self, index=0, proc=None, S=None, F=None):
        """
        Compute the rate function for a given data set
        :param index:   An integer specifying which dataset (if S is None)
        :param S:       TxK array of event counts for which we would like to
                        compute the model's rate
        :return:        TxK array of rates
        """
        # TODO: Write a Cython function to evaluate this
        if S is not None:
            assert isinstance(S, np.ndarray) and S.ndim == 2, "S must be a TxK array."
            T,K = S.shape

            # Filter the data into a TxKxB array
            if F is not None:
                assert F.shape == (T,K, self.B)
            else:
                F = self.basis.convolve_with_basis(S)

        else:
            assert len(self.data_list) > index, "Dataset %d does not exist!" % index
            data = self.data_list[index]
            T,K,S,F = data.T, data.K, data.S, data.F

        if proc is None:
            # Compute the rate
            R = np.zeros((T,K))

            # Background rate
            R += self.bias_model.lambda0[None,:]

            # Compute the sum of weighted sum of impulse responses
            H = self.weight_model.W_effective[:,:,None] * \
                self.impulse_model.g

            H = np.transpose(H, [2,0,1])

            for k2 in xrange(self.K):
                R[:,k2] += np.tensordot(F, H[:,:,k2], axes=([2,1], [0,1]))

            return R

        else:
            assert isinstance(proc, int) and proc < self.K, "Proc must be an int"
            # Compute the rate
            R = np.zeros((T,))

            # Background rate
            R += self.bias_model.lambda0[proc]

            # Compute the sum of weighted sum of impulse responses
            H = self.weight_model.W_effective[:,proc,None] * \
                self.impulse_model.g[:,proc,:]

            R += np.tensordot(F, H, axes=([1,2], [0,1]))

            return R

    def _poisson_log_likelihood(self, S, R):
        """
        Compute the log likelihood of a Poisson matrix with rates R

        :param S:   Count matrix
        :param R:   Rate matrix
        :return:    log likelihood
        """
        return (S * np.log(R) - R*self.dt).sum()

    def heldout_log_likelihood(self, S, F=None):
        """
        Compute the held out log likelihood of a data matrix S.
        :param S:   TxK matrix of event counts
        :return:    log likelihood of those counts under the current model
        """
        R = self.compute_rate(S=S, F=F)
        return self._poisson_log_likelihood(S, R)

    # def heldout_log_likelihood(self, S, F=None):
    #     self.add_data(S, F=F)
    #     hll = self.log_likelihood(indices=-1)
    #     self.data_list.pop()
    #     return hll

    def log_prior(self):
        # Get the parameter priors
        lp  = 0
        # lp += self.bias_model.log_probability()
        lp += self.weight_model.log_probability()
        # lp += self.impulse_model.log_probability()
        # lp += self.network.log_probability()

        return lp

    def log_likelihood(self, indices=None):
        """
        Compute the joint log probability of the data and the parameters
        :return:
        """
        ll = 0

        if indices is None:
            indices = np.arange(len(self.data_list))
        if isinstance(indices, int):
            indices = [indices]

        # Get the likelihood of the datasets
        for ind in indices:
            ll += self.data_list[ind].log_likelihood()

        return ll

    def log_probability(self):
        """
        Compute the joint log probability of the data and the parameters
        :return:
        """
        lp = self.log_likelihood()
        lp += self.log_prior()

        return lp



class DiscreteTimeNetworkHawkesModelSpikeAndSlab(_DiscreteTimeNetworkHawkesModelBase, ModelGibbsSampling):
    _weight_class           = SpikeAndSlabGammaWeights
    _default_weight_hypers  = {}

    _network_class          = StochasticBlockModel
    _default_network_hypers = {'C': 1, 'c': None,
                               'p': None, 'tau1': 1.0, 'tau0': 1.0,
                               'allow_self_connections': True,
                               'kappa': 1.0,
                               'v': None, 'alpha': 5.0, 'beta': 1.0,
                               'pi': 1.0}

    @line_profiled
    def resample_model(self):
        """
        Perform one iteration of the Gibbs sampling algorithm.
        :return:
        """
        # Update the parents.
        # THIS MUST BE DONE IMMEDIATELY FOLLOWING WEIGHT UPDATES!
        for p in self.data_list:
            p.resample()

        # Update the bias model given the parents assigned to the background
        self.bias_model.resample(self.data_list)

        # Update the impulse model given the parents assignments
        # self.impulse_model.resample(self.data_list)

        # Update the network model
        # self.network.resample(data=(self.weight_model.A, self.weight_model.W))

        # Update the weight model given the parents assignments
        self.weight_model.resample(self.data_list)

    def initialize_with_standard_model(self, standard_model):
        super(DiscreteTimeNetworkHawkesModelSpikeAndSlab, self).\
            initialize_with_standard_model(standard_model)

        # Update the parents.
        for d in self.data_list:
            d.resample()


class DiscreteTimeNetworkHawkesModelSpikeAndSlabFixedSparsity(DiscreteTimeNetworkHawkesModelSpikeAndSlab):
    _network_class          = StochasticBlockModelFixedSparsity
    _default_network_hypers = {'C': 1, 'c': None,
                               'p': 0.5,
                               'allow_self_connections': True,
                               'kappa': 1.0,
                               'v': None, 'alpha': 1.0, 'beta': 1.0,
                               'pi': 1.0}

class DiscreteTimeNetworkHawkesModelGammaMixture(
    _DiscreteTimeNetworkHawkesModelBase, ModelGibbsSampling, ModelMeanField):
    _weight_class           = GammaMixtureWeights
    _default_weight_hypers  = {'kappa_0': 0.1, 'nu_0': 1000.0}

    # This model uses an SBM with beta-distributed sparsity levels
    _network_class          = StochasticBlockModel
    _default_network_hypers = {'C': 1, 'c': None,
                               'p': None, 'tau1': 1.0, 'tau0': 1.0,
                               'allow_self_connections': True,
                               'kappa': 1.0,
                               'v': None, 'alpha': 1.0, 'beta': 1.0,
                               'pi': 1.0}

    def resample_model(self, resample_network=True):
        """
        Perform one iteration of the Gibbs sampling algorithm.
        :return:
        """
        # Update the parents.
        for p in self.data_list:
            p.resample()

        # Update the bias model given the parents assigned to the background
        self.bias_model.resample(self.data_list)

        # Update the impulse model given the parents assignments
        self.impulse_model.resample(self.data_list)

        # Update the weight model given the parents assignments
        self.weight_model.resample(self.data_list)

        # Update the network model
        if resample_network:
            self.network.resample(data=(self.weight_model.A, self.weight_model.W))

    def initialize_with_standard_model(self, standard_model):
        super(DiscreteTimeNetworkHawkesModelGammaMixture, self).\
            initialize_with_standard_model(standard_model)

        # Set the mean field parameters
        self.bias_model.mf_alpha = np.clip(100 * self.bias_model.lambda0, 1e-8, np.inf)
        self.bias_model.mf_beta  = 100 * np.ones(self.K)

        # Weight model
        self.weight_model.mf_kappa_0 = self.weight_model.nu_0 * self.weight_model.W.copy()
        self.weight_model.mf_v_0     = self.weight_model.nu_0 * np.ones((self.K, self.K))

        self.weight_model.mf_kappa_1 = 100 * self.weight_model.W.copy()
        self.weight_model.mf_v_1     = 100 * np.ones((self.K, self.K))

        self.weight_model.mf_p       = 0.8 * self.weight_model.A + 0.2 * (1-self.weight_model.A)

        # Set mean field parameters of the impulse model
        self.impulse_model.mf_gamma = 100 * self.impulse_model.g.copy('C')

        # Set network mean field parameters
        if self.C > 1:
            self.network.mf_m = 0.2 / (self.C-1) * np.ones((self.K, self.C))
            for c in xrange(self.C):
                self.network.mf_m[self.network.c == c, c] = 0.8
        else:
            self.network.mf_m = np.ones((self.K, self.C))

        # Update the parents.
        # for _,_,_,p in self.data_list:
        #     p.resample(self.bias_model, self.weight_model, self.impulse_model)
        #     p.meanfieldupdate(self.bias_model, self.weight_model, self.impulse_model)

    def meanfield_coordinate_descent_step(self):
        # Update the parents.
        for p in self.data_list:
            p.meanfieldupdate()

        # Update the bias model given the parents assigned to the background
        self.bias_model.meanfieldupdate(self.data_list)

        # Update the impulse model given the parents assignments
        self.impulse_model.meanfieldupdate(self.data_list)

        # Update the weight model given the parents assignments
        self.weight_model.meanfieldupdate(self.data_list)

        # Update the network model
        self.network.meanfieldupdate(self.weight_model)

        return self.get_vlb()

    def get_vlb(self):
        # Compute the variational lower bound
        vlb = 0
        for d in self.data_list:
            vlb += d.get_vlb()

        vlb += self.bias_model.get_vlb()
        vlb += self.impulse_model.get_vlb()
        vlb += self.weight_model.get_vlb()
        vlb += self.network.get_vlb()
        return vlb

    def sgd_step(self, minibatchsize, stepsize):
        # Sample a minibatch of data
        assert len(self.data_list) == 1, "We only sample from the first data set"
        S, F, T = self.data_list[0].S, self.data_list[0].F, self.data_list[0].T

        if not hasattr(self, 'sgd_offset'):
            self.sgd_offset = 0
        else:
            self.sgd_offset += minibatchsize
            if self.sgd_offset >= T:
                self.sgd_offset = 0

        # Grab a slice of S
        sgd_end = min(self.sgd_offset+minibatchsize, T)
        S_minibatch = S[self.sgd_offset:sgd_end, :]
        F_minibatch = F[self.sgd_offset:sgd_end, :, :]
        T_minibatch = S_minibatch.shape[0]
        minibatchfrac = float(T_minibatch) / T

        # Create a parent object for this minibatch
        p = self._parent_class(self, T_minibatch, S_minibatch, F_minibatch)

        # TODO: Grab one dataset from the data_list and assume
        # it has been added in minibatches

        # Update the parents using a standard mean field update
        p.meanfieldupdate()

        # Update the bias model given the parents assigned to the background
        self.bias_model.meanfield_sgdstep([p],
                                          minibatchfrac=minibatchfrac,
                                          stepsize=stepsize)

        # Update the impulse model given the parents assignments
        self.impulse_model.meanfield_sgdstep([p],
                                             minibatchfrac=minibatchfrac,
                                             stepsize=stepsize)

        # Update the weight model given the parents assignments
        # Compute the number of events in the minibatch
        self.weight_model.meanfield_sgdstep([p],
                                            minibatchfrac=minibatchfrac,
                                            stepsize=stepsize)

        # Update the network model. This only depends on the global weight model,
        # so we can just do a standard mean field update
        self.network.meanfield_sgdstep(self.weight_model,
                                       minibatchfrac=minibatchfrac,
                                       stepsize=stepsize)

        # Clear the parent buffer for this minibatch
        del p

    def resample_from_mf(self):
        self.bias_model.resample_from_mf()
        self.weight_model.resample_from_mf()
        self.impulse_model.resample_from_mf()
        self.network.resample_from_mf()


class DiscreteTimeNetworkHawkesModelGammaMixtureFixedSparsity(DiscreteTimeNetworkHawkesModelGammaMixture):
    _network_class          = StochasticBlockModelFixedSparsity
    _default_network_hypers = {'C': 1, 'c': None,
                               'p': 0.5,
                               'allow_self_connections': True,
                               'kappa': 1.0,
                               'v': None, 'alpha': 1.0, 'beta': 1.0,
                               'pi': 1.0}


class ContinuousTimeNetworkHawkesModel(ModelGibbsSampling):
    _default_bkgd_hypers = {"alpha" : 1.0, "beta" : 1.0}
    _default_impulse_hypers = {"mu_0": 0., "lmbda_0": 10.0, "alpha_0": 10.0, "beta_0" : 1.0}
    _default_weight_hypers = {}

    # This model uses an SBM with beta-distributed sparsity levels
    _network_class          = StochasticBlockModel
    _default_network_hypers = {'C': 1, 'c': None,
                               'p': None, 'tau1': 1.0, 'tau0': 1.0,
                               'allow_self_connections': True,
                               'kappa': 1.0,
                               'v': None, 'alpha': 1.0, 'beta': 1.0,
                               'pi': 1.0}

    def __init__(self, K, dt_max=10.0, B=5,
                 bkgd_hypers={},
                 impulse_hypers={},
                 weight_hypers={},
                 network_hypers={}):
        """
        Initialize a discrete time network Hawkes model with K processes.

        :param K:  Number of processes
        """
        self.K      = K
        self.dt_max = dt_max
        self.B      = B

        # Initialize the bias
        # Use the given basis hyperparameters
        self.bkgd_hypers = copy.deepcopy(self._default_bkgd_hypers)
        self.bkgd_hypers.update(bkgd_hypers)
        from pyhawkes.internals.bias import ContinuousTimeGammaBias
        self.bias_model = ContinuousTimeGammaBias(self, self.K, **self.bkgd_hypers)

        # Initialize the impulse response model
        self.impulse_hypers = copy.deepcopy(self._default_impulse_hypers)
        self.impulse_hypers.update(impulse_hypers)
        from pyhawkes.internals.impulses import ContinuousTimeImpulseResponses
        self.impulse_model = \
            ContinuousTimeImpulseResponses(self, **self.impulse_hypers)

        # Initialize the network model
        self.network_hypers = copy.deepcopy(self._default_network_hypers)
        self.network_hypers.update(network_hypers)
        self.network = \
            self._network_class(K=self.K, **self.network_hypers)

        # Initialize the weight model
        from pyhawkes.internals.weights import SpikeAndSlabContinuousTimeGammaWeights
        self.weight_hypers = copy.deepcopy(self._default_weight_hypers)
        self.weight_hypers.update(weight_hypers)
        self.weight_model = \
            SpikeAndSlabContinuousTimeGammaWeights(self,  **self.weight_hypers)

        # Initialize the data list to empty
        self.data_list = []


    @property
    def lambda0(self):
        return self.bias_model.lambda0

    @property
    def W_effective(self):
        return self.weight_model.W_effective

    def initialize_with_standard_model(self, standard_model):
        """
        Initialize with a standard Hawkes model. Typically this will have
        been fit by gradient descent or BFGS, and we just want to copy
        over the parameters to get a good starting point for MCMC or VB.
        :param W:
        :param g:
        :return:
        """
        assert isinstance(standard_model, DiscreteTimeStandardHawkesModel)
        assert standard_model.K == self.K
        assert standard_model.B == self.B

        lambda0 = standard_model.weights[:,0]

        # Get the connection weights
        Wg = standard_model.weights[:,1:].reshape((self.K, self.K, self.B))
        # Permute to out x in x basis
        Wg = np.transpose(Wg, [1,0,2])
        # Sum to get the total weight
        W = Wg.sum(axis=2) + 1e-6

        # We need to decide how to set A.
        # The simplest is to initialize it to all ones, but
        # A = np.ones((self.K, self.K))
        # Alternatively, we can start with a sparse matrix
        # of only strong connections. What sparsity? How about the
        # mean under the network model
        sparsity = self.network.tau1 / (self.network.tau0 + self.network.tau1)
        A = W > np.percentile(W, (1.0 - sparsity) * 100)

        # Set the model parameters
        self.bias_model.lambda0 = lambda0.copy('C')
        self.weight_model.A     = A.copy('C')
        self.weight_model.W     = W.copy('C')

    def add_data(self, S, C, T):
        """
        Add a data set to the list of observations.
        First, filter the data with the impulse response basis,
        then instantiate a set of parents for this data set.

        :param S: length N array of event times
        :param C: length N array of process id's for each event
        :param T: max time of
        """
        assert isinstance(T, float), "T must be a float"
        if len(S) > 0:
            assert isinstance(S, np.ndarray) and S.ndim == 1 \
                   and S.min() >= 0 and S.max() < T and \
                   S.dtype == np.float, \
                   "S must be a N array of event times"

            # Make sure S is sorted
            assert (np.diff(S) >= 0).all(), "S must be sorted!"

        if len(C) > 0:
            assert isinstance(C, np.ndarray) and C.shape == S.shape \
                   and C.min() >= 0 and C.max() < self.K and \
                   C.dtype == np.int, \
                   "C must be a N array of parent indices"

        # Instantiate corresponding parent object
        from pyhawkes.internals.parents import ContinuousTimeParents
        parents = ContinuousTimeParents(self, S, C, T, self.K, self.dt_max)

        # Add to the data list
        self.data_list.append(parents)

    def generate(self, keep=True, T=100.0, max_round=100, **kwargs):
        from pyhawkes.utils.utils import logistic
        K, dt_max = self.K, self.dt_max

        lambda0 = self.bias_model.lambda0
        W, A = self.weight_model.W, self.weight_model.A
        g_mu, g_tau = self.impulse_model.mu, self.impulse_model.tau


        def _generate_helper(S, C, s_pa, c_pa, round=0):
            # Recursively generate new generations of spikes with
            # given impulse response parameters. Takes in a single spike
            # as the parent and recursively calls itself on all children
            # spikes

            assert round < max_round, "Exceeded maximum recursion depth of %d" % max_round

            for c_ch in np.arange(K):
                w = W[c_pa, c_ch]
                a = A[c_pa, c_ch]
                if w==0 or a==0:
                    continue

                # The total area under the impulse response curve(ratE)  is w
                # Sample spikes from a homogenous poisson process with rate
                # 1 until the time exceeds w. Then transform those spikes
                # such that they are distributed under a logistic normal impulse
                n_ch = np.random.poisson(w)

                # Sample normal RVs and take the logistic of them. This is equivalent
                # to sampling uniformly from the inverse CDF
                x_ch = g_mu[c_pa, c_ch] + np.sqrt(1./g_tau[c_pa, c_ch])*np.random.randn(n_ch)

                # Spike times are logistic transformation of x
                s_ch = s_pa + dt_max * logistic(x_ch)

                # Only keep spikes within the simulation time interval
                s_ch = s_ch[s_ch < T]
                n_ch = len(s_ch)

                S.append(s_ch)
                C.append(c_ch * np.ones(n_ch, dtype=np.int))

                # Generate offspring from child spikes
                for s in s_ch:
                    _generate_helper(S, C, s, c_ch, round=round+1)

        # Initialize output arrays, a dictionary of numpy arrays
        S = []
        C = []

        # Sample background spikes
        for k in np.arange(K):
            N = np.random.poisson(lambda0[k]*T)
            S_bkgd = np.random.rand(N)*T
            C_bkgd = k*np.ones(N, dtype=np.int)
            S.append(S_bkgd)
            C.append(C_bkgd)

            # Each background spike spawns a cascade
            for s,c in zip(S_bkgd, C_bkgd):
                _generate_helper(S, C, s, c)

        # Concatenate arrays
        S = np.concatenate(S)
        C = np.concatenate(C)

        # Sort
        perm = np.argsort(S)
        S = S[perm]
        C = C[perm]

        if keep:
            self.add_data(S, C, T)

        return S, C


    def check_stability(self):
        """
        Check that the weight matrix is stable

        :return:
        """
        if self.K < 100:
            eigs = np.linalg.eigvals(self.weight_model.W_effective)
            maxeig = np.amax(np.real(eigs))
        else:
            from scipy.sparse.linalg import eigs
            maxeig = eigs(self.weight_model.W_effective, k=1)[0]

        print "Max eigenvalue: ", maxeig
        if maxeig < 1.0:
            return True
        else:
            return False

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

    def compute_rate_at_events(self, data):
        # Compute the instantaneous rate at the individual events
        # Sum over potential parents.

        # Compute it manually
        S, C, dt_max = data.S, data.C, self.dt_max
        N = S.shape[0]
        lambda0 = self.bias_model.lambda0
        W = self.weight_model.W_effective
        mu, tau = self.impulse_model.mu, self.impulse_model.tau

        # lmbda_manual = np.zeros(N)
        # impulse = self.impulse_model.impulse
        # # Resample parents
        # for n in xrange(N):
        #     # First parent is just the background rate of this process
        #     lmbda_manual[n] += lambda0[C[n]]
        #
        #     # Iterate backward from the most recent to compute probabilities of each parent spike
        #     for par in xrange(n-1, -1, -1):
        #         dt = S[n] - S[par]
        #
        #         # Since the spikes are sorted, we can stop if we reach a potential
        #         # parent that occurred greater than dt_max in the past
        #         if dt > dt_max:
        #             break
        #
        #         Wparn = W[C[par], C[n]]
        #         if Wparn > 0:
        #             lmbda_manual[n] += Wparn * impulse(dt, C[par], C[n])

        # Call cython function to evaluate instantaneous rate
        from pyhawkes.internals.continuous_time_helpers import compute_rate_at_events
        lmbda = np.zeros(N)
        compute_rate_at_events(S, C, dt_max, lambda0, W, mu, tau, lmbda)

        # assert np.allclose(lmbda_manual, lmbda)

        return lmbda

    def compute_integrated_rate(self, data, proc=None):
        """
        We can approximate this by ignoring events within dt_max of the end.
        Since each event induces an impulse response with area W, we
        simply need to count events
        :param index:
        :param proc:
        :return:
        """
        T, Ns = data.T, data.Ns
        W = self.weight_model.W_effective
        lmbda0 = self.bias_model.lambda0

        # Compute the integral (W is send x recv)
        int_lmbda = lmbda0 * T
        int_lmbda += W.T.dot(Ns)
        assert int_lmbda.shape == (self.K,)

        # TODO: Only compute for proc (probably negligible savings)
        if proc is None:
            return int_lmbda
        else:
            return int_lmbda[proc]

    def log_prior(self):
        # Get the parameter priors
        lp  = 0
        lp += self.bias_model.log_probability()
        lp += self.weight_model.log_probability()
        lp += self.impulse_model.log_probability()
        # lp += self.network.log_probability()

        return lp

    def log_likelihood(self, data=None):
        """
        Compute the joint log probability of the data and the parameters
        :return:
        """
        ll = 0

        if data is None:
            data = self.data_list

        if not isinstance(data, list):
            data = [data]

        # Get the likelihood of the datasets
        for d in data:
            # ll += -gammaln(d.N+1)
            ll -= self.compute_integrated_rate(d).sum()
            ll += np.log(self.compute_rate_at_events(d)).sum()

        return ll

    def log_probability(self):
        """
        Compute the joint log probability of the data and the parameters
        :return:
        """
        lp = self.log_likelihood()
        lp += self.log_prior()

        return lp

    def heldout_log_likelihood(self, S, C, T):
        self.add_data(S, C, T)
        data = self.data_list.pop()
        return self.log_likelihood(data)

    def compute_rate(self, S, C, T, dt=1.0):
        # TODO: Write a cythonized version of this
        # Compute rate for each process at intervals of dt
        t = np.concatenate([np.arange(0, T, step=dt), [T]])
        rate = np.zeros((t.size, self.K))
        for k in xrange(self.K):
            rate[:,k] += self.bias_model.lambda0[k]

            # Get the deltas between the time points and the spikes
            # Warning: this can be huge!
            deltas = t[:,None]-S[None,:]
            t_deltas, n_deltas = np.where((deltas>0) & (deltas < self.dt_max))
            N_deltas = t_deltas.size

            # Find the process the impulse came from
            senders = C[n_deltas]

            # Compute the impulse responses onto process k for each delta
            imps = self.impulse_model.impulse(deltas[t_deltas, n_deltas],
                                              senders,
                                              k
                                              )
            rate[t_deltas, k] += imps

        return rate, t

    def compute_impulses(self, dt=1.0):
        dt = np.concatenate([np.arange(0, self.dt_max, step=dt), [self.dt_max]])
        ir = np.zeros((dt.size, self.K, self.K))
        for k1 in xrange(self.K):
            for k2 in xrange(self.K):
                ir[:,k1,k2] = self.impulse_model.impulse(dt, k1, k2)
        return ir, dt


    ### Inference
    def resample_model(self):
        """
        Perform one iteration of the Gibbs sampling algorithm.
        :return:
        """
        # import ipdb; ipdb.set_trace()
        # Update the parents.
        # THIS MUST BE DONE IMMEDIATELY FOLLOWING WEIGHT UPDATES!
        for p in self.data_list:
            p.resample()

        # Update the bias model given the parents assigned to the background
        self.bias_model.resample(self.data_list)

        # Update the impulse model given the parents assignments
        self.impulse_model.resample(self.data_list)

        # Update the network model
        self.network.resample(data=(self.weight_model.A, self.weight_model.W))

        # Update the weight model given the parents assignments
        self.weight_model.resample(self.data_list)


