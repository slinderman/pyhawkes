"""
Top level classes for the Hawkes process model.
"""
import abc
import copy

import numpy as np
from scipy.special import gammaln
from scipy.optimize import minimize

from pyhawkes.deps.pybasicbayes.models import ModelGibbsSampling, ModelMeanField
from pyhawkes.internals.bias import GammaBias
from pyhawkes.internals.weights import SpikeAndSlabGammaWeights, GammaMixtureWeights
from pyhawkes.internals.impulses import DirichletImpulseResponses
from pyhawkes.internals.parents import SpikeAndSlabParents, GammaMixtureParents
from pyhawkes.internals.network import ErdosRenyiModel, StochasticBlockModel
from pyhawkes.utils.basis import CosineBasis


class DiscreteTimeStandardHawkesModel(object):
    """
    Discrete time standard Hawkes process model with support for
    regularized (stochastic) gradient descent.
    """
    def __init__(self, K, dt=1.0, dt_max=10.0,
                 B=5, basis=None,
                 alpha=1.0, beta=1.0,
                 l2_penalty=0.0, l1_penalty=0.0):
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
        self.B = B
        self.basis = CosineBasis(self.B, self.dt, self.dt_max, norm=True)

        # Randomly initialize parameters (bias and weights)
        # self.weights = abs((1.0/(1+self.K*self.B)) * np.random.randn(self.K, 1 + self.K * self.B))

        # Save the gamma prior
        assert alpha >= 1.0, "Alpha must be greater than 1.0 to ensure log concavity"
        self.alpha = alpha
        self.beta = beta

        # Initialize with sample from Gamma(alpha, beta)
        # self.weights = np.random.gamma(self.alpha, 1.0/self.beta, size=(self.K, 1 + self.K*self.B))
        # self.weights = self.alpha/self.beta * np.ones((self.K, 1 + self.K*self.B))
        self.weights = 1e-3 * np.ones((self.K, 1 + self.K*self.B))

        # Save the regularization penalties
        self.l2_penalty = l2_penalty
        self.l1_penalty = l1_penalty

        # Initialize the data list to empty
        self.data_list = []

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

        # DEBUG
        Wmanual = np.zeros((self.K, self.K))
        for kin in xrange(self.K):
            for kout in xrange(self.K):
                start = 1+kout*self.B
                end = 1+(kout+1)*self.B
                Wmanual[kout,kin] = self.weights[kin,start:end].sum()

        if not np.allclose(W, Wmanual):
            import pdb; pdb.set_trace()

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

            # # Check that \sum_t F[t,k,b] ~= Nk / dt
            # Fsum = F.sum(axis=0)
            # print "F_err:  ", Fsum - N/self.dt

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

        for index in indices:
            d_W_d_log_W = self._d_W_d_logW(k)
            d_rate_d_W = self._d_rate_d_W(index, k)
            d_rate_d_log_W = d_rate_d_W.dot(d_W_d_log_W)
            d_ll_d_rate = self._d_ll_d_rate(index, k)
            d_ll_d_log_W = d_ll_d_rate.dot(d_rate_d_log_W)

            grad += d_ll_d_log_W

        # Subtract the regularization penalty
        # import pdb; pdb.set_trace()
        # d_reg_d_W = self._d_reg_d_W(k)
        # grad += d_reg_d_W.dot(d_W_d_log_W)

        # Add the prior
        d_prior_d_W = self._d_prior_d_W(k)
        grad += d_prior_d_W.dot(d_W_d_log_W)

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

    def _d_reg_d_W(self, k):
        """
        Compute gradient of regularization
        d/dW  -1/2 * L2 * W^2 -L1 * |W|
            = -2*L2*W -L1

        since W >= 0
        """
        d_reg_d_W = -self.l2_penalty*self.weights[k,:] -self.l1_penalty

        # Don't penalize the bias
        d_reg_d_W[0] = 0

        return d_reg_d_W

    def _d_prior_d_W(self, k):
        """
        Use a gamma prior (it is log concave for alpha >= 1)
        Compute gradient of prior wrt W
        d/dW  (alpha-1) * log(W) - beta*W
            = (alpha-1) / W - beta
        """
        d_prior_d_W = (self.alpha-1.0) / self.weights[k,:] - self.beta

        # Don't put a prior on the bias
        d_prior_d_W[0] = 0

        return d_prior_d_W


    def fit_with_bfgs(self):
        """
        Fit the model with BFGS
        """
        def objective(x, k):
            self.weights[k,:] = np.exp(x)
            self.weights[k,:] = np.nan_to_num(self.weights[k,:])
            return np.nan_to_num(-self.log_likelihood(ks=np.array([k])))

        def gradient(x, k):
            self.weights[k,:] = np.exp(x)
            self.weights[k,:] = np.nan_to_num(self.weights[k,:])
            return np.nan_to_num(-self.compute_gradient(k))

        itr = [0]
        def callback(x):
            if itr[0] % 10 == 0:
                print "Iteration: ", itr[0], "\t LL: ", self.log_likelihood()
            itr[0] = itr[0] + 1

        for k in xrange(self.K):
            print "Optimizing process ", k
            itr[0] = 0
            x0 = np.log(self.weights[k,:])
            res = minimize(objective, x0, args=(k,), jac=gradient, callback=callback)
            self.weights[k,:] = np.exp(res.x)

    def gradient_descent_step(self, stepsz=0.01):
        grad = np.zeros((self.K, 1+self.K*self.B))

        # Compute gradient and take a step for each process
        for k in xrange(self.K):
            grad[k,:] = self.compute_gradient(k)
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
            grad[k,:] = self.compute_gradient(k, indices=[mb]) / T
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
    _weight_class = None
    _parent_class = None

    def __init__(self, K, dt=1.0, dt_max=10.0,
                 B=5, basis=None,
                 alpha0=1.0, beta0=1.0,
                 C=1, c=None,
                 kappa=1.0,
                 v=None, alpha=1, beta=1,
                 p=None, tau1=0.5, tau0=0.5,
                 gamma=1.0):
        """
        Initialize a discrete time network Hawkes model with K processes.

        :param K:  Number of processes
        """
        self.K = K
        self.dt = dt
        self.dt_max = dt_max

        # Initialize the basis
        self.B = B
        self.basis = CosineBasis(self.B, self.dt, self.dt_max, norm=True)

        # Initialize the model components
        self.bias_model = GammaBias(self.K, self.dt, alpha0, beta0)
        self.impulse_model = DirichletImpulseResponses(self.K, self.B, gamma=gamma)

        # Initialize the network model
        # self.network = ErdosRenyiModel(self.K, p=p, kappa=kappa, v=v)
        self.C = C
        # self.network = StochasticBlockModel(C=self.C, K=self.K, p=p, kappa=kappa, v=v)
        self.network = StochasticBlockModel(C=self.C, K=self.K,
                                            c=c,
                                            p=p, tau1=tau1, tau0=tau0,
                                            kappa=kappa,
                                            v=v, alpha=alpha, beta=beta,
                                            pi=1.0)

        # The weight model is dictated by whether this is for Gibbs or MF
        self.weight_model = self._weight_class(self.K, self.network)

        # Initialize the data list to empty
        self.data_list = []

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
            print "Convolving with basis"
            F = self.basis.convolve_with_basis(S)

        # # Check that \sum_t F[t,k,b] ~= Nk / dt
        # Fsum = F.sum(axis=0)
        # print "F_err:  ", Fsum - N/self.dt

        # If minibatchsize is not None, add minibatches of data
        if minibatchsize is not None:
            for offset in np.arange(T, step=minibatchsize):
                end = min(offset+minibatchsize, T)
                T_mb = end - offset
                S_mb = S[offset:end,:]
                F_mb = F[offset:end,:]
                N_mb = np.atleast_1d(S_mb.sum(axis=0))

                # Instantiate parent object for this minibatch
                parents = self._parent_class(T_mb, self.K, self.B, S_mb, F_mb)

                # Add minibatch to the data list
                self.data_list.append((S_mb, N_mb, F_mb, parents))

        else:
            # Instantiate corresponding parent object
            parents = self._parent_class(T, self.K, self.B, S, F)

            # TODO: Remove this resample as it allocates memory
            # parents.resample(self.bias_model, self.weight_model, self.impulse_model)

            # Get the event count
            N = np.atleast_1d(S.sum(axis=0))

            # Add to the data list
            self.data_list.append((S, N, F, parents))

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
        self.data_list = None

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
        H = self.weight_model.W_effective[None,:,:] * \
            G

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
            R[t+1:t+L+1,:] += dR

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
            S, _, F, _ = self.data_list[index]
            T,K = S.shape

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
        return (-gammaln(S+1) + S * np.log(R*self.dt) - R*self.dt).sum()

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
            S,_,_,_  = self.data_list[ind]
            R = self.compute_rate(index=ind)
            ll += self._poisson_log_likelihood(S,R)

        return ll

    def _log_likelihood_single_process(self, k):
        """
        Helper function to compute the log likelihood of a single process
        :param k: process to compute likelihood for
        :return:
        """
        ll = 0

        # Get the likelihood of the datasets
        for ind,(S,_,_,_)  in enumerate(self.data_list):
            Rk = self.compute_rate(index=ind, proc=k)
            ll += self._poisson_log_likelihood(S[:,k], Rk)

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
    _weight_class = SpikeAndSlabGammaWeights
    _parent_class = SpikeAndSlabParents

    def resample_model(self):
        """
        Perform one iteration of the Gibbs sampling algorithm.
        :return:
        """
        # Update the parents.
        # THIS MUST BE DONE IMMEDIATELY FOLLOWING WEIGHT UPDATES!
        for _,_,_,p in self.data_list:
            p.resample(self.bias_model, self.weight_model, self.impulse_model)

        # Update the bias model given the parents assigned to the background
        self.bias_model.resample(
            data=np.concatenate([p.Z0 for (_,_,_,p) in self.data_list]))

        # Update the impulse model given the parents assignments
        self.impulse_model.resample(
            data=np.concatenate([p.Z for (_,_,_,p) in self.data_list]))

        # Update the network model
        self.network.resample(data=(self.weight_model.A, self.weight_model.W))

        # Update the weight model given the parents assignments
        self.weight_model.resample(
            model=self,
            N=np.atleast_1d(np.sum([N for (_,N,_,_) in self.data_list], axis=0)),
            Z=np.concatenate([p.Z for (_,_,_,p) in self.data_list]))

    def initialize_with_standard_model(self, standard_model):
        super(DiscreteTimeNetworkHawkesModelSpikeAndSlab, self).\
            initialize_with_standard_model(standard_model)

        # Update the parents.
        for _,_,_,p in self.data_list:
            p.resample(self.bias_model, self.weight_model, self.impulse_model)


class DiscreteTimeNetworkHawkesModelGammaMixture(
    _DiscreteTimeNetworkHawkesModelBase, ModelGibbsSampling, ModelMeanField):
    _weight_class = GammaMixtureWeights
    _parent_class = GammaMixtureParents

    def resample_model(self, resample_network=True):
        """
        Perform one iteration of the Gibbs sampling algorithm.
        :return:
        """
        # Update the parents.
        for _,_,_,p in self.data_list:
            p.resample(self.bias_model, self.weight_model, self.impulse_model)

        # Update the bias model given the parents assigned to the background
        self.bias_model.resample(
            data=np.concatenate([p.Z0 for (_,_,_,p) in self.data_list]))

        # Update the impulse model given the parents assignments
        self.impulse_model.resample(
            data=np.concatenate([p.Z for (_,_,_,p) in self.data_list]))

        # Update the weight model given the parents assignments
        self.weight_model.resample(
            N=np.atleast_1d(np.sum([N for (_,N,_,_) in self.data_list], axis=0)),
            Z=np.concatenate([p.Z for (_,_,_,p) in self.data_list]))

        # Update the network model
        if resample_network:
            self.network.resample(data=(self.weight_model.A, self.weight_model.W))

    def initialize_with_standard_model(self, standard_model):
        super(DiscreteTimeNetworkHawkesModelGammaMixture, self).\
            initialize_with_standard_model(standard_model)

        # Set the mean field parameters
        self.bias_model.mf_alpha = 100 * self.bias_model.lambda0
        self.bias_model.mf_beta  = 100 * np.ones(self.K)

        # Weight model
        self.weight_model.mf_kappa_0 = self.weight_model.nu_0 * self.weight_model.W
        self.weight_model.mf_v_0     = self.weight_model.nu_0 * np.ones((self.K, self.K))

        self.weight_model.mf_kappa_1 = 100 * self.weight_model.W.copy()
        self.weight_model.mf_v_1     = 100 * np.ones((self.K, self.K))

        self.weight_model.mf_p       = 0.8 * self.weight_model.A + 0.2 * (1-self.weight_model.A)

        # Set mean field parameters of the impulse model
        self.impulse_model.mf_gamma = 100 * self.impulse_model.g.copy('C')

        # Set network mean field parameters
        self.network.mf_m = 0.2 / (self.C-1) * np.ones((self.K, self.C))
        for c in xrange(self.C):
            self.network.mf_m[self.network.c == c, c] = 0.8

        # Update the parents.
        # for _,_,_,p in self.data_list:
        #     p.resample(self.bias_model, self.weight_model, self.impulse_model)
        #     p.meanfieldupdate(self.bias_model, self.weight_model, self.impulse_model)

    def meanfield_coordinate_descent_step(self):
        # Update the parents.
        for _,_,_,p in self.data_list:
            p.meanfieldupdate(self.bias_model, self.weight_model, self.impulse_model)

        # Update the bias model given the parents assigned to the background
        self.bias_model.meanfieldupdate(
            EZ0=np.concatenate([p.EZ0 for (_,_,_,p) in self.data_list]))

        # Update the impulse model given the parents assignments
        self.impulse_model.meanfieldupdate(
            EZ=np.concatenate([p.EZ for (_,_,_,p) in self.data_list]))

        # Update the weight model given the parents assignments
        self.weight_model.meanfieldupdate(
            N=np.atleast_1d(np.sum([N for (_,N,_,_) in self.data_list], axis=0)),
            EZ=np.concatenate([p.EZ for (_,_,_,p) in self.data_list]))

        # Update the network model
        self.network.meanfieldupdate(self.weight_model)

        return self.get_vlb()

    def get_vlb(self):
        # Compute the variational lower bound
        vlb = 0
        for _,_,_,p in self.data_list:
            vlb += p.get_vlb(self.bias_model, self.weight_model, self.impulse_model)
        vlb += self.bias_model.get_vlb()
        vlb += self.impulse_model.get_vlb()
        vlb += self.weight_model.get_vlb()
        vlb += self.network.get_vlb()
        return vlb

    def sgd_step(self, minibatchsize, stepsize):
        # Sample a minibatch of data
        assert len(self.data_list) == 1, "We only sample from the first data set"
        S,_,F,_ = self.data_list[0]
        T = S.shape[0]

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
        N_minibatch = S_minibatch.sum(axis=0)
        T_minibatch = S_minibatch.shape[0]
        minibatchfrac = float(T_minibatch) / T

        # Create a parent object for this minibatch
        p = self._parent_class(T_minibatch, self.K, self.B, S_minibatch, F_minibatch)

        # TODO: Grab one dataset from the data_list and assume
        # it has been added in minibatches

        # Update the parents using a standard mean field update
        p.meanfieldupdate(self.bias_model, self.weight_model, self.impulse_model)

        # Update the bias model given the parents assigned to the background
        self.bias_model.meanfield_sgdstep(p.EZ0,
                                          minibatchfrac=minibatchfrac,
                                          stepsize=stepsize)

        # Update the impulse model given the parents assignments
        self.impulse_model.meanfield_sgdstep(p.EZ,
                                             minibatchfrac=minibatchfrac,
                                             stepsize=stepsize)

        # Update the weight model given the parents assignments
        # Compute the number of events in the minibatch
        self.weight_model.meanfield_sgdstep(N=N_minibatch, EZ=p.EZ,
                                            minibatchfrac=minibatchfrac,
                                            stepsize=stepsize)

        # Update the network model. This only depends on the global weight model,
        # so we can just do a standard mean field update
        self.network.meanfield_sgdstep(self.weight_model,
                                       minibatchfrac=minibatchfrac,
                                       stepsize=stepsize)

    def resample_from_mf(self):
        self.bias_model.resample_from_mf()
        self.weight_model.resample_from_mf()
        self.impulse_model.resample_from_mf()
        self.network.resample_from_mf()
