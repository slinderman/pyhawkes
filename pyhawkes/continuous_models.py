"""
Top level classes for the Hawkes process model.
"""
import copy

import numpy as np
from scipy.optimize import leastsq
from scipy.sparse.linalg import eigs
from scipy.stats import norm

from pybasicbayes.abstractions import ModelGibbsSampling
from pyhawkes.standard_models import StandardHawkesProcess
from pyhawkes.internals.parents import ContinuousTimeParents, LatentContinuousTimeParents
from pyhawkes.internals.network import ErdosRenyiFixedSparsity
from pyhawkes.internals.bias import ContinuousTimeGammaBias
from pyhawkes.internals.impulses import ContinuousTimeImpulseResponses
from pyhawkes.internals.weights import SpikeAndSlabContinuousTimeGammaWeights


class _ContinuousTimeNetworkHawkesModelBase(ModelGibbsSampling):

    _bkgd_class = None
    _default_bkgd_hypers = {}


    _impulse_class = None
    _default_impulse_hypers = {}

    _default_weight_hypers = {}

    _network_class          = None
    _default_network_hypers = {}

    def __init__(self, K, dt_max=10.0,
                 bkgd_hypers={},
                 impulse_hypers={},
                 weight_hypers={},
                 network=None, network_hypers={}):
        """
        Initialize a discrete time network Hawkes model with K processes.

        :param K:  Number of processes
        """
        self.K      = K
        self.dt_max = dt_max

        # Initialize the bias
        # Use the given basis hyperparameters
        self.bkgd_hypers = copy.deepcopy(self._default_bkgd_hypers)
        self.bkgd_hypers.update(bkgd_hypers)
        self.bias_model = self._bkgd_class(self, self.K, **self.bkgd_hypers)

        # Initialize the impulse response model
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
            self.network = self._network_class(K=self.K, **self.network_hypers)

        # Initialize the weight model
        self.weight_hypers = copy.deepcopy(self._default_weight_hypers)
        self.weight_hypers.update(weight_hypers)
        self.weight_model = \
            SpikeAndSlabContinuousTimeGammaWeights(self,  **self.weight_hypers)

        # Initialize the data list to empty
        self.data_list = []


    # Expose basic variables
    @property
    def A(self):
        return self.weight_model.A

    @property
    def W(self):
        return self.weight_model.W

    @property
    def W_effective(self):
        return self.weight_model.W_effective

    @property
    def lambda0(self):
        return self.bias_model.lambda0

    @property
    def impulses(self):
        return self.impulse_model.impulses

    def get_parameters(self):
        """
        Get a copy of the parameters of the model
        :return:
        """
        return self.A, self.W, self.lambda0, self.impulses

    def initialize_with_standard_model(self, standard_model):
        """
        Initialize with a standard Hawkes model. Typically this will have
        been fit by gradient descent or BFGS, and we just want to copy
        over the parameters to get a good starting point for MCMC or VB.
        :param W:
        :param g:
        :return:
        """
        K = self.K
        assert isinstance(standard_model, StandardHawkesProcess)
        assert standard_model.K == K

        # lambda0 = standard_model.weights[:,0]
        lambda0 = standard_model.bias

        # Get the connection weights
        W = np.clip(standard_model.W, 1e-16, np.inf)

        # Get the impulse response parameters
        G = standard_model.G
        t_basis = standard_model.basis.dt * np.arange(standard_model.basis.L)
        t_basis = np.clip(t_basis, 1e-6, self.dt_max-1e-6)
        for k1 in range(K):
            for k2 in range(K):
                std_ir = standard_model.basis.basis.dot(G[k1,k2,:])

                def loss(mutau):
                    self.impulse_model.mu[k1,k2] = mutau[0]
                    self.impulse_model.tau[k1,k2] = mutau[1]
                    ct_ir = self.impulse_model.impulse(t_basis, k1, k2)

                    return ct_ir - std_ir

                mutau0 = np.array([self.impulse_model.mu[k1,k2],
                                   self.impulse_model.tau[k1,k2]])

                mutau, _ = leastsq(loss, mutau0)
                self.impulse_model.mu[k1,k2] = mutau[0]
                self.impulse_model.tau[k1,k2] = mutau[1]

        # We need to decide how to set A.
        # The simplest is to initialize it to all ones, but
        # A = np.ones((self.K, self.K))
        # Alternatively, we can start with a sparse matrix
        # of only strong connections. What sparsity? How about the
        # mean under the network model
        # sparsity = self.network.tau1 / (self.network.tau0 + self.network.tau1)
        sparsity = self.network.p
        A = W > np.percentile(W, (1.0 - sparsity) * 100)

        # Set the model parameters
        self.bias_model.lambda0 = lambda0.copy('C')
        self.weight_model.A     = A.copy('C')
        self.weight_model.W     = W.copy('C')

    def add_data(self, S, C, T, X=None):
        """
        Add a data set to the list of observations.
        First, filter the data with the impulse response basis,
        then instantiate a set of parents for this data set.

        :param S: length N array of event times
        :param C: length N array of process id's for each event
        :param T: max time. data is in [0,T)
        :param X: marks associated with each event
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
        parents = self._parents_class(self, S, C, X, T)

        # Add to the data list
        self.data_list.append(parents)

    def generate(self, keep=True, T=100.0, max_round=100, **kwargs):
        K, dt_max = self.K, self.dt_max
        W, A = self.weight_model.W, self.weight_model.A

        def _generate_helper(S, X, C, s_pa, x_pa, c_pa, round=0):
            # Recursively generate new generations of events with
            # given impulse response parameters. Takes in a single events
            # as the parent and recursively calls itself on all children
            # events

            assert round < max_round, "Exceeded maximum recursion depth of %d" % max_round

            for c_ch in np.arange(K):
                w = W[c_pa, c_ch]
                a = A[c_pa, c_ch]
                if w==0 or a==0:
                    continue

                # The total area under the impulse response curve(ratE)  is w
                # Sample evens from a homogenous poisson process with rate
                # 1 until the time exceeds w. Then transform those event times
                # such that they are distributed under a logistic normal impulse
                n_ch = np.random.poisson(w)

                # Sample children the impulse model
                s_ch, x_ch = self.impulse_model.rvs(n_ch, s_pa, x_pa, c_pa, c_ch)

                # Only keep spikes within the simulation time interval
                valid = s_ch < T
                s_ch = s_ch[valid]
                x_ch = x_ch[valid]
                n_ch = len(s_ch)

                S.append(s_ch)
                X.append(x_ch)
                C.append(c_ch * np.ones(n_ch, dtype=np.int))

                # Generate offspring from child spikes
                for s, x in zip(s_ch, x_ch):
                    _generate_helper(S, X, C, s, x, c_ch, round=round+1)

        # Each background spike spawns a cascade
        S, X, C = [], [], []
        S_bkgd, X_bkgd, C_bkgd = self.bias_model.rvs(T)
        S.append(S_bkgd)
        X.append(X_bkgd)
        C.append(C_bkgd)
        for s, x, c in zip(S_bkgd, X_bkgd, C_bkgd):
            _generate_helper(S, X, C, s, x, c)

        # Concatenate and sort arrays
        S, X, C = [np.concatenate(a) for a in (S,X,C)]
        perm = np.argsort(S)
        S, X, C = [a[perm] for a in (S,X,C)]

        if keep:
            self.add_data(S, C, T, X=X)

        return S, X, C

    def check_stability(self):
        """
        Check that the weight matrix is stable

        :return:
        """
        if self.K < 100:
            eigvals = np.linalg.eigvals(self.weight_model.W_effective)
            maxeig = np.amax(np.real(eigvals))
        else:
            maxeig = eigs(self.weight_model.W_effective, k=1)[0]

        print("Max eigenvalue: ", maxeig)
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
            ll += d.log_likelihood()
            # ll -= self.compute_integrated_rate(d).sum()
            # ll += np.log(self.compute_rate_at_events(d)).sum()

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

    def compute_impulses(self, dt=1.0):
        dts = np.concatenate([np.arange(0, self.dt_max, step=dt), [self.dt_max]])
        ir = np.zeros((dts.size, self.K, self.K))
        for k1 in range(self.K):
            for k2 in range(self.K):
                ir[:,k1,k2] = self.impulse_model.impulse(dts, k1, k2)
        return ir, dts


    ### Inference
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

        # # Update the impulse model given the parents assignments
        self.impulse_model.resample(self.data_list)

        # Update the network model
        self.network.resample(data=(self.weight_model.A, self.weight_model.W))

        # Update the weight model given the parents assignments
        self.weight_model.resample(self.data_list)


class ContinuousTimeNetworkHawkesModel(_ContinuousTimeNetworkHawkesModelBase):
    _bkgd_class = ContinuousTimeGammaBias
    _default_bkgd_hypers = {"alpha" : 1.0, "beta" : 1.0}


    _impulse_class = ContinuousTimeImpulseResponses
    _default_impulse_hypers = {"mu_0": 0., "lmbda_0": 1.0, "alpha_0": 1.0, "beta_0" : 1.0}
    _default_weight_hypers = {}

    _network_class          = ErdosRenyiFixedSparsity
    _default_network_hypers = {'p': 0.5,
                               'allow_self_connections': True,
                               'kappa': 1.0,
                               'v': None, 'alpha': None, 'beta': None,}

    _parents_class = ContinuousTimeParents

class ContinuousTimeLatentHawkesModel(ContinuousTimeNetworkHawkesModel):
    """
    A Hawkes process with latent nodes.
    """
    _parents_class = LatentContinuousTimeParents

    def __init__(self, K, H, **kwargs):
        """
        Same as the ContinuousTimeNetworkHawkesModel, but also specify
        the number of hidden nodes.

        :param K:  Number of observed nodes.
        :param H:  Number of hidden nodes
        """
        self.H = H
        self.K_obs = K
        super(ContinuousTimeLatentHawkesModel, self).__init__(K=K+H, **kwargs)

    def add_data(self, S, C, T, X=None, includes_hidden=False):
        X = X if X is not None else [None] * len(S)
        # If the data already has spikes for the "hidden" nodes,
        # include them. Otherwise, sample from the background rate.
        if not includes_hidden:
            S_bkgd, X_bkgd, C_bkgd = self.bias_model.rvs(T)
            i_hidden = np.where(C_bkgd>=self.K_obs)[0]
            S_hidden, X_hidden, C_hidden = [a[i_hidden] for a in (S_bkgd, X_bkgd, C_bkgd)]

            S_full = np.concatenate((S, S_hidden))
            C_full = np.concatenate((C, C_hidden))
            X_full = np.concatenate((X, X_hidden))

            # Sort
            perm = np.argsort(S_full)
            S_full = S_full[perm]
            C_full = C_full[perm]
            X_full = X_full[perm]

        else:
            S_full, C_full, X_full = S, C, X

        super(ContinuousTimeLatentHawkesModel, self).add_data(S_full, C_full, T, X=X_full)
