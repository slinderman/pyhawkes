"""
Top level classes for the Hawkes process model.
"""
import numpy as np
from scipy.special import gammaln

from pyhawkes.deps.pybasicbayes.models import ModelGibbsSampling
from pyhawkes.internals.bias import GammaBias
from pyhawkes.internals.weights import SpikeAndSlabGammaWeights
from pyhawkes.internals.impulses import DirichletImpulseResponses
from pyhawkes.internals.parents import Parents
from pyhawkes.utils.basis import CosineBasis

class DiscreteTimeNetworkHawkesModel(ModelGibbsSampling):
    """
    Discrete time network Hawkes process model with support for
    Gibbs sampling inference, variational inference (TODO), and
    stochastic variational inference (TODO).
    """

    def __init__(self, K, dt=1.0, dt_max=100.0,
                 B=5, basis=None,
                 alpha0=1.0, beta0=1.0,
                 alphaW=1.0, betaW=1.0, rho=1.0,
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
        self.weight_model = SpikeAndSlabGammaWeights(self.K, rho=rho, alpha=alphaW, beta=betaW)
        self.impulse_model = DirichletImpulseResponses(self.K, self.B, gamma=gamma)

        # Initialize the data list to empty
        self.data_list = []

    def add_data(self, S):
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
        N = S.sum(axis=0)

        # Filter the data into a TxKxB array
        F = self.basis.convolve_with_basis(S)

        # Instantiate corresponding parent object
        parents = Parents(T, self.K, self.B, S, F)

        # Add to the data list
        self.data_list.append((S, N, F, parents))

    def generate(self, keep=True, T=100):
        """
        Generate a new data set with the sampled parameters

        :param keep: If True, add the generated data to the data list.
        :param T:    Number of time bins to simulate.
        :return: A TxK
        """
        assert isinstance(T, int), "T must be an integer number of time bins"

        # Initialize the output
        S = np.zeros((T, self.K))

        # Precompute the impulse responses (KxKxB array)
        G = np.tensordot(self.basis.basis, self.impulse_model.beta, axes=([1], [2]))
        L = self.basis.L
        assert G.shape == (L,self.K, self.K)
        H = self.weight_model.A[None,:,:] * \
            self.weight_model.W[None,:,:] * \
            G

        # Compute the rate matrix R
        R = np.zeros((T, self.K))

        # Add the background rate
        R += self.bias_model.lambda0[None,:]

        # Iterate over time bins
        for t in xrange(T):
            if t % 1000 == 0:
                print "t=%d" % t

            # Sample a Poisson number of events for each process
            S[t,:] = np.random.poisson(R[t,:] * self.dt)

            # For each sampled event, add a weighted impulse response to the rate
            for k in xrange(self.K):
                if S[t,k] > 0:
                    R[t+1:t+L+1,:] += H[:,k,:]

            # Check Spike limit
            if np.any(S[t,:] >= 1000):
                print "More than 1000 events in one time bin!"
                import pdb; pdb.set_trace()

        if keep:
            # Xs = [X[:T,:] for X in Xs]
            # data = np.hstack(Xs + [S])
            self.add_data(S)

        return S, R


    def resample_model(self):
        """
        Perform one iteration of the Gibbs sampling algorithm.
        :return:
        """
        # Update the bias model given the parents assigned to the background
        self.bias_model.resample(
            data=np.concatenate([p.Z0] for (_,_,_,p) in self.data_list))

        # Update the impulse model given the parents assignments
        self.impulse_model.resample(
            data=np.concatenate([p.Z] for (_,_,_,p) in self.data_list))

        # Update the weight model given the parents assignments
        self.weight_model.resample(
            N=np.sum([N for (_,N,_,_) in self.data_list]),
            Z=np.concatenate([p.Z] for (_,_,_,p) in self.data_list))

        # Update the parents.
        # THIS MUST BE DONE IMMEDIATELY FOLLOWING WEIGHT UPDATES!
        for _,_,_,p in self.data_list:
            p.resample(self.bias_model, self.weight_model, self.impulse_model)


    def compute_rate(self, S):
        """
        Compute the rate function for a given data set
        :param S:   TxK array of event counts for which we would like to
                    compute the model's rate
        :return:    TxK array of rates
        """
        raise NotImplementedError()

    def heldout_log_likelihood(self, S):
        """
        Compute the held out log likelihood of a data matrix S.
        :param S:   TxK matrix of event counts
        :return:    log likelihood of those counts under the current model
        """
        dt = self.dt
        R = self.compute_rate(S)

        return (-gammaln(S+1) + S * np.log(R*self.dt) - R*dt).sum()
