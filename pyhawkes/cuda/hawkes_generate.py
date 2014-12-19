"""
Scott Linderman
6/28/2012

Generate data from a Hawkes process with given parameters.
"""
import numpy as np
import scipy.special
import scipy.sparse.linalg

from hawkes_consts import *

from pyhawkes.utils.utils import *
import pyhawkes.utils.poisson_process as pp

def sampleHawkesModelParams(K, T, gParams):
    """
    Sample parameters for a Hawkes process from prior distributions
    """
    
    # Sample adjacency and weight matrix
    # by first sampling sparsity
    mParams = {}
    
    # Assert that matrix is stable
    stable = False
    attempts = 0
    while not stable and attempts < 25:
        mParams["rho"] = np.random.beta(gParams["a_rho"], gParams["b_rho"])
        mParams["A"] = np.random.rand(K, K) < mParams["rho"]
        mParams["W"] = np.random.gamma(gParams["a_w"], 1.0/gParams["b_w"], (K,K)) * mParams["A"]
        #(ev,_) = np.abs(scipy.sparse.linalg.eigs(mParams["W"], k=1))
        ev = np.abs(np.linalg.eigvals(mParams["W"]))
        stable = np.all(ev <= 1.0)
        attempts += 1
    assert stable, "ERROR: Failed to sample stable weight matrix in 25 attempts!"
    
    # Sample background firing rates
    mParams["mu"] = np.random.gamma(gParams["a_mu"], 1.0/gParams["b_mu"], size=(K,))
    
    # Sample parameters of the impulse response kernel    
    if gParams["G_type"] == G_LOGISTIC_NORMAL:
        mParams["g_tau"] = np.random.gamma(gParams["a_tau_0"], 1.0/gParams["b_tau_0"])
        mParams["g_mu"] = np.random.normal(gParams["mu_mu_0"], np.sqrt(1.0/(gParams["kappa_mu_0"]*mParams["g_tau"])))

    return mParams

def generateHawkesProcess(K, T, gParams, mParams):
    """
    Generate data from a Hawkes process with priors given
    in the param dict
    """
    mu = mParams["mu"]
    
    # Initialize output arrays, a dictionary of numpy arrays
    S = {}
    N = np.zeros(K)
    gParams["buff_sz"] = 10*np.max(mu)*T
    for k in np.arange(K):
        S[k] = np.zeros(gParams["buff_sz"])

    # Sample background spikes
    for k in np.arange(K):
        N[k] = np.random.poisson(mu[k]*T)
        S[k][0:N[k]] = np.random.rand(N[k])*T        
    
    # Each background spike induces a tree of child spikes
    for k in np.arange(K):
        # N vector changes each time generateHawkesHelper is called  
        Nk = N[k] 
        for s in S[k][0:Nk]:
            generateHawkesHelper(K, T, gParams, mParams, S, N, s, k, round=0)
            
    # Trim and sort spikes in chronolocial order
    for k in np.arange(K):
        S[k] = S[k][0:N[k]]
        S[k] = np.sort(S[k])
        
    return (S,N)
    
def generateTestHawkesProcess(K, T, gParams, mParams):
    """
    Generate data from a Hawkes process with priors given
    in the param dict
    """
    mParams["g_tau"] = np.float32(mParams["g_tau"])
    mu = mParams["mu"]
    
    # Initialize output arrays, a dictionary of numpy arrays
    S = {}
    N = np.zeros(K)
    gParams["buff_sz"] = 10*np.max(mu)*T
    for k in np.arange(K):
        S[k] = np.zeros((gParams["buff_sz"],))

    # Hardcode background spike on first neuron at time 1
    N[0] = 1
    S[0][0] = 1.0
    
    # Each background spike induces a tree of child spikes
    for k in np.arange(K):
        # N vector changes each time generateHawkesHelper is called  
        Nk = N[k] 
        for s in S[k][0:Nk]:
            generateHawkesHelper(K, T, gParams, mParams, S, N, s, k, round=0)
            
    # Trim and sort spikes in chronolocial order
    for k in np.arange(K):
        S[k] = S[k][0:N[k]]
        S[k] = np.sort(S[k])
        
    return (S,N)    

def generateHawkesHelper(K, T, gParams, mParams, S, N, s_pa, k_pa, round=0):
    """
    Recursively generate new generations of spikes with 
    given impulse response parameters. Takes in a single spike 
    as the parent and recursively calls itself on all children
    spikes
    """
    max_round = 1000
    assert round < max_round, "Exceeded maximum recursion depth of %d" % max_round
    
    for k_ch in np.arange(K):
        w = mParams["W"][k_pa,k_ch]
        a = mParams["A"][k_pa,k_ch] 
        
        if w==0 or a==False:
            continue
        
        # The total area under the impulse response curve(ratE)  is w
        # Sample spikes from a homogenous poisson process with rate
        # 1 until the time exceeds w. Then transform those spikes
        # such that they are distributed under a logistic normal impulse
        n_ch = np.random.poisson(w)
        # The offsets in the cdf are uniform on [0,1]
        u_ch = np.random.rand(n_ch)
        
        if gParams["G_type"] == G_LOGISTIC_NORMAL:
            # Sample normal RVs and take the logistic of them. This is equivalent
            # to sampling uniformly from the inverse CDF 
            x_ch = mParams["g_mu"] + 1/mParams["g_tau"]*np.random.randn(n_ch)
            # x_ch = scipy.special.erfinv(2*u_ch-1)*np.sqrt(2/mParams["g_tau"])+mParams["g_mu"]
            
            # Spike times are logistic transformation of x
            s_ch = s_pa + gParams["dt_max"] * 1/(1+np.exp(-1*x_ch))
            
        # Only keep spikes within the simulation time interval
        s_ch = s_ch[s_ch < T]
        n_ch = len(s_ch)
          
        # Append the new spikes to the dataset, growing buffer as necessary
        if N[k_ch] + n_ch > len(S[k_ch]):
            S[k_ch] = np.hstack((S[k_ch], np.zeros(gParams["buff_sz"])))

        
        S[k_ch][N[k_ch]:N[k_ch]+n_ch] = s_ch
        N[k_ch] += n_ch
        
        # Generate offspring from child spikes
        for s in s_ch:
            generateHawkesHelper(K, T, gParams, mParams, S, N, s, k_ch, round=round+1)
  
def flattenSpikeDict(Sdict,N,K,T):
    """
    Hawkes_mcmc expects spike times in a 1D vector S along with
    vector C identifying neurons on which spikes occur
    """
    
    S = np.empty(np.sum(N))
    C = np.empty(np.sum(N))
    
    off = 0
    for k in np.arange(K):
        S[off:off+N[k]] = Sdict[k]
        C[off:off+N[k]] = k
        off += N[k]
        
    # Sort the joint arrays
    perm = np.argsort(S)
    S = S[perm]
    C = C[perm]
    
    # Package into dict
    data = {}
    data["N"] = int(np.sum(N))
    data["Ns"] = N
    data["cumSumNs"] = np.cumsum(np.hstack(([0], N)), dtype=np.int32)
    data["maxNs"] = int(max(N))
    data["K"] = int(K)
    data["T"] = np.float32(T)
    data["C"] = C
    data["S"] = S
    
    return data

# Test
if __name__ == "__main__":
    configDir = "config"    
    configFile = "hawkes_excite_test_10.cfg"
    
    K = 2
    T = 10
    gParams = parseConfigFile(configFile, configDir)
    mParams = sampleHawkesModelParams(K, T, gParams)
    mParams["A"] = np.array([[0, 1], [0, 0]])
    mParams["W"] = np.array([[0, 10], [0, 0]])
    mParams["g_mu"] = 0
    mParams["g_tau"] = 100
    mParams["mu"] = 1 * np.ones((2,))
    (S,N) = generateTestHawkesProcess(K, T, gParams, mParams) 
    
    print N
    
#    # Plot the spike train
    import matplotlib.pyplot as plt
    plt.figure()
    plt.subplot(211)
    plt.stem(S[0], np.ones(N[0]))
    plt.xlim((0, T))
    
    # Plot the rate for neuron 0. Discretize time
    bin_sz = 0.01
    T_edges = np.arange(0,T, bin_sz)
    T_t = (T_edges[1:]+T_edges[:-1])/2
    G_t = np.arange(bin_sz, gParams["dt_max"]-bin_sz, bin_sz)
    G = np.abs(gParams["dt_max"]*G_t/(G_t**2 * (gParams["dt_max"]-G_t)))
    G = G * np.sqrt(mParams["g_tau"]/(2*np.pi))*np.exp(-1*mParams["g_tau"]/2*(-np.log((gParams["dt_max"]-G_t)/G_t)-mParams["g_mu"])**2) 
    r0 = mParams["mu"][0]*np.ones(len(T_t))
    
    print "mu: %f" % mParams["g_mu"]
    print "tau: %f" % mParams["g_tau"]
    
    print mParams["W"][:,1]
    for k in np.arange(K):
        w = mParams["W"][k,1]
        if w == 0:
            pass
    
        (spk_cts, bins) = np.histogram(S[k], T_edges)
        rk0 = np.convolve(spk_cts, w*G)
        r0 = r0 + rk0[:len(r0)]
    
    plt.subplot(212)
    plt.plot(T_t, r0)
    plt.hold(True)
    if N[1] > 0:
        plt.stem(S[1], np.ones((N[1],)), 'r')   
        plt.xlim((0,T)) 
    plt.show()
    
    print np.trapz(r0, T_t)