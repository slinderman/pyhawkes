import os
import numpy as np

def initialize_pyrngs():
    from gslrandom import PyRNG, get_omp_num_threads
    if "OMP_NUM_THREADS" in os.environ:
        num_threads = os.environ["OMP_NUM_THREADS"]
    else:
        num_threads = get_omp_num_threads()
    assert num_threads > 0

    # Choose random seeds
    seeds = np.random.randint(2**16, size=num_threads)
    return [PyRNG(seed) for seed in seeds]

def convert_discrete_to_continuous(S, dt):
    # Convert S to continuous time
    from pybasicbayes.util.general import ibincount
    T = S.shape[0] * dt
    S_ct = dt * np.concatenate([ibincount(Sk) for Sk in S.T]).astype(float)
    S_ct += dt * np.random.rand(*S_ct.shape)
    assert np.all(S_ct < T)
    C_ct = np.concatenate([k*np.ones(Sk.sum()) for k,Sk in enumerate(S.T)]).astype(int)

    # Sort the data
    perm = np.argsort(S_ct)
    S_ct = S_ct[perm]
    C_ct = C_ct[perm]
    return S_ct, C_ct, T

def get_unique_file_name(filedir, filename):
    """
    Get a unique filename by appending filename with .x, where x
    is the next untaken number
    """
    import fnmatch
    
    # Get the number of conflicting log files
    fnames = os.listdir(filedir)
    conflicts = fnmatch.filter(fnames, "%s*" % filename)
    nconflicts = len(conflicts)
    
    if nconflicts > 0:
        unique_name = "%s.%d" % (filename, nconflicts+1)
    else:
        unique_name = filename
        
    return unique_name

def logistic(x,lam_max=1.0):
    return lam_max*1.0/(1.0+np.exp(-x))

def logit(x,lam_max=1.0):
    return np.log(x/lam_max)-np.log(1-(x/lam_max))

def sample_nig(mu0, lmbda0, alpha0, beta0):
    shp = mu0.shape
    assert lmbda0.shape == alpha0.shape == beta0.shape == shp
    tau = np.array(np.random.gamma(alpha0, 1./beta0)).reshape(shp)
    mu = np.array(np.random.normal(mu0, 1./(lmbda0 * tau))).reshape(shp)
    return mu, tau