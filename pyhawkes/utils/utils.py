import os
import numpy as np


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
    tau = np.random.gamma(alpha0, 1./beta0)
    mu = np.random.normal(mu0, 1./(lmbda0 * tau))
    return mu, tau