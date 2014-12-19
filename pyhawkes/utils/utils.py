import os
import time
import io
import logging

import numpy as np
import scipy.stats as stats
import scipy

import pycuda.compiler as nvcc
import pycuda.driver as cuda

perfTimers = {}

log = logging.getLogger("global_log")

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
    

def initialize_logger(params):
    """
    Initialize a logger object. 
    Automatically detect conflicting log names.
    """
    # Get the number of conflicting log files
    log_file = get_unique_file_name(params["log_dir"], params["log_file"])
    log_path = os.path.join(params["log_dir"], log_file)

    # Initialize the global logger
    log.setLevel(logging.DEBUG)

    # create a file handler 
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(logging.DEBUG)
    
    # add formatter to the file handler
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    log.addHandler(file_handler)
    
    # add handlers to logger
    if params["print_to_console"]:
        # create console handler and set level to debug
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        log.addHandler(console_handler)
    

def startPerfTimer(perfDict, key):
    """
    Start a perf timer (update t0)
    """
    global perfTimers
    
    if perfDict != None:
        perfTimers[key] = time.clock()
    
def stopPerfTimer(perfDict, key):
    global perfTimers
    
    if perfDict != None:
        try:
            t0 = perfTimers[key]
            assert t0 != None, "startPerfTimer must be called before stopPerfTimer!"
            tf = time.clock()
            try:        
                perfDict[key] += tf-t0
            except KeyError:
                perfDict[key] = tf-t0
                
            perfTimers[key] = None
        except:
            assert False, "startPerfTimer must be called before stopPerfTimer"
                
def printPerfTimers(perfDict):
    entries = perfDict.items()
    entries.sort(lambda (k1,v1),(k2,v2): int(np.sign(v2-v1)))
    
    for (k,v) in entries:
        print "%s:  %f" % (k,v)
            
def showDeviceAttributes():

    (free,total)=cuda.mem_get_info()
    print("Global memory occupancy:%f%% free"%(free*100/total))
    
    for devicenum in range(cuda.Device.count()):
        device=cuda.Device(devicenum)
        attrs=device.get_attributes()
    
        #Beyond this point is just pretty printing
        print("\n===Attributes for device %d"%devicenum)
        for (key,value) in attrs.iteritems():
            print("%s:%s"%(str(key),str(value)))

def showKernelMemoryInfo(kernel, name=""):
    shared=kernel.shared_size_bytes
    regs=kernel.num_regs
    local=kernel.local_size_bytes
    const=kernel.const_size_bytes
    mbpt=kernel.max_threads_per_block
    print("""%s Memory USAGE:\nLocal:%d,\nShared:%d,\nRegisters:%d,\nConst:%d,\nMax Threads/B:%d"""%(name, local,shared,regs,const,mbpt))
    
def compile_kernels(srcFile, kernelNames, srcParams=None):
    """
    Load the GPU kernels from the specified CUDA C file 
    """
    
    # Read the src file into a string
    custr = ""
    with io.open(srcFile, 'r') as file:
        for l in file:
            custr += l
    
    ## Replace consts in cu file
    if srcParams != None:
        custr = custr % srcParams
    
    # Compile the CUDA Kernel
    cu_kernel_source_module = nvcc.SourceModule(custr)
    
    # Load the kernels into a dictionary
    kernels = {}
    for name in kernelNames:
        try:
            kernels[name] = cu_kernel_source_module.get_function(name)
        except:
            log.error("Failed to find kernel function: %s", name)
            exit()
        
    return kernels
        
def log_sum_exp_sample(lnp):
    """
    Sample uniformly from a vector of unnormalized log probs using 
    the log-sum-exp trick
    """
    assert np.ndim(lnp) == 1, "ERROR: logSumExpSample requires a 1-d vector"
    lnp = np.ravel(lnp)
    N = np.size(lnp)
    
    # Use logsumexp trick to calculate ln(p1 + p2 + ... + pR) from ln(pi)'s
    max_lnp = np.max(lnp)
    denom = np.log(np.sum(np.exp(lnp-max_lnp))) + max_lnp
    p_safe = np.exp(lnp - denom)
    
    # Normalize the discrete distribution over blocks
    sum_p_safe = np.sum(p_safe)
    if sum_p_safe == 0 or not np.isfinite(sum_p_safe):
        log.error("total probability for logSumExp is not valid! %f", sum_p_safe)
        log.info(p_safe)
        log.info(lnp)
        raise Exception("Invalid input. Probability infinite everywhere.")
    
    # Randomly sample a block
    choice = -1
    u = np.random.rand()
    acc = 0.0
    for n in np.arange(N):
        acc += p_safe[n]
        if u <= acc:
            choice = n
            break
    
    if choice == -1:
        raise Exception("Invalid choice in logSumExp!")
    
    return choice

def iwishrnd(dof, lmbda):
    """
    Copied from Matthew James Johnson at http://www.mit.edu/~mattjj/
    Returns a sample from the inverse Wishart distn, conjugate prior for precision matrices.
    """
    n = lmbda.shape[0]
    chol = np.linalg.cholesky(lmbda)
    
    if (dof <= 81+n) and (dof == np.round(dof)):
        x = np.random.randn(dof,n)
    else:
        d = np.sqrt(stats.chi2.rvs(dof-(np.arange(n))))
        if n==1:
            d = np.array([d])
        
        x = np.diag(d)
        x[np.triu_indices_from(x,1)] = np.random.randn(n*(n-1)/2)
    R = np.linalg.qr(x,'r')
    T = scipy.linalg.solve_triangular(R.T,chol.T).T
    return np.dot(T,T.T)

def wishrnd(dof, sigma):
    """
    Copied from Matthew James Johnson at http://www.mit.edu/~mattjj/
    Returns a sample from the Wishart distn, conjugate prior for precision matrices.
    """

    n = sigma.shape[0]
    chol = np.linalg.cholesky(sigma)

    # use matlab's heuristic for choosing between the two different sampling schemes
    if (dof <= 81+n) and (dof == round(dof)):
        # direct
        X = np.dot(chol,np.random.normal(size=(n,dof)))
    else:
        A = np.diag(np.sqrt(np.random.chisquare(dof - np.arange(0,n),size=n)))
        A[np.tri(n,k=-1,dtype=bool)] = np.random.normal(size=(n*(n-1)/2.))
        X = np.dot(chol,A)

    return np.dot(X,X.T)

def matrixNormalRnd(M,sqrtV,sqrtinvK,nsmpls=1):
    """
    Sample from a matrix normal distribution wiht mean M,
    column covariance V,
    and row covariance K 
    """
    
    assert isinstance(M,np.matrix), "matrixNormalRnd: M must be a matrix!"
    assert isinstance(sqrtV,np.matrix), "matrixNormalRnd: sqrtV must be a matrix!"
    assert isinstance(sqrtinvK,np.matrix), "matrixNormalRnd: sqrtinvK must be a matrix!"
    
    # Convert to a multivariate normal
    # Copying from matlab code requires us to reshape in column order
    mu = np.mat(np.reshape(M,(np.size(M),1),order="F"))
    
    # Additionally, matlab asssumes chol is upper tri by default
    # TODO: Can probably just reorder K and V, and then not have to transpose sqrtSigma below
    sqrtSigma = np.mat(np.kron(sqrtinvK.T, sqrtV.T))
    
    # Sample from the big multivariate normal
    z = np.mat(np.random.randn(len(mu),1))
    S = mu + sqrtSigma.T * z
    
    # Reshape into correct form    
    S = np.reshape(S,M.shape,order="F")

    return S

def logistic(x,lam_max=1.0):
    return lam_max*1.0/(1.0+np.exp(-x))

def logit(x,lam_max=1.0):
    return np.log(x/lam_max)-np.log(1-(x/lam_max))

def pprint_dict(D, name, level=logging.DEBUG):
    """
    Pretty print a dictionary
    """
    log.log(level, "Dictionary %s", name)
    
    for (key,value) in D.items():
        log.log(level, "%s:\t%s", str(key),str(value))