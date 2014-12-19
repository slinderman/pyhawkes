import os
import numpy as np
import scipy.cluster.vq
from ConfigParser import ConfigParser

import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray

from pyhawkes.utils.utils import pprint_dict, compile_kernels, wishrnd, log_sum_exp_sample
from model_extension import ModelExtension

import logging
log = logging.getLogger("global_log")

def construct_process_id_model(proc_id_model, baseModel, configFile):
    """
    Return an instance of the process ID model specified in parameters
    """
    
    if proc_id_model ==  "known":
        log.info("Creating known a priori process ID model")
        return KnownProcessIdModel(baseModel, configFile)
    elif proc_id_model ==  "spatial_gmm":
        log.info("Creating spatial GMM process ID model")
        return SpatialGmmProcessIdModel(baseModel, configFile)
    elif proc_id_model ==  "metaprocess":
        log.info("Creating meta-process ID model")
        return MetaProcessIdModel(baseModel, configFile)
    else:
        log.error("Unsupported process ID model: %s", proc_id_model)
        exit()
        
class KnownProcessIdModel(ModelExtension):
    """
    A process ID model in which the IDs are known a priori and do not change
    """
    def __init__(self, baseModel, configFile):
        self.base = baseModel
        
        # Initialize databases for this extension
        self.modelParams = baseModel.modelParams
        self.modelParams.addDatabase("proc_id_model")
        self.gpuPtrs = baseModel.gpuPtrs
        self.gpuPtrs.addDatabase("proc_id_model")
        
        # Set the base model's C and K values based on the base model's data
        if not self.base.data.proc_ids_known:
            log.error("K and C must be present in the data file for the KnownProcessIDModel!")
            exit()
            
    def getInitializationOrder(self):
        """
        Make sure the process ID model initializes first
        """
        return 0
    
    def initializeGpuMemory(self):
        """
        Allocate GPU memory for the base model parameters
        """
        N = self.base.data.N
        K = self.base.data.K
                
        self.gpuPtrs["proc_id_model","C"] = gpuarray.empty((N,), dtype=np.int32)
        self.gpuPtrs["proc_id_model","Ns"] = gpuarray.empty((K,), dtype=np.int32)
        
    def initializeModelParamsFromPrior(self):
        self.initializeModelParams()
        
    def initializeModelParamsFromDict(self, paramsDB):
        self.initializeModelParams()
        
    def initializeModelParams(self):
        self.initializeGpuMemory()
        
        self.modelParams["proc_id_model","K"] = self.base.data.K
        self.modelParams["proc_id_model","Ns"] = self.base.data.Ns
        self.modelParams["proc_id_model","C"] = self.base.data.C
        
        # Copy over the true process IDs to gpu
        self.gpuPtrs["proc_id_model","C"].set(self.base.data.C.astype(np.int32))
        self.gpuPtrs["proc_id_model","Ns"].set(self.base.data.Ns.astype(np.int32))
    
class SpatialGmmProcessIdModel(ModelExtension):
    """
    Process IDs form a GMM over the spatial location of each event.
    Assume the number of processes is held constant. Each process 
    has a spatial mean vector and covariance matrix, assumed to be 
    drawn from a Normal-Wishart prior. 
    
    A more sophisticated model might have a determinental point process
    for the mean/covariance of the mixture components. Additionally, 
    the distance between process centers can be used as a parameter
    in the prior on weights to encourage stronger interactions among 
    neighboring components. Again, this would introduce dependencies 
    among the process centers.
    """
    def __init__(self, baseModel, configFile):
        self.base = baseModel
        
        # Initialize databases for this extension
        self.modelParams = baseModel.modelParams
        self.modelParams.addDatabase("proc_id_model")
        self.gpuPtrs = baseModel.gpuPtrs
        self.gpuPtrs.addDatabase("proc_id_model")
        
        if not self.base.data.isspatial:
            log.error("SpatialGmmProcessIdModel only supports spatial datasets!")
            exit()
        
        # Check whether the data file has hard coded process IDs
        if self.base.data.proc_ids_known:
            log.warning("K and C are present in the data file but the SpatialGmmProcessIdModel ignores them!")
            
        # Parse the configuration file to get params
        self.parseConfigurationFile(configFile)
        pprint_dict(self.params, "Process ID Model Params")
        
        self.initializeGpuKernels()
        
    def getInitializationOrder(self):
        """
        Make sure the process ID model initializes first
        """
        return 0
    
    def parseConfigurationFile(self, configFile):
        """
        Parse the configuration file to get base model parameters
        """
        # Initialize defaults
        defaultParams = {}
        
        # CUDA kernels are defined externally in a .cu file
        defaultParams["cu_dir"]  = os.path.join("pyhawkes", "cuda", "cpp")
        defaultParams["cu_file"] = "process_id_kernels.cu"
        defaultParams["thin"] = 1
        defaultParams["sigma"] = 0.001
        defaultParams["kappa"] = 5
        defaultParams["nu"] = 4
        defaultParams["mu"] = "None"
                
        
        # Create a config parser object and read in the file
        cfgParser = ConfigParser(defaultParams)
        cfgParser.read(configFile)
        
        self.params = {}
        self.params["cu_dir"]  = cfgParser.get("proc_id_model", "cu_dir")
        self.params["cu_file"] = cfgParser.get("proc_id_model", "cu_file")
        self.params["thin"] = cfgParser.getint("proc_id_model", "thin")
        self.params["blockSz"] = cfgParser.getint("cuda", "blockSz")
        
        # Parse the params for the spatial GMM model
        self.params["sigma0"] = cfgParser.getfloat("proc_id_model", "sigma")
        self.params["kap0"] = cfgParser.getfloat("proc_id_model", "kappa")
        self.params["nu0"] = cfgParser.getfloat("proc_id_model", "nu")
        
        
        # Parse mu0from config file
        mu0_str = cfgParser.get("proc_id_model", "mu") 
        if mu0_str == "None":
            # If not specified, take the mean of the data
            self.params["mu0"] = np.mean(self.base.data.X, 1)
        else:
            # Filter out unwanted characters
            mu0_str = filter(lambda c: c.isdigit() or c=="," or c=="-" or c==".", mu0_str)
            self.params["mu0"] = np.fromstring(mu0_str, sep=",",dtype=np.float32)
        
        
        self.params["T0"] = self.params["sigma0"]*np.eye(self.base.data.D)
        
        # Parse the desired number of mixture components/processes
        self.params["K"] = cfgParser.getint("proc_id_model", "K")
        
    def initializeGpuKernels(self):
        kernelSrc = os.path.join(self.params["cu_dir"], self.params["cu_file"])
        kernelNames = ["computeXSum",
                       "computeXVarSum",
                       "computePerSpikePrCn"]
        src_consts = {"B" : self.params["blockSz"]}
        self.gpuKernels = compile_kernels(kernelSrc, kernelNames, src_consts)
        
    def initializeGpuMemory(self):
        K = self.params["K"]
        N = self.base.data.N
        D = self.base.data.D
        gridx = int(np.ceil(np.float32(N)/self.params["blockSz"]))
        
        self.gpuPtrs["proc_id_model","C"] = gpuarray.empty((N,), dtype=np.int32)
        self.gpuPtrs["proc_id_model","Ns"] = gpuarray.empty((K,), dtype=np.int32)
        
        self.gpuPtrs["proc_id_model","Xstats"] = gpuarray.empty((K,gridx), dtype=np.float32)
        self.gpuPtrs["proc_id_model","Xmean"] = gpuarray.empty((K,D), dtype=np.float32)
        self.gpuPtrs["proc_id_model","Xprec"] = gpuarray.empty((K,D,D), dtype=np.float32)      
        
    
    def initializeModelParamsFromPrior(self):
        """
         Get K and call the event handler framework to trigger other 
         extension updates
        """  
        self.initializeGpuMemory()
        
        K = self.params["K"]
        N = self.base.data.N
        self.modelParams["proc_id_model","K"] = K
                
#        # For each process, sample and add new parameters
#        for k in np.arange(1,K):
#            newProcParams = self.base.sampleNewProcessParams()
#            self.base.addNewProcessEventHandler(newProcParams)
            
        
#        # Update Ns -- initially all spikes are on process 0
#        self.modelParams["proc_id_model","Ns"] = np.zeros(K)
#        self.modelParams["proc_id_model","Ns"][0] = N
#        
#        # If C is given in the dataset, use this as initialization
#        if self.base.data.C != None and self.base.data.Ns != None:
#            log.info("C given in dataset, using it as initialization for Spatial GMM process model")
#            assert self.base.data.K  == self.params["K"]
#            self.gpuPtrs["proc_id_model","C"].set(self.base.data.C)
#            self.modelParams["proc_id_model","Ns"] = self.base.data.Ns

        # Run k-means to initialize the process assignments
        # Use Scipy kmeans to do the clustering
        (_,C) = scipy.cluster.vq.kmeans2(self.base.data.X.T, K)
        self.modelParams["proc_id_model","C"] = C
        self.gpuPtrs["proc_id_model","C"].set(C.astype(np.int32))
        
        
        # Update Ns
        Ns = np.zeros((K,),dtype=np.int32)
        for n in np.arange(N):
            Ns[C[n]] += 1
        log.debug("Ns0: %s", str(Ns))
        
        self.modelParams["proc_id_model","Ns"] = Ns
        self.gpuPtrs["proc_id_model","Ns"].set(Ns)
        
#        self.initializeModelParams()
#        self.initializeGpuMemory()
        
        
        self.param_iter = 0
        self.lvar_iter = 0
        
    def initializeModelParamsFromDict(self, paramsDB):
        """
        Copy the Gaussian params over, initialize process assignments according to
        probability given the Gaussians.
        """
        self.initializeGpuMemory()
        
        self.modelParams["proc_id_model","K"] = paramsDB["proc_id_model","K"]
        
        # Copy over the process mean and variances
        self.modelParams["proc_id_model","Xmean"] = paramsDB["proc_id_model","Xmean"]
        self.modelParams["proc_id_model","Xprec"] = paramsDB["proc_id_model","Xprec"]
            
        self.gpuPtrs["proc_id_model","Xmean"].set(self.modelParams["proc_id_model","Xmean"])
        self.gpuPtrs["proc_id_model","Xprec"].set(self.modelParams["proc_id_model","Xprec"])
    
        # Sample process IDs for each spike
#        self.sampleC()

        # Initialize all spikes to first process. This will be changed during burning
        C = np.zeros((self.base.data.N,), dtype=np.int32)
        self.modelParams["proc_id_model","C"] = C
        self.gpuPtrs["proc_id_model","C"].set(C)
        
        Ns = np.zeros((self.modelParams["proc_id_model","K"],), dtype=np.int32)
        Ns[0] = self.base.data.N
        self.modelParams["proc_id_model","Ns"] = Ns
        self.gpuPtrs["proc_id_model","Ns"].set(Ns)
    
    def sampleModelParameters(self):
        """
        Sample mean/covariance for each process and process affiliations for each event  
        """
        if np.mod(self.param_iter, self.params["thin"]) == 0:
            self.sampleProcessMeanCovar()
        
        self.param_iter += 1
        
    def sampleLatentVariables(self):
        if np.mod(self.lvar_iter, self.params["thin"]) == 0:
            self.sampleC()
            
        self.lvar_iter += 1
        
    def sampleProcessMeanCovar(self):
        """
        Sample the mean and covariance for each process given the current affiliations C
        """
        N = self.base.data.N
        K = self.modelParams["proc_id_model","K"]
        D = self.base.data.D
        Ns = self.modelParams["proc_id_model","Ns"]
        gridx = int(np.ceil(np.float32(N)/self.params["blockSz"]))
        
        
        # Compute the mean and variance sufficient statistics in parallel
        Xmean = np.zeros((K,D), dtype=np.float32)
        Xcovar = np.zeros((K,D,D), dtype=np.float32)
        for d in np.arange(D):
            
            self.gpuKernels["computeXSum"](np.int32(N),
                                           self.gpuPtrs["proc_id_model","C"].gpudata,
                                           np.int32(D),
                                           np.int32(d),
                                           self.base.data.gpu.X.gpudata,
                                           self.gpuPtrs["proc_id_model","Xstats"],
                                           block=(self.params["blockSz"],1,1),
                                           grid=(gridx,K))
            
            # Compute the mean along this dimension
            blockXmean = self.gpuPtrs["proc_id_model","Xstats"].get()
            blockXmean = np.sum(blockXmean, 1) 
            Xmean[Ns>0,d] = blockXmean[Ns>0] / Ns[Ns>0]
                    
        # Copy the mean to the GPU
        self.gpuPtrs["proc_id_model","Xmean"].set(Xmean)
            
        # Compute the sufficient statistics for the variance
        for d1 in np.arange(D):
            for d2 in np.arange(D):
                self.gpuKernels["computeXVarSum"](np.int32(N),
                                                  np.int32(D),
                                                  self.gpuPtrs["proc_id_model","C"].gpudata,
                                                  np.int32(d1),
                                                  np.int32(d2),
                                                  self.base.data.gpu.X.gpudata,
                                                  self.gpuPtrs["proc_id_model","Xmean"].gpudata,
                                                  self.gpuPtrs["proc_id_model","Xstats"].gpudata,
                                                  block=(self.params["blockSz"],1,1),
                                                  grid=(gridx,K))
                
                blockXvarsum = self.gpuPtrs["proc_id_model","Xstats"].get()
                blockXvarsum = np.sum(blockXvarsum, 1)
                Xcovar[Ns>0,d1,d2] =  blockXvarsum[Ns>0]/ Ns[Ns>0]
                
        
        # Sample new parameters from the Normal-Wishart posterior
        Xmean_new = np.zeros((K,D), dtype=np.float32)
        Xcov_new = np.zeros((K,D,D), dtype=np.float32)
        Xprec_new = np.zeros((K,D,D), dtype=np.float32)
        for k in np.arange(K):
            # Sample from a Normal-Wishart posterior
            if Ns[k] > 0:
                # get mean and cov for this proc
                xmean_k = Xmean[k,:].T
                xcov_k = np.reshape(Xcovar[k,:,:], (D,D)) 
                
                # Posterior parameters of the Wishart distribution on Lambda^-1
                T_post = np.copy(self.params["T0"]) 
                T_post += Ns[k]*xcov_k
                mean_op = np.dot((xmean_k-self.params["mu0"]),(xmean_k-self.params["mu0"]).T)
                T_post += (self.params["kap0"]*Ns[k])/(self.params["kap0"]+Ns[k])*mean_op
                
                nu_post = self.params["nu0"] + Ns[k]
            else:
                xmean_k = Xmean[k,:].T
                xcov_k = np.reshape(Xcovar[k,:,:], (D,D)) 
                T_post = np.copy(self.params["T0"])
                nu_post = self.params["nu0"]
                
            
            
            # Sample from the posterior Wishart with nu_new deg. of freedom and T_post ellipsoid
            # T_post is approx Ns[k]*covariance matrix, so the Wishart distribution with Ns[k] dof 
            # and a scale parameters of 1/Ns[k]*precision yields a sample precision matrix Lambda.
            # The precision is scaled up again in the sampling of the posterior mean. 
            Lam_k_new = wishrnd(nu_post, np.linalg.inv(T_post))
            
            # Add a small amount of diagonal noise to ensure Sig_k_new is nonsingular
            Lam_k_new += np.diag(1e-8*np.random.randn(D))
            # Invert to get the posterior precision
            Sig_k_new = np.linalg.inv(Lam_k_new)
                        
            # Now sample a new mean given this precision
            if Ns[k] > 0:
                kap_post = self.params["kap0"] + Ns[k]
                Sig_post = (1.0/kap_post)*Sig_k_new
                
                mu_post = (self.params["kap0"]*self.params["mu0"] + Ns[k]*xmean_k)/(self.params["kap0"]+Ns[k])
            else:
                kap_post = self.params["kap0"]
                Sig_post = (1.0/kap_post)*Sig_k_new
                mu_post = self.params["mu0"]
                
            mu_k_new = np.random.multivariate_normal(mu_post, Sig_post)
            
            # Save so we can copy to the GPU
            Xmean_new[k,:] = mu_k_new
            Xcov_new[k,:,:] = Sig_k_new
            Xprec_new[k,:,:] = Lam_k_new
            
            # DEBUG:
            
        
        # Copy the new parameters to the GPU
        self.gpuPtrs["proc_id_model","Xmean"].set(Xmean_new)
        self.gpuPtrs["proc_id_model","Xprec"].set(Xprec_new)
        
        # Save covariance for stat manager
        self.modelParams["proc_id_model","Xmean"] = Xmean_new
        self.modelParams["proc_id_model","Xprec"] = Xprec_new
        
    def sampleC(self):
        """
        Sample the process affiliations for each event. These must be done sequentially 
        since they are made dependent through the parent relationships Z.
        """
        N = self.base.data.N
        K = self.modelParams["proc_id_model","K"]
        D = self.base.data.D
        gridx = int(np.ceil(np.float32(N)/self.params["blockSz"]))
        
        for n in np.arange(self.base.data.N):
            # Sample c_n conditioned upon all other C's
            self.gpuKernels["computePerSpikePrCn"](np.int32(n),
                                                   np.int32(N),
                                                   np.int32(K),
                                                   np.int32(D),
                                                   self.base.data.gpu.X.gpudata,
                                                   self.gpuPtrs["proc_id_model","Xmean"].gpudata,
                                                   self.gpuPtrs["proc_id_model","Xprec"].gpudata,
                                                   self.gpuPtrs["parent_model","Z"].gpudata,
                                                   self.gpuPtrs["proc_id_model","C"].gpudata,
                                                   self.gpuPtrs["graph_model","A"].gpudata,
                                                   self.gpuPtrs["weight_model","W"].gpudata,
                                                   self.gpuPtrs["bkgd_model","lam"].gpudata,
                                                   self.gpuPtrs["proc_id_model","Xstats"].gpudata,
                                                   block=(self.params["blockSz"],1,1),
                                                   grid=(gridx,K))
            
            # Sum the log prob for each process and sample a new cn
            prcn = np.zeros((K,), dtype=np.float32)
            blockLogPrSum = self.gpuPtrs["proc_id_model","Xstats"].get()
            prcn[:] = np.sum(blockLogPrSum, 1)
            
            try:           
                cn = log_sum_exp_sample(prcn)
                
            except Exception as ex:
                log.info("Exception on spike %d!", n)
                log.info(ex)
                
                log.info("K=%d",K)
                
                log.info("X[n]:")
                log.info(self.base.data.gpu.X.get()[:,n])
                
                WGS = self.gpuPtrs["graph_model","WGS"].get()
                log.info("WGS[:,%d]:",n)
                log.info(WGS[:,n])
                
                C = self.gpuPtrs["proc_id_model","C"].get()
                Z = self.gpuPtrs["parent_model","Z"].get()
                A = self.gpuPtrs["graph_model","A"].get()
                W = self.gpuPtrs["weight_model","W"].get()
                
                log.info("W")
                log.info(W)
                
                if Z[n] > -1:
                    log.info("Spike %d (c=%d) parented by spike %d (c=%d)", n, C[n], Z[n], C[Z[n]])
                    
                    
#                    log.info ("A")
#                    log.info(self.gpuPtrs["graph_model","A"].get())
                    log.info("Edge exists from parent? ")
                    log.info(A[C[Z[n]],C[n]])
#                    log.info ("W")
#                    log.info(self.gpuPtrs["weight_model","A"].get())
                    log.info("Weight from parent:")
                    log.info(W[C[Z[n]],C[n]])
                    
                log.info("Spikes parented by n")
                log.info(np.count_nonzero(Z==n))
                for ch in np.nonzero(Z==n)[0]:
                    log.info("Spike %d (c=%d) parented spike %d (c=%d)", n, C[n],  ch, C[ch])
                    log.info("Edge exists to child? ")
                    log.info(A[C[n],C[ch]])
                    
                    if not A[C[n],C[ch]]:
                        log.info("WGS[:,%d]:",ch)
                        log.info(WGS[:,ch])
                    
                log.info("lam:")
                log.info(self.gpuPtrs["bkgd_model","lam"].get()[:,n])
                log.info("gaussians")
                log.info(self.gpuPtrs["proc_id_model","Xmean"].get()[C[n],:])
                log.info(self.gpuPtrs["proc_id_model","Xprec"].get()[C[n],:,:])
                
                log.info("blockLogPrSum")
                log.info(blockLogPrSum)
                log.info("prcn")
                log.info(prcn)
                
                exit()
            
            # Copy the new cn to the GPU
            cn_buff = np.array([cn], dtype=np.int32)
            cuda.memcpy_htod(self.gpuPtrs["proc_id_model","C"].ptr + int(n*cn_buff.itemsize), cn_buff)
        
        # Update Ns and C
        C = self.gpuPtrs["proc_id_model","C"].get()
        self.modelParams["proc_id_model","C"] = C
        
        Ns = np.zeros((K,), dtype=np.int32)
        for n in np.arange(N):
            Ns[C[n]] += 1
    
        self.modelParams["proc_id_model","Ns"] = Ns
        self.gpuPtrs["proc_id_model","Ns"].set(self.modelParams["proc_id_model","Ns"])
        
#        log.info("Number of non-empty processes: %d", np.count_nonzero(Ns))
        
    def registerStatManager(self, statManager):
        """
        Register callbacks with the given StatManager
        """
        N = self.base.data.N
        K = self.modelParams["proc_id_model","K"]
        D = self.base.data.D
        
        statManager.registerSampleCallback("proc_mean", 
                                           lambda: self.modelParams["proc_id_model","Xmean"],
                                           (K,D),
                                           np.float32)
        
        statManager.registerSampleCallback("proc_prec", 
                                           lambda: self.modelParams["proc_id_model","Xprec"],
                                           (K,D,D),
                                           np.float32)
        
        statManager.registerSampleCallback("C", 
                                           lambda: self.gpuPtrs["proc_id_model","C"].get(),
                                           (N,),
                                           np.int32)
        
        statManager.registerSampleCallback("Ns", 
                                           lambda: self.modelParams["proc_id_model","Ns"],
                                           (K,),
                                           np.int32)
        
        # Set a single sample documenting K
        statManager.setSingleSample("K",np.array([K]))
        
    
class MetaProcessIdModel(ModelExtension):
    """
    This process model ignores spatial extents of the data and assumes that each process
    belongs to one of a finite number of meta processes. The interactions at the graph level are 
    between meta processes, similar to a stochastic block model except that the connections
    between members of blocks are probabilistic functions of the block affiliations, whereas 
    connections between meta processes are chosen separately from their process constituents.  
    """
    def __init__(self, baseModel, configFile):
        self.base = baseModel
        self.base.data = self.base.data
                
        # Check whether the data file has hard coded process IDs
        if not self.base.data.proc_ids_known:
            log.error("Process IDs are not present in the data!")
            
        # Parse the configuration file to get params
        self.parseConfigurationFile(configFile)
        pprint_dict(self.params, "Process ID Model Params")
        
        self.initializeGpuKernels()
        
    def getInitializationOrder(self):
        """
        Make sure the process ID model initializes first
        """
        return 0
    
    def parseConfigurationFile(self, configFile):
        """
        Parse the configuration file to get base model parameters
        """
        # Initialize defaults
        defaultParams = {}
        
        # CUDA kernels are defined externally in a .cu file
        defaultParams["cu_dir"]  = os.path.join("pyhawkes", "cuda", "cpp")
        defaultParams["cu_file"] = "process_id_kernels.cu"
        defaultParams["thin"] = 1
        
        # K is now the number of meta processes. 
        defaultParams["K"] = 1
        
        # Create a config parser object and read in the file
        cfgParser = ConfigParser(defaultParams)
        cfgParser.read(configFile)
        
        self.params = {}
        self.params["cu_dir"]  = cfgParser.get("proc_id_model", "cu_dir")
        self.params["cu_file"] = cfgParser.get("proc_id_model", "cu_file")
        self.params["thin"] = cfgParser.getint("proc_id_model", "thin")
        self.params["blockSz"] = cfgParser.getint("cuda", "blockSz")
        
        # Parse the params for the meta process model
        self.params["K"] = cfgParser.getint("proc_id_model", "K")
        
    def initializeGpuKernels(self):
        kernelSrc = os.path.join(self.params["cu_dir"], self.params["cu_file"])
        kernelNames = ["computeLogQratio"]
        src_consts = {"B" : self.params["blockSz"]}
        self.gpuKernels = compile_kernels(kernelSrc, kernelNames, src_consts)
    
    def initializeModelParamsFromPrior(self):
        # Copy over the true process IDs
        # R is the number of real processes specified in the data
        R = self.base.data.K
        N = self.base.data.N
        C = self.base.data.C
        
        # K is the number of meta processes
        K = self.params["K"]
        self.modelParams["proc_id_model", "K"] = K
        
        # Assign the R processes to K metaprocesses randomly
        self.modelParams["proc_id_model", "Y"] = np.random.randint(0,K,R).astype(np.int32)
                    
        # Each spike is now associated with the metaprocess affiliated with its 
        # actual process
        self.params["C_meta"] = np.zeros((N,), dtype=np.int32)
        for r in np.arange(R):
            self.params["C_meta"][C==r] = self.modelParams["proc_id_model", "Y"][r]
            
        # Also save C_data and Ns_data
        self.params["C_data"] = self.base.data.C
        self.params["Ns_data"] = self.base.data.Ns

    
        # Populate M meta processes in the base model
        # For each process, sample and add new parameters
#        for k in np.arange(1,K):
#            newProcParams = self.base.sampleNewProcessParams()
#            self.base.addNewProcessEventHandler(newProcParams)
        
        # Update C on the gpu
        self.gpuPtrs["proc_id_model","C"].set(self.params["C_meta"])
        
        # Update Ns
        self.modelParams["proc_id_model","Ns"] = np.zeros((K,),dtype=np.int32)
        for n in np.arange(N):
            self.modelParams["proc_id_model","Ns"][self.params["C_meta"][n]] += 1
        self.gpuPtrs["proc_id_model","Ns"].set(self.modelParams["proc_id_model","Ns"])
        log.debug("Ns0: %s", str(self.modelParams["proc_id_model","Ns"]))
        
        self.initializeModelParams()
        self.initializeGpuMemory()
        
    def initializeGpuMemory(self):
        K = self.modelParams["proc_id_model","K"]
        N = self.base.data.N
        
        # Store Y and Cdata on the GPU as well
        self.gpuPtrs["proc_id_model","Y"] = gpuarray.to_gpu(self.modelParams["proc_id_model", "Y"].astype(np.int32))
        self.gpuPtrs["proc_id_model","C_data"] = gpuarray.to_gpu(self.params["C_data"].astype(np.int32))
        
        # Allocate space for the log Q-ratio sum
        grid_w = int(np.ceil(np.float(N)/ self.params["blockSz"]))
        self.gpuPtrs["proc_id_model","logQratio"] = gpuarray.empty((grid_w,), dtype=np.float32)
        
    def initializeModelParams(self):
        self.iter = 0
        
    def computeLogProbability(self):
        # TODO
        return 0
    
    def sampleModelParameters(self):
        """
        Sample mean/covariance for each process and process affiliations for each event  
        """
        if np.mod(self.iter, self.params["thin"]) == 0:
            self.sampleY()
            
            
    def sampleY(self):
        """
        Make a Metropolis Hastings step that changes one process's metaprocess affiliation.
        """
        # Propose to change Y_k = m uniformly over all k and m
        N = self.base.data.N
        R = self.base.data.K
        Cdata = self.base.data.C
        K = self.modelParams["proc_id_model","K"]
        A = self.modelParams["graph_model","A"]
        W = self.modelParams["weight_model","W"]
        
        r = np.random.randint(0,R)
        k = np.random.randint(0,K)
        k_old = self.modelParams["proc_id_model", "Y"][r] 
        
        if k_old != k: 
            log.debug("proposal: Y_%d: %d --> %d", r,k_old,k)
            Nr = self.params["Ns_data"][r]
            logPratio = Nr * np.sum(A[k_old,:]*W[k_old,:]-A[k,:]*W[k,:])
            
            # Use the GPU to compute the logQratio
            grid_w = int(np.ceil(np.float(N)/ self.params["blockSz"]))
            self.gpuKernels["computeLogQratio"](np.int32(r),
                                                np.int32(k),
                                                np.int32(N),
                                                np.int32(K),
                                                self.gpuPtrs["proc_id_model","Y"].gpudata,
                                                self.gpuPtrs["proc_id_model","C_data"].gpudata,
                                                self.gpuPtrs["bkgd_model","lam"].gpudata,
                                                self.gpuPtrs["graph_model","A"].gpudata,
                                                self.gpuPtrs["weight_model","W"].gpudata,
                                                self.gpuPtrs["impulse_model","GS"].gpudata,
                                                self.base.dSS["colPtrs"].gpudata,
                                                self.base.dSS["rowIndices"].gpudata,
                                                self.gpuPtrs["proc_id_model","logQratio"].gpudata,
                                                block=(self.params["blockSz"],1,1),
                                                grid=(grid_w,1)
                                                )
            
            blockLogQratio = self.gpuPtrs["proc_id_model","logQratio"].get()
            assert np.all(np.isfinite(blockLogQratio))
            
            logQratio = np.sum(blockLogQratio)
            
            # accept or reject depending on exp(logPratio + logQratio)
            if np.log(np.random.rand()) < logPratio + logQratio:
                log.debug("Y_%d: %d --> %d", r,k_old,k)
                
                # Update Y, C, and Ns on host and on GPU
                self.modelParams["proc_id_model", "Y"][r]=k
                self.gpuPtrs["proc_id_model","Y"].set(self.modelParams["proc_id_model", "Y"])
                
                self.modelParams["proc_id_model","C"][Cdata==r] = k
                self.gpuPtrs["proc_id_model","C"].set(self.modelParams["proc_id_model","C"])
                
                self.modelParams["proc_id_model","Ns"][k_old] -= Nr
                self.modelParams["proc_id_model","Ns"][k] += Nr
                self.gpuPtrs["proc_id_model","Ns"].set(self.modelParams["proc_id_model","Ns"])
                
                assert np.sum(self.modelParams["proc_id_model","Ns"])==N
                
                # If the meta process affiliations are updated, the parent assignments
                # must be updated as well
                self.base.extensions["parent_model"].sampleZ()
                
    def registerStatManager(self, statManager):
        """
        Register callbacks with the given StatManager
        """
        R = self.base.data.K
        K = self.modelParams["proc_id_model","K"]
                
        statManager.registerSampleCallback("Y_meta", 
                                           lambda: self.modelParams["proc_id_model", "Y"],
                                           (R,),
                                           np.int32)
        
        # Set a single sample documenting K
        statManager.setSingleSample("K",np.array([K]))
        
            
            