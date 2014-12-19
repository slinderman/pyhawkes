"""
Base model handles the basic parameters of the Hawkes model, most importantly 
the parents of each spike. To sample these the model must have access to  Model extensions handle 
"""
import numpy as np
import os
import logging
import ConfigParser

import pycuda.gpuarray as gpuarray
import pycuda.curandom as curandom

from weight_models import construct_weight_model
from impulse_models import construct_impulse_response_model
from process_id_models import construct_process_id_model
from parent_models import construct_parent_model
from model_factory import mf_construct_background_rate_model, mfConstructGraphModel, \
                          mf_construct_model_extensions, construct_spatial_model

from pyhawkes.utils.utils import pprint_dict, compile_kernels
import pyhawkes.utils.poisson_process as pp
from pyhawkes.utils.param_db import ParamsDatabase

log = logging.getLogger("global_log")

class BaseModel:
    def __init__(self, dataManager, configFile, data):
        """
        Construct a new base model.
        """
        self.parseConfigFile(configFile)
        pprint_dict(self.params, "Base Model Params")
        
        # Save a pointer to the data and compute the intervals between spikes
        self.dm = dataManager
        self.data = data
        self.prepareData()
    
        
        # Initialize parameter and GPU pointer databases
        self.modelParams = ParamsDatabase()
        self.gpuPtrs = ParamsDatabase()
        
        # Initialize model parameters on host and gpu
        self.initializeRandomness()
        self.initializeGpuKernels()
        self.constructModelExtensions(configFile)
        self.computeSampleOrder()
        
        # Initialization of the model state has to be done subsequently
#        self.initializeModelParameters()
#        self.initializeGpuMemory()
#        self.initializeModelExtensions(configFile)
    
    def parseConfigFile(self, configFile):
        """
        Parse the configuration file to get base model parameters
        """
        # Initialize defaults
        defaultParams = {}
        
        # CUDA kernels are defined externally in a .cu file
        defaultParams["cu_dir"]  = os.path.join("pyhawkes", "cuda", "cpp")
        defaultParams["cu_file"] = "base_model.cu"
        
        # Block size
        defaultParams["blockSz"] = 1024
        
        # Metropolis Hastings prob of proposing a birth
#        defaultParams["gamma"]    = 0.5
        
        # Default extensions
        defaultParams["bkgd_model"] = "homogenous"
        defaultParams["graph_model"] = "erdos_renyi"
        defaultParams["weight_model"] = "homogenous"
        defaultParams["impulse_model"] = "logistic_normal"
        defaultParams["proc_id_model"] = "known"
        defaultParams["parent_model"] = "default"
        
        # Default random seed
        defaultParams["seed"] = -1
        
    
        # Create a config parser object and read in the file
        cfgParser = ConfigParser(defaultParams)
        cfgParser.read(configFile)
        
        self.params = {}
        self.params["blockSz"] = cfgParser.getint("cuda", "blockSz")
        self.params["cu_dir"]  = cfgParser.get("base_model", "cu_dir")
        self.params["cu_file"] = cfgParser.get("base_model", "cu_file")
        self.params["seed"] = cfgParser.getint("base_model", "seed")
        self.params["dt_max"] = cfgParser.getfloat("preprocessing", "dt_max")
        self.params["sim_inhibition"]   = bool(cfgParser.getint("base_model", "sim_inhibition"))

        self.params["parent_model"]   = cfgParser.get("base_model", "parent_model")
        self.params["bkgd_model"]   = cfgParser.get("base_model", "bkgd_model")
        self.params["graph_model"]   = cfgParser.get("base_model", "graph_model")
        self.params["weight_model"]   = cfgParser.get("base_model", "weight_model")
        self.params["impulse_model"]   = cfgParser.get("base_model", "impulse_model")
        self.params["proc_id_model"]   = cfgParser.get("base_model", "proc_id_model")
        self.params["parent_model"]   = cfgParser.get("base_model", "parent_model")

        if cfgParser.has_option("base_model", "spatial_model"):
            self.params["spatial_model"] = cfgParser.get("base_model", "spatial_model")
        else:
            self.params["spatial_model"] = "default"

        
    
    def prepareData(self):
        """
        If we are simulating inhibition we only need interspike intervals
        from real spikes. Fantasized spikes for inhibited processes cannot induce
        child spikes.
        """
        if self.params["sim_inhibition"]:
            log.debug("Simulating inhibition")
            # TODO: Generate a set of fantasized spikes for each process such that 
            # the joint set of real and fantasized spikes is drawn from a homogeneous
            # process with rate lambda_max (the approximate max firing rate of the process
            
            # For now assume that the latter K/2 processes are the fantasized spikes
            # corresponding to the first K/2 processes
            assert np.mod(self.data.K, 2) == 0, "K must be even in order to simulate inhibition"
            
            # We only need to keep intervals from real spikes
            Sreal_inds = np.nonzero(self.data.C < self.data.K/2)[0]
            Sreal = self.data.S[Sreal_inds]
            
            log.debug("N real spikes: %d", len(Sreal))
            self.dSS = self.dm.compute_sparse_spike_intvl_matrix_unknown_procs(Sreal, self.data.S)
            
            # Update the row indices 
            rowInds = self.dSS["rowIndices"].get()
            rowIndsNew = Sreal_inds[rowInds]
            self.dSS["rowIndices"].set(rowIndsNew.astype(np.int32))
            
        else:
            self.dSS = self.dm.compute_sparse_spike_intvl_matrix_unknown_procs(self.data.S, self.data.S)
    
    def initializeRandomness(self):
        """
        Initialize the random number generator used on GPU
        """
        # Define a random seed_getter for curandom
        seed_getter = lambda N: gpuarray.to_gpu(np.random.randint(-2**30,2**30,size=(N,)).astype(np.int32))
        self.rand_gpu = curandom.XORWOWRandomNumberGenerator(seed_getter)
    
    def initializeGpuKernels(self):
        kernelSrc = os.path.join(self.params["cu_dir"], self.params["cu_file"])
        
        kernelNames = ["computeLogLkhdPerSpike",
                       "computeConditionalIntensity"]
        
        src_consts = {"B" : self.params["blockSz"]}
        
        self.gpuKernels = compile_kernels(kernelSrc, kernelNames, src_consts)
        
    def constructModelExtensions(self, configFile):
        """
        Initialize model extensions, e.g.:
            background rate model
            weight model
            impulse response model
            graph model
        """
        self.extensions = {}
        self.extensions["bkgd_model"] = mf_construct_background_rate_model(self.params["bkgd_model"], self, configFile)
        self.extensions["graph_model"] = mfConstructGraphModel(self.params["graph_model"], self, configFile)
        self.extensions["weight_model"] = construct_weight_model(self.params["weight_model"], self, configFile)
        self.extensions["impulse_model"] = construct_impulse_response_model(self.params["impulse_model"], self, configFile)
        self.extensions["proc_id_model"] = construct_process_id_model(self.params["proc_id_model"], self, configFile)
        self.extensions["parent_model"] = construct_parent_model(self.params["parent_model"], self, configFile)
        self.extensions["spatial_model"] = construct_spatial_model(self.params["spatial_model"], self, configFile)
        
        # TODO: Replace this with fully fledged model factory
        more_extensions = mf_construct_model_extensions(self, configFile)
        self.extensions.update(more_extensions)
        
    def initializeFromDict(self, paramsDB):
        """
        Initialize the model with the specified parameters.
        """
        # Sort the extensions by their initialization order
        sortedExtensions = sorted(self.extensions.keys(), lambda e1,e2: self.extensions[e1].getInitializationOrder() - self.extensions[e2].getInitializationOrder())
        
        for extension in sortedExtensions:
            log.debug("Initializing extension: %s", extension)
            self.extensions[extension].initializeModelParamsFromDict(paramsDB)
            
    def initializeFromPrior(self):
        """
        Initialize the model with a draw from the prior
        """
        # Sort the extensions by their initialization order
        sortedExtensions = sorted(self.extensions.keys(), lambda e1,e2: self.extensions[e1].getInitializationOrder() - self.extensions[e2].getInitializationOrder())
        
        for extension in sortedExtensions:
            log.debug("Initializing extension: %s", extension)
            self.extensions[extension].initializeModelParamsFromPrior()
                   
    def registerStatManager(self, statManager):
        """
        Register callbacks with the given StatManager
        """
        # TODO: How should we handle changing K?!
        K = int(self.modelParams["proc_id_model","K"])
        
        statManager.registerSampleCallback("ll", 
                                           self.computeLogLikelihood,
                                           (1,),
                                           np.float32,
                                           burnin=True)
#        statManager.registerSampleCallback("logprob", 
#                                           self.computeLogPosterior,
#                                           (1,),
#                                           np.float32,
#                                           burnin=True)

        for extension in self.extensions.values():
            extension.registerStatManager(statManager)
        
    def computeLogLikelihood(self):
        """
        Compute the log likelihood of the data given the current parameters
        """
        N = int(self.data.N)
        K = self.modelParams["proc_id_model","K"]
        Ns = self.modelParams["proc_id_model","Ns"]
        A = self.modelParams["graph_model","A"]
        W = self.modelParams["weight_model","W"]
        
        ll = 0.0
        
        # LL += -1 * \int_0^T lam0(t)dt
        ll += np.sum(-1.0 * self.extensions["bkgd_model"].integrateBkgdRates())
#        log.info("ll_int: %f", ll)
        if not np.isfinite(ll):
            log.error("integrated background rates not finite!")
            exit()
        
        # LL += \sum_{n=1}^N \sum_{k=1}^K  -1*\int_{s_n}^T A_{c_n,k}*W_{c_n,k}*g(t)dt
        #     = \sum_{n=1}^N \sum_{k=1}^K -1*A_{c_n,k}*W_{c_n,k}
        #     = \sum_{k1=1}^K \sum_{k2=1}^K  -1*Ns_{k1}*A_{k1,k2}*W_{k1,k2}
        #import pdb; pdb.set_trace()
        for k1 in np.arange(K):
            for k2 in np.arange(K):
                ll += -1.0*Ns[k1]*A[k1,k2]*W[k1,k2] 
                
        if not np.isfinite(ll):
            log.error("induced IRs not finite!")
            exit()
                
        
        if N > 0:
            # LL += sum_{n=1}^N log(lam0[s_n] + \sum_n'=1^n A_{c_n',c_n}*W_{c_n',c_n}*g(s_n-s_n'))
            grid_w = min(int(np.ceil(float(N)/1024)), 2**16-1)
            grid_h = int(np.ceil(float(N)/(grid_w*1024)))
            block_ll = gpuarray.zeros((grid_h,grid_w), dtype=np.float32)
            
    #        log.info(K)
    #        log.info(N)
    #        log.info(self.gpuPtrs["proc_id_model","C"].get())
    #        log.info(self.gpuPtrs["bkgd_model","lam"].get())
    #        log.info(self.gpuPtrs["graph_model","A"].get())
    #        log.info(self.gpuPtrs["weight_model","W"].get())
    #        log.info(self.gpuPtrs["impulse_model","GS"].get())
            
            self.gpuKernels["computeLogLkhdPerSpike"](np.int32(K),
                                                      np.int32(N),
                                                      self.gpuPtrs["proc_id_model","C"].gpudata,
                                                      self.gpuPtrs["bkgd_model","lam"].gpudata,
                                                      self.gpuPtrs["graph_model","A"].gpudata,
                                                      self.gpuPtrs["weight_model","W"].gpudata,
                                                      self.gpuPtrs["impulse_model","GS"].gpudata,
                                                      self.dSS["colPtrs"].gpudata,
                                                      self.dSS["rowIndices"].gpudata,
                                                      block_ll.gpudata,
                                                      block=(1024,1,1),
                                                      grid=(grid_w,grid_h)
                                                      )

            ll += gpuarray.sum(block_ll).get()

            # Compute the spatial probability if it exists
            ll += np.sum(self.extensions["spatial_model"].computeSpatialLogLkhdPerSpike())

#        log.info("ll_spks: %f", gpuarray.sum(block_ll).get())

        if not np.isfinite(ll):
            log.error("lkhd per spike not finite!")
            exit()

        return ll
        
    def computeLogPosterior(self):
        """
        Compute the log posterior probability - the log likelihood plus the log prior
        """
        logprob = self.computeLogLikelihood()
                
        for (name,extension) in self.extensions.items():
            lp = extension.computeLogProbability()
            if not np.isscalar(lp):
                if np.size(lp)==1:
                    lp = np.int32(lp)
                else:
                    log.error("Extension %s returned vector logprob", name)
                
            if not np.isfinite(lp):
                log.info("Extension %s returned infinite logprob", name)
                
            logprob += lp 
        
        #assert np.isfinite(logprob), "ERROR: logprob is %f" % logprob
        return logprob
        
    def computeConditionalIntensityFunction(self, t=None):
        """
        In order to test the statistical significance of the resulting model
        we will use the Kolmogorov-Smirnoff (KS) test. Under the renewal process model
        spikes should be uniformly distributed in "rescaled time." In order to compute
        rescaled time we need to integrate the conditional intensity function.  
        """
        N = self.data.N
        K = self.modelParams["proc_id_model","K"]
        A = self.modelParams["graph_model", "A"]
        W = self.modelParams["weight_model", "W"]
        C = self.modelParams["proc_id_model","C"]
        
        if t is None:
            t = np.linspace(self.data.Tstart,self.data.Tstop,1000)
        
        n_pts = len(t)
        
        # Allocate an array for the conditional intensity function
        cond_int  = np.zeros((K,n_pts))
        
        # evaluate the background rate
        cond_int += self.extensions["bkgd_model"].evaluateBkgdRate(t)
        
        # Add each spike's impulse responses to the conditional intensity
        # Leverage the dataManager code to compute interspike intervals 
        # now between spikes and time points.
        dST = self.dm.compute_sparse_spike_intvl_matrix_unknown_procs(self.data.S, t)
                
        # Compute the impulse density for each spike-time pt pair
        gST = self.extensions["impulse_model"].computeIrDensity(dST["dS"])
        
        # Calculate the conditional intensity at each time point by summing the contribution
        # from each preceding spike
        cond_int_gpu = gpuarray.empty((K,n_pts), dtype=np.float32)
        grid_w = int(np.ceil(np.float32(n_pts)/self.params["blockSz"]))
        self.gpuKernels["computeConditionalIntensity"](np.int32(K),
                                                       np.int32(n_pts),
                                                       self.gpuPtrs["proc_id_model","C"].gpudata,
                                                       self.gpuPtrs["graph_model","A"].gpudata,
                                                       self.gpuPtrs["weight_model","W"].gpudata,
                                                       gST.gpudata,
                                                       dST["colPtrs"].gpudata,
                                                       dST["rowIndices"].gpudata,
                                                       cond_int_gpu.gpudata,
                                                       block=(self.params["blockSz"],1,1),
                                                       grid=(grid_w,K)
                                                       )
        
        cond_int += cond_int_gpu.get()
        
#        g = self.extensions["impulse_model"].getImpulseFunction()
#        for n in np.arange(N):
#            s_n = self.data.S[n]
#            c_n = C[n]
#            # Determine which time points would be affected by this spike
#            t_aff_inds = np.nonzero(np.bitwise_and(t>s_n, t<s_n+self.params["dt_max"]))
#            dt_aff = t[t_aff_inds] - s_n
#            g_aff = g(dt_aff)
#                        
#            for k in np.arange(K):
#                cond_int[k,t_aff_inds] += A[c_n,k]*W[c_n,k]*g_aff 
#        
        return cond_int

    def computeRescaledSpikeTimes(self):
        """
        Compute rescaled spike times by computing the cumulative integral of 
        the conditional intensity function at each spike
        """
        N = self.data.N
        K = self.modelParams["proc_id_model","K"]
        C = self.modelParams["proc_id_model","C"]
        S = self.data.S
        
        # Define a grid of time points at which to compute the conditional intensity
        n_pts = 20000
        t = np.linspace(self.data.Tstart, self.data.Tstop, n_pts)
        dt = (self.data.Tstop - self.data.Tstart)/(n_pts-1)
        t_center = (t[:-1]+t[1:])/2
        
        # evaluate conditional intensity at t_center
        cond = self.computeConditionalIntensityFunction(t)
        cond_center = (cond[:,:-1]+cond[:,1:])/2
         
        # Cumulative sum (trapezoidal integration) to get cumulative intensity       
        cum_int_cond = np.hstack((np.zeros((K,1)),np.cumsum(dt*cond_center, axis=1)))
        
        # Interpolate to get the integral of the conditional intensity at individual
        # spike times
        cum_int_spikes = np.zeros(N)
        for k in np.arange(K):
            Ck = np.nonzero(C==k)
            cum_int_spikes[Ck] = np.interp(S[Ck],t,cum_int_cond[k,:])
            
        return cum_int_spikes
        
    def sampleNewProcessParams(self):
        """
        Call each extension to sample parameters for a new process
        """
        newProcParams = {}
        for (name,extension) in self.extensions.items():
            extension.sampleNewProcessParams(newProcParams)
            
        return newProcParams
        
    def addNewProcessEventHandler(self, newProcParams):
        """
        Add a new process (K+=1). Call each extension to update the new parameters.
        """
        
        # Manually delete previously allocated gpuarray instances (just in case the GC
        # takes a while  and we run out of memory
#        del self.gpuData["WGS"]
#        del self.gpuData["lam"]
#        
#        # Update the base model params
#        self.modelParams["K"] += 1
#        self.gpuData["WGS"] = gpuarray.empty((self.modelParams["K"], self.data.N), dtype=np.float32)
#        self.gpuData["lam"] = gpuarray.empty((self.modelParams["K"], self.data.N), dtype=np.float32)
#        
#        # Update Ns
#        Ns_new = np.hstack((self.gpuData["Ns"].get(),np.array([0],dtype=np.int32)))
#        self.gpuData["Ns"] = gpuarray.empty((self.modelParams["K"],), dtype=np.int32)
#        self.gpuData["Ns"].set(Ns_new)
        
        # Iterate over extensions
        for (name,extension) in self.extensions.items():
            extension.addNewProcessEventHandler(newProcParams)
        
    def removeProcessEventHandler(self, procId):
        """
        Remove process procId from the set of processes
        """
#        # Update the base model params
#        self.modelParams["K"] -= 1
#        self.gpuData["WGS"] = gpuarray.empty((self.modelParams["K"], self.data.N), dtype=np.float32)
#        self.gpuData["lam"] = gpuarray.empty((self.modelParams["K"], self.data.N), dtype=np.float32)
#        
#        # Update Ns
#        Ns_old = self.gpuData["Ns"].get()
#        self.gpuData["Ns"] = gpuarray.empty((self.modelParams["K"],), dtype=np.int32)
#        self.gpuData["Ns"].set(Ns_old[:-1])
#        
        # Iterate over extensions
        for (name,extension) in self.extensions.items():
            extension.removeProcessEventHandler(procId)
        
    def computeSampleOrder(self):
        """
        TODO: Could make this cleaner by having a dependency list in each extension
        There are some hard constraints on the sampling order, namely
        sampling A in the graph model and Ymeta for the MetaProcessIdModel
        require a subsequent sampling of Z
        """
        self.sampleOrder = self.extensions.keys()
        
        # Remove parent model from the list
        self.sampleOrder.remove("parent_model")
        
        # The graph model requires a resampling of Z
        self.sampleOrder.insert(self.sampleOrder.index("graph_model")+1, "parent_model")
        
        # If we are using the metaprocess model we need to sample Z
        if self.params["proc_id_model"] == "metaprocess":
            self.sampleOrder.insert(self.sampleOrder.index("proc_id_model")+1, "parent_model")
        
#        log.info(self.sampleOrder)
        
    def sampleModelParametersAndLatentVars(self):
        """
        Gibbs sample all model parameters conditional upon one another
        """  
        # Iterate over model extensions and sample their parameters
        for name in self.sampleOrder:
            extension = self.extensions[name]
            extension.sampleModelParameters()
            extension.sampleLatentVariables()
            
    def sampleLatentVariables(self):
        """
        Gibbs sample all latent variables, conditional upon one another,
        and keeping the model parameters fixed
        """
        # Iterate over model extensions and sample their parameters
        for name in self.sampleOrder:
            extension = self.extensions[name]
            extension.sampleLatentVariables()
        

    def generateData(self, (T_start, T_stop), N_t=100):
        """
        The Hawkes process is a generative model. Given the parameters of 
        this model instance, sample spikes in the given range. For example,
        suppose the model has been trained on spikes in [0, T_start] and we would like
        to predict the spikes in [T_start, T_stop]. This can be done by "rolling the 
        model forward in time." 
        """
        K = self.modelParams["proc_id_model","K"]
        
        # TODO: Extend this to spatiotemporal models with spikes in greater than 1 dimension
        D = 1
        
        # Discretize time in order to evaluate the background rate
        # N_t specifies the numbre of time points (resolution) at which
        # we evaluate the background rate
        assert T_stop > T_start, "ERROR: Invalid generation time interval"
        tt = np.linspace(T_start, T_stop, N_t)
        
        # Check stability of the weight matrix
        ev = np.abs(np.linalg.eigvals(self.modelParams["graph_model","A"]*self.modelParams["weight_model","W"]))
        stable = np.all(ev <= 1.0)
        if not stable:
            raise Exception("Error: Connectivity matrix A*W is not stable: Eigenvalues = %s" % str(ev))
        
        # Initialize output arrays, a dictionary of numpy arrays
        S = {}
        N = np.zeros(K)
        
        # Allocate a buffer for each process's spike train
        lam_max = 100
        buff_sz = 10*lam_max*(T_stop - T_start)
        for k in np.arange(K):
            S[k] = np.zeros((D,buff_sz))
            
#        # Generate children of spikes in the training window. First find the
#        # set of possible parents
#        max_offset = -1
#        while T_start - self.data.S[max_offset] < self.params["dt_max"]: 
#            max_offset -= 1
#            # Don't allow cycling
#            if max_offset < -1 * self.data.N:
#                break
#        
#        log.info("Training period has %d possible parent spikes.", np.abs(max_offset)+1)
#        
#        # Each training spike induces a tree of child pp's on each process
#        for offset in np.arange(max_offset+1,0):
#            for k in np.arange(K):
#                s = np.reshape(self.data.S[offset], (D,1))
#                k = self.data.C[offset]
#                self.__generateDataHelper((T_start,T_stop), S, N, s, k, round=0, buff_sz=buff_sz)
                
        # First we sample the background rate. Depending on the 
        # background model, this may be homogenous or time varying GP.
        lam_pred = self.extensions["bkgd_model"].generateBkgdRate(tt)
    
        # Sample spikes from background rate
        Nbkgd_offset = np.copy(N)
        debug_Nbkgd = 0
        for k in np.arange(K):
            # Draw spikes from the backgruond rate
            Sbkgd = pp.sampleInhomogeneousPoissonProc(tt, lam_pred[k,:])
            Nbkgd = len(Sbkgd)
            debug_Nbkgd += Nbkgd
            S[k][:,N[k]:N[k]+Nbkgd] = Sbkgd
            N[k] += Nbkgd
            
        log.info("Background rate induced %d spikes", debug_Nbkgd)
        Nbkgd_end = np.copy(N)
        
        log.debug("Nbkgd_offset: %s", str(Nbkgd_offset))
        log.debug("Nbkgd_end: %s", str(Nbkgd_end))
        log.debug("Nbkgd: %s", str(Nbkgd_end-Nbkgd_offset))
         
        # Each background spike induces a tree of child spikes
        for k in np.arange(K):
            for pa_ind in np.arange(Nbkgd_offset[k],Nbkgd_end[k]):
                s = S[k][:,pa_ind]
                self.__generateDataHelper((T_start,T_stop), S, N, s, k, round=0, buff_sz=buff_sz)
                
        # Trim and sort spikes in chronolocial order
        for k in np.arange(K):
            if N[k] > 0:
                S[k] = S[k][:,0:N[k]]
                # Sort spikes according to time (the first dimension by default)
                perm = np.argsort(S[k][0,:])
                S[k] = S[k][:,perm]
            else:
                S[k] = []
            
        # Flatten the output
        Nsum = int(np.sum(N))
        Sflat = np.empty(Nsum, dtype=np.float32)
        Cflat = np.empty(Nsum, dtype=np.int32)
        
        off = 0
        for k in np.arange(K):
            if N[k] > 0:
                Sflat[off:off+N[k]] = S[k]
                Cflat[off:off+N[k]] = k
                off += N[k]
            
        # Sort the joint arrays
        perm = np.argsort(Sflat)
        Sflat = Sflat[perm]
        Cflat = Cflat[perm]
        
        return (Sflat,Cflat,Nsum)
    
    def __generateDataHelper(self, (T_start,T_stop), S, N, s_pa, k_pa, round=0, buff_sz=1000):
        """
        Recursively generate new generations of spikes with 
        given impulse response parameters. Takes in a single spike 
        as the parent and recursively calls itself on all children
        spikes
        """
        D = 1
        K = self.modelParams["proc_id_model","K"]
        
        max_round = 1000
        assert round < max_round, "Exceeded maximum recursion depth of %d" % max_round
        
        for k_ch in np.arange(K):
            w = self.modelParams["weight_model","W"][k_pa,k_ch]
            a = self.modelParams["graph_model","A"][k_pa,k_ch] 
            
            if w==0 or a==False:
                continue
            
            # The total area under the impulse response curve(ratE)  is w
            # Sample spikes from a homogenous poisson process with rate
            # 1 until the time exceeds w. Then transform those spikes
            # such that they are distributed under a logistic normal impulse
            n_ch = np.random.poisson(w)
            
            if n_ch > 0:
                # Generate spikes with the impulse response model
                s_ch = self.extensions["impulse_model"].generateSpikeOffset(s_pa,k_pa,k_ch, n_ch)
                assert np.shape(s_ch)  ==  (D,n_ch), "ERROR: generateSpikeOffset returned invalid shape %s" % str(np.shape(s_ch))
                                                                                                                  
                # Only keep spikes within the simulation time interval
                valid_ch = np.bitwise_and(s_ch[0,:] > T_start, s_ch[0,:] < T_stop)
                s_ch = s_ch[:,valid_ch]
                n_ch = np.size(s_ch, 1)
                  
                # Append the new spikes to the dataset, growing buffer as necessary
                if N[k_ch] + n_ch > np.size(S[k_ch],1):
                    S[k_ch] = np.hstack((S[k_ch], np.zeros((D,buff_sz))))
        
                
                S[k_ch][:,N[k_ch]:N[k_ch]+n_ch] = s_ch
                N[k_ch] += n_ch
                
                # Generate offspring from child spikes
                for ch_ind in np.arange(n_ch):
                    s = s_ch[:,ch_ind]
                    self.__generateDataHelper((T_start,T_stop), S, N, s, k_ch, round=round+1, buff_sz=buff_sz)
