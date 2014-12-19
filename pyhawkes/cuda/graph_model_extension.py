import numpy as np
import os
import logging
import scipy.io
from ConfigParser import ConfigParser

import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray

from pyhawkes.utils.utils import pprint_dict, compile_kernels
from model_extension import ModelExtension

log = logging.getLogger("global_log")

## Mimic CUDA defines
MH_ADD = 0
MH_DEL = 1
MH_NOP = 2

class GraphModelExtension(ModelExtension):
    
    def __init__(self, baseModel, configFile):
        # Store pointer to base model
        self.base = baseModel
        
        
        # Initialize databases for this extension
        self.modelParams = baseModel.modelParams
        self.modelParams.addDatabase("graph_model")
        self.gpuPtrs = baseModel.gpuPtrs
        self.gpuPtrs.addDatabase("graph_model")
        
        self.params = self.parseGeneralGraphParameters(configFile)
        self.initializeGeneralGraphKernels()
    
    def parseGeneralGraphParameters(self,configFile):
        """
        Parse parameters common to all graph models
        """
        # Initialize defaults
        defaultParams = {}
        defaultParams["cu_dir"]  = os.path.join("pyhawkes", "cuda", "cpp")
        defaultParams["cu_file"] = "hawkes_mcmc_kernels.cu"
        defaultParams["blockSz"] = 1024
        
        defaultParams["allow_self_excitation"] = False
        defaultParams["force_dense_matrix"] = False
        defaultParams["burnin_with_dense_matrix"] = True
        # Metropolis Hastings prob of proposing a new edge
#        defaultParams["gamma"]    = 0.5
        
    
        # Create a config parser object and read in the file
        cfgParser = ConfigParser(defaultParams)
        cfgParser.read(configFile)
        
        # Create an output params dict. The config file is organized into
        # sections. Read them one at a time
        params = {}
        params["blockSz"] = cfgParser.getint("cuda", "blockSz")
        params["cu_dir"]  = cfgParser.get("base_model", "cu_dir")
        params["cu_file"] = cfgParser.get("base_model", "cu_file")
        params["allow_self_excitation"]    = bool(cfgParser.getint("graph_prior", "allow_self_excitation"))
        params["force_dense_matrix"]       = bool(cfgParser.getint("graph_prior", "force_dense_matrix"))
        params["burnin_with_dense_matrix"] = bool(cfgParser.getint("graph_prior", "burnin_with_dense_matrix"))
#        params["gamma"]                    = cfgParser.getfloat("graph_prior", "gamma")
        
        # Check if a mask is provided
        if cfgParser.has_option("graph_prior", "mask_file"):
            mask_file = cfgParser.get("graph_prior", "mask_file")
            mask_data = scipy.io.loadmat(mask_file, appendmat=True)
            self.modelParams["graph_model","mask"] = mask_data["mask"].astype(np.int)
        else:
            self.modelParams["graph_model","mask"] = None
        
        return params
    
    def initializeGeneralGraphKernels(self):
        kernelSrc = os.path.join(self.params["cu_dir"], self.params["cu_file"])
        kernelNames = ["computeWGSForAllSpikes", 
                       "computeWGSForNewEdge",
                       "clearWGSForDeletedEdge",
                       "computeProdQratio",
                       "computeLkhdRatioA",
                       "sampleA"]
        src_consts = {"B" : self.params["blockSz"]}
        self.gpuKernels = compile_kernels(kernelSrc, kernelNames, src_consts)
        
    
    def initializeGpuMemory(self):
        K = self.modelParams["proc_id_model","K"]
        N = self.base.data.N
        
        self.gpuPtrs["graph_model","A"] = gpuarray.empty((K,K), dtype=np.bool)
        self.gpuPtrs["graph_model","WGS"]   = gpuarray.empty((K,N), dtype=np.float32)
        
        qratio_width = int(np.ceil(np.float32(self.base.data.N)/self.params["blockSz"]))
        self.gpuPtrs["graph_model","qratio"] = gpuarray.empty((qratio_width,), dtype=np.float64)
        self.gpuPtrs["graph_model","lkhd_ratio"] = gpuarray.empty((1,), dtype=np.float32)
    
    def mh_sample_A(self, is_symmetric=False):
        """
        Sample new adjacency matrix and relevant spike parents using MH 
        Determine whether or not to propose a birth
        Choose an edge to propose
        Determine whether to accept using cuComputeProdQratio to do the parallel computation. 
        If accept, use cuSampleSingleProcessZ to choose new parents
        """
        N = self.base.data.N
        
        K = self.modelParams["proc_id_model","K"]
        Ns = self.modelParams["proc_id_model","Ns"]
                
        # Determine whether to propose a new edge or a removal of an existing edge
        op = MH_ADD if np.random.rand() < self.params["gamma"] else MH_DEL
           
        # Choose a row (ki) and column (kj) to update
        # They must be selected randomly, otherwise the transition probabilities 
        # do not cancel properly as derived in the paper, and the distribution is 
        # not left invariant after the MH operation
#        ki = np.random.randint(K)
#        # If this is a symmetric graph model, only choose from the upper diagonal
#        if is_symmetric:
#            kj = np.random.randint(ki,K)
#        else:
#            kj = np.random.randint(K)
            
        # Choose one of the unspecified edges in the graph
        if self.modelParams["graph_model","mask"] == None:
            ki = np.random.randint(K)
            # If this is a symmetric graph model, only choose from the upper diagonal
            if is_symmetric:
                kj = np.random.randint(ki,K)
            else:
                kj = np.random.randint(K)
        else:
            (kis,kjs) = np.nonzero(self.modelParams["graph_model","mask"]==-1)
            ind = np.random.randint(0,len(kis))
            ki = kis[ind]
            kj = kjs[ind]
            assert self.modelParams["graph_model","mask"][ki,kj]==-1
            
#        # Check if this entry is set in the mask already
#        if self.modelParams["graph_model","mask"] != None:
#            if self.modelParams["graph_model","mask"][ki,kj] != -1:
#                return
        
        # Get the current weight for this entry
        currWBuffer = np.zeros((1,), dtype=np.float32)
        cuda.memcpy_dtoh(currWBuffer,
                         self.gpuPtrs["weight_model","W"].ptr + int((ki*K+kj)*currWBuffer.itemsize))
        currW = currWBuffer[0]
        
        
        # Compute WGS using the current set of weights
        grid_w = int(np.ceil(np.float32(N)/self.params["blockSz"]))
        
#        startPerfTimer(perfDict, "computeWGSForAllSpikes")
#        log.info(self.gpuPtrs["graph_model","WGS"].get())
        self.gpuKernels["computeWGSForAllSpikes"](np.int32(K),
                                                  np.int32(N),
                                                 self.gpuPtrs["proc_id_model","C"].gpudata,
                                                 self.gpuPtrs["impulse_model","GS"].gpudata,
                                                 self.base.dSS["colPtrs"].gpudata,
                                                 self.base.dSS["rowIndices"].gpudata,
                                                 self.gpuPtrs["weight_model","W"].gpudata,
                                                 self.gpuPtrs["graph_model","A"].gpudata,
                                                 self.gpuPtrs["graph_model","WGS"].gpudata,
                                                 block=(1024, 1, 1), 
                                                 grid=(grid_w,K)
                                                 )
#        stopPerfTimer(perfDict, "computeWGSForAllSpikes")
        
        # Now determine whether or not to accept the change
        # If the proposal does not change the adjacency matrix
        # then this is easy, otherwise we enlist the GPU
        logQratio = 0.0
        logPratio = 0.0
        
        if op == MH_ADD:
            # We are proposing a new edge. Calculate WGS with the 
            # new edge and determine the P and Q ratios
            
            # This now runs on all N spikes but only counts spikes affected by the new edge
            # namely, those on process kj
            self.gpuKernels["computeWGSForNewEdge"](np.int32(ki),
                                               np.int32(kj),
                                               np.int32(K),
                                               np.int32(N),
                                               self.gpuPtrs["proc_id_model","C"].gpudata,
                                               self.gpuPtrs["impulse_model","GS"].gpudata,
                                               self.base.dSS["colPtrs"].gpudata,
                                               self.base.dSS["rowIndices"].gpudata,
                                               self.gpuPtrs["weight_model","W"].gpudata,
                                               self.gpuPtrs["graph_model","WGS"].gpudata,
                                               block=(1024, 1, 1), 
                                               grid=(grid_w,1)
                                               )
            
            if is_symmetric and ki!=kj:
                self.gpuKernels["computeWGSForNewEdge"](np.int32(kj),
                                                       np.int32(ki),
                                                       np.int32(K),
                                                       np.int32(N),
                                                       self.gpuPtrs["proc_id_model","C"].gpudata,
                                                       self.gpuPtrs["impulse_model","GS"].gpudata,
                                                       self.base.dSS["colPtrs"].gpudata,
                                                       self.base.dSS["rowIndices"].gpudata,
                                                       self.gpuPtrs["weight_model","W"].gpudata,
                                                       self.gpuPtrs["graph_model","WGS"].gpudata,
                                                       block=(1024, 1, 1), 
                                                       grid=(grid_w,1)
                                                       )
#            stopPerfTimer(perfDict, "computeWGSForNewEdge")
            
            # Compute the P ratio
            rho = self.getConditionalEdgePr(ki,kj)
            if rho == 1.0:
                logPratio = np.Inf
            elif rho == 0.0:
                logPratio = -np.Inf
            elif is_symmetric and ki!=kj:
                logPratio = -1.0*Ns[ki]*currW + -1.0*Ns[kj]*currW + np.log(rho) - np.log(1-rho)
            else:
                logPratio = -1.0*Ns[ki]*currW + np.log(rho) - np.log(1-rho)
                
            # Compute the Q ratio
            if ki==kj and not self.params["allow_self_excitation"]:
                # Do not allow self-excitation
                logQratio = np.float64(-1.0*np.Inf)
            else:
                logQratio = np.log(1-self.params["gamma"]) - np.log(self.params["gamma"])
                self.gpuKernels["computeProdQratio"](np.int32(kj),
                                                     np.int32(K),
                                                     np.int32(N),
                                                     self.gpuPtrs["proc_id_model","C"].gpudata,
                                                     self.gpuPtrs["graph_model","WGS"].gpudata,
                                                     self.gpuPtrs["bkgd_model","lam"].gpudata,
                                                     np.int32(ki),
                                                     np.int32(MH_ADD),
                                                     self.gpuPtrs["graph_model","qratio"].gpudata,
                                                     block=(1024, 1, 1), 
                                                     grid=(grid_w,1)
                                                     )
                
                blockLogQratio = self.gpuPtrs["graph_model","qratio"].get()
                logQratio += np.sum(blockLogQratio)
                
                if is_symmetric and ki!=kj:
                    self.gpuKernels["computeProdQratio"](np.int32(ki),
                                                         np.int32(K),
                                                         np.int32(N),
                                                         self.gpuPtrs["proc_id_model","C"].gpudata,
                                                         self.gpuPtrs["graph_model","WGS"].gpudata,
                                                         self.gpuPtrs["bkgd_model","lam"].gpudata,
                                                         np.int32(kj),
                                                         np.int32(MH_ADD),
                                                         self.gpuPtrs["graph_model","qratio"].gpudata,
                                                         block=(1024, 1, 1), 
                                                         grid=(grid_w,1)
                                                         )
                    
                    blockLogQratio = self.gpuPtrs["graph_model","qratio"].get()
                    logQratio += np.sum(blockLogQratio)
            
            # Decide whether or not to accept this change
            logPrAccept = logPratio + logQratio
            accept = np.log(np.random.rand()) < logPrAccept 
            if accept:
#                if is_symmetric and ki!=kj:
#                    log.debug("+A[%d,%d],+A[%d,%d]",ki,kj,kj,ki)
#                else:
#                    log.debug("+A[%d,%d]",ki,kj)
                
                # Update the adjacency matrix on host and GPU
                self.modelParams["graph_model","A"][ki,kj] = True
                A_buff = np.array([True], dtype=np.bool)
                cuda.memcpy_htod(self.gpuPtrs["graph_model","A"].ptr + int((ki*K+kj)*A_buff.itemsize), A_buff)
                
                if is_symmetric and ki!=kj:
                    self.modelParams["graph_model","A"][kj,ki] = True
                    A_buff = np.array([True], dtype=np.bool)
                    cuda.memcpy_htod(self.gpuPtrs["graph_model","A"].ptr + int((kj*K+ki)*A_buff.itemsize), A_buff)
    
            else:
                # Clear the WGS changes
                self.gpuKernels["clearWGSForDeletedEdge"](np.int32(ki),
                                                     np.int32(kj),
                                                     np.int32(N),
                                                     self.gpuPtrs["proc_id_model","C"].gpudata,
                                                     self.gpuPtrs["graph_model","WGS"].gpudata,
                                                     block=(1024, 1, 1), 
                                                     grid=(grid_w,1)
                                                     )
                
                if is_symmetric and ki!=kj:
                    self.gpuKernels["clearWGSForDeletedEdge"](np.int32(kj),
                                                             np.int32(ki),
                                                             np.int32(N),
                                                             self.gpuPtrs["proc_id_model","C"].gpudata,
                                                             self.gpuPtrs["graph_model","WGS"].gpudata,
                                                             block=(1024, 1, 1), 
                                                             grid=(grid_w,1)
                                                             )
        
        if op == MH_DEL:
            # We are proposing to delete an edge. WGS was calculated 
            # with the edge present. 
                    
            # Compute the P ratio
            
            rho = self.getConditionalEdgePr(ki,kj)
            if rho == 1.0:
                logPratio = -np.Inf
            elif rho == 0.0:
                logPratio = np.Inf
            elif is_symmetric and ki!=kj:
                logPratio = Ns[ki]*currW + Ns[kj]*currW + np.log(1-rho) - np.log(rho)
            else:
                logPratio = Ns[ki]*currW + np.log(1-rho) - np.log(rho)
                
            # Compute the Q ratio
            logQratio = np.log(self.params["gamma"]) - np.log(1-self.params["gamma"])
            self.gpuKernels["computeProdQratio"](np.int32(kj),
                                                 np.int32(K),
                                                 np.int32(N),
                                                 self.gpuPtrs["proc_id_model","C"].gpudata,
                                                 self.gpuPtrs["graph_model","WGS"].gpudata,
                                                 self.gpuPtrs["bkgd_model","lam"].gpudata,
                                                 np.int32(ki),
                                                 np.int32(MH_DEL),
                                                 self.gpuPtrs["graph_model","qratio"].gpudata,
                                                 block=(1024, 1, 1), 
                                                 grid=(grid_w,1)
                                                 )
            blockLogQratio = self.gpuPtrs["graph_model","qratio"].get()
            logQratio += np.sum(blockLogQratio) 
            
            if is_symmetric and ki!=kj:
                self.gpuKernels["computeProdQratio"](np.int32(ki),
                                                     np.int32(K),
                                                     np.int32(N),
                                                     self.gpuPtrs["proc_id_model","C"].gpudata,
                                                     self.gpuPtrs["graph_model","WGS"].gpudata,
                                                     self.gpuPtrs["bkgd_model","lam"].gpudata,
                                                     np.int32(kj),
                                                     np.int32(MH_DEL),
                                                     self.gpuPtrs["graph_model","qratio"].gpudata,
                                                     block=(1024, 1, 1), 
                                                     grid=(grid_w,1)
                                                     )
                blockLogQratio = self.gpuPtrs["graph_model","qratio"].get()
                logQratio += np.sum(blockLogQratio) 
                    
            # Decide whether or not to accept this change
            logPrAccept = logPratio + logQratio
            accept = np.log(np.random.rand()) < logPrAccept 
            if accept:
#                if is_symmetric and ki!=kj:
#                    log.debug("-A[%d,%d],-A[%d,%d]",ki,kj,kj,ki)
#                else:
#                    log.debug("-A[%d,%d]",ki,kj)
                    
                
                # Update the adjacency matrix
                self.modelParams["graph_model","A"][ki,kj] = False
                A_buff = np.array([False], dtype=np.bool)
                cuda.memcpy_htod(self.gpuPtrs["graph_model","A"].ptr + int((ki*K+kj)*A_buff.itemsize), A_buff)
                
                if is_symmetric and ki!=kj:
                    self.modelParams["graph_model","A"][kj,ki] = False
                    A_buff = np.array([False], dtype=np.bool)
                    cuda.memcpy_htod(self.gpuPtrs["graph_model","A"].ptr + int((kj*K+ki)*A_buff.itemsize), A_buff)
                
                # Clear the WGS changes
                self.gpuKernels["clearWGSForDeletedEdge"](np.int32(ki),
                                                     np.int32(kj),
                                                     np.int32(N),
                                                     self.gpuPtrs["proc_id_model","C"].gpudata,
                                                     self.gpuPtrs["graph_model","WGS"].gpudata,
                                                     block=(1024, 1, 1), 
                                                     grid=(grid_w,1)
                                                     )
                
                if is_symmetric and ki!=kj:
                    self.gpuKernels["clearWGSForDeletedEdge"](np.int32(kj),
                                                             np.int32(ki),
                                                             np.int32(N),
                                                             self.gpuPtrs["proc_id_model","C"].gpudata,
                                                             self.gpuPtrs["graph_model","WGS"].gpudata,
                                                             block=(1024, 1, 1), 
                                                             grid=(grid_w,1)
                                                             )
                
            else:
                # Nothing changes if we do not delete the edge
                pass
            
        # After sampling A we must resample Z 
        # (well, must if we change, but we always do anyway)
#        self.base.extensions["parent_model"].sampleZ(useWgsFromA=True)
        
    def sampleA(self):
        #def collapsed_sample_A():
        """
        Sample the entries in A using a collapsed Gibbs sampler
        """
        N = self.base.data.N
        K = self.modelParams["proc_id_model","K"]
        Ns = self.modelParams["proc_id_model","Ns"]
        
        # If there are no spikes then the posterior is the same as the prior
        # because there are no induced impulse responses
        if N == 0:
            self.modelParams["graph_model","A"] = self.sampleGraphFromPrior() 
            self.gpuPtrs["graph_model","A"].set(self.modelParams["graph_model","A"].astype(np.bool))
            return
        
        # Compute the sum of weighted impulse responses for each pair of neurons
        
        ## TODO!! WE SHOULD REFACTOR HOW WGS IS USED,
        #         IT SHOULD BE POPULATED REGARDLESS OF WHETHER OR NOT AN EDGE IS PRESENT
        #         OTHER FUNCTIONS (EG PARENT SELECTION) SHOULD BE MODIFIED TO ACCOUNT FOR THIS
        grid_w = int(np.ceil(np.float32(N)/self.params["blockSz"]))
        self.gpuKernels["computeWGSForAllSpikes"](np.int32(K),
                                                  np.int32(N),
                                                 self.gpuPtrs["proc_id_model","C"].gpudata,
                                                 self.gpuPtrs["impulse_model","GS"].gpudata,
                                                 self.base.dSS["colPtrs"].gpudata,
                                                 self.base.dSS["rowIndices"].gpudata,
                                                 self.gpuPtrs["weight_model","W"].gpudata,
                                                 self.gpuPtrs["graph_model","A"].gpudata,
                                                 self.gpuPtrs["graph_model","WGS"].gpudata,
                                                 block=(1024, 1, 1), 
                                                 grid=(grid_w,K)
                                                 )
        
        # Sample each entrie in A from its collapsed distribution 
        # That is, from its marginal distribution after summing out Z
#        log.info("Spawing threads")
#        threads = []
#        for kj in np.arange(K):
##            log.info("Sampling column %d", kj)
##            th = threading.Thread(target=self.thread_sampleA, args=(kj))
#            th = ColumnSampler(kj,self)
#            th.start()
#            threads.append(th)
#        
#        # Wait for threads to finish
#        log.info("Waiting for threads to finish")
#        for th in threads:
#            th.join()
            
#             Each column could be parallelized 
#             Randomly shuffle the order in which we sample rows ki
        for kj in np.arange(K):
            row_perm = np.random.permutation(K)
            for ki in row_perm:
#                log.info("Sampling A[%d,%d]. Current Value: %d", ki,kj, self.modelParams["graph_model","A"][ki,kj])
                # Compute the ratio of prior probabilities for A[k,k']
                rho = self.getConditionalEdgePr(ki,kj)
                self.gpuPtrs["graph_model","lkhd_ratio"].fill(0.0)
                if rho == 1.0:
                    logPratio = np.Inf
                elif rho == 0.0:
                    logPratio = -np.Inf
                elif ki==kj and not self.params["allow_self_excitation"]:
                    # Do not allow self-excitation
                    logPratio = np.float64(-np.Inf)
                else:
                    logPratio = -1.0*Ns[ki]*self.modelParams["weight_model","W"][ki,kj] + np.log(rho) - np.log(1-rho)
                
                self.gpuPtrs["graph_model","lkhd_ratio"].set(np.float32([0.0]))
                self.gpuKernels["computeLkhdRatioA"](np.int32(ki),
                                                     np.int32(kj),
                                                     np.int32(K),
                                                     np.int32(N),
                                                     self.gpuPtrs["graph_model","A"].gpudata,
                                                     self.gpuPtrs["proc_id_model","C"].gpudata,
                                                     self.gpuPtrs["graph_model","WGS"].gpudata,
                                                     self.gpuPtrs["bkgd_model","lam"].gpudata,
                                                     self.gpuPtrs["graph_model","lkhd_ratio"].gpudata,
                                                     block=(1024, 1, 1), 
                                                     grid=(grid_w,1)
                                                     )
            
                
                self.gpuKernels["sampleA"](np.int32(ki),
                                         np.int32(kj),
                                         np.int32(K),
                                         self.gpuPtrs["graph_model","A"].gpudata,
                                         np.float32(logPratio),
                                         self.gpuPtrs["graph_model","lkhd_ratio"].gpudata,
                                         np.float32(np.random.rand()),
                                         block=(1,1,1),
                                         grid=(1,1)
                                         )
                
               
        A_old = np.copy(self.modelParams["graph_model","A"])    
        self.modelParams["graph_model","A"] = self.gpuPtrs["graph_model","A"].get()
        
        log.info("Nnz A: %d\tNum changed: %d", np.count_nonzero(self.modelParams["graph_model","A"]), np.count_nonzero(A_old-self.modelParams["graph_model","A"]))
            
    def getConditionalEdgePr(self, ki, kj):
        """
        Override this in the base class
        """
        return 0.0
            
    def registerStatManager(self, statManager):
        """
        Register callbacks with the given StatManager
        """
        K = self.modelParams["proc_id_model","K"]

        def get_A():
#            import pdb; pdb.set_trace()
            return self.gpuPtrs["graph_model","A"].get()
        statManager.registerSampleCallback("A", 
#                                           lambda: self.gpuPtrs["graph_model","A"].get(),
                                           get_A,
                                           (K,K),
                                           np.bool)
