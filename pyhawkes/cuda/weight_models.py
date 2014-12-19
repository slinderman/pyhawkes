"""
Define the weight models that will be used by the Hawkes process inference algorithm.
Abstract away the details of generating each weight model and sampling its params. 
For now the only model is a time-homogenous weight matrix, but in the future this 
could be extended to include time-varying weights.
"""
import sys
import os
import numpy as np
import logging
import ConfigParser

import pycuda.gpuarray as gpuarray

from graph_models import StochasticBlockModel, ErdosRenyiModel

from pyhawkes.utils.utils import pprint_dict, compile_kernels
from model_extension import ModelExtension

log = logging.getLogger("global_log")

def construct_weight_model(weight_model, baseModel, configFile):
    """
    Return an instance of the graph model specified in parameters
    """
    
    if weight_model ==  "homogenous":
        log.info("Creating homogenous weight model")
        return HomogenousWeightModel(baseModel, configFile)
    elif weight_model == "symmetric":
        log.info("Creating symmetric weight model")
        return SymmetricWeightModel(baseModel, configFile)
    elif weight_model == "coupled_sbm_w":
        # Graph model will create the true instance
        return ModelExtension()
    else:
        log.error("Unsupported weight model: %s", weight_model)
        exit()
        
class HomogenousWeightModel(ModelExtension):
    def __init__(self, baseModel, configFile):
        # Store pointer to base model
        self.base = baseModel
        
        # Keep a local pointer to the data manager for ease of notation
        self.base.data = baseModel.data
        
        # Initialize databases for this extension
        self.modelParams = baseModel.modelParams
        self.modelParams.addDatabase("weight_model")
        self.gpuPtrs = baseModel.gpuPtrs
        self.gpuPtrs.addDatabase("weight_model")
        
        # Load the GPU kernels necessary for background rate inference
        # Allocate memory on GPU for background rate inference
        # Initialize host params        
        self.parseConfigurationFile(configFile)
        pprint_dict(self.params, "Weight Model Params")
        
        
        self.initializeGpuKernels()
        
        
        
    def parseConfigurationFile(self, configFile):
        """
        Parse the configuration file to get base model parameters
        """
        # Initialize defaults
        defaultParams = {}
        
        # CUDA kernels are defined externally in a .cu file
        defaultParams["cu_dir"]  = os.path.join("pyhawkes", "cuda", "cpp")
        defaultParams["cu_file"] = "weight_model.cu"
        
        defaultParams["a"] = 2.0
        defaultParams["numThreadsPerGammaRV"] = 32
        defaultParams["thin"] = 1
        
        # Create a config parser object and read in the file
        cfgParser = ConfigParser(defaultParams)
        cfgParser.read(configFile)
        
        self.params = {}
        self.params["cu_dir"]  = cfgParser.get("weight_prior", "cu_dir")
        self.params["cu_file"] = cfgParser.get("weight_prior", "cu_file")
        self.params["numThreadsPerGammaRV"] = cfgParser.getint("cuda", "numThreadsPerGammaRV")
        self.params["blockSz"] = cfgParser.getint("cuda", "blockSz")
        self.params["max_hist"]     = cfgParser.getint("preprocessing", "max_hist")
        
        self.params["thin"] = cfgParser.getint("weight_prior", "thin")
        self.params["a_w"] = cfgParser.getfloat("weight_prior", "a")
        
        # If b is not specified, weight until initialization so that we can choose it
        # based on the number of processes and the assumption of a complete matrix
        if cfgParser.has_option("weight_prior", "b"):
            self.params["b_initialized"] = True
            self.params["b_w"] = cfgParser.getfloat("weight_prior", "b")
        else:
            self.params["b_initialized"] = False
            
    def computeWeightScale(self):
        """
        Determine an appropriate scale paramter for the weights based on the
        type of graph prior in use
        """
        graph_model = self.base.extensions["graph_model"]
        K = self.modelParams["proc_id_model", "K"]
        b_w = 1
        if isinstance(graph_model,ErdosRenyiModel):
            rho = graph_model.rho_v
            b_w = K * self.params["a_w"] * rho / 0.7
            log.info("Set b_w=%f based on number of processes and specified alpha and rho", b_w)
            
        elif isinstance(graph_model,StochasticBlockModel):
            # Try to approximate the in-degree based on the average 
            # prob of connection between blocks
            b0 = graph_model.params["b0"]
            b1 = graph_model.params["b1"]
            rho = graph_model.rho
            
            rho *= b0/ (b0+b1)
            # Set the expected out-degree to mean+1 std
#            stdev = np.sqrt(b0*b1/((b0+b1)**2)/(b0+b1+1))
#            rho += stdev
            
            b_w = K * self.params["a_w"] * rho / 0.7
            log.info("Set b_w=%f based on number of processes and specified alpha and SBM params", b_w)
        else:
            # Default to weights appropriate for a complete graph        
            b_w = K * self.params["a_w"] / 0.7
            log.debug("Set b_w=%f based on number of processes and specified alpha and complete graph model", b_w)
        
        return b_w 
            
    def initializeGpuKernels(self):
        kernelSrc = os.path.join(self.params["cu_dir"], self.params["cu_file"])
        
        kernelNames = ["sumNnzZPerBlock",
                       "computeWPosterior",
                       "sampleGammaRV"]
        
        src_consts = {"B" : self.params["blockSz"]}
        
        # Before compiling, make sure utils.cu is in the sys path
        sys.path.append(self.params["cu_dir"])
        self.gpuKernels = compile_kernels(kernelSrc, kernelNames, src_consts)
                
    def initializeGpuMemory(self):
        K = self.modelParams["proc_id_model","K"]
        
        self.gpuPtrs["weight_model","W"] = gpuarray.empty((K,K), dtype=np.float32)
        
        self.gpuPtrs["weight_model","nnz_Z_gpu"]   = gpuarray.empty((K,K), dtype=np.int32)
        self.gpuPtrs["weight_model","aW_post_gpu"] = gpuarray.empty((K,K), dtype=np.float32)
        self.gpuPtrs["weight_model","bW_post_gpu"] = gpuarray.empty((K,K), dtype=np.float32)
        
        self.gpuPtrs["weight_model","urand_KxK_gpu"] = gpuarray.empty((K,K,self.params["numThreadsPerGammaRV"]), dtype=np.float32)
        self.gpuPtrs["weight_model","nrand_KxK_gpu"] = gpuarray.empty((K,K,self.params["numThreadsPerGammaRV"]), dtype=np.float32)
        
        # Result of GPU gamma sampling
        self.gpuPtrs["weight_model","sample_status_gpu"] = gpuarray.empty((K,K), dtype=np.int32)
        
    def initializeModelParamsFromPrior(self):
        """
        Initialize the background rate with a draw from the prior
        """
        self.initializeGpuMemory()
        
        K = self.modelParams["proc_id_model","K"]
        
        # Set b if necessary 
        if not self.params["b_initialized"]:
            self.params["b_w"] = self.computeWeightScale()
        
        W0 = np.random.gamma(self.params["a_w"],
                              1.0/self.params["b_w"], 
                              size=(K,K)).astype(np.float32)
        
        self.modelParams["weight_model","W"] = W0
        self.gpuPtrs["weight_model","W"].set(W0)
        
        # Keep track of number of calls to sampleModelParameters.
        # Thin with specified frequency
        self.iter = 0
        
    def initializeModelParamsFromDict(self, paramsDB):
        self.initializeGpuMemory()
        
        # Set b if necessary 
        if not self.params["b_initialized"]:
            self.params["b_w"] = self.computeWeightScale()
        
        self.modelParams["weight_model","W"] = paramsDB["weight_model","W"]
        self.gpuPtrs["weight_model","W"].set(paramsDB["weight_model","W"])
        self.iter = 0
                
    def sampleNewProcessParams(self, newProcParams):
        """
        If the Process ID Model wants to add a new process it will call this function to 
        get parameters from the prior. Sample and add to the given dict.
        """
        # Get the current number of processes
        K = self.modelParams["proc_id_model","K"]
        
        # Copy over the existing W
        Wold = self.gpuPtrs["weight_model","W"].get()
        
        # Update with the new params
        Wnew = np.zeros((K+1,K+1), dtype=np.float32)
                
        newRow = np.random.gamma(self.params["a_w"],
                              1.0/self.params["b_w"], 
                              size=(K,)).astype(np.float32)
                              
        newCol = np.random.gamma(self.params["a_w"],
                                 1.0/self.params["b_w"], 
                                 size=(K,)).astype(np.float32)
                                 
        newDiag = np.random.gamma(self.params["a_w"],
                                 1.0/self.params["b_w"], 
                                 size=(1,1)).astype(np.float32)
                                 
        Wnew[:-1,:-1] = Wold
        Wnew[-1,:-1] = newRow
        Wnew[-1,-1] = newDiag
        Wnew[:-1,-1] = newCol
                                 
        newProcParams["weight_model"] = {"W":Wnew}
        
    def addNewProcessEventHandler(self, newProcParams):
        """
        If a new process is added the parameters will be in the given dict.
        We need to update all our data structures that depend on K. We can
        assume that the base model has updated K
        """
        # Delete existing gpuData pointers to free space on GPU immediately
        del self.gpuPtrs["weight_model","nnz_Z_gpu"]
        del self.gpuPtrs["weight_model","aW_post_gpu"]
        del self.gpuPtrs["weight_model","bW_post_gpu"]
        del self.gpuPtrs["weight_model","urand_KxK_gpu"]
        del self.gpuPtrs["weight_model","nrand_KxK_gpu"]
        del self.gpuPtrs["weight_model","sample_status_gpu"]
        
        K = self.modelParams["proc_id_model","K"]
                
        # Update with the new params
        self.base.modelParams["W"] = newProcParams["weight_model"]["W"]
        
        # Copy over to the GPU
        self.gpuPtrs["weight_model","W"] = gpuarray.to_gpu(newProcParams["weight_model"]["W"])
        
        self.gpuPtrs["weight_model","nnz_Z_gpu"]   = gpuarray.empty((K,K), dtype=np.int32)
        self.gpuPtrs["weight_model","aW_post_gpu"] = gpuarray.empty((K,K), dtype=np.float32)
        self.gpuPtrs["weight_model","bW_post_gpu"] = gpuarray.empty((K,K), dtype=np.float32)
        
        self.gpuPtrs["weight_model","urand_KxK_gpu"] = gpuarray.empty((K,K,self.params["numThreadsPerGammaRV"]), dtype=np.float32)
        self.gpuPtrs["weight_model","nrand_KxK_gpu"] = gpuarray.empty((K,K,self.params["numThreadsPerGammaRV"]), dtype=np.float32)
        
        # Result of GPU gamma sampling
        self.gpuPtrs["weight_model","sample_status_gpu"] = gpuarray.empty((K,K), dtype=np.int32)
        
    def removeProcessEventHandler(self, procId):
        """
        Remove process procID from the set of processes and update data structures
        accordingly. We can assume that the base model has updated K.
        """
        K = self.modelParams["proc_id_model","K"]
        
        # Copy over the existing W
        Wold = self.gpuPtrs["weight_model","W"].get()
        
        # Remove a row and a column from the matrix
        Wnew = np.zeros((K,K), dtype=np.float32)
        Wnew[:procId,:procId] = Wold[:procId,:procId]
        Wnew[:procId,procId:] = Wold[:procId,(procId+1):]
        Wnew[procId:,:procId] = Wold[(procId+1):,:procId]
        Wnew[procId:,procId:] = Wold[(procId+1):,(procId+1):]
        
        # Copy over to the GPU
        self.gpuPtrs["weight_model","W"] = gpuarray.to_gpu(Wnew, dtype=np.float32)
        
        self.gpuPtrs["weight_model","nnz_Z_gpu"]   = gpuarray.empty((K,K), dtype=np.int32)
        self.gpuPtrs["weight_model","aW_post_gpu"] = gpuarray.empty((K,K), dtype=np.float32)
                     
        self.gpuPtrs["weight_model","bW_post_gpu"] = gpuarray.empty((K,K), dtype=np.float32)
        
        self.gpuPtrs["weight_model","urand_KxK_gpu"] = gpuarray.empty((K,K,self.params["numThreadsPerGammaRV"]), dtype=np.float32)
        self.gpuPtrs["weight_model","nrand_KxK_gpu"] = gpuarray.empty((K,K,self.params["numThreadsPerGammaRV"]), dtype=np.float32)
        
        # Result of GPU gamma sampling
        self.gpuPtrs["weight_model","sample_status_gpu"] = gpuarray.empty((K,K), dtype=np.int32)
        
    def computeLogProbability(self):
        """
        Compute the log prob of W under the gamma prior
        """
        W_flat = np.ravel(self.gpuPtrs["weight_model","W"].get())
        ll = np.sum((self.params["a_w"]-1)*np.log(W_flat) - self.params["b_w"]*W_flat)
        return ll
    
    def sampleModelParameters(self):
        """
        Sample W on the GPU. Use CURAND to generate uniform and 
        standard normal random variates. These are fed to the Marsaglia
        algorithm on the GPU.
        """
        if np.mod(self.iter, self.params["thin"]) == 0:
            K = self.modelParams["proc_id_model","K"]
            N = self.base.data.N
            
            if N==0:
                a_post = self.params["a_w"]
                b_post = self.params["b_w"]
                W = np.random.gamma(a_post,
                                    1.0/b_post, 
                                    size=(K,K,)).astype(np.float32)
                                      
                self.modelParams["weight_model","W"] = W
                self.gpuPtrs["weight_model","W"].set(W)
            else:
            
                self.gpuPtrs["weight_model","nnz_Z_gpu"].set(np.ones((K,K), dtype=np.int32))
                
                # Sum up the spikes attributed to spikes on other processes
                self.gpuKernels["sumNnzZPerBlock"](np.int32(K),
                                                   np.int32(N),
                                                   self.gpuPtrs["parent_model","Z"].gpudata,
                                                   self.gpuPtrs["proc_id_model","C"].gpudata,
                                                   self.gpuPtrs["weight_model","nnz_Z_gpu"].gpudata,
                                                   block=(1024, 1, 1), 
                                                   grid=(K,K)
                                                   )
                
                # Compute the posterior parameters for W's
                grid_sz = int(np.ceil(float(K)/32))
                self.gpuKernels["computeWPosterior"](np.int32(K),
                                                     self.gpuPtrs["weight_model","nnz_Z_gpu"].gpudata,
                                                      self.gpuPtrs["proc_id_model","Ns"].gpudata,
                                                      self.gpuPtrs["graph_model","A"].gpudata,
                                                      np.float32(self.params["a_w"]),
                                                      np.float32(self.params["b_w"]),
                                                      self.gpuPtrs["weight_model","aW_post_gpu"].gpudata,
                                                      self.gpuPtrs["weight_model","bW_post_gpu"].gpudata,
                                                      block=(32,32,1),
                                                      grid=(grid_sz,grid_sz)
                                                      )
                
                
                self.base.rand_gpu.fill_uniform(self.gpuPtrs["weight_model","urand_KxK_gpu"])
                self.base.rand_gpu.fill_normal(self.gpuPtrs["weight_model","nrand_KxK_gpu"])
                
                self.gpuKernels["sampleGammaRV"](self.gpuPtrs["weight_model","urand_KxK_gpu"].gpudata,
                                              self.gpuPtrs["weight_model","nrand_KxK_gpu"].gpudata,
                                              self.gpuPtrs["weight_model","aW_post_gpu"].gpudata,
                                              self.gpuPtrs["weight_model","bW_post_gpu"].gpudata,
                                              self.gpuPtrs["weight_model","W"].gpudata,
                                              self.gpuPtrs["weight_model","sample_status_gpu"].gpudata,
                                              block=(self.params["numThreadsPerGammaRV"],1,1),
                                              grid=(K,K)
                                              )
                
                # Copy W to host
                self.modelParams["weight_model","W"] = self.gpuPtrs["weight_model","W"].get()
        
        #    assert np.all(gpuData["sample_status_gpu"].get() == 0), "Sampling W failed!"
            
        self.iter += 1
        
    def registerStatManager(self, statManager):
        """
        Register callbacks with the given StatManager
        """
        K = self.modelParams["proc_id_model","K"]
        statManager.registerSampleCallback("W", 
                                           lambda: self.gpuPtrs["weight_model","W"].get(),
                                           (K,K),
                                           np.float32)
        
        
        statManager.registerSampleCallback("nnz_Z", 
                                           lambda: self.gpuPtrs["weight_model","nnz_Z_gpu"].get(),
                                           (K,K),
                                           np.int32)
        
        
class SymmetricWeightModel(ModelExtension):
    """
    The entries in this matrix are i.i.d. from a Gamma prior, but the cross diagonal
    entries are restricted to be the same.  
    """
    def __init__(self, baseModel, configFile):
        # Store pointer to base model
        self.base = baseModel
                
        # Initialize databases for this extension
        self.modelParams = baseModel.modelParams
        self.modelParams.addDatabase("weight_model")
        self.gpuPtrs = baseModel.gpuPtrs
        self.gpuPtrs.addDatabase("weight_model")
        
        # Load the GPU kernels necessary for background rate inference
        # Allocate memory on GPU for background rate inference
        # Initialize host params        
        self.parseConfigurationFile(configFile)
        pprint_dict(self.params, "Weight Model Params")
        
        self.initializeGpuKernels()
        self.initializeGpuMemory()
                
        
    def parseConfigurationFile(self, configFile):
        """
        Parse the configuration file to get base model parameters
        """
        # Initialize defaults
        defaultParams = {}
        
        # CUDA kernels are defined externally in a .cu file
        defaultParams["cu_dir"]  = os.path.join("pyhawkes", "cuda", "cpp")
        defaultParams["cu_file"] = "weight_model.cu"
        
        defaultParams["a"] = 2.0
        defaultParams["b"] = 5.0
        defaultParams["numThreadsPerGammaRV"] = 32
        defaultParams["thin"] = 1
        
        # Create a config parser object and read in the file
        cfgParser = ConfigParser(defaultParams)
        cfgParser.read(configFile)
        
        self.params = {}
        self.params["cu_dir"]  = cfgParser.get("weight_prior", "cu_dir")
        self.params["cu_file"] = cfgParser.get("weight_prior", "cu_file")
        self.params["a_w"] = cfgParser.getfloat("weight_prior", "a")
        self.params["b_w"] = cfgParser.getfloat("weight_prior", "b")
        self.params["thin"] = cfgParser.getint("weight_prior", "thin")
        self.params["numThreadsPerGammaRV"] = cfgParser.getint("cuda", "numThreadsPerGammaRV")
        self.params["blockSz"] = cfgParser.getint("cuda", "blockSz")
        self.params["max_hist"]     = cfgParser.getint("preprocessing", "max_hist")
    
    def initializeGpuKernels(self):
        kernelSrc = os.path.join(self.params["cu_dir"], self.params["cu_file"])
        
        kernelNames = ["sumNnzZPerBlock",
                       "computeWPosterior",
                       "sampleGammaRV",
                       "computeSymmWPosterior",
                       "copySymmW"]
        
        src_consts = {"B" : self.params["blockSz"]}
        
        # Before compiling, make sure utils.cu is in the sys path
        sys.path.append(self.params["cu_dir"])
        self.gpuKernels = compile_kernels(kernelSrc, kernelNames, src_consts)
                
    def initializeGpuMemory(self):
        K = self.modelParams["proc_id_model","K"]
        
        self.gpuData = {}
        self.gpuPtrs["weight_model","nnz_Z_gpu"]   = gpuarray.empty((K,K), dtype=np.int32)
        self.gpuPtrs["weight_model","aW_post_gpu"] = gpuarray.empty((K,K), dtype=np.float32)
                     
        self.gpuPtrs["weight_model","bW_post_gpu"] = gpuarray.empty((K,K), dtype=np.float32)
        
        self.gpuPtrs["weight_model","urand_KxK_gpu"] = gpuarray.empty((K,K,self.params["numThreadsPerGammaRV"]), dtype=np.float32)
        self.gpuPtrs["weight_model","nrand_KxK_gpu"] = gpuarray.empty((K,K,self.params["numThreadsPerGammaRV"]), dtype=np.float32)
        
        # Result of GPU gamma sampling
        self.gpuPtrs["weight_model","sample_status_gpu"] = gpuarray.empty((K,K), dtype=np.int32)
        
    def initializeModelParamsFromPrior(self):
        """
        Initialize the background rate with a draw from the prior.
        """
        self.initializeGpuMemory()
        
        K = self.modelParams["proc_id_model","K"]
        W0 = np.zeros((K,K), dtype=np.float32)
        
        # Draw the diagonal
        W0[np.diag_indices(K)] = np.random.gamma(self.params["a_w"],
                                                 1.0/self.params["b_w"], 
                                                 size=(K,))
        # Draw a lower-triangular matrix
        W0_lower = np.tril(np.random.gamma(self.params["a_w"],
                                           1.0/self.params["b_w"], 
                                           size=(K,K)).astype(np.float32), 1)
        
        # Copy the lower triangular matrix to the upper triangle
        W0 += W0_lower
        W0 += W0_lower.T
        
        self.gpuPtrs["weight_model","W"].set(W0)
        
        # Keep track of number of calls to sampleModelParameters.
        # Thin with specified frequency
        self.iter = 0
        
    def initializeModelParamsFromDict(self, paramsDB):
        self.initializeGpuMemory()
        
        self.modelParams["weight_model","W"] = paramsDB["weight_model","W"]
        self.gpuPtrs["weight_model","W"].set(paramsDB["weight_model","W"])
        self.iter = 0
        
        
    def sampleNewProcessParams(self, newProcParams):
        """
        If the Process ID Model wants to add a new process it will call this function to 
        get parameters from the prior. Sample and add to the given dict.
        """
        # Get the current number of processes
        K = self.modelParams["proc_id_model","K"]
        
        # Copy over the existing W
        Wold = self.gpuPtrs["weight_model","W"].get()
        
        # Update with the new params
        Wnew = np.zeros((K+1,K+1), dtype=np.float32)
                
        newRow = np.random.gamma(self.params["a_w"],
                              1.0/self.params["b_w"], 
                              size=(K,)).astype(np.float32)
                              
        newCol = newRow.T
                                 
        newDiag = np.random.gamma(self.params["a_w"],
                                 1.0/self.params["b_w"], 
                                 size=(1,1)).astype(np.float32)
                                 
        Wnew[:-1,:-1] = Wold
        Wnew[-1,:-1] = newRow
        Wnew[-1,-1] = newDiag
        Wnew[:-1,-1] = newCol
                                 
        newProcParams["weight_model"] = {"W":Wnew}
        
    def addNewProcessEventHandler(self, newProcParams):
        """
        If a new process is added the parameters will be in the given dict.
        We need to update all our data structures that depend on K. We can
        assume that the base model has updated K
        """
        K = self.modelParams["proc_id_model","K"]
                
        # Update with the new params
        # Copy over to the GPU
        self.gpuPtrs["weight_model","W"] = gpuarray.to_gpu(newProcParams["weight_model"]["W"])
        
        self.gpuPtrs["weight_model","nnz_Z_gpu"]   = gpuarray.empty((K,K), dtype=np.int32)
        self.gpuPtrs["weight_model","aW_post_gpu"] = gpuarray.empty((K,K), dtype=np.float32)
        self.gpuPtrs["weight_model","bW_post_gpu"] = gpuarray.empty((K,K), dtype=np.float32)
        
        self.gpuPtrs["weight_model","urand_KxK_gpu"] = gpuarray.empty((K,K,self.params["numThreadsPerGammaRV"]), dtype=np.float32)
        self.gpuPtrs["weight_model","nrand_KxK_gpu"] = gpuarray.empty((K,K,self.params["numThreadsPerGammaRV"]), dtype=np.float32)
        
        # Result of GPU gamma sampling
        self.gpuPtrs["weight_model","sample_status_gpu"] = gpuarray.empty((K,K), dtype=np.int32)
        
    def removeProcessEventHandler(self, procId):
        """
        Remove process procID from the set of processes and update data structures
        accordingly. We can assume that the base model has updated K.
        """
        K = self.modelParams["proc_id_model","K"]
        
        # Copy over the existing W
        Wold = self.gpuPtrs["weight_model","W"].get()
        
        # Remove a row and a column from the matrix
        Wnew = np.zeros((K,K), dtype=np.float32)
        Wnew[:procId,:procId] = Wold[:procId,:procId]
        Wnew[:procId,procId:] = Wold[:procId,(procId+1):]
        Wnew[procId:,:procId] = Wold[(procId+1):,:procId]
        Wnew[procId:,procId:] = Wold[(procId+1):,(procId+1):]
        
        # Copy over to the GPU
        self.gpuPtrs["weight_model","W"] = gpuarray.to_gpu(Wnew, dtype=np.float32)
        
        self.gpuPtrs["weight_model","nnz_Z_gpu"]   = gpuarray.empty((K,K), dtype=np.int32)
        self.gpuPtrs["weight_model","aW_post_gpu"] = gpuarray.empty((K,K), dtype=np.float32)
        self.gpuPtrs["weight_model","bW_post_gpu"] = gpuarray.empty((K,K), dtype=np.float32)
        
        self.gpuPtrs["weight_model","urand_KxK_gpu"] = gpuarray.empty((K,K,self.params["numThreadsPerGammaRV"]), dtype=np.float32)
        self.gpuPtrs["weight_model","nrand_KxK_gpu"] = gpuarray.empty((K,K,self.params["numThreadsPerGammaRV"]), dtype=np.float32)
        
        # Result of GPU gamma sampling
        self.gpuPtrs["weight_model","sample_status_gpu"] = gpuarray.empty((K,K), dtype=np.int32)
        
    def computeLogProbability(self):
        """
        Compute the log prob of W under the gamma prior
        """
        W = self.gpuPtrs["weight_model","W"].get()
        W_lower = np.tril(W)
        # A gamma r.v. is 0 with probability 0, thus we can safely take only the nonzero entries
        W_flat = np.ravel(W_lower[W_lower>0])
        ll = np.sum((self.params["a_w"]-1)*np.log(W_flat) - self.params["b_w"]*W_flat)
        return ll
    
    def sampleModelParameters(self):
        """
        Sample W on the GPU. Use CURAND to generate uniform and 
        standard normal random variates. These are fed to the Marsaglia
        algorithm on the GPU.
        """
        if np.mod(self.iter, self.params["thin"]) == 0:
            K = self.modelParams["proc_id_model","K"]
            N = self.base.data.N
            
            self.gpuPtrs["weight_model","nnz_Z_gpu"].set(np.ones((K,K), dtype=np.int32))
            
            # Sum up the spikes attributed to spikes on other processes
            self.gpuKernels["sumNnzZPerBlock"](np.int32(K),
                                               np.int32(N),
                                               self.gpuPtrs["parent_model","Z"].gpudata,
                                               self.gpuPtrs["proc_id_model","C"].gpudata,
                                               self.gpuPtrs["weight_model","nnz_Z_gpu"].gpudata,
                                               block=(1024, 1, 1), 
                                               grid=(K,K)
                                               )

            # DEBUG
            if np.sum(self.gpuPtrs["weight_model","nnz_Z_gpu"].get()) > 0:
                if np.sum(self.modelParams["graph_model","A"]) == 0:
                    import pdb; pdb.set_trace()
                    raise Exception("Spikes assigned over nonexistent edges!")
            # END DEBUG

            #startPerfTimer(perfDict, "sample_W")
            # Compute the posterior parameters for W's
            grid_sz = int(np.ceil(float(K)/32))
            self.gpuKernels["computeWPosterior"](np.int32(K),
                                                 self.gpuPtrs["weight_model","nnz_Z_gpu"].gpudata,
                                                  self.gpuPtrs["proc_id_model","Ns"].gpudata,
                                                  self.gpuPtrs["graph_model","A"].gpudata,
                                                  np.float32(self.params["a_w"]),
                                                  np.float32(self.params["b_w"]),
                                                  self.gpuPtrs["weight_model","aW_post_gpu"].gpudata,
                                                  self.gpuPtrs["weight_model","bW_post_gpu"].gpudata,
                                                  block=(32,32,1),
                                                  grid=(grid_sz,grid_sz)
                                                  )
            
            # The posterior params of a symmetric model are the sum of the cross diagonal terms
            self.gpuKernels["computeSymmWPosterior"](np.int32(K),
                                                     np.float32(self.params["a_w"]),
                                                     np.float32(self.params["b_w"]),
                                                     self.gpuPtrs["weight_model","aW_post_gpu"].gpudata,
                                                     self.gpuPtrs["weight_model","bW_post_gpu"].gpudata,
                                                     block=(32,32,1),
                                                     grid=(grid_sz,grid_sz)
                                                     )
            
            
            self.base.rand_gpu.fill_uniform(self.gpuPtrs["weight_model","urand_KxK_gpu"])
            self.base.rand_gpu.fill_normal(self.gpuPtrs["weight_model","nrand_KxK_gpu"])
            
            self.gpuKernels["sampleGammaRV"](self.gpuPtrs["weight_model","urand_KxK_gpu"].gpudata,
                                          self.gpuPtrs["weight_model","nrand_KxK_gpu"].gpudata,
                                          self.gpuPtrs["weight_model","aW_post_gpu"].gpudata,
                                          self.gpuPtrs["weight_model","bW_post_gpu"].gpudata,
                                          self.gpuPtrs["weight_model","W"].gpudata,
                                          self.gpuPtrs["weight_model","sample_status_gpu"].gpudata,
                                          block=(self.params["numThreadsPerGammaRV"],1,1),
                                          grid=(K,K)
                                          )
            
            # Make the new matrix symmetric
            self.gpuKernels["copySymmW"](np.int32(K),
                                         self.gpuPtrs["weight_model","W"].gpudata,
                                         block=(32,32,1),
                                         grid=(grid_sz,grid_sz)
                                         )
            
            # Copy W to host
            self.modelParams["weight_model","W"] = self.gpuPtrs["weight_model","W"].get()
            
        #    assert np.all(gpuData["sample_status_gpu"].get() == 0), "Sampling W failed!"
            
            #stopPerfTimer(perfDict, "sample_W")
        self.iter += 1
        
    def registerStatManager(self, statManager):
        """
        Register callbacks with the given StatManager
        """
        K = self.modelParams["proc_id_model","K"]
        statManager.registerSampleCallback("W", 
                                           lambda: self.gpuPtrs["weight_model","W"].get(),
                                           (K,K),
                                           np.float32)
        
