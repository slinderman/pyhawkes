"""
The default parent model is to choose among the potential parents
according to their relative probability. I can't really imagine
another model, but for consistency we define this as an extension.
"""

"""
Impulse response models include a shared-parameter logistic normal model
and a logistic normal model with unique params for every process pair
"""

import numpy as np
import logging

import pycuda.autoinit
import pycuda.compiler as nvcc
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.curandom as curandom

from pyhawkes.utils.utils import *
from model_extension import ModelExtension

from graph_models import EmptyGraphModel

log = logging.getLogger("global_log")

def constructParentModel(parent_model, baseModel, configFile):
    """
    Return an instance of the impulse response model specified in parameters
    """
    
    if parent_model ==  "default":
        log.info("Creating default parent model")
        return DefaultParentModel(baseModel, configFile)
    else:
        log.error("Unsupported parent model: %s", parent_model)
        exit()
        
class DefaultParentModel(ModelExtension):
    
    def __init__(self, baseModel, configFile):
        # Store pointer to base model
        self.base = baseModel
        
        # Initialize databases for this extension
        self.modelParams = baseModel.modelParams
        self.modelParams.addDatabase("parent_model")
        self.gpuPtrs = baseModel.gpuPtrs
        self.gpuPtrs.addDatabase("parent_model")
        
        # Parse config file
        self.parseConfigurationFile(configFile)
        
        # Initialize GPU kernels and memory
        self.initializeGpuKernels()
        
    def parseConfigurationFile(self, configFile):
        """
        Parse the configuration file to get base model parameters
        """
        # Initialize defaults
        defaultParams = {}
        
        # CUDA kernels are defined externally in a .cu file
        defaultParams["cu_dir"]  = os.path.join("pyhawkes", "cuda", "cpp")
        defaultParams["cu_file"] = "hawkes_mcmc_kernels.cu"
        defaultParams["thin"] = 1
        
        # Create a config parser object and read in the file
        cfgParser = ConfigParser(defaultParams)
        cfgParser.read(configFile)
        
        self.params = {}
        self.params["cu_dir"]  = cfgParser.get("parent_model", "cu_dir")
        self.params["cu_file"] = cfgParser.get("parent_model", "cu_file")
        self.params["thin"] = cfgParser.get("parent_model", "thin")
        self.params["blockSz"] = cfgParser.getint("cuda", "blockSz")
        
    def initializeGpuKernels(self):
        kernelSrc = os.path.join(self.params["cu_dir"], self.params["cu_file"])
        
        kernelNames = ["computeWGSForAllSpikes",
                       "sampleNewParentProcs",
                       "sampleNewParentSpikes"]
        
        src_consts = {"B" : self.params["blockSz"]}
        
        self.gpuKernels = compileKernels(kernelSrc, kernelNames, src_consts)
    
    def initializeGpuMemory(self):
        """
        Allocate GPU memory for the base model parameters
        """
        N = self.base.data.N
        K = self.modelParams["proc_id_model","K"]
        
        self.gpuPtrs["parent_model","Z"] = gpuarray.empty((N,), dtype=np.int32)
        self.gpuPtrs["parent_model","WGS"]   = gpuarray.empty((K,N), dtype=np.float32)
        self.gpuPtrs["parent_model","urand_Z_gpu"] = gpuarray.empty((N,), dtype=np.float32)
        self.gpuPtrs["parent_model","Zi_temp_gpu"] = gpuarray.empty((N,), dtype=np.int32)
        
    def initializeModelParamsFromPrior(self):
        """
        Initialize parents to the background process for all spikes.
        """
        self.initializeModelParams()
    
    def initializeModelParamsFromDict(self, paramsDB):
        """
        Initialize parents to the background process for all spikes.
        """
        self.initializeModelParams()
        
    def initializeModelParams(self):
        """
        Initialize parents to the background process for all spikes.
        """
        self.initializeGpuMemory()
        
        N = self.base.data.N
        Z0 = -1*np.ones((N,), dtype=np.int32)
        self.gpuPtrs["parent_model","Z"].set(Z0)
        
    def sampleLatentVariables(self):
        self.sampleZ()
        
    def sampleZ(self, useWgsFromA=False):
        """
        Sample new parents for every spike. There is not much point in updating W 
        if only one process's parents are updated at a time
        """
        K = self.modelParams["proc_id_model","K"]
        
        # If the graph model is empty then we don't need to do anything
        # because all spikes will belong to the background process
        if isinstance(self.base.extensions["graph_model"],EmptyGraphModel):
            return
        
        N = self.base.data.N
        if N == 0:
            return
        
        # The width of the grid is determined by the number of spikes on process j
        grid_w = int(np.ceil(np.float32(N)/self.params["blockSz"]))
        
        # We can get substantial speed up by using the WGS computed in the sampling of A
        if useWgsFromA:
            pWGS = self.gpuPtrs["graph_model","WGS"]
        else:
            # Otherwise we need to recompute
            pWGS = self.gpuPtrs["parent_model","WGS"]
            
            self.gpuKernels["computeWGSForAllSpikes"](np.int32(K),
                                                      np.int32(N),
                                                     self.gpuPtrs["proc_id_model","C"].gpudata,
                                                     self.gpuPtrs["impulse_model","GS"].gpudata,
                                                     self.base.dSS["colPtrs"].gpudata,
                                                     self.base.dSS["rowIndices"].gpudata,
                                                     self.gpuPtrs["weight_model","W"].gpudata,
                                                     self.gpuPtrs["graph_model","A"].gpudata,
                                                     pWGS.gpudata,
                                                     block=(1024, 1, 1), 
                                                     grid=(grid_w,K)
                                                     )
            
        self.base.rand_gpu.fill_uniform(self.gpuPtrs["parent_model","urand_Z_gpu"])
        
        # Call the SampleNewParentProcs kernel to update parent procs in a
        # temporary structure Zi_temp. In the second step we choose
        # actual parent spikes
        self.gpuKernels["sampleNewParentProcs"](np.int32(K),
                                                np.int32(N),
                                                self.gpuPtrs["proc_id_model","C"].gpudata,
                                                self.gpuPtrs["graph_model","A"].gpudata,
                                                self.gpuPtrs["weight_model","W"].gpudata,
                                                pWGS.gpudata,
                                                self.gpuPtrs["bkgd_model","lam"].gpudata,
                                                self.gpuPtrs["parent_model","urand_Z_gpu"].gpudata,
                                                self.gpuPtrs["parent_model","Zi_temp_gpu"].gpudata,
                                                block=(1024, 1, 1), 
                                                grid=(grid_w,1)
                                                )
        
        # Use new random variables to sample the spikes. We cannot reuse
        # the variates used to sample parent processes because, since they are 
        # now biased by the fact that they were greater than the background rate
        self.base.rand_gpu.fill_uniform(self.gpuPtrs["parent_model","urand_Z_gpu"])
        
        self.gpuKernels["sampleNewParentSpikes"](np.int32(K),
                                                 np.int32(N),
                                                 self.gpuPtrs["proc_id_model","C"].gpudata,
                                                 self.gpuPtrs["parent_model","Zi_temp_gpu"].gpudata,
                                                 self.gpuPtrs["impulse_model","GS"].gpudata,
                                                 self.base.dSS["rowIndices"].gpudata,
                                                 self.base.dSS["colPtrs"].gpudata,
                                                 self.gpuPtrs["weight_model","W"].gpudata,
                                                 pWGS.gpudata,
                                                 self.gpuPtrs["parent_model","Z"].gpudata,
                                                 self.gpuPtrs["parent_model","urand_Z_gpu"].gpudata,
                                                 block=(1024, 1, 1), 
                                                 grid=(grid_w,K)
                                                 )
        # DEBUG
#        import pdb; pdb.set_trace()
#        self.debugParentAssignments()
        
    def debugParentAssignments(self):
        """
        Check if the parent assignment is valid.
        """
        N = self.base.data.N
        K = self.modelParams["proc_id_model","K"]
        
        Z = self.gpuPtrs["parent_model","Z"].get()
        C = self.gpuPtrs["proc_id_model","C"].get()
        A = self.gpuPtrs["graph_model","A"].get()
        WGS = self.gpuPtrs["parent_model","WGS"].get()
        for n in np.arange(N):
            if Z[n] > -1:
                if not A[C[Z[n]], C[n]]:
                    log.error("Parent assigned over nonexistent edge!")
                    log.info("Spike %d(c=%d) parented by %d(c=%d)",n,C[n],Z[n],C[Z[n]])
                    log.info("WGS[:,n]")
                    log.info(WGS[:,n])
                    log.info("Zi_temp[n]")
                    log.info(self.gpuData["Zi_temp_gpu"].get()[n])
                    
                    rowInds = self.dSS["rowIndices"].get()
                    cols = self.dSS["colPtrs"].get()
                    log.info("parent spikes")
                    log.info(rowInds[np.arange(cols[n],cols[n+1])])
                    log.info("parent spike procs")
                    log.info(C[rowInds[np.arange(cols[n],cols[n+1])]])
                    exit()
        
        
