"""
Impulse response models include a shared-parameter logistic normal model
and a logistic normal model with unique params for every process pair
"""

import numpy as np
import string 
import logging

import pycuda.autoinit
import pycuda.compiler as nvcc
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.curandom as curandom

from ConfigParser import ConfigParser

from hawkes_consts import *

import sys
import os
sys.path.append(os.path.join("..","utils"))
from utils import *

sys.path.append(os.path.join("..","common"))
from model_extension import ModelExtension

log = logging.getLogger("global_log")

def constructImpulseResponseModel(impulse_model, baseModel, configFile):
    """
    Return an instance of the impulse response model specified in parameters
    """
    
    if impulse_model ==  "logistic_normal_shared":
        log.info("Creating logistic normal shared IR model")
        return LogisticNormalSharedIRModel(baseModel, configFile)
    if impulse_model ==  "logistic_normal":
        log.info("Creating logistic normal IR model")
        return LogisticNormalIRModel(baseModel, configFile)
    elif impulse_model ==  "uniform":
        log.info("Creating uniform IR model")
        return UniformIRModel(baseModel, configFile)
#    elif impulse_model == "logistic_normal_unique":
#        log.info("Creating logistic normal unique model")
#        return LogisticNormalUniqueIRModel(baseModel, configFile)
    else:
        log.error("Unsupported weight model: %s", weight_model)
        exit()
        
class LogisticNormalSharedIRModel(ModelExtension):
    """
    Logistic Normal impulse responses with shared parameters for every pair
    of processes
    """
    def __init__(self, baseModel, configFile):
        # Store pointer to base model
        self.base = baseModel
        
        # Keep a local pointer to the data for ease of notation
        self.base.data = baseModel.data
        
        # Initialize databases for this extension
        self.modelParams = baseModel.modelParams
        self.modelParams.addDatabase("impulse_model")
        self.gpuPtrs = baseModel.gpuPtrs
        self.gpuPtrs.addDatabase("impulse_model")
        
        # Load the GPU kernels necessary for impulse response param inference
        # Allocate memory on GPU for inference
        # Initialize host params        
        self.parseConfigurationFile(configFile)
        pprintDict(self.params, "Impulse Model Params")
        
        
        self.initializeGpuKernels()
        
        
    def parseConfigurationFile(self, configFile):
        """
        Parse the configuration file to get base model parameters
        """
        # Initialize defaults
        defaultParams = {}
        
        # CUDA kernels are defined externally in a .cu file
        defaultParams["cu_dir"]  = os.path.join("pyhawkes", "cuda", "cpp")
        defaultParams["cu_file"] = "ir_model.cu"
        
        defaultParams["dt_max"] = 5.0
        
        defaultParams["mu_mu"] = 0.0
        defaultParams["kappa_mu"] = 10.0
        defaultParams["a_tau"] = 10.0
        defaultParams["b_tau"] = 10.0
        
        defaultParams["thin"] = 50
        
        # Create a config parser object and read in the file
        cfgParser = ConfigParser(defaultParams)
        cfgParser.read(configFile)
        
        self.params = {}
        self.params["cu_dir"]  = cfgParser.get("ir_prior", "cu_dir")
        self.params["cu_file"] = cfgParser.get("ir_prior", "cu_file")
        
        if cfgParser.has_option("ir_prior", "mu") and cfgParser.has_option("ir_prior", "tau"):
            self.params["mu_tau_given"] = True
            self.modelParams["impulse_model","g_mu"] = cfgParser.getfloat("ir_prior", "mu")
            self.modelParams["impulse_model","g_tau"] = cfgParser.getfloat("ir_prior", "tau")
        else:
            self.params["mu_tau_given"] = False
        
            self.params["mu_mu_0"]    = cfgParser.getfloat("ir_prior", "mu_mu")
            self.params["kappa_mu_0"] = cfgParser.getfloat("ir_prior", "kappa_mu")
            # Second the parameters of the gamma prior on g_prec
            self.params["a_tau_0"]    = cfgParser.getfloat("ir_prior", "a_tau")
            self.params["b_tau_0"]    = cfgParser.getfloat("ir_prior", "b_tau")
        
        self.params["thin"] = cfgParser.getint("ir_prior", "thin")
        self.params["dt_max"]       = cfgParser.getfloat("preprocessing", "dt_max")
        self.params["blockSz"] = cfgParser.getint("cuda", "blockSz")
        self.params["max_hist"]     = cfgParser.getint("preprocessing", "max_hist")
    
    def getImpulseFunction(self):
        """
        Return an impulse response as a function of time offset
        """
        dt_max = self.params["dt_max"]
        tau = self.modelParams["impulse_model","g_tau"]
        mu = self.modelParams["impulse_model","g_mu"]
        g = lambda dt: (dt>0)*(dt<dt_max)*dt_max/(dt*(dt_max-dt)) * np.sqrt(tau/2/np.pi)*np.exp(-tau/2*(np.log(dt)-np.log(dt_max-dt))**2)
        
        return g 
    
    def computeIrDensity(self, dS_gpu):
        """
        Compute the impulse response density at the time intervals in dS_gpu 
        """
        gS_gpu = gpuarray.empty_like(dS_gpu)
        
        # Update GS using the impulse response parameters
        grid_w = min(int(np.ceil(dS_gpu.size)/1024), 2**16-1)
        grid_h = int(np.ceil(np.float32(dS_gpu.size)/(grid_w*1024)))
        self.gpuKernels["computeLogisticNormalGS"](dS_gpu.gpudata,
                                                   np.int32(dS_gpu.size),
                                                   np.float32(self.modelParams["impulse_model","g_mu"]),
                                                   np.float32(self.modelParams["impulse_model","g_tau"]),
                                                   np.float32(self.params["dt_max"]),
                                                   gS_gpu.gpudata,
                                                   block=(1024, 1, 1), 
                                                   grid=(grid_h,grid_w)
                                                   )
        
        return gS_gpu
    
    def initializeGpuKernels(self):
        kernelSrc = os.path.join(self.params["cu_dir"], self.params["cu_file"])
        
        kernelNames = ["sumNnzZPerBlock",
                       "computeGSuffStatistics", 
                       "computeLogisticNormalGS"]
        
        src_consts = {"B" : self.params["blockSz"]}
        
        self.gpuKernels = compileKernels(kernelSrc, kernelNames, src_consts)
        
    def initializeGpuMemory(self):
        K = self.modelParams["proc_id_model","K"]
        
        # Sufficient statistics for the parameters of G kernels
        self.gpuPtrs["impulse_model","nnz_Z"] = gpuarray.empty((K,K), dtype=np.int32)
        self.gpuPtrs["impulse_model","g_suff_stats"] = gpuarray.empty((K,K), dtype=np.float32) 
        self.gpuPtrs["impulse_model","GS"] = gpuarray.empty_like(self.base.dSS["dS"])
        
    def updateGs(self):
        """
        Update the impulse for each ISI
        """
        if self.base.dSS["dS_size"]==0:
            return
        
        # Update GS using the new parameters
        grid_w = int(np.ceil(np.float32(self.base.dSS["dS_size"])/1024))
        grid_w = min(2**16-1,grid_w)
        grid_h = int(np.ceil(np.float32(self.base.dSS["dS_size"])/(grid_w*1024)))
        self.gpuKernels["computeLogisticNormalGS"](self.base.dSS["dS"].gpudata,
                                                np.int32(self.base.dSS["dS_size"]),
                                                np.float32(self.modelParams["impulse_model","g_mu"]),
                                                np.float32(self.modelParams["impulse_model","g_tau"]),
                                                np.float32(self.params["dt_max"]),
                                                self.gpuPtrs["impulse_model","GS"].gpudata,
                                                block=(1024, 1, 1), 
                                                grid=(grid_h,grid_w)
                                                )
        
    def initializeModelParamsFromPrior(self):
        self.initializeGpuMemory()
        
        if not self.params["mu_tau_given"]:
            self.modelParams["impulse_model","g_tau"] = np.random.gamma(self.params["a_tau_0"], 1.0/self.params["b_tau_0"])
            self.modelParams["impulse_model","g_mu"] = np.random.normal(self.params["mu_mu_0"], np.sqrt(1.0/(self.params["kappa_mu_0"]*self.modelParams["impulse_model","g_tau"])))
        
        self.updateGs()

        
        # Keep track of number of calls to sampleModelParameters.
        # Thin with specified frequency
        self.iter = 0 
        
    def initializeModelParamsFromDict(self, paramsDB):
        """
        Copy the learned impulse model params
        """
        self.initializeGpuMemory()
        
        self.modelParams["impulse_model","g_tau"] = paramsDB["impulse_model","g_tau"]
        self.modelParams["impulse_model","g_mu"] = paramsDB["impulse_model","g_mu"]
        
        self.updateGs()
        
        # Keep track of number of calls to sampleModelParameters.
        # Thin with specified frequency
        self.iter = 0 
        
    def computeLogProbability(self):
        """
        Compute log probability of the params under the prior. 
        That is, log gamma density at g_tau and log normal density at g_mu
        """
        ll = 0.0
        ll += (self.params["a_tau_0"]-1)*np.log(self.modelParams["impulse_model","g_tau"]) - self.params["b_tau_0"]*self.modelParams["impulse_model","g_tau"]
        ll += -0.5*self.params["kappa_mu_0"]*self.modelParams["impulse_model","g_tau"]*(self.modelParams["impulse_model","g_mu"]-self.params["mu_mu_0"])**2
        
        return ll
    
    def sampleNewProcessParams(self, newProcParams):
        """
        If the Process ID Model wants to add a new process it will call this function to 
        get parameters from the prior. Sample and add to the given dict.
        """
        # There are no per-process params for this impulse model
        pass
    
    def addNewProcessEventHandler(self, newProcParams):
        """
        If a new process is added the parameters will be in the given dict.
        We need to update all our data structures that depend on K. We can
        assume that the base model has updated K
        """
        del self.gpuPtrs["impulse_model","nnz_Z"]
        del self.gpuPtrs["impulse_model","g_suff_stats"]
        
        K = self.modelParams["proc_id_model","K"]
        
        self.gpuPtrs["impulse_model","nnz_Z"] = gpuarray.empty((K,K), dtype=np.int32)
        self.gpuPtrs["impulse_model","g_suff_stats"] = gpuarray.empty((K,K), dtype=np.float32) 
        
    def removeProcessEventHandler(self, procId):
        """
        Remove process procID from the set of processes and update data structures
        accordingly. We can assume that the base model has updated K.
        """
        K = self.modelParams["proc_id_model","K"]
        
        self.gpuPtrs["impulse_model","nnz_Z"] = gpuarray.empty((K,K), dtype=np.int32)
        self.gpuPtrs["impulse_model","g_suff_stats"] = gpuarray.empty((K,K), dtype=np.float32) 
    
    def sampleModelParameters(self):
        """
        Sample the parameters of G depending on the type of impulse response
        in use
        """
        if self.params["mu_tau_given"]:
            return
        
        if np.mod(self.iter, self.params["thin"]) == 0:
            K = self.modelParams["proc_id_model","K"]
            N = self.base.data.N
            
            if N == 0:
                m = 0
            else:
                # All impulse responses require the total amount of delta spike time
                # per process pair under the current parent assignment
                grid_sz = int(np.ceil(float(K)/32))
                
               
                # Calculate sufficient statistics
                # m is the number of spikes parented by other spikes
                
                # Sum up the spikes attributed to spikes on other processes
                self.gpuKernels["sumNnzZPerBlock"](np.int32(K),
                                                   np.int32(N),
                                                   self.gpuPtrs["parent_model","Z"].gpudata,
                                                   self.gpuPtrs["proc_id_model","C"].gpudata,
                                                   self.gpuPtrs["impulse_model","nnz_Z"].gpudata,
                                                   block=(1024, 1, 1), 
                                                   grid=(K,K)
                                                   )
                
                m = gpuarray.sum(self.gpuPtrs["impulse_model","nnz_Z"]).get()
            
            # If all spikes are assigned to the background processes
            # then we just draw from the prior
            if m==0:
                kappa_mu  = self.params["kappa_mu_0"]
                mu_mu     = self.params["mu_mu_0"]
                alpha_tau = self.params["a_tau_0"]
                beta_tau  = self.params["b_tau_0"]
                
            else:
                nullParam = gpuarray.zeros((K,K), dtype=np.float32)
                # Calculate the sum of transformed interspike intervals first
                self.gpuKernels["computeGSuffStatistics"](np.int32(K),
                                                          np.int32(N),
                                                          np.int32(G_LOGISTIC_NORMAL),
                                                           np.int32(LOGISTIC_NORMAL_SUM),
                                                           nullParam.gpudata,
                                                           np.float32(self.params["dt_max"]),
                                                           self.gpuPtrs["proc_id_model","C"].gpudata,
                                                           self.base.dSS["dS"].gpudata,
                                                           self.base.dSS["rowIndices"].gpudata,
                                                           self.base.dSS["colPtrs"].gpudata,
                                                           self.gpuPtrs["parent_model","Z"].gpudata,
                                                           self.gpuPtrs["graph_model","A"].gpudata,
                                                           self.gpuPtrs["impulse_model","g_suff_stats"].gpudata,
                                                           block=(1024, 1, 1), 
                                                           grid=(K,K)
                                                           )
                
                # x is the sum of inter-spike intervals transformed back from logistic space     
                x_sum = gpuarray.sum(self.gpuPtrs["impulse_model","g_suff_stats"]).get()
                
                if not np.isfinite(x_sum):
                    log.error("x_sum not finite!")
                    log.info(self.gpuPtrs["impulse_model","g_suff_stats"].get())
                    exit()
                    
                
                # x_bar is the mean of x
                x_bar = x_sum/m;
                x_bar_gpu = gpuarray.to_gpu(np.tile(x_bar,[K,K]).astype(np.float32))
                
                # the parameters mu and tau of the kernel are Normal-Gamma distributed 
                # with params mu_mu, kappa_mu, alpha_tau, beta_tau
                kappa_mu = self.params["kappa_mu_0"] + m
                mu_mu = (self.params["kappa_mu_0"]*self.params["mu_mu_0"] + x_sum)/kappa_mu
                
                # Calculate the variance of transformed interspike intervals
                self.gpuKernels["computeGSuffStatistics"](np.int32(K),
                                                          np.int32(N),
                                                          np.int32(G_LOGISTIC_NORMAL),
                                                           np.int32(LOGISTIC_NORMAL_VARIANCE),
                                                           x_bar_gpu.gpudata,
                                                           np.float32(self.params["dt_max"]),
                                                           self.gpuPtrs["proc_id_model","C"].gpudata,
                                                           self.base.dSS["dS"].gpudata,
                                                           self.base.dSS["rowIndices"].gpudata,
                                                           self.base.dSS["colPtrs"].gpudata,
                                                           self.gpuPtrs["parent_model","Z"].gpudata,
                                                           self.gpuPtrs["graph_model","A"].gpudata,
                                                           self.gpuPtrs["impulse_model","g_suff_stats"].gpudata,
                                                           block=(1024, 1, 1), 
                                                           grid=(K,K)
                                                           )
                x_var = gpuarray.sum(self.gpuPtrs["impulse_model","g_suff_stats"]).get()
                assert np.isfinite(x_var), "ERROR: x_Var is not finite!"
                
                alpha_tau = self.params["a_tau_0"] + m/2.0
                beta_tau  = self.params["b_tau_0"] + x_var/2.0 + (self.params["kappa_mu_0"]*m*(x_bar-self.params["mu_mu_0"])**2.0)/(2.0*kappa_mu)
                
            
            # sample new precision for logistic normal impulse response
            self.modelParams["impulse_model","g_tau"] = np.random.gamma(alpha_tau, 1.0/beta_tau)
            
            # sample new mean for logistic normal impulse response
            self.modelParams["impulse_model","g_mu"] = np.random.normal(mu_mu, np.sqrt(1.0/(kappa_mu*self.modelParams["impulse_model","g_tau"])))
            
            self.updateGs()
    
        # Update iteration count
        self.iter += 1
        
    def generateSpikeOffset(self, s_pa,k_pa,k_ch,n_ch):
        """
        Generate n_ch child spike offsets on process k_ch given a 
        parent spike on process k_pa at time s_pa
        """
        # Sample normal RVs and take the logistic of them. This is equivalent
        # to sampling uniformly from the inverse CDF 
        x_ch = self.modelParams["impulse_model","g_mu"] + np.sqrt(1.0/self.modelParams["impulse_model","g_tau"])*np.random.randn(n_ch)
        # x_ch = scipy.special.erfinv(2*u_ch-1)*np.sqrt(2/mParams["g_tau"])+mParams["g_mu"]
        
        # Spike times are logistic transformation of x
        s_ch = s_pa[0] + self.params["dt_max"] * 1.0/(1.0+np.exp(-1*x_ch))
    
        return np.reshape(s_ch, (1,n_ch))
    
    def registerStatManager(self, statManager):
        """
        Register callbacks with the given StatManager
        """
        statManager.registerSampleCallback("g_mu", 
                                           lambda: self.modelParams["impulse_model","g_mu"],
                                           (1,),
                                           np.float32)
        
        statManager.registerSampleCallback("g_tau", 
                                           lambda: self.modelParams["impulse_model","g_tau"],
                                           (1,),
                                           np.float32)

class LogisticNormalIRModel(ModelExtension):
    """
    Logistic Normal impulse responses with unique parameters for every pair
    of processes
    """
    def __init__(self, baseModel, configFile):
        # Store pointer to base model
        self.base = baseModel
        
        # Keep a local pointer to the data for ease of notation
        self.base.data = baseModel.data
        
        # Initialize databases for this extension
        self.modelParams = baseModel.modelParams
        self.modelParams.addDatabase("impulse_model")
        self.gpuPtrs = baseModel.gpuPtrs
        self.gpuPtrs.addDatabase("impulse_model")
        
        # Load the GPU kernels necessary for impulse response param inference
        # Allocate memory on GPU for inference
        # Initialize host params        
        self.parseConfigurationFile(configFile)
        pprintDict(self.params, "Impulse Model Params")
        
        
        self.initializeGpuKernels()
        
        
    def parseConfigurationFile(self, configFile):
        """
        Parse the configuration file to get base model parameters
        """
        # Initialize defaults
        defaultParams = {}
        
        # CUDA kernels are defined externally in a .cu file
        defaultParams["cu_dir"]  = os.path.join("pyhawkes", "cuda", "cpp")
        defaultParams["cu_file"] = "ir_model.cu"
        
        defaultParams["dt_max"] = 5.0
        
        defaultParams["mu_mu"] = 0.0
        defaultParams["kappa_mu"] = 10.0
        defaultParams["a_tau"] = 10.0
        defaultParams["b_tau"] = 10.0
        
        defaultParams["thin"] = 50
        
        # Create a config parser object and read in the file
        cfgParser = ConfigParser(defaultParams)
        cfgParser.read(configFile)
        
        self.params = {}
        self.params["cu_dir"]  = cfgParser.get("ir_prior", "cu_dir")
        self.params["cu_file"] = cfgParser.get("ir_prior", "cu_file")
        
        if cfgParser.has_option("ir_prior", "mu") and cfgParser.has_option("ir_prior", "tau"):
            self.params["mu_tau_given"] = True
            self.modelParams["impulse_model","g_mu"] = cfgParser.getfloat("ir_prior", "mu")
            self.modelParams["impulse_model","g_tau"] = cfgParser.getfloat("ir_prior", "tau")
        else:
            self.params["mu_tau_given"] = False
        
            self.params["mu_mu_0"]    = cfgParser.getfloat("ir_prior", "mu_mu")
            self.params["kappa_mu_0"] = cfgParser.getfloat("ir_prior", "kappa_mu")
            # Second the parameters of the gamma prior on g_prec
            self.params["a_tau_0"]    = cfgParser.getfloat("ir_prior", "a_tau")
            self.params["b_tau_0"]    = cfgParser.getfloat("ir_prior", "b_tau")
        
        self.params["thin"] = cfgParser.getint("ir_prior", "thin")
        self.params["dt_max"]       = cfgParser.getfloat("preprocessing", "dt_max")
        self.params["blockSz"] = cfgParser.getint("cuda", "blockSz")
        self.params["max_hist"]     = cfgParser.getint("preprocessing", "max_hist")
    
    def getImpulseFunction(self):
        """
        Return an impulse response as a function of time offset
        """
        dt_max = self.params["dt_max"]
        tau = self.modelParams["impulse_model","g_tau"]
        mu = self.modelParams["impulse_model","g_mu"]
        g = lambda dt: (dt>0)*(dt<dt_max)*dt_max/(dt*(dt_max-dt)) * np.sqrt(tau/2/np.pi)*np.exp(-tau/2*(np.log(dt)-np.log(dt_max-dt))**2)
        
        return g 
    
    def computeIrDensity(self, dS_gpu):
        """
        Compute the impulse response density at the time intervals in dS_gpu 
        """
        K = self.modelParams["proc_id_model","K"]
        gS_gpu = gpuarray.empty_like(dS_gpu)
        
        # Update GS using the impulse response parameters
        grid_w = int(np.ceil(N/1024.0))
        self.gpuKernels["computeLogisticNormalGSIndiv"](np.int32(K),
                                                        np.int32(self.base.data.N),
                                                        self.gpuPtrs["proc_id_model","C"].gpudata,
                                                        self.base.dSS["rowIndices"].gpudata,
                                                        self.base.dSS["colPtrs"].gpudata,
                                                        self.gpuPtrs["impulse_model","g_mu"].gpudata,
                                                        self.gpuPtrs["impulse_model","g_tau"].gpudata,
                                                        np.float32(self.params["dt_max"]),
                                                        dS_gpu.gpudata,
                                                        gS_gpu.gpudata,
                                                        block=(1024, 1, 1), 
                                                        grid=(grid_w,1)
                                                        )
        
        return gS_gpu
    
    def initializeGpuKernels(self):
        kernelSrc = os.path.join(self.params["cu_dir"], self.params["cu_file"])
        
        kernelNames = ["sumNnzZPerBlock",
                       "computeGSuffStatistics", 
                       "computeLogisticNormalGSIndiv"]
        
        src_consts = {"B" : self.params["blockSz"]}
        
        self.gpuKernels = compileKernels(kernelSrc, kernelNames, src_consts)
        
    def initializeGpuMemory(self):
        K = self.modelParams["proc_id_model","K"]
        
        # Sufficient statistics for the parameters of G kernels
        self.gpuPtrs["impulse_model","nnz_Z"] = gpuarray.empty((K,K), dtype=np.int32)
        self.gpuPtrs["impulse_model","g_suff_stats"] = gpuarray.empty((K,K), dtype=np.float32)
        self.gpuPtrs["impulse_model","g_mu"] = gpuarray.empty((K,K), dtype=np.float32)
        self.gpuPtrs["impulse_model","g_tau"] = gpuarray.empty((K,K), dtype=np.float32) 
        self.gpuPtrs["impulse_model","GS"] = gpuarray.empty_like(self.base.dSS["dS"])
        
    def updateGs(self):
        """
        Update the impulse for each ISI
        """
        if self.base.dSS["dS_size"]==0:
            return
        
        K = self.modelParams["proc_id_model","K"]
        N = self.base.data.N
        
        # Update GS using the new parameters
        grid_w = int(np.ceil(N/1024.0))
        self.gpuKernels["computeLogisticNormalGSIndiv"](np.int32(K),
                                                        np.int32(self.base.data.N),
                                                        self.gpuPtrs["proc_id_model","C"].gpudata,
                                                        self.base.dSS["rowIndices"].gpudata,
                                                        self.base.dSS["colPtrs"].gpudata,
                                                        self.gpuPtrs["impulse_model","g_mu"].gpudata,
                                                        self.gpuPtrs["impulse_model","g_tau"].gpudata,
                                                        np.float32(self.params["dt_max"]),
                                                        self.base.dSS["dS"].gpudata,
                                                        self.gpuPtrs["impulse_model","GS"].gpudata,
                                                        block=(1024, 1, 1), 
                                                        grid=(grid_w,1)
                                                        )
        
        GS = self.gpuPtrs["impulse_model","GS"].get()
        if not np.all(np.isfinite(GS)):
            import pdb
            (nans,_) = np.nonzero(np.isnan(GS))
            (infs,_) = np.nonzero(np.isinf(GS)) 
        
            pdb.set_trace()
            
    def initializeModelParamsFromPrior(self):
        self.initializeGpuMemory()
        K = self.modelParams["proc_id_model","K"]
        if not self.params["mu_tau_given"]:
            self.modelParams["impulse_model","g_tau"] = np.random.gamma(self.params["a_tau_0"], 
                                                                        1.0/self.params["b_tau_0"],
                                                                        size=(K,K)).astype(np.float32)
            
            self.modelParams["impulse_model","g_mu"] = np.random.normal(self.params["mu_mu_0"], 
                                                                        np.sqrt(1.0/(self.params["kappa_mu_0"]*self.modelParams["impulse_model","g_tau"])),
                                                                        size=(K,K)).astype(np.float32)
        
        self.gpuPtrs["impulse_model","g_mu"].set(self.modelParams["impulse_model","g_mu"])
        self.gpuPtrs["impulse_model","g_tau"].set(self.modelParams["impulse_model","g_tau"])
        
        self.updateGs()
        
        # Keep track of number of calls to sampleModelParameters.
        # Thin with specified frequency
        self.iter = 0 
        
    def initializeModelParamsFromDict(self, paramsDB):
        """
        Copy the learned impulse model params
        """
        self.initializeGpuMemory()
        
        self.modelParams["impulse_model","g_tau"] = paramsDB["impulse_model","g_tau"]
        self.modelParams["impulse_model","g_mu"] = paramsDB["impulse_model","g_mu"]
        
        self.gpuPtrs["impulse_model","g_mu"].set(
                self.modelParams["impulse_model","g_mu"].astype(np.float32))
        self.gpuPtrs["impulse_model","g_tau"].set(
                self.modelParams["impulse_model","g_tau"].astype(np.float32))
                
        self.updateGs()
        
        # Keep track of number of calls to sampleModelParameters.
        # Thin with specified frequency
        self.iter = 0 
        
    def computeLogProbability(self):
        """
        Compute log probability of the params under the prior. 
        That is, log gamma density at g_tau and log normal density at g_mu
        """
        ll = 0.0
        ll += np.sum((self.params["a_tau_0"]-1)*np.log(self.modelParams["impulse_model","g_tau"]) - self.params["b_tau_0"]*self.modelParams["impulse_model","g_tau"])
        ll += np.sum(-0.5*self.params["kappa_mu_0"]*self.modelParams["impulse_model","g_tau"]*(self.modelParams["impulse_model","g_mu"]-self.params["mu_mu_0"])**2)
        
        return ll
    
    def sampleNewProcessParams(self, newProcParams):
        """
        If the Process ID Model wants to add a new process it will call this function to 
        get parameters from the prior. Sample and add to the given dict.
        """
        # There are no per-process params for this impulse model
        pass
    
    def addNewProcessEventHandler(self, newProcParams):
        """
        If a new process is added the parameters will be in the given dict.
        We need to update all our data structures that depend on K. We can
        assume that the base model has updated K
        """
        del self.gpuPtrs["impulse_model","nnz_Z"]
        del self.gpuPtrs["impulse_model","g_suff_stats"]
        
        K = self.modelParams["proc_id_model","K"]
        
        self.gpuPtrs["impulse_model","nnz_Z"] = gpuarray.empty((K,K), dtype=np.int32)
        self.gpuPtrs["impulse_model","g_suff_stats"] = gpuarray.empty((K,K), dtype=np.float32) 
        
    def removeProcessEventHandler(self, procId):
        """
        Remove process procID from the set of processes and update data structures
        accordingly. We can assume that the base model has updated K.
        """
        K = self.modelParams["proc_id_model","K"]
        
        self.gpuPtrs["impulse_model","nnz_Z"] = gpuarray.empty((K,K), dtype=np.int32)
        self.gpuPtrs["impulse_model","g_suff_stats"] = gpuarray.empty((K,K), dtype=np.float32) 
    
    def sampleModelParameters(self):
        """
        Sample the parameters of G depending on the type of impulse response
        in use
        """
        if self.params["mu_tau_given"]:
            return
        
        if np.mod(self.iter, self.params["thin"]) == 0:
            K = self.modelParams["proc_id_model","K"]
            N = self.base.data.N
            
            if N == 0:
                m = 0
            else:
                # All impulse responses require the total amount of delta spike time
                # per process pair under the current parent assignment
                grid_sz = int(np.ceil(float(K)/32))
                
               
                # Calculate sufficient statistics
                # m is the number of spikes parented by other spikes
                
                # Sum up the spikes attributed to spikes on other processes
                self.gpuKernels["sumNnzZPerBlock"](np.int32(K),
                                                   np.int32(N),
                                                   self.gpuPtrs["parent_model","Z"].gpudata,
                                                   self.gpuPtrs["proc_id_model","C"].gpudata,
                                                   self.gpuPtrs["impulse_model","nnz_Z"].gpudata,
                                                   block=(1024, 1, 1), 
                                                   grid=(K,K)
                                                   )
                
                m = self.gpuPtrs["impulse_model","nnz_Z"].get()
            
            # Calculate the sum of transformed interspike intervals first
            nullParam = gpuarray.zeros((K,K), dtype=np.float32)
            self.gpuKernels["computeGSuffStatistics"](np.int32(K),
                                                      np.int32(N),
                                                      np.int32(G_LOGISTIC_NORMAL),
                                                      np.int32(LOGISTIC_NORMAL_SUM),
                                                      nullParam.gpudata,
                                                      np.float32(self.params["dt_max"]),
                                                      self.gpuPtrs["proc_id_model","C"].gpudata,
                                                      self.base.dSS["dS"].gpudata,
                                                      self.base.dSS["rowIndices"].gpudata,
                                                      self.base.dSS["colPtrs"].gpudata,
                                                      self.gpuPtrs["parent_model","Z"].gpudata,
                                                      self.gpuPtrs["graph_model","A"].gpudata,
                                                      self.gpuPtrs["impulse_model","g_suff_stats"].gpudata,
                                                      block=(1024, 1, 1), 
                                                      grid=(K,K)
                                                      )
            
            # x is the sum of inter-spike intervals transformed back from logistic space     
            x_sum = self.gpuPtrs["impulse_model","g_suff_stats"].get()
            
            if not np.all(np.isfinite(x_sum)):
                log.error("x_sum not finite!")
                log.info(self.gpuPtrs["impulse_model","g_suff_stats"].get())
                exit()
                
            
            # x_bar is the mean of x
            x_bar = x_sum/m;
            x_bar_gpu = gpuarray.to_gpu(x_bar.astype(np.float32))
            
            # the parameters mu and tau of the kernel are Normal-Gamma distributed 
            # with params mu_mu, kappa_mu, alpha_tau, beta_tau
            kappa_mu = self.params["kappa_mu_0"] + m
            mu_mu = (self.params["kappa_mu_0"]*self.params["mu_mu_0"] + x_sum)/kappa_mu
            
            # Calculate the variance of transformed interspike intervals
            self.gpuKernels["computeGSuffStatistics"](np.int32(K),
                                                      np.int32(N),
                                                      np.int32(G_LOGISTIC_NORMAL),
                                                       np.int32(LOGISTIC_NORMAL_VARIANCE),
                                                       x_bar_gpu.gpudata,
                                                       np.float32(self.params["dt_max"]),
                                                       self.gpuPtrs["proc_id_model","C"].gpudata,
                                                       self.base.dSS["dS"].gpudata,
                                                       self.base.dSS["rowIndices"].gpudata,
                                                       self.base.dSS["colPtrs"].gpudata,
                                                       self.gpuPtrs["parent_model","Z"].gpudata,
                                                       self.gpuPtrs["graph_model","A"].gpudata,
                                                       self.gpuPtrs["impulse_model","g_suff_stats"].gpudata,
                                                       block=(1024, 1, 1), 
                                                       grid=(K,K)
                                                       )
            x_var = self.gpuPtrs["impulse_model","g_suff_stats"].get()
            assert np.all(np.isfinite(x_var)), "ERROR: x_Var is not finite!"
            
            alpha_tau = self.params["a_tau_0"] + m/2.0
            beta_tau  = self.params["b_tau_0"] + x_var/2.0 + (self.params["kappa_mu_0"]*m*(x_bar-self.params["mu_mu_0"])**2.0)/(2.0*kappa_mu)
        
            
            # If all spikes are assigned to the background processes
            # then we just draw from the prior        
            kappa_mu[m==0]  = self.params["kappa_mu_0"]
            mu_mu[m==0]     = self.params["mu_mu_0"]
            alpha_tau[m==0] = self.params["a_tau_0"]
            beta_tau[m==0]  = self.params["b_tau_0"]
            
            
            # sample new precision for logistic normal impulse response
            self.modelParams["impulse_model","g_tau"] = np.random.gamma(alpha_tau, 1.0/beta_tau)
            
            # sample new mean for logistic normal impulse response
            self.modelParams["impulse_model","g_mu"] = np.random.normal(mu_mu, np.sqrt(1.0/(kappa_mu*self.modelParams["impulse_model","g_tau"])))
            
            # Update GPU
            self.gpuPtrs["impulse_model","g_mu"].set(
                self.modelParams["impulse_model","g_mu"].astype(np.float32))
            self.gpuPtrs["impulse_model","g_tau"].set(
                self.modelParams["impulse_model","g_tau"].astype(np.float32))
            
            self.updateGs()
    
        # Update iteration count
        self.iter += 1
        
    def generateSpikeOffset(self, s_pa,k_pa,k_ch,n_ch):
        """
        Generate n_ch child spike offsets on process k_ch given a 
        parent spike on process k_pa at time s_pa
        """
        # Sample normal RVs and take the logistic of them. This is equivalent
        # to sampling uniformly from the inverse CDF 
        mu = self.modelParams["impulse_model","g_mu"][k_pa,k_ch] 
        tau = self.modelParams["impulse_model","g_tau"][k_pa,k_ch]
        x_ch = mu + np.sqrt(1.0/tau)*np.random.randn(n_ch)
        # x_ch = scipy.special.erfinv(2*u_ch-1)*np.sqrt(2/mParams["g_tau"])+mParams["g_mu"]
        
        # Spike times are logistic transformation of x
        s_ch = s_pa[0] + self.params["dt_max"] * 1.0/(1.0+np.exp(-1*x_ch))
    
        return np.reshape(s_ch, (1,n_ch))
    
    def registerStatManager(self, statManager):
        """
        Register callbacks with the given StatManager
        """
        K = self.modelParams["proc_id_model","K"]
        statManager.registerSampleCallback("g_mu", 
                                           lambda: self.modelParams["impulse_model","g_mu"],
                                           (K,K),
                                           np.float32)
        
        statManager.registerSampleCallback("g_tau", 
                                           lambda: self.modelParams["impulse_model","g_tau"],
                                           (K,K),
                                           np.float32)

        
class UniformIRModel(ModelExtension):
    """
    Impulse response model which is uniform density over the maximum interspike interval.
    """
    def __init__(self, baseModel, configFile):
        # Store pointer to base model
        self.base = baseModel
                
        # Initialize databases for this extension
        self.modelParams = baseModel.modelParams
        self.modelParams.addDatabase("impulse_model")
        self.gpuPtrs = baseModel.gpuPtrs
        self.gpuPtrs.addDatabase("impulse_model")
        
        # Load the GPU kernels necessary for impulse response param inference
        # Allocate memory on GPU for inference
        # Initialize host params        
        self.parseConfigurationFile(configFile)
        pprintDict(self.params, "Impulse Model Params")
        
    def parseConfigurationFile(self, configFile):
        
        # Create a config parser object and read in the file
        defaultParams = {}
        cfgParser = ConfigParser(defaultParams)
        cfgParser.read(configFile)
        
        self.params = {}
        self.params["dt_max"] = cfgParser.getfloat("preprocessing", "dt_max")
    
    def getImpulseFunction(self):
        """
        Return an impulse response as a function of time offset
        """
        dt_max = self.params["dt_max"]
        g = lambda dt: (dt>0)*(dt<dt_max)*1/dt_max
        
        return g
    
    def computeIrDensity(self, dS_gpu):
        """
        Compute the impulse response density at the time intervals in dS_gpu 
        """
        gS_gpu = gpuarray.empty_like(dS_gpu)
        gS_gpu.fill(self.params["density"])
        
        return gS_gpu
    
    def initializeGpuMemory(self):
        self.gpuPtrs["impulse_model","GS"] = gpuarray.empty_like(self.base.dSS["dS"])
        
    def initializeModelParamsFromPrior(self):
        self.initializeModelParams()
        
    def initializeModelParamsFromDict(self, paramsDB):
        self.initializeModelParams()
        
    def initializeModelParams(self):
        self.initializeGpuMemory()
        # Fill the GS array with a uniform value
        self.params["density"] = 1.0/self.params["dt_max"]
        self.gpuPtrs["impulse_model","GS"].fill(self.params["density"])
        
    def computeLogProbability(self):
        """
        Compute log probability of the params under the prior. All params are 
        fixed and therefore have probability 1.
        """
        ll = 0.0
        
        return ll
    
        
        