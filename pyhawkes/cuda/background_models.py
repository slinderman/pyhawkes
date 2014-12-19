"""
ImplemeN_knotss differeN_knots models for the background rate. Namely:
- a homogenous background rate
- a Gaussian process background rate with fixed iN_knotservals between knots

This abstracts the details of sampling
"""

import numpy as np
import scipy.linalg

import string
import logging

import pycuda.autoinit
import pycuda.compiler as nvcc
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.curandom as curandom

from ConfigParser import ConfigParser

import sys
import os
sys.path.append(os.path.join("..","utils"))
from pyhawkes.utils.utils import *
from pyhawkes.utils.elliptical_slice import *
from model_extension import ModelExtension

log = logging.getLogger("global_log")

#def constructBackgroundRateModel(bkgd_model, baseModel, configFile):
#    """
#    Generic factory method to be called by base model
#    """
#    if bkgd_model == "homogenous":
#        log.info("Creating homogenous background rate model")
#        return HomogenousRateModel(baseModel, configFile)
#    elif bkgd_model == "gp":
#        log.info("Creating Gaussian process weight model")
#        return GaussianProcRateModel(baseModel, configFile)
#    elif bkgd_model == "glm":
#        log.info("Creating GLM background model")
#        return GlmBackgroundModel(baseModel, configFile)
#    elif bkgd_model == "shared_gp":
#        log.info("Creating Shared Gaussian process weight model")
#        return SharedGpBkgdModel(baseModel, configFile)
#    else:
#        raise Exception("Unsupported background rate model: %s" % bkgd_model)
#

class HomogenousRateModel(ModelExtension):
    def __init__(self, baseModel, configFile):
        # Store pointer to base model
        self.base = baseModel

        # Initialize databases for this extension
        self.modelParams = baseModel.modelParams
        self.modelParams.addDatabase("bkgd_model")
        self.gpuPtrs = baseModel.gpuPtrs
        self.gpuPtrs.addDatabase("bkgd_model")

        # Load the GPU kernels necessary for background rate inference
        # Allocate memory on GPU for background rate inference
        # Initialize host params
        self.parseConfigurationFile(configFile)
        pprintDict(self.params, "Background Model Params")

        self.initializeGpuKernels()
#        self.initializeGpuMemory()

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
        defaultParams["a"] = 1.0
        #defaultParams["b"] = 4.0
        defaultParams["numThreadsPerGammaRV"] = 32

        # Create a config parser object and read in the file
        cfgParser = ConfigParser(defaultParams)
        cfgParser.read(configFile)

        self.params = {}
        self.params["cu_dir"]  = cfgParser.get("bkgd_model", "cu_dir")
        self.params["cu_file"] = cfgParser.get("bkgd_model", "cu_file")
        self.params["thin"] = cfgParser.getint("bkgd_model", "thin")
        self.params["a_lam"] = cfgParser.getfloat("bkgd_model", "a")
        self.params["numThreadsPerGammaRV"] = cfgParser.getint("cuda", "numThreadsPerGammaRV")


        # If b is not specified, weight until initialization so that we can choose it
        # based on the number of processes and the assumption of a complete matrix
        if cfgParser.has_option("bkgd_model", "b"):
            self.params["b_initialized"] = True
            self.params["b_lam"] = cfgParser.getfloat("bkgd_model", "b")
        else:
            self.params["b_initialized"] = False

        self.params["blockSz"] = cfgParser.getint("cuda", "blockSz")

    def initializeGpuKernels(self):
        kernelSrc = os.path.join(self.params["cu_dir"], self.params["cu_file"])

        kernelNames = ["sumZBkgd",
                       "computeLamHomogPosterior",
                       "computeLambdaHomogPerSpike",
                       "sampleGammaRV"]

        src_consts = {"B" : self.params["blockSz"]}

        self.gpuKernels = compileKernels(kernelSrc, kernelNames, src_consts)


    def initializeGpuMemory(self):
        K = self.modelParams["proc_id_model","K"]
        N = self.base.data.N

        if N>0:
            self.gpuPtrs["bkgd_model","lam"] = gpuarray.empty((K,N), dtype=np.float32)
        self.gpuPtrs["bkgd_model","lam_homog"] = gpuarray.empty((K,), dtype=np.float32)

        # Posterior parameters of p(mu)
        self.gpuPtrs["bkgd_model","a_lam_post"] = gpuarray.empty((K,), dtype=np.float32)
        self.gpuPtrs["bkgd_model","b_lam_post"] = gpuarray.empty((K,), dtype=np.float32)

        self.gpuPtrs["bkgd_model","Z_bkgd"]   = gpuarray.empty((K,), dtype=np.int32)

        # Space for gamma RVs
        self.gpuPtrs["bkgd_model","urand_Kx1"] = gpuarray.empty((K,self.params["numThreadsPerGammaRV"]), dtype=np.float32)
        self.gpuPtrs["bkgd_model","nrand_Kx1"] = gpuarray.empty((K,self.params["numThreadsPerGammaRV"]), dtype=np.float32)
        self.gpuPtrs["bkgd_model","sample_status"] = gpuarray.empty((K,), dtype=np.int32)

    def computeBkgdRateScale(self):
        """
        Determine an appropriate scale paramter for the background rate based on the
        number of spikes. Center the background rate on the mean number of spikes/process
        """
        graph_model = self.base.extensions["graph_model"]
        K = self.modelParams["proc_id_model", "K"]
        N = self.base.data.N
        T = (self.base.data.Tstop - self.base.data.Tstart)

        b_lam = self.params["a_lam"] / (float(N)/K/T)
        log.info("Setting b_lam=%f based on number of spikes in dataset", b_lam)

        return b_lam

    def initializeModelParamsFromPrior(self):
        """
        Initialize the background rate with a draw from the prior
        """
        self.initializeGpuMemory()

        # Set b if necessary
        if not self.params["b_initialized"]:
            self.params["b_lam"] = self.computeBkgdRateScale()

        lam0 = np.random.gamma(self.params["a_lam"],
                              1.0/self.params["b_lam"],
                              size=(self.modelParams["proc_id_model","K"],)).astype(np.float32)

        self.modelParams["bkgd_model","lam_homog"] = lam0
        self.gpuPtrs["bkgd_model","lam_homog"].set(lam0)

        self.sampleLamPerSpike()

        self.iter = 0

    def initializeModelParamsFromDict(self, paramsDB):
        self.initializeGpuMemory()

        # Set b if necessary
        if not self.params["b_initialized"]:
            self.params["b_lam"] = self.computeBkgdRateScale()

        self.modelParams["bkgd_model","lam_homog"] = paramsDB["bkgd_model","lam_homog"]
        self.gpuPtrs["bkgd_model","lam_homog"].set(paramsDB["bkgd_model","lam_homog"])

        self.sampleLamPerSpike()

        self.iter = 0


    def integrateBkgdRates(self):
        """
        Integrate the background rates. For homogenous rates this is
        the integral of a constant from 0 to T
        """
        return (self.base.data.Tstop-self.base.data.Tstart) * self.gpuPtrs["bkgd_model","lam_homog"].get()

    def evaluateBkgdRate(self, t):
        """
        Evaluate the background rate at the specified time points
        """
        K = self.modelParams["proc_id_model","K"]
        N_t = len(t)

        lam_homog = self.gpuPtrs["bkgd_model","lam_homog"].get()
        lam_homog = np.reshape(lam_homog, (K,1))

        lam_interp = np.repeat(lam_homog, N_t, axis=1)
        return lam_interp

    def cumIntegrateBkgdRates(self, t):
        """
        Return the cumulative integral evaluated at the specified times.
        The GP background rate is linearly interpolated between knots, so we
        can evaluate the cumulative integral at  the knots and interpolate at t.
        """
        K = self.modelParams["proc_id_model","K"]
        N_t = len(t)

        # Get the time relative to Tstart as a row vector
        t_rel = t-self.base.data.Tstart
        t_rel = np.reshape(t, (1,N_t))

        # The cumulative integral is the outerproduct of lam_homog and t_rel
        lam = self.gpuPtrs["bkgd_model","lam_homog"].get()
        lam = np.reshape(lam, (K,1))

        return np.dot(lam,t_rel)

    def computeLogProbability(self):
        """
        Compute the log probability of the background model params given the prior
        For GP backgrounod rates this is a Gaussian probability. We drop the terms
        that do not depend on the background rate x
        """
        # The log prob is the probability under the Gamma prior. The terms depending on
        # lambda are (a_lam-1)*log(lam) + (-b_lam*lam)
        lam = self.gpuPtrs["bkgd_model","lam_homog"].get()
        ll = 0.0
        for k in np.arange(self.modelParams["proc_id_model","K"]):
            ll += (self.params["a_lam"]-1)*np.log(lam[k]) - self.params["b_lam"]*lam[k]

        return ll

    def sampleNewProcessParams(self, newProcParams):
        """
        If the Process ID Model wants to add a new process it will call this function to
        get parameters from the prior. Sample and add to the given dict.
        """
        # Copy over the existing processes' background rates
        lam_homog_old = self.gpuPtrs["bkgd_model","lam_homog"].get()
        lam_new_proc = np.random.gamma(self.params["a_lam"], 1.0/self.params["b_lam"])
        lam_homog_new = np.hstack((lam_homog_old, [lam_new_proc])).astype(np.float32)

        newProcParams["bkgd_model"] = {"lam_homog":lam_homog_new}

    def addNewProcessEventHandler(self, newProcParams):
        """
        If a new process is added the parameters will be in the given dict.
        We need to update all our data structures that depend on K. We can
        assume that the base model has updated K
        """
        K = self.modelParams["proc_id_model","K"]

        # Copy over the existing processes' background rates
        self.modelParams["bkgd_model","lam_homog"] = newProcParams["bkgd_model"]["lam_homog"]
        self.gpuPtrs["bkgd_model","lam_homog"] = gpuarray.to_gpu(newProcParams["bkgd_model"]["lam_homog"])

        # Posterior parameters of p(mu)
        self.gpuPtrs["bkgd_model","a_lam_post"] = gpuarray.empty((K,), dtype=np.float32)
        self.gpuPtrs["bkgd_model","b_lam_post"] = gpuarray.empty((K,), dtype=np.float32)

        self.gpuPtrs["bkgd_model","Z_bkgd"]   = gpuarray.empty((K,), dtype=np.int32)

        # Space for gamma RVs
        self.gpuPtrs["bkgd_model","urand_Kx1"] = gpuarray.empty((K,self.params["numThreadsPerGammaRV"]), dtype=np.float32)
        self.gpuPtrs["bkgd_model","nrand_Kx1"] = gpuarray.empty((K,self.params["numThreadsPerGammaRV"]), dtype=np.float32)
        self.gpuPtrs["bkgd_model","sample_status"] = gpuarray.empty((K,), dtype=np.int32)

        # TODO Update lam memory on GPU

    def removeProcessEventHandler(self, procId):
        """
        Remove process procID from the set of processes and update data structures
        accordingly. We can assume that the base model has updated K.
        """
        K = self.modelParams["proc_id_model","K"]

        # Copy over the existing processes' background rates
        lam_homog_old = self.gpuPtrs["bkgd_model","lam_homog"].get()
        lam_homog_new = np.hstack((lam_homog_old[:procId], lam_homog_old[(procId+1):]))
        self.modelParams["bkgd_model","lam_homog"] = lam_homog_new
        self.gpuPtrs["bkgd_model","lam_homog"] = gpuarray.to_gpu(lam_homog_new.astype(np.float32))

        # Posterior parameters of p(mu)
        self.gpuPtrs["bkgd_model","a_lam_post"] = gpuarray.empty((K,), dtype=np.float32)
        self.gpuPtrs["bkgd_model","b_lam_post"] = gpuarray.empty((K,), dtype=np.float32)

        self.gpuPtrs["bkgd_model","Z_bkgd"]   = gpuarray.empty((K,), dtype=np.int32)

        # Space for gamma RVs
        self.gpuPtrs["bkgd_model","urand_Kx1"] = gpuarray.empty((K,self.params["numThreadsPerGammaRV"]), dtype=np.float32)
        self.gpuPtrs["bkgd_model","nrand_Kx1"] = gpuarray.empty((K,self.params["numThreadsPerGammaRV"]), dtype=np.float32)
        self.gpuPtrs["bkgd_model","sample_status"] = gpuarray.empty((K,), dtype=np.int32)

        # TODO: Update lam memory on GPU

    def sampleModelParameters(self):
        """
        Sample homogenous background rate on the GPU. Use CURAND to generate uniform and
        standard normal random variates. These are fed into the Marsaglia
        algorithm on the GPU.
        """
        if np.mod(self.iter, self.params["thin"]) == 0:
            K = self.modelParams["proc_id_model","K"]
            N = self.base.data.N

            if N==0:
                a_post = self.params["a_lam"]
                b_post = self.params["b_lam"]+self.base.data.Tstop-self.base.data.Tstart
                lam0 = np.random.gamma(a_post,
                                      1.0/b_post,
                                      size=(self.modelParams["proc_id_model","K"],)).astype(np.float32)

                self.modelParams["bkgd_model","lam_homog"] = lam0
                self.gpuPtrs["bkgd_model","lam_homog"].set(lam0)
            else:

                self.gpuKernels["sumZBkgd"](np.int32(N),
                                            self.gpuPtrs["parent_model","Z"].gpudata,
                                            self.gpuPtrs["proc_id_model","C"].gpudata,
                                            self.gpuPtrs["bkgd_model","Z_bkgd"].gpudata,
                                            block=(1024, 1, 1),
                                            grid=(K,1)
                                            )

                # Compute the posterior parameters for each mu
                grid_sz = int(np.ceil(float(K)/1024))
                self.gpuKernels["computeLamHomogPosterior"](np.int32(K),
                                                            self.gpuPtrs["bkgd_model","Z_bkgd"].gpudata,
                                                            np.float32(self.params["a_lam"]),
                                                            np.float32(self.params["b_lam"]),
                                                            np.float32(self.base.data.Tstop-self.base.data.Tstart),
                                                            self.gpuPtrs["bkgd_model","a_lam_post"].gpudata,
                                                            self.gpuPtrs["bkgd_model","b_lam_post"].gpudata,
                                                            block=(1024,1,1),
                                                            grid=(grid_sz,1)
                                                            )


                self.base.rand_gpu.fill_uniform(self.gpuPtrs["bkgd_model","urand_Kx1"])
                self.base.rand_gpu.fill_normal(self.gpuPtrs["bkgd_model","nrand_Kx1"])

                self.gpuKernels["sampleGammaRV"](self.gpuPtrs["bkgd_model","urand_Kx1"].gpudata,
                                                   self.gpuPtrs["bkgd_model","nrand_Kx1"].gpudata,
                                                   self.gpuPtrs["bkgd_model","a_lam_post"].gpudata,
                                                   self.gpuPtrs["bkgd_model","b_lam_post"].gpudata,
                                                   self.gpuPtrs["bkgd_model","lam_homog"].gpudata,
                                                   self.gpuPtrs["bkgd_model","sample_status"].gpudata,
                                                   block=(self.params["numThreadsPerGammaRV"],1,1),
                                                   grid=(K,1)
                                                   )

                # Update model params
                self.modelParams["bkgd_model","lam_homog"] = self.gpuPtrs["bkgd_model","lam_homog"].get()
    #            assert np.all(self.gpuData["sample_status"].get()==0)


                self.sampleLamPerSpike()


        self.iter += 1

    def sampleLamPerSpike(self):
        K = self.modelParams["proc_id_model","K"]
        N = self.base.data.N

        if N==0:
            return

        # Update the lam vector on the GP
        grid_w = int(np.ceil(N/1024.0))
        self.gpuKernels["computeLambdaHomogPerSpike"](np.int32(K),
                                                      self.gpuPtrs["bkgd_model","lam_homog"].gpudata,
                                                       np.int32(N),
                                                       self.gpuPtrs["proc_id_model","C"].gpudata,
                                                       self.gpuPtrs["bkgd_model","lam"].gpudata,
                                                       block=(1024,1,1),
                                                       grid=(grid_w,1)
                                                       )

    def registerStatManager(self, statManager):
        """
        Register callbacks with the given StatManager
        """
        K = int(self.modelParams["proc_id_model","K"])
        statManager.registerSampleCallback("lam_homog",
                                           lambda: self.gpuPtrs["bkgd_model","lam_homog"].get(),
                                           (K,),
                                           np.float32)

    def generateBkgdRate(self, tt):
        """
        Predict the background rate at the specified times. For a homogenous
        background rate this is simply the parameter lambda
        """
        K = self.modelParams["proc_id_model","K"]
        lam_homog = self.gpuPtrs["bkgd_model","lam_homog"].get()

        lam = np.empty((K, len(tt)))
        for k in range(K):
            lam[k,:] = lam_homog[k]

        return lam

class GaussianProcRateModel(ModelExtension):
    def __init__(self, baseModel, configFile):
        self.base = baseModel

        # Initialize databases for this extension
        self.modelParams = baseModel.modelParams
        self.modelParams.addDatabase("bkgd_model")
        self.gpuPtrs = baseModel.gpuPtrs
        self.gpuPtrs.addDatabase("bkgd_model")

        self.parseConfigurationFile(configFile)
        pprintDict(self.params, "Background Model Params")

#        self.calculateNumKnots()
        self.initializeGpuKernels()

        self.iter = 0

    def parseConfigurationFile(self, configFile):
        """
        Parse the configuration file to get base model parameters
        """
        # Initialize defaults
        defaultParams = {}

        # CUDA kernels are defined externally in a .cu file
        defaultParams["cu_dir"]  = os.path.join("pyhawkes", "cuda", "cpp")
        defaultParams["cu_file"] = "bkgd_model.cu"
        defaultParams["thin"] = 1

        # Create a config parser object and read in the file
        cfgParser = ConfigParser(defaultParams)
        cfgParser.read(configFile)

        self.params = {}
        self.params["cu_dir"]  = cfgParser.get("bkgd_model", "cu_dir")
        self.params["cu_file"] = cfgParser.get("bkgd_model", "cu_file")

        self.params["kernel_type"] = cfgParser.get("bkgd_model", "kernel")
        self.params["lam_dt"] = cfgParser.getfloat("bkgd_model", "dt")
        self.params["lam_tau"] = cfgParser.getfloat("bkgd_model", "tau")
        self.params["lam_sigma"] = cfgParser.getfloat("bkgd_model", "sigma")

        if cfgParser.has_option("bkgd_model", "T"):
            self.params["lam_T"] = cfgParser.getfloat("bkgd_model", "T")
        self.params["mu_mu"] = np.log(cfgParser.getfloat("bkgd_model", "mu_mu"))
        self.params["sig_mu"] = cfgParser.getfloat("bkgd_model", "sig_mu")
        self.params["thin"] = cfgParser.getint("bkgd_model", "thin")

        self.params["blockSz"] = cfgParser.getint("cuda", "blockSz")
        self.params["max_hist"]     = cfgParser.getint("preprocessing", "max_hist")

    def calculateNumKnots(self):
        self.modelParams["bkgd_model","N_knots"] = int(np.floor((self.base.data.Tstop-self.base.data.Tstart)/self.params["lam_dt"]+1))
#        log.info("Approximating background rate with %d knots.", self.modelParams["bkgd_model","N_knots"])

    def initializeGpuKernels(self):
        kernelSrc = os.path.join(self.params["cu_dir"], self.params["cu_file"])

        kernelNames = ["computeLamOffsetAndFrac",
                       "computeLambdaGPPerSpike",
                       "computeLamLogLkhd"]

        src_consts = {"B" : self.params["blockSz"]}

        self.gpuKernels = compileKernels(kernelSrc, kernelNames, src_consts)

    def initializeGpuMemory(self):
        K = self.modelParams["proc_id_model","K"]
        N = self.base.data.N

        self.gpuPtrs["bkgd_model","lam"]   = gpuarray.empty((K,N), dtype=np.float32)

        self.gpuPtrs["bkgd_model","lam_knots"]   = gpuarray.empty((K,self.modelParams["bkgd_model","N_knots"]), dtype=np.float32)

        # Each spike needs to know its offset in lam and the fractional distance between
        # the preceding and following lam grid poiN_knotss
        self.gpuPtrs["bkgd_model","lam_offset"] = gpuarray.empty((N,), dtype=np.int32)
        self.gpuPtrs["bkgd_model","lam_frac"]   = gpuarray.empty((N,), dtype=np.float32)


        # Compute each spike's bin background rate parameters
        # Temporarily copy S and C to the GPU
        # The bin is determined by the spike's offset from Tstart
        S_gpu = gpuarray.to_gpu(self.base.data.S.astype(np.float32) - self.base.data.Tstart)

        grid_w = int(np.ceil(float(N)/self.params["blockSz"]))
        self.gpuKernels["computeLamOffsetAndFrac"](np.float32(self.params["lam_dt"]),
                                                   np.int32(N),
                                                   np.float32(0),
                                                   S_gpu.gpudata,
                                                   self.gpuPtrs["bkgd_model","lam_offset"].gpudata,
                                                   self.gpuPtrs["bkgd_model","lam_frac"].gpudata,
                                                   block=(1024,1,1),
                                                   grid=(grid_w,1)
                                                   )

    def initialize_gp_cov_kernel(self):
        """
        Initialize the covariance kernel of the Gaussian process
        """
        kernel = None
        if self.params["kernel_type"] == "ou":
            kernel = lambda dt: np.exp(-dt/self.params["lam_tau"]) * self.params["lam_sigma"]
        elif self.params["kernel_type"] == "rbf":
            kernel = lambda dt: np.exp(-dt**2/self.params["lam_tau"]**2/2) * self.params["lam_sigma"]
        elif self.params["kernel_type"] == "periodic":
            kernel = lambda dt: np.exp(-2*np.sin(np.pi*dt/self.params["lam_T"])**2/self.params["lam_tau"]**2) * self.params["lam_sigma"]
        else:
            raise Exception("Unsupported kernel type: %s" % self.params["kernel_type"])

        return kernel

    def initializeModelParamsFromPrior(self):
        """
        The GP has an exponential kernel decaying in time with precision lam_tau and
        with overall variance scaled by lam_sigma
        """
        self.calculateNumKnots()
        self.initializeGpuMemory()

        K = self.modelParams["proc_id_model","K"]
#
#        self.modelParams["bkgd_model","N_knots"] = int(np.floor(self.base.data.T/self.params["lam_dt"]+1))
#        log.info("Approximating background rate with %d knots.", self.modelParams["bkgd_model","N_knots"])
        self.kernel = self.initialize_gp_cov_kernel()

        # Evaluate the kernel at evenly spaced "knots" in the interval [0,T]
        self.modelParams["bkgd_model","knots"] = np.linspace(self.base.data.Tstart,self.base.data.Tstop, self.modelParams["bkgd_model","N_knots"])

        # Since the knots are evenly spaced, the covariance matrix is Toeplitz
        self.modelParams["bkgd_model","lam_C"] = scipy.linalg.toeplitz(self.kernel(self.modelParams["bkgd_model","knots"]))

        # Add diagonal offset to prevent instability
        self.modelParams["bkgd_model","lam_C"] += 1e-6*np.eye(self.modelParams["bkgd_model","N_knots"])

        # Save the cholesky of the kernel for generating normal random variates
        self.modelParams["bkgd_model","lam_L"] = scipy.linalg.cholesky(self.modelParams["bkgd_model","lam_C"], lower=True)

        # Save the inverse of the covariance matrix for use in computing log probs
        L_inv = scipy.linalg.inv(self.modelParams["bkgd_model","lam_L"])
        self.modelParams["bkgd_model","inv_lam_C"] = np.dot(L_inv.T, L_inv)

        # Set the mean of each GP.
        # TODO: Make the mean a random variable
#        self.modelParams["bkgd_model","lam_mu_mu"] = np.random.normal(self.params["mu_mu"], self.params["sig_mu"], size=(K,))
        self.modelParams["bkgd_model","lam_mu_mu"] = self.params["mu_mu"]*np.ones((K,))
        self.modelParams["bkgd_model","lam_mu"] = np.ones((K, self.modelParams["bkgd_model","N_knots"]), dtype=np.float32)
        for k in np.arange(K):
            self.modelParams["bkgd_model","lam_mu"][k,:] *= self.modelParams["bkgd_model","lam_mu_mu"][k]


        # Initialize the background rate at the knots with a random draw from the prior
        self.modelParams["bkgd_model","lam_knots"] = self.modelParams["bkgd_model","lam_mu"] + np.dot(self.modelParams["bkgd_model","lam_L"], np.random.randn(self.modelParams["bkgd_model","N_knots"],self.modelParams["proc_id_model","K"])).T
        self.gpuPtrs["bkgd_model","lam_knots"].set(self.modelParams["bkgd_model","lam_knots"].astype(np.float32))

        # Initialize the background rate at each spike
        for k in np.arange(K):
            grid_sz = int(np.ceil(float(self.base.data.N)/1024))
            self.gpuKernels["computeLambdaGPPerSpike"](np.int32(k),
                                                       np.int32(self.base.data.N),
                                                       np.int32(self.modelParams["bkgd_model","N_knots"]),
                                                       self.gpuPtrs["bkgd_model","lam_knots"].gpudata,
                                                       self.gpuPtrs["bkgd_model","lam_offset"].gpudata,
                                                       self.gpuPtrs["bkgd_model","lam_frac"].gpudata,
                                                       self.gpuPtrs["bkgd_model","lam"].gpudata,
                                                       block=(1024,1,1),
                                                       grid=(grid_sz,1)
                                                       )

        self.iter = 0


    def initializeModelParamsFromDict(self, paramsDB):
        """
        In the prediction tests we need to take into account the value of the
        background rate at previous knots as learned during training. The predicted
        rate will be affected by the inferred rate at these points.
        """
        self.calculateNumKnots()
        self.initializeGpuMemory()

        K = self.modelParams["proc_id_model","K"]

        # The modelParams ParamsDatabase is already populated with firing rates at
        # knots from the learning phase. We want to incorporate those into a GP distribution
        # over the firing rate at the predictive knots.
        prev_knots = paramsDB["bkgd_model", "knots"]
        prev_lam_knots = paramsDB["bkgd_model", "lam_knots"]
        N_prev_knots  = len(prev_knots)
        self.modelParams["bkgd_model","lam_mu_mu"] = paramsDB["bkgd_model","lam_mu_mu"]

        # The paramsDB contains the background rate at a set of previous knots
        # Calculate the covariance matrix and its inverse at the previous knots
        C11 = scipy.linalg.toeplitz(self.kernel(prev_knots))
        L11 = scipy.linalg.cholesky(C11, lower=True)
        invL11 = scipy.linalg.inv(L11)
        invC11 = np.dot(invL11.T, invL11)

        # Initialize knots for the predicted data
#        self.modelParams["bkgd_model","N_knots"] = int(np.floor((self.base.data.Tstop-self.base.data.Tstart)/self.params["lam_dt"]+1))
#        log.info("Approximating background rate with %d knots.", self.modelParams["bkgd_model","N_knots"])

        # Evaluate the kernel at evenly spaced "knots" in the interval [0,T]
        self.modelParams["bkgd_model","knots"] = np.linspace(self.base.data.Tstart, self.base.data.Tstop, self.modelParams["bkgd_model","N_knots"])

        # Initialize kernel function
        self.kernel = self.initialize_gp_cov_kernel()

        # To compute C12 first create a meshgrid of knots and tt of shape (N_knots, length(tt))
        G12 = np.meshgrid(prev_knots, self.modelParams["bkgd_model","knots"])
        # [i,j]th entry is prev_knots[i] - knots[j]
        dt12 = np.abs(G12[1]-G12[0])
        C12 = np.asmatrix(self.kernel(dt12))

        # TODO: Currently using numpy 1.6.1. The next version (1.7) has a keyword for "ij" indexing
        # which would return the array of the desired shape
        C12 = C12.T
        assert np.shape(C12) == (N_prev_knots,self.modelParams["bkgd_model","N_knots"]), "ERROR: invalid shape for covariance matrix C12! %s. Expected (%d,%d)" % (str(np.shape(C12)),N_prev_knots, self.modelParams["bkgd_model","N_knots"])

        # Do the same for C22
        G22 = np.meshgrid(self.modelParams["bkgd_model","knots"], self.modelParams["bkgd_model","knots"])
        dt22 = np.abs(G22[1]-G22[0]).T
        C22 = np.asmatrix(self.kernel(dt22))

        assert np.shape(C22) == (self.modelParams["bkgd_model","N_knots"],self.modelParams["bkgd_model","N_knots"]), "ERROR: invalid shape for covariance matrix C12! %s" % str(np.shape(C22))

        # Calculate the conditional covariance matrix
        self.modelParams["bkgd_model","lam_C"] = C22 - C12.T*invC11*C12

        # Add diagonal noise to ensure SPD
        self.modelParams["bkgd_model","lam_C"] += np.diag(1e-10*np.ones(self.modelParams["bkgd_model","N_knots"]), 0)

#        assert np.allclose(self.modelParams["bkgd_model","lam_C"][0,0], [0.0])

        self.modelParams["bkgd_model","lam_L"] = scipy.linalg.cholesky(self.modelParams["bkgd_model","lam_C"], lower=True)

        # Each process may have different mean due to the differing histories of each process
        self.modelParams["bkgd_model","lam_mu"] = np.zeros((K, self.modelParams["bkgd_model","N_knots"]), dtype=np.float32)
        for k in np.arange(K):
            # Compute the parameters of the conditional normal distribution
            # Incorporate mu into the conditional mean
            prev_lam_knots_k = np.asmatrix(np.reshape(prev_lam_knots[k,:], (N_prev_knots,1)))
#            lam_mu_k = logit_lam_homog_mu + C12.T*invC11*(prev_lam_knots_k-logit_lam_homog_mu)
            lam_mu_k = self.modelParams["bkgd_model","lam_mu_mu"][k] + C12.T*invC11*(prev_lam_knots_k-self.modelParams["bkgd_model","lam_mu_mu"][k])

            self.modelParams["bkgd_model","lam_mu"][k,:] = np.reshape(lam_mu_k,(self.modelParams["bkgd_model","N_knots"],))

            assert np.allclose(self.modelParams["bkgd_model","lam_mu"][k,0],prev_lam_knots[k,-1])

        # Initialize the background rate at the knots with a random draw from the prior
        self.modelParams["bkgd_model","lam_knots"] = self.modelParams["bkgd_model","lam_mu"] + np.dot(self.modelParams["bkgd_model","lam_L"], np.random.randn(self.modelParams["bkgd_model","N_knots"],self.modelParams["proc_id_model","K"])).T
        self.gpuPtrs["bkgd_model","lam_knots"].set(self.modelParams["bkgd_model","lam_knots"].astype(np.float32))

        # DEBUG
#        if not np.allclose(self.modelParams["bkgd_model","lam_knots"][:,0],prev_lam_knots[:,-1],1e-4,1e-4):
#            log.error("Discontinuity at T split")
#            log.info(self.modelParams["bkgd_model","lam_knots"][:,0])
#            log.info(prev_lam_knots[:,-1])
#            exit()

        # Initialize the background rate at each spike
        for k in np.arange(K):
            grid_sz = int(np.ceil(float(self.base.data.N)/1024))
            self.gpuKernels["computeLambdaGPPerSpike"](np.int32(k),
                                                       np.int32(self.base.data.N),
                                                       np.int32(self.modelParams["bkgd_model","N_knots"]),
                                                       self.gpuPtrs["bkgd_model","lam_knots"].gpudata,
                                                       self.gpuPtrs["bkgd_model","lam_offset"].gpudata,
                                                       self.gpuPtrs["bkgd_model","lam_frac"].gpudata,
                                                       self.gpuPtrs["bkgd_model","lam"].gpudata,
                                                       block=(1024,1,1),
                                                       grid=(grid_sz,1)
                                                       )

        self.iter = 0

    def sampleNewProcessParams(self, newProcParams):
        """
        If the Process ID Model wants to add a new process it will call this function to
        get parameters from the prior. Sample and add to the given dict.
        """
        lam_knots_new = np.copy(self.modelParams["bkgd_model","lam_knots"])
        lam_knots_new = np.vstack((lam_knots_new, \
                                   np.log(self.modelParams["bkgd_model","lam_mu_mu"]) + np.dot(self.modelParams["bkgd_model","lam_L"], np.random.randn(self.modelParams["bkgd_model","N_knots"],)).T))

        newProcParams["bkgd_model"] = {"lam_knots":lam_knots_new}

    def addNewProcessEventHandler(self, newProcParams):
        """
        If a new process is added the parameters will be in the given dict.
        We need to update all our data structures that depend on K. We can
        assume that the base model has updated K
        """
        del self.gpuPtrs["bkgd_model","lam_knots"]

        K = self.modelParams["proc_id_model","K"]

        # Update lam_knots
        self.gpuPtrs["bkgd_model","lam_knots"]   = gpuarray.empty((K,self.modelParams["bkgd_model","N_knots"]), dtype=np.float32)
        self.modelParams["bkgd_model","lam_knots"] = newProcParams["bkgd_model"]["lam_knots"]
        self.gpuPtrs["bkgd_model","lam_knots"].set(self.modelParams["bkgd_model","lam_knots"].astype(np.float32))

        # Calculate the background rate at each spike
        for k in np.arange(K):
            grid_sz = int(np.ceil(float(self.base.data.N)/1024))
            self.gpuKernels["computeLambdaGPPerSpike"](np.int32(k),
                                                       np.int32(self.base.data.N),
                                                       np.int32(self.modelParams["bkgd_model","N_knots"]),
                                                       self.gpuPtrs["bkgd_model","lam_knots"].gpudata,
                                                       self.gpuPtrs["bkgd_model","lam_offset"].gpudata,
                                                       self.gpuPtrs["bkgd_model","lam_frac"].gpudata,
                                                       self.gpuPtrs["bkgd_model","lam"].gpudata,
                                                       block=(1024,1,1),
                                                       grid=(grid_sz,1)
                                                       )



    def removeProcessEventHandler(self, procId):
        """
        Remove process procID from the set of processes and update data structures
        accordingly. We can assume that the base model has updated K.
        """
        K = self.modelParams["proc_id_model","K"]

        # Update lam_knots
        self.gpuPtrs["bkgd_model","lam_knots"]   = gpuarray.empty((K,self.modelParams["bkgd_model","N_knots"]), dtype=np.float32)
        self.modelParams["bkgd_model","lam_knots"] = np.vstack((self.modelParams["bkgd_model","lam_knots"][:procId,:],\
                                              self.modelParams["bkgd_model","lam_knots"][(procId+1),:]))
        self.gpuPtrs["bkgd_model","lam_knots"].set(self.modelParams["bkgd_model","lam_knots"])

        # Calculate the background rate at each spike
        for k in np.arange(K):
            grid_sz = int(np.ceil(float(self.base.data.N)/1024))
            self.gpuKernels["computeLambdaGPPerSpike"](np.int32(k),
                                                       np.int32(self.base.data.N),
                                                       np.int32(self.modelParams["bkgd_model","N_knots"]),
                                                       self.gpuPtrs["bkgd_model","lam_knots"].gpudata,
                                                       self.gpuPtrs["bkgd_model","lam_offset"].gpudata,
                                                       self.gpuPtrs["bkgd_model","lam_frac"].gpudata,
                                                       self.gpuPtrs["bkgd_model","lam"].gpudata,
                                                       block=(1024,1,1),
                                                       grid=(grid_sz,1)
                                                       )

    def computeLamLogLikelihood(self, lam, k):
        """
        Compute the log likelihood of a given background rate. Leverage the GPU
        to calculate per-spike contributions.
        """

        # Copy the proposed lambda to the GPU
        # The prior is over zero-mean gaussian processes, but the actual value is
        # centered at the offset specified in the parameters
        # The parameter is specified in rate space (i.e. exponential of the gp))
        # so we first convert it to gp space by taking the log


        assert np.size(lam) == self.modelParams["bkgd_model","N_knots"]
        # TODO: Fix up the hacky offset calculation (4 bytes per float32)
        dest = int(self.gpuPtrs["bkgd_model","lam_knots"].ptr + k*self.modelParams["bkgd_model","N_knots"]*4)
#        cuda.memcpy_htod(dest, np.array(lam+self.modelParams["bkgd_model","lam_mu"][k,:], dtype=np.float32))
        cuda.memcpy_htod(dest, np.array(lam, dtype=np.float32))

        # Calculate the background rate at each spike
        grid_sz = int(np.ceil(float(self.base.data.N)/1024))
        self.gpuKernels["computeLambdaGPPerSpike"](np.int32(k),
                                                   np.int32(self.base.data.N),
                                                   np.int32(self.modelParams["bkgd_model","N_knots"]),
                                                   self.gpuPtrs["bkgd_model","lam_knots"].gpudata,
                                                   self.gpuPtrs["bkgd_model","lam_offset"].gpudata,
                                                   self.gpuPtrs["bkgd_model","lam_frac"].gpudata,
                                                   self.gpuPtrs["bkgd_model","lam"].gpudata,
                                                   block=(1024,1,1),
                                                   grid=(grid_sz,1)
                                                   )



        # Calculate the log likelihood of the proposed background rate
        ll_gpu = np.array([0.0], dtype=np.float32)
        self.gpuKernels["computeLamLogLkhd"](np.int32(k),
                                             np.int32(self.base.data.N),
                                             self.gpuPtrs["bkgd_model","lam_knots"].gpudata,
                                             np.float32(self.params["lam_dt"]),
                                             np.int32(self.modelParams["bkgd_model","N_knots"]),
                                             self.gpuPtrs["parent_model","Z"].gpudata,
                                             self.gpuPtrs["proc_id_model","C"].gpudata,
                                             self.gpuPtrs["bkgd_model","lam"].gpudata,
                                             cuda.Out(ll_gpu),
                                             block=(1024,1,1),
                                             grid=(1,1)
                                             )

        ll = ll_gpu[0]
        if np.isnan(ll):
            log.error("Log likelihood is not finite for process %d!", k)
            log.info(ll)
            log.info(lam)
            log.info(np.min(lam))
            log.info(np.max(lam))
            exit()
#        log.info('Log likelihood %f', ll)
        return ll

    def integrateBkgdRates(self):
        """
        Integrate the background rates. For piecewise linear function this is
        the trapezoidal quadrature of the rate
        """
        lam = np.exp(self.gpuPtrs["bkgd_model","lam_knots"].get())
        trap_coeff = self.params["lam_dt"] * np.ones((self.modelParams["bkgd_model","N_knots"],1))
        trap_coeff[0] *= 0.5
        trap_coeff[-1] *= 0.5

        # Matrix vector multiply to get trapezoidal integration of each bkgd rate
        # (K x N_knots) * (N_knots x 1) = (K x 1)
        return np.dot(lam, trap_coeff)

    def evaluateBkgdRate(self, t):
        """
        Evaluate the background rate at the specified time points
        """
        K = self.modelParams["proc_id_model","K"]
        N_t = len(t)
        knots = self.modelParams["bkgd_model","knots"]
        lam = np.exp(self.gpuPtrs["bkgd_model","lam_knots"].get())

        lam_interp = np.zeros((K,N_t))
        for k in np.arange(K):
            lam_interp[k,:] = np.interp(t,knots,lam[k,:])

        return lam_interp

    def cumIntegrateBkgdRates(self, t):
        """
        Return the cumulative integral evaluated at the specified times.
        The GP background rate is linearly interpolated between knots, so we
        can evaluate the cumulative integral at  the knots and interpolate at t.
        """
        K = self.modelParams["proc_id_model","K"]
        N_knots = self.modelParams["bkgd_model","N_knots"]
        N_t = len(t)

        knots = self.modelParams["bkgd_model","knots"]
        lam = np.exp(self.gpuPtrs["bkgd_model","lam_knots"].get())
        trap_coeff = self.params["lam_dt"] * np.ones((N_knots,1))
        trap_coeff[0] *= 0.5
        trap_coeff[-1] *= 0.5

        cumIntegral = np.zeros((K,N_t))

        for k in np.arange(K):
            lam_int = np.cumsum(trap_coeff*lam[k,:])
            cumIntegral[k,:] = np.interp(t,knots,lam_int)

        return cumIntegral

    def computeLogProbability(self):
        """
        Compute the log probability of the background model params given the prior
        For GP backgrounod rates this is a Gaussian probability. We drop the terms
        that do not depend on the background rate x
        """
        # The log prob is 1/2*x^T*inv(L*L.T)*x which requires the inverse of the
        # covariance matrix for computing likelihood.
        ll = 0.0
        for k in np.arange(self.modelParams["proc_id_model","K"]):
            lam = np.reshape(self.modelParams["bkgd_model","lam_knots"][k,:], (self.modelParams["bkgd_model","N_knots"],1))
            ll += -0.5 * np.dot(np.dot(lam.T, self.modelParams["bkgd_model","inv_lam_C"]), lam)

        return ll

    def sampleMuMu(self):
        """
        Sample the homogeneous background rates
        """

        # Sample the mean of the GP means
        inv_C = self.modelParams["bkgd_model","inv_lam_C"]
        for k in np.arange(self.modelParams["proc_id_model","K"]):
            x = self.modelParams["bkgd_model", "lam_knots"][k,:]
            [X1,X2] = np.meshgrid(x, x)
            xsum = X1+X2

            # The posterior is normal
            sig_hat = 1.0/2.0/(np.sum(inv_C) + 1/self.params["sig_mu"])
            mu_hat = (np.sum(xsum*np.asarray(inv_C)) + 2*self.params["mu_mu"]/self.params["sig_mu"])*sig_hat
            self.modelParams["bkgd_model","lam_mu_mu"][k] = np.random.normal(mu_hat,sig_hat)

            # Update mean vector
            self.modelParams["bkgd_model","lam_mu"][k,:] = self.modelParams["bkgd_model","lam_mu_mu"][k]*np.ones(self.modelParams["bkgd_model","N_knots"])

    def sampleModelParameters(self):
        """
        Sample the homogeneous background rates
        """
        if np.mod(self.iter, self.params["thin"]) == 0:
            self.sampleMuMu()

    def sampleLatentVariables(self):
        """
        Sample new background rates lambda using elliptical slice sampling
        The idea is that the number of points at which lambda is evaluated
        will be much smaller than the number of spikes. We leverage the GPU
        to calculate the per-spike contribution to lambda's likelihood, and
        we compute the trapezoidal integration on the host.
        """
        if np.mod(self.iter, self.params["thin"]) == 0:
            # Update each background rate in turn
            for k in np.arange(self.modelParams["proc_id_model","K"]):
                # Create a single param function to compute the log likelihood of a
                # given background rate
                ll_fn = lambda lam,args: self.computeLamLogLikelihood(lam, k)
                lam_cur = self.modelParams["bkgd_model","lam_knots"][k,:]

                # DEBUG: Call ll_fn multiple timse and assert that we get the same ans
#                ll_test = np.zeros((10,), dtype=np.float32)
#                lam_test = np.zeros((10, self.modelParams["bkgd_model", "N_knots"]))
#                for j in np.arange(10):
#                    ll_test[j] = ll_fn(lam_cur,None)
##                    lam_per_spk = self.gpuPtrs["bkgd_model","lam"].get()
#                    lam_knots = self.gpuPtrs["bkgd_model","lam_knots"].get()
#                    lam_test[j,:] = np.copy(lam_knots[k,:])
#                if not np.allclose(ll_test-ll_test[0], np.zeros(10)):
#                    log.info("k=%d",k)
#                    log.error("ll_fn returned varying values!")
#                    log.info(ll_test)
#                    for j in np.arange(10):
#                        if not np.allclose(lam_test[j,:],lam_test[0,:]):
#                            log.error("lambda per knot differs on trials %d and 0", j)
#                        else:
#                            log.info("lam_per_knot[%d,:]=lam_per_knot[0,:]", j)
#                    exit()


                try:
                    (lam_new, ll_new) = elliptical_slice(lam_cur, self.modelParams["bkgd_model","lam_L"], ll_fn,  mu=self.modelParams["bkgd_model","lam_mu"][k,:])
                except Exception as e:
                    log.error("Exception in Elliptical Slice sampling!")
                    log.error(e.message)
                    exit()

                # Update background rate on host and gpu
                self.modelParams["bkgd_model","lam_knots"][k,:] = lam_new

        self.iter += 1

    def registerStatManager(self, statManager):
        """
        Register callbacks with the given StatManager
        """
        statManager.registerSampleCallback("lam",
                                           lambda: np.exp(self.gpuPtrs["bkgd_model","lam_knots"].get()),
                                           (self.modelParams["proc_id_model","K"],self.modelParams["bkgd_model","N_knots"]),
                                           np.float32)

        statManager.registerSampleCallback("lam_mu",
                                           lambda: np.exp(self.modelParams["bkgd_model","lam_mu_mu"]),
                                           (self.modelParams["proc_id_model","K"],),
                                           np.float32)


    def generateBkgdRate(self, tt):
        """
        Predict the background rate at the specified times. For a GP rate this
        is a draw from the conditional distribution, which is normal. The joint
        distribution of the GP over the observed and unobserved times is given by

        lam[knots, tt] ~ N(mu, [[C11, C12], [C12.T, C22]])

        where C11 is the kernel evaluated at deltas between each pair of knots
              C12 is the kernel evaluated at deltas between each pair of knot and tt
              C22 is the kernel evaluated at deltas between each pair in tt

        The conditional distribution of tt given knots is then
        lam[tt | knots] ~ N(mu + C12.T * inv(C11) * (knots-mu), C22 - C12.T * inv(C11) * C12)
        """
        K = self.modelParams["proc_id_model","K"]
        N_t = len(tt)

        # We already have inv(C11) from initialization
        invC11 = np.asmatrix(self.modelParams["bkgd_model","inv_lam_C"])

        # To compute C12 first create a meshgrid of knots and tt of shape (N_knots, length(tt))
        G12 = np.meshgrid(self.modelParams["bkgd_model","knots"], tt)
        # [i,j]th entry is knots[i] - tt[j] (due to the transpose)
        dt12 = np.abs(G12[1]-G12[0]).T
        C12 = np.asmatrix(self.kernel(dt12))

        assert np.shape(C12) == (self.modelParams["bkgd_model","N_knots"], N_t), "ERROR: invalid shape for covariance matrix C12! %s" % str(np.shape(C12))

        # Do the same for C22
        G22 = np.meshgrid(tt, tt)
        dt22 = np.abs(G22[1]-G22[0]).T
        C22 = np.asmatrix(self.kernel(dt22))

        assert np.shape(C22) == (N_t,N_t), "ERROR: invalid shape for covariance matrix C12! %s" % str(np.shape(C22))

        # Sample from the conditional distribution for each process K
        lam_tt = np.empty((K,N_t))
        for k in np.arange(K):
            # Get current lam_knots. Remember is centered, mean 0
            lam_knots = np.reshape(self.modelParams["bkgd_model","lam_knots"][k,:], (self.modelParams["bkgd_model","N_knots"],1))

            # Compute the parameters of the conditional normal distribution
            # Incorporate mu into the conditional mean
            mu_tt = np.log(self.modelParams["bkgd_model","lam_mu_mu"][k]) + C12.T*invC11*lam_knots

            C_tt = C22 - C12.T*invC11*C12
            # Add a small amount of diagonal noise to make covariance matrix positive definite
            C_tt += 10e-6*np.eye(N_t)

            # Sample from the conditional. C_tt = L_tt * L_tt.T
            L_tt = scipy.linalg.cholesky(C_tt, lower=True)

            # Sample multivariate normal by rotating vector of standard normals
            Z = np.asmatrix(np.random.randn(N_t,1))
            lam_tt[k,:] = np.exp(mu_tt + L_tt*Z).T


        return lam_tt

class GlmBackgroundModel(ModelExtension):
    def __init__(self, baseModel, configFile):
        self.base = baseModel

        # Initialize databases for this extension
        self.modelParams = baseModel.modelParams
        self.modelParams.addDatabase("bkgd_model")
        self.gpuPtrs = baseModel.gpuPtrs
        self.gpuPtrs.addDatabase("bkgd_model")

        self.parseConfigurationFile(configFile)
        pprintDict(self.params, "Background Model Params")

#        self.calculateNumKnots()
        self.initializeGpuKernels()

        self.iter = 0

    def parseConfigurationFile(self, configFile):
        """
        Parse the configuration file to get base model parameters
        """
        # Initialize defaults
        defaultParams = {}

        # CUDA kernels are defined externally in a .cu file
        defaultParams["cu_dir"]  = os.path.join("pyhawkes", "cuda", "cpp")
        defaultParams["cu_file"] = "bkgd_model.cu"
        defaultParams["thin"] = 1

        # Create a config parser object and read in the file
        cfgParser = ConfigParser(defaultParams)
        cfgParser.read(configFile)

        self.params = {}
        self.params["cu_dir"]  = cfgParser.get("bkgd_model", "cu_dir")
        self.params["cu_file"] = cfgParser.get("bkgd_model", "cu_file")
        self.params["blockSz"] = cfgParser.getint("cuda", "blockSz")

        # Load stimulus vars
        self.params["stim_file"] = cfgParser.get("bkgd_model", "stim_file")
        self.params["thin"] = cfgParser.getint("bkgd_model", "thin")

        # Get prior params
        self.params["sigma"] = cfgParser.getfloat("bkgd_model", "sigma")

    def initializeGpuKernels(self):
        kernelSrc = os.path.join(self.params["cu_dir"], self.params["cu_file"])

        kernelNames = ["computeLamOffsetAndFrac",
                       "computeLambdaGPPerSpike",
                       "computeLamLogLkhd"]

        src_consts = {"B" : self.params["blockSz"]}

        self.gpuKernels = compileKernels(kernelSrc, kernelNames, src_consts)

    def initializeGpuMemory(self):
        K = self.modelParams["proc_id_model","K"]
        N = self.base.data.N

        self.gpuPtrs["bkgd_model","lam"]   = gpuarray.empty((K,N), dtype=np.float32)
        self.gpuPtrs["bkgd_model","lam_knots"] = gpuarray.empty((K,
                                                                 self.modelParams["bkgd_model","N_frames"]),
                                                                dtype=np.float32)

        # Each spike needs to know its offset in lam and the fractional distance between
        # the preceding and following lam grid poiN_knotss
        self.gpuPtrs["bkgd_model","lam_offset"] = gpuarray.empty((N,), dtype=np.int32)
        self.gpuPtrs["bkgd_model","lam_frac"]   = gpuarray.empty((N,), dtype=np.float32)


        # Compute each spike's bin background rate parameters
        # Temporarily copy S and C to the GPU
        # The bin is determined by the spike's offset from Tstart
        S_gpu = gpuarray.to_gpu(self.base.data.S.astype(np.float32) - self.base.data.Tstart)

        grid_w = int(np.ceil(float(N)/self.params["blockSz"]))
        self.gpuKernels["computeLamOffsetAndFrac"](np.float32(self.modelParams["bkgd_model", "dt"]),
                                                   np.int32(N),
                                                   np.float32(0),
                                                   S_gpu.gpudata,
                                                   self.gpuPtrs["bkgd_model","lam_offset"].gpudata,
                                                   self.gpuPtrs["bkgd_model","lam_frac"].gpudata,
                                                   block=(1024,1,1),
                                                   grid=(grid_w,1)
                                                   )

    def load_stim(self, stim_file,(T_start,T_stop)):
        """
        Load the stimulus mat file
        """
        mat_data = scipy.io.loadmat(stim_file)
        mat_stim = mat_data["stim"]

        # Reshape the stimulus to be 2D, where the 2nd index is the frame number
        N_frames = mat_stim.shape[-1]
        mat_stim = np.reshape(mat_stim, (-1,N_frames))
        self.modelParams["bkgd_model", "stim"] = mat_stim
        self.modelParams["bkgd_model", "stim_shape"] = mat_stim.shape

        # Compute frame rate
        if "dt" in mat_data:
            dt = float(mat_data["dt"])
        elif "fr" in mat_data:
            # If frame rate specified, use dt as 1/fr
            dt = float(1.0/mat_data["fr"])
        else:
            raise Exception("Either dt or fr must be specified in stimulus file")

        self.modelParams["bkgd_model", "dt"] = dt

        # Compute time at each stimulus frame
        self.modelParams["bkgd_model", "stim_t"] = np.arange(0,N_frames) * dt

        # Only keep the desired range
        start = int(np.floor(T_start/dt))
        stop = int(np.ceil(T_stop/dt))+1
        log.info("Using stimulus frames in range [%d,%d)", start, stop)

        self.modelParams["bkgd_model", "stim"] = self.modelParams["bkgd_model", "stim"][...,start:stop]
        self.modelParams["bkgd_model", "stim_t"] = self.modelParams["bkgd_model", "stim_t"][start:stop]

        # Initialize background rate params
        self.modelParams["bkgd_model", "N_features"] = self.modelParams["bkgd_model", "stim"].shape[0]
        self.modelParams["bkgd_model", "N_frames"] = self.modelParams["bkgd_model", "stim"].shape[1]

        # Augment data with a "bias" feature
        self.modelParams["bkgd_model", "stim"] = np.concatenate((self.modelParams["bkgd_model", "stim"],
                                                                 np.ones((1,self.modelParams["bkgd_model", "N_frames"]))),
                                                                axis=0)
        self.modelParams["bkgd_model", "N_features"] += 1

        log.info("Stimulus has %d features and %d frames",
                 self.modelParams["bkgd_model", "N_features"],
                 self.modelParams["bkgd_model", "N_frames"] )

    def initializeModelParamsFromPrior(self):
        """
        The GP has an exponential kernel decaying in time with precision lam_tau and
        with overall variance scaled by lam_sigma
        """
        K = self.modelParams["proc_id_model","K"]
        self.load_stim(self.params["stim_file"], (self.base.data.Tstart,self.base.data.Tstop))

        self.initializeGpuMemory()

        # Initialize parameters
        self.modelParams["bkgd_model","beta"] = np.zeros((K,self.modelParams["bkgd_model", "N_features"]))

        # TODO: Initialize with a fit from a GLM
        try:
            log.info("Initializing GLM backround model with parameters from a Poisson GLM")
            import statsmodels.api as sm

            # Bin the spike train
            bin_edges = self.modelParams['bkgd_model','stim_t']
            N_bins = bin_edges.shape[0]-1
            binned_spikes = np.zeros((K,N_bins))

            for k in np.arange(K):
                (binned_spikes[k,:],_) = np.histogram(self.base.data.S[self.base.data.C==k], bin_edges)

                assert np.sum(binned_spikes[k,:]) == np.count_nonzero(self.base.data.C==k)

                poiss_glm = sm.GLM(np.reshape(binned_spikes[k,:], (N_bins,1)),
                                   self.modelParams['bkgd_model','stim'][:,:N_bins].T,
                                   family=sm.families.Poisson())
                poiss_fit = poiss_glm.fit()
                self.modelParams['bkgd_model', 'beta'][k,:] = poiss_fit.params
        except Exception as e:
            log.warning("Failed to initialize with GLM")
            log.warning(e.message)

        # Calculate the background rate at each spike
        grid_sz = int(np.ceil(float(self.base.data.N)/1024))
        for k in np.arange(K):
            self.gpuKernels["computeLambdaGPPerSpike"](np.int32(k),
                                                       np.int32(self.base.data.N),
                                                       np.int32(self.modelParams["bkgd_model","N_frames"]),
                                                       self.gpuPtrs["bkgd_model","lam_knots"].gpudata,
                                                       self.gpuPtrs["bkgd_model","lam_offset"].gpudata,
                                                       self.gpuPtrs["bkgd_model","lam_frac"].gpudata,
                                                       self.gpuPtrs["bkgd_model","lam"].gpudata,
                                                       block=(1024,1,1),
                                                       grid=(grid_sz,1)
                                                       )
        self.iter = 0


    def initializeModelParamsFromDict(self, paramsDB):
        """
        In the prediction tests we need to take into account the value of the
        background rate at previous knots as learned during training. The predicted
        rate will be affected by the inferred rate at these points.
        """
        K = self.modelParams["proc_id_model","K"]
        self.initializeGpuMemory()

        self.modelParams["bkgd_model","beta"] = paramsDB["bkgd_model","beta"]

        # Calculate the background rate at each spike
        grid_sz = int(np.ceil(float(self.base.data.N)/1024))
        for k in np.arange(K):
            beta = np.reshape(self.modelParams["bkgd_model","beta"][k,:],
                              (1,self.modelParams["bkgd_model","N_features"]))
            lam = np.dot(beta,self.modelParams["bkgd_model","stim"])

            # TODO: Fix up the hacky offset calculation (4 bytes per float32)
            dest = int(self.gpuPtrs["bkgd_model","lam_knots"].ptr + k*self.modelParams["bkgd_model","N_frames"]*4)
    #        cuda.memcpy_htod(dest, np.array(lam+self.modelParams["bkgd_model","lam_mu"][k,:], dtype=np.float32))
            cuda.memcpy_htod(dest, np.array(lam, dtype=np.float32))

            self.gpuKernels["computeLambdaGPPerSpike"](np.int32(k),
                                                       np.int32(self.base.data.N),
                                                       np.int32(self.modelParams["bkgd_model","N_frames"]),
                                                       self.gpuPtrs["bkgd_model","lam_knots"].gpudata,
                                                       self.gpuPtrs["bkgd_model","lam_offset"].gpudata,
                                                       self.gpuPtrs["bkgd_model","lam_frac"].gpudata,
                                                       self.gpuPtrs["bkgd_model","lam"].gpudata,
                                                       block=(1024,1,1),
                                                       grid=(grid_sz,1)
                                                       )
        self.iter = 0


    def computeBetaLogLikelihood(self, beta, k):
        """
        Compute the log likelihood of a given set of features. Leverage the GPU
        to calculate per-spike contributions.
        """

        # Copy the proposed lambda to the GPU
        # The prior is over zero-mean gaussian processes, but the actual value is
        # centered at the offset specified in the parameters
        # The parameter is specified in rate space (i.e. exponential of the gp))
        # so we first convert it to gp space by taking the log
        beta = np.reshape(beta,(1,self.modelParams["bkgd_model","N_features"]))
        lam = np.dot(beta,self.modelParams["bkgd_model","stim"])

        # TODO: Fix up the hacky offset calculation (4 bytes per float32)
        dest = int(self.gpuPtrs["bkgd_model","lam_knots"].ptr + k*self.modelParams["bkgd_model","N_frames"]*4)
#        cuda.memcpy_htod(dest, np.array(lam+self.modelParams["bkgd_model","lam_mu"][k,:], dtype=np.float32))
        cuda.memcpy_htod(dest, np.array(lam, dtype=np.float32))

        # Calculate the background rate at each spike
        grid_sz = int(np.ceil(float(self.base.data.N)/1024))
        self.gpuKernels["computeLambdaGPPerSpike"](np.int32(k),
                                                   np.int32(self.base.data.N),
                                                   np.int32(self.modelParams["bkgd_model","N_frames"]),
                                                   self.gpuPtrs["bkgd_model","lam_knots"].gpudata,
                                                   self.gpuPtrs["bkgd_model","lam_offset"].gpudata,
                                                   self.gpuPtrs["bkgd_model","lam_frac"].gpudata,
                                                   self.gpuPtrs["bkgd_model","lam"].gpudata,
                                                   block=(1024,1,1),
                                                   grid=(grid_sz,1)
                                                   )



        # Calculate the log likelihood of the proposed background rate
        ll_gpu = np.array([0.0], dtype=np.float32)
        self.gpuKernels["computeLamLogLkhd"](np.int32(k),
                                             np.int32(self.base.data.N),
                                             self.gpuPtrs["bkgd_model","lam_knots"].gpudata,
                                             np.float32(self.modelParams["bkgd_model","dt"]),
                                             np.int32(self.modelParams["bkgd_model","N_frames"]),
                                             self.gpuPtrs["parent_model","Z"].gpudata,
                                             self.gpuPtrs["proc_id_model","C"].gpudata,
                                             self.gpuPtrs["bkgd_model","lam"].gpudata,
                                             cuda.Out(ll_gpu),
                                             block=(1024,1,1),
                                             grid=(1,1)
                                             )

        ll = ll_gpu[0]
        if np.isnan(ll):
            import pdb
            pdb.set_trace()
            log.info(ll)
            log.info(lam)
            log.info(np.min(lam))
            log.info(np.max(lam))
            raise Exception("Log likelihood is not finite for process %d!", k)
        return ll

    def integrateBkgdRates(self):
        """
        Integrate the background rates. For piecewise linear function this is
        the trapezoidal quadrature of the rate
        """
        lam = np.exp(self.gpuPtrs["bkgd_model","lam_knots"].get())
        trap_coeff = self.modelParams["bkgd_model","dt"] * np.ones((self.modelParams["bkgd_model","N_frames"],1))
        trap_coeff[0] *= 0.5
        trap_coeff[-1] *= 0.5

        # Matrix vector multiply to get trapezoidal integration of each bkgd rate
        # (K x N_knots) * (N_knots x 1) = (K x 1)
        return np.dot(lam, trap_coeff)

    def evaluateBkgdRate(self, t):
        """
        Evaluate the background rate at the specified time points
        """
        K = self.modelParams["proc_id_model","K"]
        N_t = len(t)
        knots = self.modelParams["bkgd_model","stim_t"]
        lam = np.exp(self.gpuPtrs["bkgd_model","lam_knots"].get())

        lam_interp = np.zeros((K,N_t))
        for k in np.arange(K):
            lam_interp[k,:] = np.interp(t,knots,lam[k,:])

        return lam_interp

    def cumIntegrateBkgdRates(self, t):
        """
        Return the cumulative integral evaluated at the specified times.
        The GP background rate is linearly interpolated between knots, so we
        can evaluate the cumulative integral at  the knots and interpolate at t.
        """
        K = self.modelParams["proc_id_model","K"]
        N_knots = self.modelParams["bkgd_model","N_knots"]
        N_t = len(t)

        knots = self.modelParams["bkgd_model", "stim_t"]
        lam = np.exp(self.gpuPtrs["bkgd_model","lam_knots"].get())
        trap_coeff = self.params["lam_dt"] * np.ones((N_knots,1))
        trap_coeff[0] *= 0.5
        trap_coeff[-1] *= 0.5

        cumIntegral = np.zeros((K,N_t))

        for k in np.arange(K):
            lam_int = np.cumsum(trap_coeff*lam[k,:])
            cumIntegral[k,:] = np.interp(t,knots,lam_int)

        return cumIntegral

    def computeLogProbability(self):
        """
        Compute the log probability of the background model params given the prior
        For GP backgrounod rates this is a Gaussian probability. We drop the terms
        that do not depend on the background rate x
        """
        # The log prob is 1/2*x^T*inv(L*L.T)*x which requires the inverse of the
        # covariance matrix for computing likelihood.
        ll = 0.0

        return ll

    def sampleModelParameters(self):
        """
        Sample new background rates lambda using elliptical slice sampling
        The idea is that the number of points at which lambda is evaluated
        will be much smaller than the number of spikes. We leverage the GPU
        to calculate the per-spike contribution to lambda's likelihood, and
        we compute the trapezoidal integration on the host.
        """
        if np.mod(self.iter, self.params["thin"]) == 0:
            # Update each background rate in turn
            for k in np.arange(self.modelParams["proc_id_model","K"]):
                # Create a single param function to compute the log likelihood of a
                # given background rate
                ll_fn = lambda beta,args: self.computeBetaLogLikelihood(beta, k)
                beta_cur = self.modelParams["bkgd_model","beta"][k,:]

                try:
                    (beta_new, ll_new) = elliptical_slice(beta_cur,
                                                          self.params["sigma"]*np.eye(self.modelParams["bkgd_model","N_features"]),
                                                          ll_fn)
                except Exception as e:
                    log.error("Exception in Elliptical Slice sampling!")
                    log.error(e.message)
                    raise e

                # Update background rate on host and gpu
                self.modelParams["bkgd_model","beta"][k,:] = beta_new

        self.iter += 1

    def sampleLatentVariables(self):
        pass


    def registerStatManager(self, statManager):
        """
        Register callbacks with the given StatManager
        """
        statManager.registerSampleCallback("beta",
                                           lambda: self.modelParams["bkgd_model","beta"],
                                           (self.modelParams["proc_id_model","K"],
                                            self.modelParams["bkgd_model","N_features"]),
                                           np.float32)

        statManager.registerSampleCallback("lam_knots",
                                           lambda: np.exp(self.gpuPtrs["bkgd_model","lam_knots"].get()),
                                           (self.modelParams["proc_id_model","K"],
                                            self.modelParams["bkgd_model","N_frames"]),
                                            np.float32)


class SharedGpBkgdModel(ModelExtension):
    """
    A shared period background rate representing common fluctuations in rate.
    This is shifted up and down by process-specific mean rates.
    """
    def __init__(self, baseModel, configFile):
        self.base = baseModel
        self.data = self.base.data

        # Initialize databases for this extension
        self.modelParams = baseModel.modelParams
        self.modelParams.addDatabase("bkgd_model")
        self.gpuPtrs = baseModel.gpuPtrs
        self.gpuPtrs.addDatabase("bkgd_model")

        self.parseConfigurationFile(configFile)
        pprintDict(self.params, "Background Model Params")

#        self.calculateNumKnots()
        self.initializeGpuKernels()

        self.iter = 0

    def parseConfigurationFile(self, configFile):
        """
        Parse the configuration file to get base model parameters
        """
        # Initialize defaults
        defaultParams = {}

        # CUDA kernels are defined externally in a .cu file
        defaultParams["cu_dir"]  = os.path.join("pyhawkes", "cuda", "cpp")
        defaultParams["cu_file"] = "bkgd_model.cu"
        defaultParams["thin"] = 1

        # Create a config parser object and read in the file
        cfgParser = ConfigParser(defaultParams)
        cfgParser.read(configFile)

        self.params = {}
        self.params["cu_dir"]  = cfgParser.get("bkgd_model", "cu_dir")
        self.params["cu_file"] = cfgParser.get("bkgd_model", "cu_file")

        self.params["kernel_type"] = cfgParser.get("bkgd_model", "kernel")
        self.params["lam_dt"] = cfgParser.getfloat("bkgd_model", "dt")
        self.params["lam_tau"] = cfgParser.getfloat("bkgd_model", "tau")
        self.params["lam_sigma"] = cfgParser.getfloat("bkgd_model", "sigma")
        self.params["lam_T"] = cfgParser.getfloat("bkgd_model", "T")
        self.params["lam_ell"] = cfgParser.getfloat("bkgd_model", "ell")
        self.params["mu_mu"] = np.log(cfgParser.getfloat("bkgd_model", "mu_mu"))
        self.params["sig_mu"] = cfgParser.getfloat("bkgd_model", "sig_mu")
        self.params["thin"] = cfgParser.getint("bkgd_model", "thin")

        self.params["use_tod"] = bool(cfgParser.getint("bkgd_model", "use_tod"))

        self.params["blockSz"] = cfgParser.getint("cuda", "blockSz")
        self.params["max_hist"]     = cfgParser.getint("preprocessing", "max_hist")

    def calculateNumKnots(self):
        """
        Initialize the 'knots' where the background rate will be evaluated
        """
        # Evaluate the kernel at evenly spaced "knots" in the interval [0,T]
        N_knots = max(2,np.floor((self.data.Tstop-self.data.Tstart)/self.params["lam_dt"])+1)
        N_knots = int(N_knots)
        self.modelParams["bkgd_model","N_knots"] = N_knots
        knots = np.linspace(self.data.Tstart,self.data.Tstop, N_knots)
        self.modelParams["bkgd_model","knots"] = knots

        # The true spacing is not exactly lam_dt
        self.params["lam_dt"] = float(self.data.Tstop-self.data.Tstart)/(N_knots-1)

        log.debug("Approximating background rate with %d knots.", N_knots)

    def initializeGpuKernels(self):
        kernelSrc = os.path.join(self.params["cu_dir"], self.params["cu_file"])

        kernelNames = ["computeLamOffsetAndFrac",
                       "computeLambdaGPPerSpike",
                       "computeLamLogLkhd",
                       "computeTimeOfDayPr"]

        src_consts = {"B" : self.params["blockSz"]}

        self.gpuKernels = compileKernels(kernelSrc, kernelNames, src_consts)

    def initializeGpuMemory(self):
        K = self.modelParams["proc_id_model","K"]
        N = self.data.N
        N_knots = self.modelParams["bkgd_model","N_knots"]

        self.gpuPtrs["bkgd_model","lam"]   = gpuarray.empty((K,N), dtype=np.float32)
        self.gpuPtrs["bkgd_model","lam_knots"]   = gpuarray.empty((K,N_knots), dtype=np.float32)

        # Each spike needs to know its offset in lam and the fractional distance between
        # the preceding and following lam grid poiN_knotss
        self.gpuPtrs["bkgd_model","lam_offset"] = gpuarray.empty((N,), dtype=np.int32)
        self.gpuPtrs["bkgd_model","lam_frac"]   = gpuarray.empty((N,), dtype=np.float32)


        # Compute each spike's bin background rate parameters
        # Temporarily copy S and C to the GPU
        # The bin is determined by the spike's offset from Tstart
        S_gpu = gpuarray.to_gpu(self.data.S.astype(np.float32) - self.data.Tstart)

        grid_w = int(np.ceil(float(N)/self.params["blockSz"]))
        self.gpuKernels["computeLamOffsetAndFrac"](np.float32(self.params["lam_dt"]),
                                                   np.int32(N),
                                                   np.float32(0),
                                                   S_gpu.gpudata,
                                                   self.gpuPtrs["bkgd_model","lam_offset"].gpudata,
                                                   self.gpuPtrs["bkgd_model","lam_frac"].gpudata,
                                                   block=(1024,1,1),
                                                   grid=(grid_w,1)
                                                   )

        # Save the time of day of each spike on the GPU
        self.gpuPtrs["bkgd_model","tod"] = gpuarray.to_gpu(np.floor(24*np.mod(self.data.S,1.0)).astype(np.int32))

    def initialize_pr_time_of_day(self):
        """
        Compute the empirical time of day probability which
        multiplies the instantaneous background rates, but does not
        change the overall integral of the rate since it is normalized.
        """
        if self.params["use_tod"]:
            tod = np.linspace(0,1,25)
            (pr_tod,_) = np.histogram(np.mod(self.data.S,1), tod, density=True)
            if not np.allclose(np.sum(pr_tod),24.0):
                log.info("Sum pr_tod: %f", np.sum(pr_tod))
                raise Exception("ERROR: time of day prob not properly normalized")
        else:
            pr_tod = np.ones((24,))

        self.modelParams["bkgd_model","pr_tod"] = pr_tod
        self.gpuPtrs["bkgd_model","pr_tod"] = gpuarray.to_gpu(pr_tod.astype(np.float32))

    def initialize_gp_cov_kernel_fn(self, type):
        """
        Initialize the covariance kernel of the Gaussian process
        """
        kernel = None
        if type == "ou":
            kernel = lambda dt: np.exp(-dt/self.params["lam_tau"]) * self.params["lam_sigma"]
        elif type == "rbf":
            kernel = lambda dt: np.exp(-dt**2/self.params["lam_tau"]**2/2) * self.params["lam_sigma"]
        elif type == "periodic":
            kernel = lambda dt: np.exp(-2*np.sin(np.pi*dt/self.params["lam_T"])**2/self.params["lam_ell"]**2) * self.params["lam_sigma"]
        else:
            raise Exception("Unsupported kernel type: %s" % self.params["kernel_type"])

        return kernel

    def initialize_gp_mean_and_cov(self, kernel, knots, mu0, prev_knots=None, prev_vals=None):
        """
        Intialize the covariance matrices for the mean background rate
        """
        N_knots = self.modelParams["bkgd_model","N_knots"]
        K = self.modelParams["proc_id_model","K"]

        # Since the knots are evenly spaced, the covariance matrix is Toeplitz
        G22 = np.meshgrid(knots, knots)
        dt22 = np.abs(G22[1]-G22[0])
        C = np.asmatrix(kernel(dt22))

        if prev_knots is None or prev_vals is None:
            # If no previous info is given, we do not need to add a
            # correction to the covariance matrix
            # Since the knots are evenly spaced, the covariance matrix is Toeplitz
            C_corr = 0
            mu_corr = 0
        else:
            # The paramsDB contains the background rate at a set of previous knots
            # Calculate the covariance matrix and its inverse at the previous knots
            N_prev_knots = len(prev_knots)
            prev_vals = np.asmatrix(np.reshape(prev_vals,(N_prev_knots,1)))

            G11 = np.meshgrid(prev_knots, prev_knots)
            dt11 = np.abs(G11[1]-G11[0])
            C11 = np.asmatrix(kernel(dt11))
            C11 += np.diag(1e-6*np.random.randn(N_prev_knots))
            try:
                L11 = scipy.linalg.cholesky(C11, lower=True)
            except Exception as e:
                # Add noise if we were not PSD
                C11 += np.diag(1e-3*np.ones(N_prev_knots))
                L11 = scipy.linalg.cholesky(C11, lower=True)

            invL11 = scipy.linalg.inv(L11)
            invL11 = np.asmatrix(invL11)
            invC11 = invL11.T*invL11

            # To compute C12 first create a meshgrid of knots and tt of shape (N_knots, length(tt))
            G12 = np.meshgrid(prev_knots, knots)
            # [i,j]th entry is prev_knots[i] - knots[j]
            dt12 = np.abs(G12[1]-G12[0])
            C12 = np.asmatrix(kernel(dt12))

            # TODO: Currently using numpy 1.6.1. The next version (1.7) has a keyword for "ij" indexing
            # which would return the array of the desired shape
            C12 = C12.T
            assert np.shape(C12) == (N_prev_knots, N_knots), "ERROR: invalid shape for covariance matrix C12! %s. Expected (%d,%d)" % (str(np.shape(C12)),N_prev_knots, self.modelParams["bkgd_model","N_knots"])

            C_corr = -C12.T*invC11*C12
            mu_corr = C12.T*invC11*(prev_vals-mu0)


        # Compute the final GP covariance
        C += C_corr
        C += np.diag(1e-6*np.random.randn(N_knots))
        C += np.diag(1e-3*np.ones(N_knots))
        L = scipy.linalg.cholesky(C, lower=True)

        # Compute the final mean
        mu = mu0 + mu_corr

        return (mu, C, L)

    def initializeModelParamsFromPrior(self):
        """
        The GP has an exponential kernel decaying in time with precision lam_tau and
        with overall variance scaled by lam_sigma
        """
        self.calculateNumKnots()
        self.initializeGpuMemory()

        K = self.modelParams["proc_id_model","K"]
        N_knots = self.modelParams["bkgd_model","N_knots"]
        knots = self.modelParams["bkgd_model","knots"]

        # Initialize the time of day inhomogeneity
        self.initialize_pr_time_of_day()

        # Initialize process-specific mean levels
        self.modelParams["bkgd_model","lam_mu"] = np.random.normal(self.params["mu_mu"],
                                                                    self.params["sig_mu"],
                                                                    size=(K,))


        # Initialize a Gaussian process for the seasonal variation
        seasonal_kernel = self.initialize_gp_cov_kernel_fn("periodic")
        (seas_mu, seas_C, seas_L) = self.initialize_gp_mean_and_cov(seasonal_kernel, knots, 0)
        self.modelParams["bkgd_model","seas_L"] = seas_L
        seas_L_inv = scipy.linalg.inv(seas_L)
        self.modelParams["bkgd_model","inv_seas_C"] = np.dot(seas_L_inv.T, seas_L_inv)

        # Sample randomly from the Gaussian process prioirs
        self.modelParams["bkgd_model","lam_shared"] = seas_mu + np.dot(seas_L, np.random.randn(N_knots)).T

        # Add the background rate with process specific means
        for k in np.arange(K):
            dest = int(self.gpuPtrs["bkgd_model","lam_knots"].ptr + k*N_knots*4)
            cuda.memcpy_htod(dest,
                             np.array(self.modelParams["bkgd_model","lam_shared"]+
                                      self.modelParams["bkgd_model","lam_mu"][k],
                                      dtype=np.float32)
                            )

            # Initialize the background rate at each spike
            self.compute_bkgd_rate_at_spikes(k)

        self.iter = 0


    def initializeModelParamsFromDict(self, paramsDB):
        """
        In the prediction tests we need to take into account the value of the
        background rate at previous knots as learned during training. The predicted
        rate will be affected by the inferred rate at these points.
        """
        K = self.modelParams["proc_id_model","K"]
        self.calculateNumKnots()
        self.initializeGpuMemory()

        N_knots = self.modelParams["bkgd_model","N_knots"]
        knots = self.modelParams["bkgd_model","knots"]

        # Initialize the time of day inhomogeneity
        if ("bkgd_model", "pr_tod") in paramsDB:
            self.modelParams["bkgd_model","pr_tod"] = paramsDB["bkgd_model","pr_tod"]
            self.gpuPtrs["bkgd_model","pr_tod"] = gpuarray.to_gpu(self.modelParams["bkgd_model","pr_tod"].astype(np.float32))
        else:
            self.initialize_pr_time_of_day()

        # Copy the process-specific means
        self.modelParams["bkgd_model","lam_mu"] = paramsDB["bkgd_model","lam_mu"]

        # Check if the knots have changed. If not, copy the old background
        # rates over directly. Otherwise we have to make a prediction
        if np.allclose(knots, paramsDB["bkgd_model", "knots"]):
            self.modelParams["bkgd_model","lam_shared"] = paramsDB["bkgd_model","lam_knots"]
            seasonal_kernel = self.initialize_gp_cov_kernel_fn("periodic")
            (seas_mu, seas_C, seas_L) = self.initialize_gp_mean_and_cov(seasonal_kernel,
                                                                        knots,
                                                                        0)
            self.modelParams["bkgd_model","seas_L"] = seas_L
            self.modelParams["bkgd_model","inv_seas_C"] = np.linalg.inv(seas_C)
        else:
            # The modelParams ParamsDatabase is already populated with firing rates at
            # knots from the learning phase. We want to incorporate those into a GP distribution
            # over the firing rate at the predictive knots.
            prev_knots = paramsDB["bkgd_model", "knots"]
            prev_lam_shared = paramsDB["bkgd_model", "lam_shared"]

            # Initialize a Gaussian process for the seasonal variation
            seasonal_kernel = self.initialize_gp_cov_kernel_fn("periodic")
            (seas_mu, seas_C, seas_L) = self.initialize_gp_mean_and_cov(seasonal_kernel,
                                                                        knots,
                                                                        0,
                                                                        prev_knots=prev_knots,
                                                                        prev_vals=prev_lam_shared
                                                                        )

            # Sample randomly from the Gaussian process prioirs
            seas = np.ravel(seas_mu)+np.dot(seas_L, np.random.randn(N_knots)).T
            self.modelParams["bkgd_model","lam_shared"] = seas

            # Set the covariance matrix for subsequent sampling wihtout regard to past?
            (seas_mu, seas_C, seas_L) = self.initialize_gp_mean_and_cov(seasonal_kernel,
                                                                        knots,
                                                                        0
                                                                        )

            self.modelParams["bkgd_model","seas_L"] = seas_L
            self.modelParams["bkgd_model","inv_seas_C"] = np.linalg.inv(seas_C)


        # Add the seasonal background rate with process specific rates
        for k in np.arange(K):
            dest = int(self.gpuPtrs["bkgd_model","lam_knots"].ptr + k*N_knots*4)
            cuda.memcpy_htod(dest, np.array(self.modelParams["bkgd_model","lam_shared"]+
                                            self.modelParams["bkgd_model","lam_mu"][k],
                                            dtype=np.float32))

            # Initialize the background rate at each spike
            self.compute_bkgd_rate_at_spikes(k)

        self.iter = 0

    def computeLamLogLikelihoodSingleNeuron(self, lam, k, mu=None):
        """
        Compute the log likelihood of a given background rate. Leverage the GPU
        to calculate per-spike contributions.
        """

        # Copy the proposed lambda to the GPU
        # The prior is over zero-mean gaussian processes, but the actual value is
        # centered at the offset specified in the parameters
        # The parameter is specified in rate space (i.e. exponential of the gp))
        # so we first convert it to gp space by taking the log
        K = self.modelParams["proc_id_model","K"]
        N_knots = self.modelParams["bkgd_model","N_knots"]
        ll_gpu = gpuarray.zeros(1,dtype=np.float32)

        if mu is None:
            mu = self.modelParams["bkgd_model","lam_mu"][k]

        assert np.size(lam) == N_knots
        # TODO: Fix up the hacky offset calculation (4 bytes per float32)
        dest = int(self.gpuPtrs["bkgd_model","lam_knots"].ptr + k*N_knots*4)
        cuda.memcpy_htod(dest, (lam+mu).astype(np.float32))

        # Calculate the background rate at each spike
        self.compute_bkgd_rate_at_spikes(k)

        # Calculate the log likelihood of the proposed background rate
        self.gpuKernels["computeLamLogLkhd"](np.int32(k),
                                             np.int32(self.data.N),
                                             self.gpuPtrs["bkgd_model","lam_knots"].gpudata,
                                             np.float32(self.params["lam_dt"]),
                                             np.int32(self.modelParams["bkgd_model","N_knots"]),
                                             self.gpuPtrs["parent_model","Z"].gpudata,
                                             self.gpuPtrs["proc_id_model","C"].gpudata,
                                             self.gpuPtrs["bkgd_model","lam"].gpudata,
                                             ll_gpu.gpudata,
                                             block=(1024,1,1),
                                             grid=(1,1)
                                             )

        ll = ll_gpu.get()

        if np.isnan(ll):
            log.info(ll)
            log.info(lam)
            log.info(np.min(lam))
            log.info(np.max(lam))
            log.error("Log likelihood is not finite for process %d!" % k)
#            raise Exception("Log likelihood is not finite for process %d!" % k)
            ll = -np.Inf
        return ll

    def computeLamLogLikelihood(self, lam):
        """
        Compute the log likelihood of a given background rate. Leverage the GPU
        to calculate per-spike contributions.
        """
        K = self.modelParams["proc_id_model","K"]
        ll = 0
        for k in np.arange(K):
            ll+= self.computeLamLogLikelihoodSingleNeuron(lam, k)

        return ll

    def compute_bkgd_rate_at_spikes(self, k):
        """
        Compute the background rate at the spikes
        """
        # Calculate the background rate at each spike
        grid_sz = int(np.ceil(float(self.data.N)/1024))
        self.gpuKernels["computeLambdaGPPerSpike"](np.int32(k),
                                                   np.int32(self.data.N),
                                                   np.int32(self.modelParams["bkgd_model","N_knots"]),
                                                   self.gpuPtrs["bkgd_model","lam_knots"].gpudata,
                                                   self.gpuPtrs["bkgd_model","lam_offset"].gpudata,
                                                   self.gpuPtrs["bkgd_model","lam_frac"].gpudata,
                                                   self.gpuPtrs["bkgd_model","lam"].gpudata,
                                                   block=(1024,1,1),
                                                   grid=(grid_sz,1)
                                                   )

        # Account for time of day at each spike
        self.gpuKernels["computeTimeOfDayPr"](np.int32(k),
                                               np.int32(self.data.N),
                                               self.gpuPtrs["bkgd_model","tod"].gpudata,
                                               self.gpuPtrs["bkgd_model","pr_tod"].gpudata,
                                               self.gpuPtrs["bkgd_model","lam"].gpudata,
                                               block=(1024,1,1),
                                               grid=(grid_sz,1)
                                               )

    def integrateBkgdRates(self):
        """
        Integrate the background rates. For piecewise linear function this is
        the trapezoidal quadrature of the rate
        """
        lam = np.exp(self.gpuPtrs["bkgd_model","lam_knots"].get())
        trap_coeff = self.params["lam_dt"] * np.ones((self.modelParams["bkgd_model","N_knots"],1))
        trap_coeff[0] *= 0.5
        trap_coeff[-1] *= 0.5

        # Matrix vector multiply to get trapezoidal integration of each bkgd rate
        # (K x N_knots) * (N_knots x 1) = (K x 1)
        return np.dot(lam, trap_coeff)

    def evaluateBkgdRate(self, t):
        """
        Evaluate the background rate at the specified time points
        """
        K = self.modelParams["proc_id_model","K"]
        N_t = len(t)
        knots = self.modelParams["bkgd_model","knots"]
        lam = np.exp(self.gpuPtrs["bkgd_model","lam_knots"].get())

        lam_interp = np.zeros((K,N_t))
        for k in np.arange(K):
            lam_interp[k,:] = np.interp(t,knots,lam[k,:])

        return lam_interp

    def cumIntegrateBkgdRates(self, t):
        """
        Return the cumulative integral evaluated at the specified times.
        The GP background rate is linearly interpolated between knots, so we
        can evaluate the cumulative integral at  the knots and interpolate at t.
        """
        K = self.modelParams["proc_id_model","K"]
        N_knots = self.modelParams["bkgd_model","N_knots"]
        N_t = len(t)

        knots = self.modelParams["bkgd_model","knots"]
        lam = np.exp(self.gpuPtrs["bkgd_model","lam_knots"].get())
        trap_coeff = self.params["lam_dt"] * np.ones((N_knots,1))
        trap_coeff[0] *= 0.5
        trap_coeff[-1] *= 0.5

        cumIntegral = np.zeros((K,N_t))

        for k in np.arange(K):
            lam_int = np.cumsum(trap_coeff*lam[k,:])
            cumIntegral[k,:] = np.interp(t,knots,lam_int)

        return cumIntegral

    def computeLogProbability(self):
        """
        Compute the log probability of the background model params given the prior
        For GP backgrounod rates this is a Gaussian probability. We drop the terms
        that do not depend on the background rate x
        """
        K = self.modelParams["proc_id_model","K"]
        N_knots = self.modelParams["bkgd_model","N_knots"]
        # The log prob is 1/2*x^T*inv(L*L.T)*x which requires the inverse of the
        # covariance matrix for computing likelihood.
        ll = 0.0

        seas = np.reshape(self.modelParams["bkgd_model","lam_shared"], (N_knots,1))
        ll += -0.5 * np.dot(np.dot(seas.T, self.modelParams["bkgd_model","inv_seas_C"]), seas)

        for k in np.arange(K):
            ll += -0.5/self.params['sig_mu']**2 * (self.modelParams["bkgd_model","lam_mu"] - self.params["mu_mu"])**2

        return ll

    def sample_mean_bkgd_rates(self):
        """
        Sample process specific background rates
        """
        K = self.modelParams["proc_id_model","K"]
        N_knots = self.modelParams["bkgd_model","N_knots"]

        # Sample the mean of the GP means
        lam = self.modelParams["bkgd_model","lam_shared"]
        for k in np.arange(K):
            ll_fn = lambda mu,args: self.computeLamLogLikelihoodSingleNeuron(lam, k, mu=mu)
            mu_cur = self.modelParams["bkgd_model","lam_mu"][k]

            try:
                (mu_new, ll_new) = elliptical_slice(mu_cur,
                                                    self.params["sig_mu"],
                                                    ll_fn,
                                                    mu=self.params["mu_mu"])
            except Exception as e:
                log.error("Exception in Elliptical Slice sampling! (mu)")
                log.error(e.message)
                exit()

            # Update mean vector
            self.modelParams["bkgd_model","lam_mu"][k] = mu_new

    def sample_shared_bkgd_rate(self):
        """
        Sample the seasonal background rate
        """
        ll_fn = lambda lam,args: self.computeLamLogLikelihood(lam)
        lam_cur = self.modelParams["bkgd_model","lam_shared"]

        try:
            (lam_new, ll_new) = elliptical_slice(lam_cur,
                                                 self.modelParams["bkgd_model","seas_L"],
                                                 ll_fn)
        except Exception as e:
            log.error(e.message)
            raise Exception("Exception in Elliptical Slice sampling!")

        # Update background rate on host
        # The GPU is already updated by calls to computeLamLogLkhd
        self.modelParams["bkgd_model","lam_shared"] = lam_new

    def sampleModelParameters(self):
        """
        Sample the homogeneous background rates
        """
        pass

    def sampleLatentVariables(self):
        """
        Sample new background rates lambda using elliptical slice sampling
        The idea is that the number of points at which lambda is evaluated
        will be much smaller than the number of spikes. We leverage the GPU
        to calculate the per-spike contribution to lambda's likelihood, and
        we compute the trapezoidal integration on the host.
        """
        if np.mod(self.iter, self.params["thin"]) == 0:
            self.sample_shared_bkgd_rate()
            self.sample_mean_bkgd_rates()

        self.iter += 1

    def registerStatManager(self, statManager):
        """
        Register callbacks with the given StatManager
        """
        statManager.setSingleSample("lam_knots", self.modelParams["bkgd_model","knots"])
        statManager.setSingleSample("pr_tod", self.modelParams["bkgd_model","pr_tod"])
        statManager.registerSampleCallback("lam",
                                           lambda: np.exp(self.gpuPtrs["bkgd_model","lam_knots"].get()),
                                           (self.modelParams["proc_id_model","K"],self.modelParams["bkgd_model","N_knots"]),
                                           np.float32)

        statManager.registerSampleCallback("lam_shared",
                                           lambda: self.modelParams["bkgd_model","lam_shared"],
                                           (self.modelParams["bkgd_model","N_knots"],),
                                           np.float32)

        statManager.registerSampleCallback("lam_mu",
                                           lambda: self.modelParams["bkgd_model","lam_mu"],
                                           (self.modelParams["proc_id_model","K"],),
                                           np.float32)


    def generateBkgdRate(self, tt):
        """
        Predict the background rate at the specified times. For a GP rate this
        is a draw from the conditional distribution, which is normal. The joint
        distribution of the GP over the observed and unobserved times is given by

        lam[knots, tt] ~ N(mu, [[C11, C12], [C12.T, C22]])

        where C11 is the kernel evaluated at deltas between each pair of knots
              C12 is the kernel evaluated at deltas between each pair of knot and tt
              C22 is the kernel evaluated at deltas between each pair in tt

        The conditional distribution of tt given knots is then
        lam[tt | knots] ~ N(mu + C12.T * inv(C11) * (knots-mu), C22 - C12.T * inv(C11) * C12)
        """
        K = self.modelParams["proc_id_model","K"]
        N_t = len(tt)

        # Initialize a Gaussian process for the seasonal activity
        seasonal_kernel = self.initialize_gp_cov_kernel_fn("periodic")
        (seas_mu, seas_C, seas_L) = self.initialize_gp_mean_and_cov(seasonal_kernel,
                                                                    tt,
                                                                    0,
                                                                    prev_knots=self.modelParams["bkgd_model","knots"],
                                                                    prev_vals=self.modelParams["bkgd_model","lam_shared"]
                                                                    )

        # Sample randomly from the Gaussian process prioirs
        lam_seas = seas_mu+np.dot(seas_L, np.random.randn(N_t)).T

        lam_mu = self.modelParams["bkgd_model","lam_mu"]

        return np.tile(np.reshape(lam_seas,[1,N_t]),(K,1)) + lam_mu
