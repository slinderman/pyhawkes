import numpy as np

from ConfigParser import ConfigParser

from model_extension import ModelExtension
from pyhawkes.utils.utils import *
from pyhawkes.utils.elliptical_slice import *

import scipy.io


log = logging.getLogger("global_log")


class LocationModel(ModelExtension):
    """
    A (parametric) cluster model with a fixed number of clusters.
    Calls into child nodes (e.g. weight model or background model) to
    get the likelihood of the current child variable values given a new
    cluster assignment and combines this with the prior over clusters
    to sample new clusters.
    """
    def __init__(self, baseModel, configFile, name):
        self.name = name
        
        # Store pointer to base model
        self.base = baseModel
        
        # Keep a local pointer to the data manager for ease of notation
        self.data = baseModel.data
        
        # Initialize databases for this extension
        self.modelParams = baseModel.modelParams
        self.modelParams.addDatabase(self.name)
        self.gpuPtrs = baseModel.gpuPtrs
        self.gpuPtrs.addDatabase(self.name)
        
        self.parseConfigurationFile(configFile)
        
        # Initialize callbacks
        self.callbacks = []
        
        self.iter = 0
        
    def parseConfigurationFile(self, configFile):
        """
        Parse the configuration file to get base model parameters
        """
        # Initialize defaults
        defaultParams = {}
        defaultParams["thin"] = 1
        defaultParams["delay"] = 0
        
        # Create a config parser object and read in the file
        cfgParser = ConfigParser(defaultParams)
        cfgParser.read(configFile)
        
        self.params = {}
        self.params["thin"]  = cfgParser.getint("location_model", "thin")
        self.params["delay"]  = cfgParser.getint("location_model", "delay")
        self.params["D"]  = cfgParser.getint("location_model", "D")
        
        # Mean location mu can be given as a list, in which case
        # it is interpreted as a vector mean that is shared across
        # all neurons. 
        # Alternatively, it can be a path to a .mat file with a 
        # K x D array of means for each individual neurons
        if cfgParser.has_option("location_model", "mu"):
            mu_str = cfgParser.get("location_model", "mu")
            if mu_str.endswith(".mat"):
                log.debug("Looking for array 'mu' in %s", mu_str)
                if not os.path.exists(mu_str):
                    log.error("Specified path does not exist! %s", mu_str)
                    exit()
                    
                mu_mat = scipy.io.loadmat(mu_str)
                if not "mu" in mu_mat.keys():
                    log.error("Specified mat file does not contain field 'mu'!")
                    exit()
                    
                self.params["mu"] = mu_mat["mu"]
                
                mu_shape = self.params["mu"].shape
                if not mu_shape[1] == self.params["D"]:
                    raise Exception("mu shape %s does not have correct dimensionality [*,%d]", 
                                    str(mu_shape),
                                    self.params["D"])
                
                # Important! Make sure the array is loaded in C order
                # Matlab arrays default to Fortran order
                self.params["mu"] = self.params["mu"].copy(order="C")
                
            else:
                # Parse the alpha parameter
                mu_list = mu_str.split(',')
                self.params["mu"] = np.array(map(float, alpha_list))
                if not self.params["mu"].shape == (D,):
                    log.error("mu has incorrect shape: %s", str(self.params["mu"].shape))
                    exit()
                self.params["mu"] = np.reshape(self.params["mu"], (1,D))
        else:
            log.info("mu not specified. Using default of zero")
            self.params["mu"] = np.zeros((1,self.params["D"]))
        
        
        # For now we only support spherical Gaussians with covariance sigma
        if cfgParser.has_option("location_model", "sigma"):
            self.params["sigma"] = cfgParser.getfloat("location_model","sigma")
        else:
            log.info("Using default sigma = 0.2")
            self.params["sigma"] = 0.2
            
    def getInitializationOrder(self):
        """
        The cluster model has to initialize before its consumers
        """
        return 2
    
    def initializeModelParamsFromPrior(self):
        """
        Initialize the parameters of the model.
        """
        K = self.modelParams["proc_id_model","K"]
        D = self.params["D"]
        
        if self.params["mu"].shape == (K,D):
            self.modelParams[self.name, "L_mean"] = self.params["mu"]
        elif self.params["mu"].shape == (1,D):
            self.modelParams[self.name, "L_mean"] = np.repeat(self.params["mu"], K, 0)
        else:
            log.error("Invalid shape for mu: %s", self.params["mu"].shape)
            exit()
        
        self.modelParams[self.name, "L_std"] = self.params["sigma"]*np.eye(D)
        
        # Initialize locations
        self.modelParams[self.name, "L"] = np.zeros((K,D), dtype=np.float32) 
        
#        for k in np.arange(K):
#            self.modelParams[self.name, "L"][k,:] = np.random.multivariate_normal(self.modelParams[self.name, "L_mean"][k,:],
#                                                                                  self.modelParams[self.name, "L_std"])

        self.modelParams[self.name, "L"] = self.modelParams[self.name, "L_mean"]
        
    def initializeModelParamsFromDict(self, paramsDB):
        """
        Initialize from a dictionary
        """
        # HACK!
        if ("location1","L") in paramsDB:
            self.modelParams[self.name,"L"] = paramsDB["location1","L"]
        else:
            self.initializeModelParamsFromPrior()
        
        
    def register_consumer(self, callback):
        """
        Add a consumer of the cluster assignments, Y. Whenever Y is sampled
        the callback will be called to get a likelihood of the new cluster.
        """
        self.callbacks.append(callback)
        
    def sampleModelParameters(self):
        if np.mod(self.iter,self.params["thin"])==0 and self.iter >= self.params["delay"]:
            self.sampleL()
        self.iter += 1
        
    
    def computeLogLkhdLk(self, Lk, k):
        """
        Compute the log likelihood of the new location X[k,:]
        """
        ll = 0.0
        for callback in self.callbacks:
            ll  += callback(k,Lk)
        return ll
    
    def sampleL(self):
        """
        Sample new latent locations and update the distance matriL. Since the prior
        on locations is multivariate Gaussian, we can use elliptical slice sampling.
        """
        K = self.modelParams["proc_id_model","K"]
        L = self.modelParams[self.name, "L"]
        
        L_mean = self.modelParams[self.name, "L_mean"]
        L_std = self.modelParams[self.name, "L_std"]
        
        L0 = np.copy(L)
        
        for k in np.arange(K):
            ll_fn = lambda x,args: self.computeLogLkhdLk(x, k)
            L_cur = L[k,:]
            
            try:
                (L_new, ll_new) = elliptical_slice(L_cur, 
                                                   L_std, 
                                                   ll_fn, 
                                                   mu=L_mean[k,:])
            except Exception as e:
                log.error("Caught exception in elliptical slice!")
                log.error(e.message)
                log.info(traceback.format_exc())
                exit()
            
            # Update the location
            L[k,:] = L_new
            
#        log.info("Average change in L: %f", np.mean(np.sqrt(np.sum((L-L0)**2,1))))
        
    def registerStatManager(self, statManager):
        """
        Register callbacks with the given StatManager
        """
        K = self.modelParams["proc_id_model","K"]
        D = self.params["D"]
        
        statManager.registerSampleCallback("L", 
                                           lambda: self.modelParams[self.name,"L"],
                                           (K,D),
                                           np.float32)
        