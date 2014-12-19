import numpy as np
from ConfigParser import ConfigParser


from model_extension import ModelExtension
from pyhawkes.utils.utils import *


log = logging.getLogger("global_log")


class ClusterModel(ModelExtension):
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
        
    def getInitializationOrder(self):
        """
        The cluster model has to initialize before its consumers
        """
        return 2
        
    def parseConfigurationFile(self, configFile):
        """
        Parse the configuration file to get base model parameters
        """
        # Initialize defaults
        defaultParams = {}
        defaultParams["thin"] = 1
        
        # Create a config parser object and read in the file
        cfgParser = ConfigParser(defaultParams)
        cfgParser.read(configFile)
        
        self.params = {}
        self.params["thin"]  = cfgParser.getint("cluster_model", "thin")
        self.params["R"]  = cfgParser.getint("cluster_model", "R")
        
        if cfgParser.has_option("cluster_model", "Y"):
            Y_str = cfgParser.get("cluster_model", "Y")
            if Y_str.endswith(".mat"):
                log.info("Looking for array 'Y' in %s", Y_str)
                if not os.path.exists(Y_str):
                    log.error("Specified path does not exist! %s", Y_str)
                    exit()
                    
                Y_mat = scipy.io.loadmat(Y_str)
                if not "Y" in Y_mat.keys():
                    log.error("Specified mat file does not contain field 'Y'!")
                    exit()
                    
                self.params["Y0"] = np.ravel(Y_mat["Y"])
                self.params["Y_given"] = True
        else:
            self.params["Y_given"] = False
        
        # If alpha is given we will not sample it
        self.params["alpha_given"] = True
        if cfgParser.has_option("cluster_model", "alpha"):
            log.debug("alpha is given. Will not sample.")
        
            # Parse the alpha parameter
            alpha_str = cfgParser.get("cluster_model", "alpha")
            alpha_list = alpha_str.split(',')
            self.params["alpha"] = np.array(map(float, alpha_list))
        else:
            log.debug("Alpha not specified. Using all ones.")
            self.params["alpha"] = np.ones(self.params["R"])
        
    def initializeModelParamsFromPrior(self):
        """
        Initialize the parameters of the model.
        """
        K = self.modelParams["proc_id_model","K"]
        
        self.modelParams[self.name,"R"] = self.params["R"]
        
        if self.params["alpha_given"]:
            self.modelParams[self.name,"alpha"] = self.params["alpha"]
            # Make sure alpha is normalized
            self.modelParams[self.name,"alpha"] /= np.sum(self.modelParams[self.name,"alpha"])
        else:
            self.modelParams[self.name,"alpha"] = np.random.dirichlet(self.params["alpha0"]).astype(np.float32)
        
        
        # Each process 1:K is endowed with a latent block 1:R. Let Y be a vector
        # representing the block identity for each process.
        self.modelParams[self.name,"Y"] = np.zeros((K,), dtype=np.int32)
        
        if self.params["Y_given"]:
            if np.min(self.params["Y0"]) < 0 or np.max(self.params["Y0"]) > self.params["R"]-1:
                log.error("Specified Y has values out of range [%d,%d]",0,self.params["R"]-1)
                exit()
            self.modelParams[self.name,"Y"] = self.params["Y0"]
        else:
            # Assign blocks randomly
            for k in np.arange(K):
                mn = np.random.multinomial(1,self.modelParams[self.name,"alpha"]).astype(np.int32)
                self.modelParams[self.name,"Y"][k] = np.nonzero(mn)[0][0]
        
    def initializeModelParamsFromDict(self, paramsDB):
        """
        Initialize from a dictionary
        """
        # HACK!
        if ("cluster1","Y") in paramsDB:
            self.modelParams[self.name,"Y"] = paramsDB["cluster1","Y"]
        else:
            self.initializeModelParamsFromPrior()
            if self.modelParams[self.name,"R"] == self.modelParams["proc_id_model","K"]:
                log.info("Assigning neurons to individual clusters")
                self.modelParams[self.name,"Y"] = np.arange(self.modelParams["proc_id_model","K"])
        
    def register_consumer(self, callback):
        """
        Add a consumer of the cluster assignments, Y. Whenever Y is sampled
        the callback will be called to get a likelihood of the new cluster.
        """
        self.callbacks.append(callback)
        
    def sampleModelParameters(self):
        return
        if np.mod(self.iter, self.params["thin"]) == 0:
            if not self.params["Y_given"]:
                self.sampleY()
                self.computeJaccardCoeff()
        self.iter += 1
        
    def sampleY(self):
        """
        Gibbs sample Y conditioned upon all other model params
        We seem to run into problems with all of the neurons flipping at the same time. 
        Just choose a subset to update at any time.
        """
        K = self.modelParams["proc_id_model","K"]
        Y = self.modelParams[self.name,"Y"]
        
        Y0 = np.copy(Y)
        for k in np.arange(K):
            # The prior distribution is simply alpha
            ln_pYk = np.log(self.modelParams[self.name,"alpha"])
            
            # likelihood of cluster assignment is given by callbacks
            for r in np.arange(self.modelParams[self.name,"R"]):
                for callback in self.callbacks:
                    ln_pYk[r] += callback(k,r)
                    
            # Sample block assignment with log-sum-exp trick 
            try:
                Y[k] = logSumExpSample(ln_pYk)
            except Exception as e:
                log.error("Caught exception in log-sum-exp")
                log.info(ln_pYk)
                raise e
        
        log.debug("Num Y changes: %d", np.count_nonzero((Y-Y0)!=0)) 
    # Compute Jaccard coeff
    def computeJaccardCoeff(self):
        Y = self.modelParams[self.name, "Y"]
        
        for r in np.arange(self.modelParams[self.name,"R"]):
            # Compute Jaccard (OFF, {k: Y[k]==r})
            intsct = np.count_nonzero(Y[:16]==r)
            union = 16 + np.count_nonzero(Y[16:]==r)
            log.info("Jaccard(OFF,Y==%d)=%f", r, float(intsct)/float(union))
            
        for r in np.arange(self.modelParams[self.name,"R"]):
            # Compute Jaccard (OFF, {k: Y[k]==r})
            intsct = np.count_nonzero(Y[16:]==r)
            union = 11 + np.count_nonzero(Y[:16]==r)
            log.info("Jaccard(ON,Y==%d)=%f", r, float(intsct)/float(union))

        
    def registerStatManager(self, statManager):
        """
        Register callbacks with the given StatManager
        """
        K = self.modelParams["proc_id_model","K"]
        
        statManager.registerSampleCallback("Y", 
                                           lambda: self.modelParams[self.name,"Y"],
                                           (K,),
                                           np.int32)
        
        statManager.registerSampleCallback("R", 
                                           lambda: self.modelParams[self.name,"R"],
                                           (1,),
                                           np.int32)