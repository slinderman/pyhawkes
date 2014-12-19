"""
Define the graph models that will be used by the Hawkes process inference algorithm.
Abstract away the details of generating each graph model and sampling its params. 
Each class supports the following functions
    - getConditionalEdgePr
    - sampleModelParameters
    - sampleGraphFromPrior
"""

import numpy as np
import string 
import logging


from ConfigParser import ConfigParser

from pyhawkes.utils.utils import *
from pyhawkes.utils.elliptical_slice import *
from pyhawkes.utils.hmc import *

from graph_model_extension import GraphModelExtension

# Import other graph models
from coupled_sbm_w_model import StochasticBlockModelCoupledWithW

log = logging.getLogger("global_log")


def constructGraphModel(graph_model, baseModel, configFile):
    """
    Return an instance of the graph model specified in parameters
    """
    
    if graph_model ==  "complete":
        log.info("Creating complete graph model")
        return CompleteGraphModel(baseModel, configFile)
    elif graph_model ==  "empty":
        log.info("Creating disconnected graph model")
        return EmptyGraphModel(baseModel, configFile)
    elif graph_model == "erdos_renyi":
        log.info("Creating Erdos-Renyi graph model")
        return ErdosRenyiModel(baseModel, configFile)
    elif graph_model == "symmetric":
        log.info("Creating Symmetric Erdos-Renyi graph model")
        return ErdosRenyiModel(baseModel, configFile)
    elif graph_model == "sbm":
        log.info("Creating Stochastic Block model")
        return StochasticBlockModel(baseModel, configFile)
    elif graph_model == "distance":
        log.info("Creating Latent Distance model")
        return LatentDistanceModel(baseModel, configFile)
    elif graph_model == "coupled_sbm_w":
        log.info("Creating coupled SBM+Weight prior")
        return StochasticBlockModelCoupledWithW(baseModel, configFile)
    else:
        log.error("Unrecognized graph model: %s", graph_model)
        exit()

    
class CompleteGraphModel(GraphModelExtension):
    """
    A deterministic, completely connected graph model. 
    """
    def __init__(self, baseModel, configFile):
        super(CompleteGraphModel,self).__init__(baseModel, configFile)
        
        pprintDict(self.params, "Graph Model Params")
                        
    def getConditionalEdgePr(self, ki, kj):
        """
        Return the conditional probability of edge A[ki,kj] given the model
        parameters. In this case the probability is simply rho. 
        """
        if ki >=  self.modelParams["proc_id_model","K"] or kj >=  self.modelParams["proc_id_model","K"]:
            log.error("Complete Graph: Index out of bounds! %ki=d, kj=%d, K=%d", ki, kj,  self.modelParams["proc_id_model","K"])
            exit()
            
        if self.params["allow_self_excitation"]:
            return 1.0
        elif ki != kj:
            return 1.0
        else:
            return 0.0
        
    def initializeModelParamsFromPrior(self):
        self.initializeGpuMemory()
        # Override the base model's default adjacency matrix with a draw from the prior
        self.modelParams["graph_model","A"] = self.sampleGraphFromPrior()
        self.gpuPtrs["graph_model","A"].set(self.modelParams["graph_model","A"])
        
    def initializeModelParamsFromDict(self, paramsDB):
        self.modelParams["graph_model","A"] = paramsDB["graph_model","A"]
        self.gpuPtrs["graph_model","A"].set(self.modelParams["graph_model","A"])
                
    def sampleNewProcessParams(self, newProcParams):
        """
        If the Process ID Model wants to add a new process it will call this function to 
        get parameters from the prior. Sample and add to the given dict.
        """
        # Get the current number of processes
        K = self.modelParams["proc_id_model","K"]
        
        # Copy over the existing W
        Aold = self.gpuPtrs["graph_model","A"].get()
        
        # Update with the new params
        Anew = np.zeros((K+1,K+1), dtype=np.bool)
        newRow = np.random.rand(K) < 1.0
        newCol = np.random.rand(K) < 1.0                                 
        newDiag = np.random.rand(1) < 1.0
                                 
        Anew[:-1,:-1] = Aold
        Anew[-1,:-1] = newRow
        Anew[-1,-1] = newDiag
        Anew[:-1,-1] = newCol
                                 
        newProcParams["graph_model"] = {"A":Anew}
        
    def addNewProcessEventHandler(self, newProcParams):
        """
        If a new process is added the parameters will be in the given dict.
        We need to update all our data structures that depend on K. We can
        assume that the base model has updated K
        """
        K = self.modelParams["proc_id_model","K"]
        
        # Update with the new params
        self.modelParams["graph_model","A"] = newProcParams["graph_model"]["A"]
        # Copy over to the GPU
        self.gpuPtrs["graph_model","A"] = gpuarray.to_gpu(newProcParams["graph_model"]["A"])
        
    def removeProcessEventHandler(self, procId):
        """
        Remove process procID from the set of processes and update data structures
        accordingly. We can assume that the base model has updated K.
        """
        K = self.modelParams["proc_id_model","K"]
        
        # Copy over the existing W
        Aold = self.gpuPtrs["graph_model","A"].get()
        
        # Remove a row and a column from the matrix
        Anew = np.zeros((K,K), dtype=np.bool)
        Anew[:procId,:procId] = Aold[:procId,:procId]
        Anew[:procId,procId:] = Aold[:procId,(procId+1):]
        Anew[procId:,:procId] = Aold[(procId+1):,:procId]
        Anew[procId:,procId:] = Aold[(procId+1):,(procId+1):]
        
        # Copy over to the GPU
        self.gpuPtrs["graph_model","A"] = gpuarray.to_gpu(Anew, dtype=np.bool)
    
    def sampleModelParameters(self):
        pass
    
    def sampleGraphFromPrior(self):
        """
        Generate a random sample from the prior
        """
        if self.params["allow_self_excitation"]:
            return np.ones(( self.modelParams["proc_id_model","K"], self.modelParams["proc_id_model","K"]), dtype=np.bool)
        else:
            return np.ones(( self.modelParams["proc_id_model","K"], self.modelParams["proc_id_model","K"]), dtype=np.bool) - np.eye( self.modelParams["proc_id_model","K"], dtype=np.bool)
    
    def registerStatManager(self, statManager):
        """
        Register callbacks with the given StatManager
        """
        super(CompleteGraphModel,self).registerStatManager(statManager)
        
class EmptyGraphModel(GraphModelExtension):
    """
    A deterministic, disconnected graph model for testing the log-Gaussian cox process. 
    """
    def __init__(self, baseModel, configFile):
        super(EmptyGraphModel,self).__init__(baseModel, configFile)
        
        pprintDict(self.params, "Graph Model Params")
        
        
    
    def initializeModelParamsFromPrior(self):
        self.initializeGpuMemory()
        # Override the base model's default adjacency matrix with a draw from the prior
        self.modelParams["graph_model","A"] = self.sampleGraphFromPrior()
        self.gpuPtrs["graph_model","A"].set(self.modelParams["graph_model","A"])
        
    def initializeModelParamsFromDict(self, paramsDB):
        self.initializeGpuMemory()
        self.modelParams["graph_model","A"] = paramsDB["graph_model","A"]
        self.gpuPtrs["graph_model","A"].set(self.modelParams["graph_model","A"])
        
    def getModelParamsDict(self):
        return {"A" : self.modelParams["graph_model","A"]}
                        
    def getConditionalEdgePr(self, ki, kj):
        """
        Return the conditional probability of edge A[ki,kj] given the model
        parameters. In this case the probability is simply rho. 
        """
        if ki >= self.modelParams["proc_id_model","K"] or kj >=  self.modelParams["proc_id_model","K"]:
            log.error("Complete Graph: Index out of bounds! %ki=d, kj=%d, K=%d", ki, kj,  self.modelParams["proc_id_model","K"])
            exit()
            
        return 0.0
    
    def sampleNewProcessParams(self, newProcParams):
        """
        If the Process ID Model wants to add a new process it will call this function to 
        get parameters from the prior. Sample and add to the given dict.
        """
        # Get the current number of processes
        K = self.modelParams["proc_id_model","K"]
        
        # Copy over the existing W
        Aold = self.gpuPtrs["graph_model","A"].get()
        
        # Update with the new params
        Anew = np.zeros((K+1,K+1), dtype=np.bool)
        newRow = np.random.rand(K) < 0.0
        newCol = np.random.rand(K) < 0.0                                 
        newDiag = np.random.rand(1) < 0.0
                                 
        Anew[:-1,:-1] = Aold
        Anew[-1,:-1] = newRow
        Anew[-1,-1] = newDiag
        Anew[:-1,-1] = newCol
                                 
        newProcParams["graph_model"] = {"A":Anew}
        
    def addNewProcessEventHandler(self, newProcParams):
        """
        If a new process is added the parameters will be in the given dict.
        We need to update all our data structures that depend on K. We can
        assume that the base model has updated K
        """
        K = self.modelParams["proc_id_model","K"]
        
        # Update with the new params
        self.modelParams["graph_model","A"] = newProcParams["graph_model"]["A"]
        # Copy over to the GPU
        self.gpuPtrs["graph_model","A"] = gpuarray.to_gpu(newProcParams["graph_model"]["A"])
        
    def removeProcessEventHandler(self, procId):
        """
        Remove process procID from the set of processes and update data structures
        accordingly. We can assume that the base model has updated K.
        """
        K = self.modelParams["proc_id_model","K"]
        
        # Copy over the existing W
        Aold = self.gpuPtrs["graph_model","A"].get()
        
        # Remove a row and a column from the matrix
        Anew = np.zeros((K,K), dtype=np.bool)
        Anew[:procId,:procId] = Aold[:procId,:procId]
        Anew[:procId,procId:] = Aold[:procId,(procId+1):]
        Anew[procId:,:procId] = Aold[(procId+1):,:procId]
        Anew[procId:,procId:] = Aold[(procId+1):,(procId+1):]
        
        # Copy over to the GPU
        self.gpuPtrs["graph_model","A"] = gpuarray.to_gpu(Anew, dtype=np.bool)
    
    def sampleModelParameters(self):
        pass
    
    def sampleGraphFromPrior(self):
        """
        Generate a random sample from the prior
        """
        return np.zeros(( self.modelParams["proc_id_model","K"], self.modelParams["proc_id_model","K"]), dtype=np.bool)
    
    def registerStatManager(self, statManager):
        """
        Register callbacks with the given StatManager
        """
        super(EmptyGraphModel,self).registerStatManager(statManager)

        
class ErdosRenyiModel(GraphModelExtension):
    """
    Prior for an Erdos Renyi random graph model. Each edge is drawn
    i.i.d. Bernoulli with parameter rho
    """
    def __init__(self, baseModel, configFile):
        super(ErdosRenyiModel,self).__init__(baseModel, configFile)
        
        self.parseConfigurationFile(configFile)
        pprintDict(self.params, "Graph Model Params")
        
        
    def parseConfigurationFile(self, configFile):
        # Set defaults
        defaultParams = {}
        
        # Parse config file
        cfgParser = ConfigParser(defaultParams)
        cfgParser.read(configFile)
        
        erparams = {}
        if cfgParser.has_option("graph_prior", "rho"):
            erparams["rho"] = cfgParser.getfloat("graph_prior", "rho")
        elif cfgParser.has_option("graph_prior", "a") and \
             cfgParser.has_option("graph_prior", "b"):
            
            erparams["a_rho"] = cfgParser.getfloat("graph_prior", "a")
            erparams["b_rho"] = cfgParser.getfloat("graph_prior", "b")
        else:
            log.error("Either rho or both a and b must be specified for Erdos-Renyi graph model")
            exit()
            
        # Check for a parameter allowing/disallowing excitatory connections
        # between the uptick and downtick processes as in the financial data
        if cfgParser.has_option("graph_prior", "allow_updown_excitation"):
            erparams["allow_updown_excitation"] = cfgParser.getfloat("graph_prior", "allow_updown_excitation")
        else:
            erparams["allow_updown_excitation"] = True
        
        # Combine this with the general param dict
        self.params.update(erparams)
        
    def initializeModelParamsFromPrior(self):
        self.initializeGpuMemory()
        self.initializeModelParameters()
        
        # Override the base model's default adjacency matrix with a draw from the prior
        self.modelParams["graph_model","A"] = self.sampleGraphFromPrior()
        self.gpuPtrs["graph_model","A"].set(self.modelParams["graph_model","A"])
    
    def initializeModelParamsFromDict(self, paramsDB):
        self.initializeGpuMemory()
        self.initializeModelParameters()
        self.modelParams["graph_model","A"] = paramsDB["graph_model","A"]
        self.gpuPtrs["graph_model","A"].set(self.modelParams["graph_model","A"])
        
    
    def initializeModelParameters(self):
        K = self.modelParams["proc_id_model","K"]
        
        # Use given rho if specified
        if "rho" in self.params:
            self.rho_v = self.params["rho"]
        else:
            self.rho_v = np.random.beta(self.params["a_rho"], self.params["b_rho"]) *np.ones((K,K))
        
        # Rho can either be a scalar in [0,1] or a matrix of size [K,K]   
        if np.size(self.rho_v) == 1:
            self.rho = self.rho_v*np.ones((K,K), dtype=np.float32)
        elif np.shape(self.rho_v) != (K,K):
            log.error("Invalid shape for rho! Must be %dx%d matrix, instead received %s", K,K,str(np.shape(rho)))
        else:
            self.rho = self.rho_v
            
        # Override diagonal if we're not allowing self excitation
        if not self.params["allow_self_excitation"]:
            self.rho[np.diag_indices(K)] = 0.0
            
        # Override diagonal if we're not allowing self excitation
        if not self.params["allow_updown_excitation"]:
            half = K/2
            self.rho[np.arange(0,half), np.arange(half,K)] = 0.0
            self.rho[np.arange(half,K), np.arange(0,half)] = 0.0
        
    def computeLogProbability(self):
        """
        The log probability is the sum of individual edge log probs
        """
        K = self.modelParams["proc_id_model","K"]
        A = self.modelParams["graph_model","A"]
        
        ll = 0.0
        for k1 in np.arange(K):
            for k2 in np.arange(K):
                if self.rho[k1,k2]==0.0:
                    if A[k1,k2]:
                        log.error("ERROR: invalid edge (%d,%d) where rho=0.0", k1,k2)
                        ll = -np.Inf
                elif self.rho[k1,k2]==1.0:
                    if not A[k1,k2]:
                        log.error("ERROR: invalid lack of edge (%d,%d) where rho=1.0", k1,k2)
                        ll = -np.Inf
                else:
                    ll +=  A[k1,k2]*np.log(self.rho[k1,k2]) + (1-A[k1,k2])*np.log(1-self.rho[k1,k2])
                      
#        ll = np.sum(np.ravel(A*np.log(self.rho) + (1-A)*np.log(1-self.rho)))
        
        return ll
    
    def getConditionalEdgePr(self, ki, kj):
        """
        Return the conditional probability of edge A[ki,kj] given the model
        parameters. In this case the probability is simply rho. 
        """
        return self.rho[ki,kj]
    
    def sampleNewProcessParams(self, newProcParams):
        """
        If the Process ID Model wants to add a new process it will call this function to 
        get parameters from the prior. Sample and add to the given dict.
        """
        # Get the current number of processes
        K = self.modelParams["proc_id_model","K"]
        
        # Keep the existing A and rho
        # Update rho
        rho_new = np.zeros((K+1,K+1), dtype=np.float32)
        rho_new[:-1,:-1] = self.rho
        rho_new[-1,:-1] = self.rho_v
        rho_new[:-1,-1] = self.rho_v
        if self.params["allow_self_excitation"]:                                
            rho_new[-1,-1] = self.rho_v
        else:
            rho_new[-1,-1] = 0.0
          
        # Update A
        Aold = self.gpuPtrs["graph_model","A"].get()
        Anew = np.zeros((K+1,K+1), dtype=np.bool)                       
        Anew[:-1,:-1] = Aold
        Anew[-1,:-1] = np.random.rand(K) < rho_new[-1,:-1]
        Anew[:-1,-1] = np.random.rand(K) < rho_new[:-1,-1]
        Anew[-1,-1] = np.random.rand(1) < rho_new[-1,-1]
                                 
        newProcParams["graph_model"] = {"A":Anew, "rho":rho_new}
        
    def addNewProcessEventHandler(self, newProcParams):
        """
        If a new process is added the parameters will be in the given dict.
        We need to update all our data structures that depend on K. We can
        assume that the base model has updated K
        """
        K = self.modelParams["proc_id_model","K"]
        
        # Update with the new params
        self.modelParams["graph_model","A"] = newProcParams["graph_model"]["A"]
        self.rho = newProcParams["graph_model"]["rho"]
        
        # Copy over to the GPU
        self.gpuPtrs["graph_model","A"] = gpuarray.to_gpu(newProcParams["graph_model"]["A"])
        
    def removeProcessEventHandler(self, procId):
        """
        Remove process procID from the set of processes and update data structures
        accordingly. We can assume that the base model has updated K.
        """
        K = self.modelParams["proc_id_model","K"]
        
        rho_new = np.zeros((K,K), dtype=np.bool)
        rho_new[:procId,:procId] = self.rho[:procId,:procId]
        rho_new[:procId,procId:] = self.rho[:procId,(procId+1):]
        rho_new[procId:,:procId] = self.rho[(procId+1):,:procId]
        rho_new[procId:,procId:] = self.rho[(procId+1):,(procId+1):]
        
        self.rho = rho_new
        
        # Copy over the existing W
        Aold = self.gpuPtrs["graph_model","A"].get()
        
        # Remove a row and a column from the matrix
        Anew = np.zeros((K,K), dtype=np.bool)
        Anew[:procId,:procId] = Aold[:procId,:procId]
        Anew[:procId,procId:] = Aold[:procId,(procId+1):]
        Anew[procId:,:procId] = Aold[(procId+1):,:procId]
        Anew[procId:,procId:] = Aold[(procId+1):,(procId+1):]
        
        # Copy over to the GPU
        self.gpuPtrs["graph_model","A"] = gpuarray.to_gpu(Anew, dtype=np.bool)
    
    def sampleModelParameters(self):
        """
        Sample model parameters given the current matrix A. None of the other Hawkes
        parameters play into the likelihood of the model params since, by design, 
        A is conditionally independent given the model.
        """
        if "rho" not in self.params:
            self.sampleRho()
            print "rho: %.4f" % self.rho[0,0]
        self.sampleA()
        
    def sampleRho(self):
        A = self.modelParams["graph_model","A"]
        K = self.modelParams["proc_id_model","K"]

        nnz_A = sum(sum(A))
        a_rho_post = self.params["a_rho"] + nnz_A
        b_rho_post = self.params["b_rho"] +  self.modelParams["proc_id_model","K"]**2 - nnz_A
        self.rho = np.random.beta(a_rho_post, b_rho_post) * np.ones((K,K))
        
    def sampleGraphFromPrior(self):
        """
        Sample graph from the prior
        """
        K = self.modelParams["proc_id_model","K"]
        A = np.random.rand(K,K) < self.rho
        
        # Override entries set in the mask
        if self.modelParams["graph_model","mask"] != None:
            for ki in np.arange(K):
                for kj in np.arange(K):
                    if self.modelParams["graph_model","mask"][ki,kj] != -1:
                        A[ki,kj] = np.bool(self.modelParams["graph_model","mask"][ki,kj])
        
        return A
    
    def registerStatManager(self, statManager):
        """
        Register callbacks with the given StatManager
        """
        super(ErdosRenyiModel,self).registerStatManager(statManager)
        
        statManager.registerSampleCallback("rho", 
                                           lambda: self.rho.astype(np.float32),
                                           (self.modelParams["proc_id_model","K"],self.modelParams["proc_id_model","K"]),
                                           np.float32)
    
class StochasticBlockModel(GraphModelExtension):
    """
    Prior for a stochastic block model of connectivity. Each process is assigned a 
    latent "block"; probability of connectivity between a pair of processes is 
    determined by their block identities. Each pair of blocks is given a bernoulli
    probability of an edge being present.
    """
    def __init__(self, baseModel, configFile):
        super(StochasticBlockModel,self).__init__(baseModel, configFile)
        
        self.parseConfigurationFile(configFile)
        pprintDict(self.params, "Graph Model Params")
    
    def parseConfigurationFile(self, configFile):
        
        # Set defaults
        defaultParams = {}
        defaultParams["R"] = 2
        defaultParams["b0"] = 1.0
        defaultParams["b1"] = 1.0
        defaultParams["rho"] = 0.1
        defaultParams["allow_block_self_excitation"] = 1
        defaultParams["thin"] = 50
        
        # Parse config file
        cfgParser = ConfigParser(defaultParams)
        cfgParser.read(configFile)
        
        sbmparams = {}
        sbmparams["R"] = cfgParser.getint("graph_prior", "R")
        sbmparams["b0"] = cfgParser.getfloat("graph_prior", "b0")
        sbmparams["b1"] = cfgParser.getfloat("graph_prior", "b1")
        sbmparams["rho"] = cfgParser.getfloat("graph_prior", "rho")
        
        sbmparams["allow_block_self_excitation"]    = bool(cfgParser.getint("graph_prior", "allow_block_self_excitation"))
        
        sbmparams["thin"] = cfgParser.getint("graph_prior", "thin")
        
        # Check for a parameter allowing/disallowing excitatory connections
        # between the uptick and downtick processes as in the financial data
        if cfgParser.has_option("graph_prior", "allow_updown_excitation"):
            sbmparams["allow_updown_excitation"] = cfgParser.getfloat("graph_prior", "allow_updown_excitation")
        else:
            sbmparams["allow_updown_excitation"] = True
        
        # Combine the two param dicts
        self.params.update(sbmparams)
        
    def initializeModelParamsFromPrior(self):
        self.initializeGpuMemory()
        self.initializeModelParameters()
        
        # Override the base model's default adjacency matrix with a draw from the prior
        self.modelParams["graph_model","A"] = self.sampleGraphFromPrior()
        self.gpuPtrs["graph_model","A"].set(self.modelParams["graph_model","A"])
    
    def initializeModelParamsFromDict(self, paramsDB):
        self.initializeGpuMemory()
        self.modelParams["graph_model","A"] = paramsDB["graph_model","A"]
        self.gpuPtrs["graph_model","A"].set(self.modelParams["graph_model","A"])
        
        # Copy over training parameters
        self.modelParams["graph_model","B"] = paramsDB["graph_model","B"]
        self.modelParams["graph_model","Y"] = paramsDB["graph_model","Y"]
        
        
        
    def initializeModelParameters(self):
        K = self.modelParams["proc_id_model","K"]
        
        # Prior params
        self.R      = self.params["R"]           # number of blocks
        
        # Number of fake instances of alpha0 observed
        n_alpha0    = 100
        self.alpha0 = n_alpha0 * 1.0/self.R*np.ones((self.R,), dtype=np.float32)  # Dirichlet prior on block membership
        
        self.b0     = self.params["b0"]          # prior on beta distribution for B
        self.b1     = self.params["b1"]
        
        self.rho    = self.params["rho"]         # prior on sparsity
        
        # To model sparsity in the adjacency matrix we introduce a mixture with a 
        # Bernoulli random graph with sparsity rho
        self.Nu = np.random.rand(K,K) < self.rho
        
        # There is a matrix B representing the Bernoulli probability of an edge between 
        # processes on two blocks. It has a Beta prior.
        self.modelParams["graph_model","B"] = np.random.beta(self.b0, self.b1, (self.R,self.R)).astype(np.float32)
        if not self.params["allow_block_self_excitation"]:
            self.modelParams["graph_model","B"][np.diag_indices(self.R)] = 0
        
        # There is a multinomial distribution over each process's block identity
        # It has a dirichlet prior
        self.modelParams["graph_model","alpha"] = np.random.dirichlet(self.alpha0).astype(np.float32)
        
        # Each process 1:K is endowed with a latent block 1:R. Let Y be a vector
        # representing the block identity for each process.
        self.modelParams["graph_model","Y"] = np.zeros((K,), dtype=np.int32)
        
        for k in np.arange(K):
            mn = np.random.multinomial(1,self.modelParams["graph_model","alpha"]).astype(np.int32)
            self.modelParams["graph_model","Y"][k] = np.nonzero(mn)[0][0]
        
        # WLOG distribute multinomial samples in order
#        init_blocksz = np.random.multinomial(K,self.modelParams["graph_model","alpha"]).astype(np.int32)
#        offset = 0
#        for r in np.arange(self.R):
#            self.modelParams["graph_model","Y"][offset:offset+init_blocksz[r]] = r
#            offset += init_blocksz[r] 
            
        
        # Keep a counter of number of calls to sampleModelParameters
        # We thin the sampling according to the specified parameter.
        self.iter = 0
            
    def sampleNewProcessParams(self, newProcParams):
        """
        If the Process ID Model wants to add a new process it will call this function to 
        get parameters from the prior. Sample and add to the given dict.
        """
        # Get the current number of processes
        K = self.modelParams["proc_id_model","K"]
        
        # Keep the existing Y, Nu, and A
        # Update Y
        Ynew = np.zeros((K+1,), dtype=np.int32)
        Ynew[:-1] = self.modelParams["graph_model","Y"]
        # Sample a block for the new process
        p = np.random.rand()*np.sum(self.modelParams["graph_model","alpha"])
        acc = 0.0
        for r in np.arange(self.R):
            acc += self.modelParams["graph_model","alpha"][r]
            if p <= acc:
                Ynew[-1] = r
                break
            
        # Update Nu
        Nu_new = np.zeros((K+1,K+1), dtype=np.bool)
        Nu_new[:-1,:-1] = self.Nu
        Nu_new[-1,:-1] = np.random.rand(K) < self.rho
        Nu_new[:-1,-1] = np.random.rand(K) < self.rho
        Nu_new[-1,-1] = np.random.rand(1) < self.rho
          
        # Update A
        Aold = self.gpuPtrs["graph_model","A"].get()
        Anew = np.zeros((K+1,K+1), dtype=np.bool)                       
        Anew[:-1,:-1] = Aold
        for k in np.arange(K):
            Anew[-1,k] = Nu_new[-1,k] and (np.random.rand() < self.modelParams["graph_model","B"][Ynew[-1],Ynew[k]])
            Anew[k,-1] = Nu_new[k,-1] and (np.random.rand() < self.modelParams["graph_model","B"][Ynew[k],Ynew[-1]])
        Anew[-1,-1] = Nu_new[-1,-1] and (np.random.rand() < self.modelParams["graph_model","B"][Ynew[-1],Ynew[-1]])
                                 
        newProcParams["graph_model"] = {"A":Anew, "Y":Ynew, "Nu":Nu_new}
        
    def addNewProcessEventHandler(self, newProcParams):
        """
        If a new process is added the parameters will be in the given dict.
        We need to update all our data structures that depend on K. We can
        assume that the base model has updated K
        """
        K = self.modelParams["proc_id_model","K"]
        
        # Update with the new params
        self.modelParams["graph_model","A"] = newProcParams["graph_model"]["A"]
        self.modelParams["graph_model","Y"] = newProcParams["graph_model"]["Y"]
        self.Nu = newProcParams["graph_model"]["Nu"]
        
        # Copy over to the GPU
        self.gpuPtrs["graph_model","A"] = gpuarray.to_gpu(newProcParams["graph_model"]["A"])
        
    def removeProcessEventHandler(self, procId):
        """
        Remove process procID from the set of processes and update data structures
        accordingly. We can assume that the base model has updated K.
        """
        K = self.modelParams["proc_id_model","K"]
        
        # Update Y
        Ynew = np.zeros((K,), dtype=np.int32)
        Ynew[:procId] = self.modelParams["graph_model","Y"][:procId]
        Ynew[procId:] = self.modelParams["graph_model","Y"][(procId+1):]
        self.modelParams["graph_model","Y"] = Ynew
        
        # Update Nu
        Nu_new = np.zeros((K,K), dtype=np.bool)
        Nu_new[:procId,:procId] = self.Nu[:procId,:procId]
        Nu_new[:procId,procId:] = self.Nu[:procId,(procId+1):]
        Nu_new[procId:,:procId] = self.Nu[(procId+1):,:procId]
        Nu_new[procId:,procId:] = self.Nu[(procId+1):,(procId+1):]
        self.Nu = Nu_new
        
        # Copy over the existing A
        Aold = self.gpuPtrs["graph_model","A"].get()
        
        # Remove a row and a column from the matrix
        Anew = np.zeros((K,K), dtype=np.bool)
        Anew[:procId,:procId] = Aold[:procId,:procId]
        Anew[:procId,procId:] = Aold[:procId,(procId+1):]
        Anew[procId:,:procId] = Aold[(procId+1):,:procId]
        Anew[procId:,procId:] = Aold[(procId+1):,(procId+1):]
        
        # Copy over to the GPU
        self.gpuPtrs["graph_model","A"] = gpuarray.to_gpu(Anew, dtype=np.bool)
        
    def computeLogProbability(self):
        """
        The log probability is the sum of individual edge log probs
        """
        A = self.modelParams["graph_model","A"]
        nnz_A = sum(sum(A))
        K = self.modelParams["proc_id_model","K"]
        
        ll = 0.0
        
        # Log likelihood of alpha
        ll += np.sum((self.alpha0-1)*np.log(self.modelParams["graph_model","alpha"]))
        
        # Log likelihood of B
        if not self.params["allow_block_self_excitation"]:
            off_diag_mask = np.ones((self.R,self.R), dtype=np.bool)
            off_diag_mask[np.diag_indices(self.R)]=False
            B_flat = self.modelParams["graph_model","B"][off_diag_mask]
        else:
            B_flat = np.ravel(self.modelParams["graph_model","B"])  
            
        ll += np.sum((self.b0-1)*np.log(B_flat) + (self.b1-1)*np.log(1-B_flat))
        
        # Log likelihood of Y
        for k in np.arange(K):
            ll += np.log(self.modelParams["graph_model","alpha"][self.modelParams["graph_model","Y"][k]])
        
        # Log likelihood of A
        for k1 in np.arange(K):
            for k2 in np.arange(K):
                if A[k1,k2]:
                    # Edge present implies Nu[k1,k2] = 1 
                    ll += np.log(self.rho) + np.log(self.modelParams["graph_model","B"][self.modelParams["graph_model","Y"][k1],self.modelParams["graph_model","Y"][k2]])
                elif self.Nu[k1,k2]:
                    # Edge was possible but did not occur because B was too low
                    ll += np.log(self.rho) + np.log(1-self.modelParams["graph_model","B"][self.modelParams["graph_model","Y"][k1],self.modelParams["graph_model","Y"][k2]])
                else:
                    # Edge was not possible due to Nu
                    ll += np.log(1-self.rho)
        
        return ll 
        
    def getConditionalEdgePr(self, ki, kj):
        """
        Return the conditional probability of edge A[ki,kj] given the model
        parameters. In this case the probability is simply rho. 
        """
        K = self.modelParams["proc_id_model","K"]
        
        # The probability of an edge is simply the entry in B corresponding to the 
        # block identities for processes ki and kj, unless we disallow self edges
        if not self.params["allow_self_excitation"] and ki==kj:
            return 0.0
        elif not self.params["allow_updown_excitation"] and np.abs(ki-kj)==K/2:
            return 0.0
        else:
            return self.rho * self.modelParams["graph_model","B"][self.modelParams["graph_model","Y"][ki], self.modelParams["graph_model","Y"][kj]]
            
    
    def sampleModelParameters(self):
        """
        Sample model parameters given the current matrix A. None of the other Hawkes
        parameters play into the likelihood of the model params since, by design, 
        A is conditionally independent given the model.
        """
        if np.mod(self.iter, self.params["thin"]) == 0:
            A = self.modelParams["graph_model","A"]
            
            # Sample Nu - if Nu[i,j] = 1 then an edge may exist between processes i and j
            # otherwise the edge is not present due to sparsity
            self.sampleNu(A)
            
            # Sample Y -- the block identities - given the rest of the parameters
            self.sampleY(A)
            
            # Sample B - the Bernoulli matrix representing edge probabilities between
            # each pair of blocks
            self.sampleB(A)
            
            # Sample alpha - the multinomial distribution over block membership
            self.sampleAlpha(A)
            
        # Sample a new graph on every iteration
        self.sampleA()
        
        self.iter += 1
        
    def sampleGraphFromPrior(self):
        """
        Sample graph from the prior
        """
        
        
        # There is a matrix B representing the Bernoulli probability of an edge between 
        # processes on two blocks. It has a Beta prior.
        self.modelParams["graph_model","B"] = np.random.beta(self.b0, self.b1, (self.R,self.R))
        if not self.params["allow_block_self_excitation"]:
            self.modelParams["graph_model","B"][np.diag_indices(self.R)] = 0
        
        # There is a multinomial distribution over each process's block identity
        # It has a beta prior
        #self.modelParams["graph_model","alpha"] = np.random.dirichlet(self.alpha0)
#        self.modelParams["graph_model","alpha"] = self.alpha0
        
        # Each process 1:K is endowed with a latent block 1:R. Let Y be a vector
        # representing the block identity for each process.
        for k in np.arange( self.modelParams["proc_id_model","K"]):
            p = np.random.rand()
            acc = 0.0
            for r in np.arange(self.R):
                acc += self.modelParams["graph_model","alpha"][r]
                if p <= acc:
                    self.modelParams["graph_model","Y"][k] = r
                    break
                
        self.Nu = np.random.rand( self.modelParams["proc_id_model","K"], self.modelParams["proc_id_model","K"]) < self.rho 
        # Sample adjacency matrix
        A = np.zeros(( self.modelParams["proc_id_model","K"], self.modelParams["proc_id_model","K"]), dtype=np.bool)
        for k1 in np.arange( self.modelParams["proc_id_model","K"]):
            for k2 in np.arange( self.modelParams["proc_id_model","K"]):
                if self.Nu[k1,k2]:
                    A[k1,k2] = np.random.rand() < self.modelParams["graph_model","B"][self.modelParams["graph_model","Y"][k1],self.modelParams["graph_model","Y"][k2]]
    
        return A
            
    def sampleNu(self, A):
        """
        Sample Nu conditioned upon all other model params
        """
        K =  self.modelParams["proc_id_model","K"]
        self.Nu = np.zeros((K,K), dtype=np.bool)
        for k1 in np.arange(K):
            for k2 in np.arange(K):
#                self.Nu[A] = True
#                pNu = (1-self.modelParams["graph_model","B"][np.ix_(self.modelParams["graph_model","Y"],self.modelParams["graph_model","Y"])])*self.rho / ((1-self.modelParams["graph_model","B"][np.ix_(self.modelParams["graph_model","Y"],self.modelParams["graph_model","Y"])])*self.rho + (1-self.rho))
#                self.Nu[1-A] = (np.random.rand(K,K) < pNu)[1-A]
                if A[k1,k2]:
                    self.Nu[k1,k2] = True
                else:
                    # Probability the edge was possible is the probability the edge
                    # was not present due to B over the total probability. It is more likely
                    # for Nu to be 1 when the corresponding entry in B is large
                    pNu = (1-self.modelParams["graph_model","B"][self.modelParams["graph_model","Y"][k1],self.modelParams["graph_model","Y"][k2]])*self.rho / ((1-self.modelParams["graph_model","B"][self.modelParams["graph_model","Y"][k1],self.modelParams["graph_model","Y"][k2]])*self.rho + (1-self.rho))
                    
                    self.Nu[k1,k2] = np.random.rand() < pNu
    
    def sampleY(self, A):
        """
        Gibbs sample Y conditioned upon all other model params
        """
        Y = self.modelParams["graph_model","Y"]
        for k in np.arange( self.modelParams["proc_id_model","K"]):
            
            # The prior distribution is simply alpha
            ln_pYk = np.log(self.modelParams["graph_model","alpha"])
            
            # likelihood of block membership is affected by the presence and absence
            # of edges, both incoming and outgoing
            for r in np.arange(self.R):
                # Block IDs of nodes we connect to
                o1 = np.bitwise_and(A[k,:], self.Nu[k,:]) 
                if np.any(o1):
                    ln_pYk[r] += np.sum(np.log(self.modelParams["graph_model","B"][np.ix_([r],Y[o1])]))
                
                # Block IDs of nodes we don't connect to
                o2 = np.bitwise_and(np.logical_not(A[k,:]), self.Nu[k,:])
                if np.any(o2):
                    ln_pYk[r] += np.sum(np.log(1-self.modelParams["graph_model","B"][np.ix_([r],Y[o2])]))
                
                # Block IDs of nodes that connect to us
                i1 = np.bitwise_and(A[:,k], self.Nu[:,k])
                if np.any(i1):
                    ln_pYk[r] += np.sum(np.log(self.modelParams["graph_model","B"][np.ix_(Y[i1],[r])]))
    
                # Block IDs of nodes that do not connect to us
                i2 = np.bitwise_and(np.logical_not(A[:,k]), self.Nu[:,k])
                if np.any(i2):
                    ln_pYk[r] += np.sum(np.log(1-self.modelParams["graph_model","B"][np.ix_(Y[i2],[r])]))
            
            # Use logsumexp trick to calculate ln(p1 + p2 + ... + pR) from ln(pi)'s
#            max_lnp = np.max(ln_pYk)
#            denom = np.log(np.sum(np.exp(ln_pYk-max_lnp))) + max_lnp
#            pYk_safe = np.exp(ln_pYk - denom)
#            
#            # Normalize the discrete distribution over blocks
#            sum_pYk = np.sum(pYk_safe)
#            if sum_pYk == 0 or not np.isfinite(sum_pYk):
#                log.error("total probability of block assignment is not valid! %f", sum_pYk)
#                log.info(pYk_safe)
#                exit()
#            
#            # Randomly sample a block
#            p = np.random.rand()
#            acc = 0.0
#            for r in np.arange(self.R):
#                acc += pYk_safe[r]
#                if p <= acc:
#                    Y[k] = r
#                    break
            try:
                Y[k] = logSumExpSample(ln_pYk)
            except:
                log.info(self.modelParams["graph_model","B"])
                exit()
                
    def sampleB(self, A):
        self.modelParams["graph_model","B"] = np.zeros((self.R,self.R), dtype=np.float32)
        
        for r1 in np.arange(self.R):
            for r2 in np.arange(self.R):
                if not self.params["allow_block_self_excitation"] and r1==r2:
                    self.modelParams["graph_model","B"][r1,r2] = 0
                else:
                    b0star = self.b0
                    b1star = self.b1
                    
                    Ar1r2 = A[np.ix_(self.modelParams["graph_model","Y"]==r1, self.modelParams["graph_model","Y"]==r2)]
                    Nur1r2 = self.Nu[np.ix_(self.modelParams["graph_model","Y"]==r1, self.modelParams["graph_model","Y"]==r2)]
                    if np.size(Ar1r2) > 0:
                        b0star += np.sum(Ar1r2[Nur1r2])
                        b1star += np.sum(1-Ar1r2[Nur1r2])
                    
                    self.modelParams["graph_model","B"][r1,r2] = np.random.beta(b0star, b1star)
                
    def sampleAlpha(self, A):
        alpha_star = np.zeros((self.R,), dtype=np.float32)
        for r in np.arange(self.R):
            alpha_star[r] = self.alpha0[r] + np.sum(self.modelParams["graph_model","Y"]==r)
            
        self.modelParams["graph_model","alpha"] = np.random.dirichlet(alpha_star).astype(np.float32)
        
    def sampleGraphGivenAlphaBRho(self,K,R,alpha,B,rho):
        """
        Sample a graph with given parameters. This is statically callable
        """
        
        # Each process 1:K is endowed with a latent block 1:R. Let Y be a vector
        # representing the block identity for each process.
        Y = np.zeros((K,), dtype=np.int32)
        for k in np.arange(K):
            p = np.random.rand()
            acc = 0.0
            for r in np.arange(R):
                acc += alpha[r]
                if p <= acc:
                    Y[k] = r
                    break
                
        Nu = np.random.rand(K,K) < rho 
        # Sample adjacency matrix
        A = np.zeros((K,K), dtype=np.bool)
        for k1 in np.arange(K):
            for k2 in np.arange(K):
                if Nu[k1,k2]:
                    A[k1,k2] = np.random.rand() < B[Y[k1],Y[k2]]
    
        return (Y,Nu,A)
    
    def registerStatManager(self, statManager):
        """
        Register callbacks with the given StatManager
        """
        super(StochasticBlockModel,self).registerStatManager(statManager)
        
        statManager.registerSampleCallback("Y", 
                                           lambda: self.modelParams["graph_model","Y"],
                                           ( self.modelParams["proc_id_model","K"],),
                                           np.int32)
        
        statManager.registerSampleCallback("B", 
                                           lambda: self.modelParams["graph_model","B"],
                                           (self.R,self.R),
                                           np.float32)
        
        statManager.registerSampleCallback("alpha", 
                                           lambda: self.modelParams["graph_model","alpha"],
                                           (self.R,),
                                           np.float32)
        
        statManager.registerSampleCallback("rho", 
                                           lambda: self.rho,
                                           (1,),
                                           np.float32)
        

class LatentDistanceModel(GraphModelExtension):
    """
    Prior for a latent distance model of connectivity. Each process is given 
    a location in some latent space, and the probability of connectivity is 
    exponentially decreasing with distance between two processes. 
    """
    def __init__(self, baseModel, configFile):
        super(LatentDistanceModel,self).__init__(baseModel, configFile)
        
        self.parseConfigurationFile(configFile)
        pprintDict(self.params, "Graph Model Params")
        
        self.params["registered"] = False 
        
    def parseConfigurationFile(self, configFile):
        
        # Set defaults
        defaultParams = {}
        defaultParams["thin"] = 50
        
        # Parse config file
        cfgParser = ConfigParser(defaultParams)
        cfgParser.read(configFile)
        
        distparams = {}
        distparams["location_name"] = cfgParser.get("graph_prior", "location")
        distparams["mu_theta"] = cfgParser.getfloat("graph_prior", "mu_theta")
        distparams["sig_theta"] = cfgParser.getfloat("graph_prior", "sig_theta")
        distparams["thin"] = cfgParser.getint("graph_prior", "thin")
        distparams["rho"] = cfgParser.getfloat("graph_prior", "rho")
        
        # Combine the two param dicts
        self.params.update(distparams)
        
    def initializeModelParamsFromPrior(self):
        self.register_providers()
        
        self.initializeGpuMemory()
        
        K = self.modelParams["proc_id_model","K"]
        
        # Initialize tau with draw from exponential prior
        self.modelParams["graph_model", "tau"] = np.exp(np.random.normal(self.params["mu_theta"],
                                                                         self.params["sig_theta"]))
        
        # Initialize the pairwise distance matrix
        self.computeDistanceMatrix()
        
        # Override the base model's default adjacency matrix with a draw from the prior
        self.modelParams["graph_model","A"] = self.sampleGraphFromPrior()
        self.gpuPtrs["graph_model","A"].set(self.modelParams["graph_model","A"])
        
        self.iter = 0
    
    def initializeModelParamsFromDict(self, paramsDB):
        self.register_providers()
        self.initializeGpuMemory()
        self.modelParams["graph_model","A"] = paramsDB["graph_model","A"]
        self.gpuPtrs["graph_model","A"].set(self.modelParams["graph_model","A"])
        
        self.modelParams["graph_model", "tau"] = np.float(paramsDB["graph_model","tau"])
        
        self.computeDistanceMatrix()
        
    def register_providers(self):
        """
        Register the cluster and location providers.
        """
        if self.params["registered"]:
            return
        
        # Now find the correct location model
        location = None
        location_name = self.params["location_name"]
        location_list = self.base.extensions["location_model"]
        location_list = location_list if isinstance(location_list, type([])) else [location_list]
        for location_model in location_list:
            if location_model.name == location_name:
                # Found the location model!
                location = location_model
        if location is None:
            raise Exception("Failed to find location model '%s' in extensions!" % location_name)
            
        self.params["location"] = location
        
        # Add the location callback
        self.params["location"].register_consumer(self.compute_log_lkhd_new_location)
        
        self.params["registered"] = True
    
    
    def computeDistanceMatrix(self):
        """
        compute the pairwise distances between each process
        """
        K = self.modelParams["proc_id_model","K"]
        L = self.modelParams[self.params["location"].name, "L"]
        
        dist = np.zeros((K,K))
        for i in np.arange(K):
            for j in np.arange(i+1,K):
                d_ij = np.linalg.norm(L[i,:]-L[j,:], 2)
                dist[i,j] = d_ij
                dist[j,i] = d_ij
                
        self.modelParams["graph_model", "dist"] = dist
    
    def getConditionalEdgePr(self, ki, kj):
        """
        Return the conditional probability of edge A[ki,kj] given the model
        parameters.  
        """
        if not self.params["allow_self_excitation"] and ki==kj:
            return 0.0
        
        else:
            return self.params["rho"]*np.exp(-1/self.modelParams["graph_model","tau"]*self.modelParams["graph_model","dist"][ki,kj])
    
    def sampleModelParameters(self):
        """
        Sample process locations and edges in adjacency matrix
        """
        self.computeDistanceMatrix()
        if np.mod(self.iter, self.params["thin"]) == 0:
            self.sampleTau()
        
        # Sample a new graph on every iteration
        self.sampleA()
        
        self.iter += 1
    
    def sampleTau(self):
        """
        Sample tau using Hybrid Monte Carlo. The log likelihood is a function of the 
        current graph and the distances between connected and disconnected nodes.
        """
        # Set HMC params
        epsilon = 0.001
        n_steps = 10
        
        # By convention hmc minimizes the negative log likelihood,
        # so negate the logprob and gradient calculations
        theta_new = hmc(lambda t: -1.0*self.computeLogProbTau(t), 
                        lambda t: -1.0*self.computeGradLogProbTau(t), 
                        epsilon,
                        n_steps,
                        np.log(self.modelParams["graph_model", "tau"]))
        
        tau_new = np.exp(theta_new)
        
        self.modelParams["graph_model","tau"] = tau_new
    
    def computeLogProbTau(self, theta):
        """
        Compute the log likelihood of the current graph given theta = log(tau)
        """
        tau = np.exp(theta)
        
        K = self.modelParams["proc_id_model", "K"]
        
        # Get the distances between connected neurons and bw disconnected neurons
        # Ignore the diagonal since the distances are equal to zero
        dist_conn = self.modelParams["graph_model", "dist"][self.modelParams["graph_model", "A"]]
        dist_conn = dist_conn[dist_conn>0]
        N_conn = np.size(dist_conn)
        dist_noconn = self.modelParams["graph_model", "dist"][np.bitwise_not(self.modelParams["graph_model", "A"])]
        dist_noconn = dist_noconn[dist_noconn>0]
        N_noconn = np.size(dist_noconn)
        
        # Compute the logprob
        lpr = 0.0
        if N_conn > 0:
            lpr += -1.0*np.sum(dist_conn)/tau
        if N_noconn > 0:
            lpr += np.sum(np.log(1-self.params["rho"]*np.exp(-dist_noconn/tau))) 
        
        # Contribution from prior
        lpr += -0.5*(theta - self.params["mu_theta"])**2/self.params["sig_theta"]**2
        
        return lpr
    
    def computeGradLogProbTau(self, theta):
        """
        Compute the gradient of the log likelihood wrt theta = log(tau)
        """
        tau = np.exp(theta)
        K = self.modelParams["proc_id_model", "K"]
        
        # Get the distances between connected neurons and bw disconnected neurons
        dist_conn = self.modelParams["graph_model", "dist"][self.modelParams["graph_model", "A"]]
        dist_conn = dist_conn[dist_conn>0]
#        N_conn = np.size(dist_conn)
        dist_noconn = self.modelParams["graph_model", "dist"][np.bitwise_not(self.modelParams["graph_model", "A"])]
        dist_noconn = dist_noconn[dist_noconn>0]
#        N_noconn = K**2- N_conn
        
        grad_lpr = 0.0
        grad_lpr += np.sum(dist_conn)/(tau**2) 
        
        try:
            grad_lpr += np.sum(dist_noconn/(tau**2)*(-1.0*self.params["rho"]*np.exp(-dist_noconn/tau))/(1-self.params["rho"]*np.exp(-dist_noconn/tau)))
        except Exception as e:
            # Catch FloatingPointErrors
            log.error("Caught FloatingPointError (underflow?) in GradLogProbTau")
            log.info(dist_noconn)
            log.info(tau)
            
        # The above derivative is with respect to tau. Multiply by dtau/dtheta
        grad_lpr *= np.exp(theta)
        
        # Add the gradient of the prior over theta
        grad_lpr += -(theta-self.params["mu_theta"])/(self.params["sig_theta"]**2)
        
        return grad_lpr
        
    def compute_log_lkhd_new_location(self, k, Lk):
        """
        Compute the log likelihood of A given X[k,:] = x 
        This affects edges into and out of process k
        """
        A = self.modelParams["graph_model", "A"]
        K = self.modelParams["proc_id_model","K"]
        L = self.modelParams[self.params["location"].name, "L"]
        tau = self.modelParams["graph_model","tau"]
        
        # Compute the updated distance vector from k to other nodes
        dist_k = np.zeros((K,))
        for j in np.arange(K):
            if j != k:
                dist_k[j] = np.linalg.norm(Lk-L[j,:], 2)
        
        # Compute the log likelihood
        try:
            ll = 0
            for j in np.arange(K):
                if j != k:
                    ll += (A[j,k]+A[k,j])*(-1/tau*dist_k[j])
                    
                    if dist_k[j] == 0:
                        # If the distance is zero then there must be an edge
                        if not A[j,k] or not A[k,j]:
                            ll = -np.Inf
                    else:   
                        ll += (2-A[j,k]-A[k,j])*np.log(1-np.exp(-1/tau*dist_k[j]))
                    
        except Exception as e:
            log.info("compute_lkhd_(%d,%s)", k, str(Lk))
            log.info("L")
            log.info(L)
            log.info("dist")     
            log.info(dist_k)
            log.info("tau: %f", tau)
            raise e
        return ll
    
    def sampleGraphFromPrior(self):
        """
        Sample a graph from the prior, assuming model params have been set.
        """
        K = self.modelParams["proc_id_model","K"]
        dist = self.modelParams["graph_model","dist"]
        tau = self.modelParams["graph_model","tau"]
        
        A = np.random.rand(K, K) < np.exp(-1/tau*dist)
        return A
    
    def registerStatManager(self, statManager):
        """
        Register callbacks with the given StatManager
        """
        super(LatentDistanceModel,self).registerStatManager(statManager)
        
        K = self.modelParams["proc_id_model","K"]
        
        statManager.registerSampleCallback("A_tau", 
                                           lambda: self.modelParams["graph_model","tau"],
                                           (1,),
                                           np.float32)
        
