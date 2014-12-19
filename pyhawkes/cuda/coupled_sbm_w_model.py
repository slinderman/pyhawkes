import numpy as np
import logging

import os
import sys

import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray

from pyhawkes.utils.utils import pprint_dict, compile_kernels, log_sum_exp_sample
from pyhawkes.utils.elliptical_slice import elliptical_slice

from graph_model_extension import GraphModelExtension

from ConfigParser import ConfigParser

log = logging.getLogger("global_log")

class StochasticBlockModelCoupledWithW(GraphModelExtension):
    """
    Prior for a stochastic block model of connectivity. Couple this graph 
    prior with the prior on W. The mean of the weights is a separate for 
    each pair of blocks.
    """
    def __init__(self, baseModel, configFile):
        super(StochasticBlockModelCoupledWithW,self).__init__(baseModel, configFile)
        
        # Initialize databases for this extension
        self.modelParams.addDatabase("weight_model")
        self.gpuPtrs.addDatabase("weight_model")
        
        # Load the GPU kernels necessary for background rate inference
        # Allocate memory on GPU for background rate inference
        # Initialize host params        
        self.parseConfigurationFile(configFile)
        pprint_dict(self.params, "Weight and Graph Model Params")
        
        self.initializeGpuKernels()
    
    def parseConfigurationFile(self, configFile):
        
        # Set defaults
        defaultParams = {}
        defaultParams["R"] = 2
        defaultParams["b0"] = 1.0
        defaultParams["b1"] = 1.0
        defaultParams["rho"] = 0.1
        defaultParams["a_w"] = 2.0
        defaultParams["thin"] = 10
        defaultParams["Y_given"] = 0
        defaultParams["thin"] = 1
        defaultParams["allow_self_excitation"] = 0
        
        # Parse config file
        cfgParser = ConfigParser(defaultParams)
        cfgParser.read(configFile)
        
        sbmw_params = {}
        sbmw_params["R"] = cfgParser.getint("graph_prior", "R")
        sbmw_params["b0"] = cfgParser.getfloat("graph_prior", "b0")
        sbmw_params["b1"] = cfgParser.getfloat("graph_prior", "b1")
        sbmw_params["rho"] = cfgParser.getfloat("graph_prior", "rho")
        sbmw_params["a_w"] = cfgParser.getfloat("weight_prior", "a")
        sbmw_params["allow_self_excitation"]    = bool(cfgParser.getint("graph_prior", "allow_self_excitation"))
        
        # If alpha is given we will not sample it
        if cfgParser.has_option("graph_prior", "alpha"):
            log.debug("alpha is given. Will not sample.")
        
            # Parse the alpha parameter
            alpha_str = cfgParser.get("graph_prior", "alpha")
            alpha_list = alpha_str.split(',')
            sbmw_params["alpha"] = np.array(map(float, alpha_list))
            sbmw_params["alpha_given"] = True
            
        elif cfgParser.has_option("graph_prior", "alpha0"):
            log.debug("alpha0 prior specified.")
            # Parse the prior on alpha
            alpha_str = cfgParser.get("graph_prior", "alpha0")
            alpha_list = alpha_str.split(',')
            sbmw_params["alpha0"] = np.array(map(float, alpha_list))
            sbmw_params["alpha_given"] = False
        else:
            log.debug("Neither alpha nor alpha0 was specified. Using uniform prior.")
            sbmw_params["alpha0"] = 1.0/sbmw_params["R"]*np.ones(sbmw_params["R"])
            sbmw_params["alpha_given"] = False
        
        sbmw_params["cu_dir"] = cfgParser.get("weight_prior", "cu_dir")
        sbmw_params["cu_file"] = cfgParser.get("weight_prior", "cu_file")
        sbmw_params["Y_given"] = bool(cfgParser.getint("weight_prior", "Y_given"))
        sbmw_params["thin"] = cfgParser.getint("weight_prior", "thin")
        sbmw_params["blockSz"] = cfgParser.getint("cuda", "blockSz")
        sbmw_params["numThreadsPerGammaRV"] = cfgParser.getint("cuda", "numThreadsPerGammaRV")
        
        # Set the initial weight scale to something stable     
        rho = sbmw_params["rho"]   
        rho *= sbmw_params["b0"]/ (sbmw_params["b0"]+sbmw_params["b1"])
        
        sbmw_params["b_w"] = self.base.data.K * sbmw_params["a_w"] * rho / 0.7
        log.info("Set b_w=%f based on number of processes and specified a_w and SBM params", sbmw_params["b_w"])
        
        # Update param dict
        self.params.update(sbmw_params)
    
#    def initializeGpuKernels(self):
#        kernelSrc = os.path.join(self.params["cu_dir"], self.params["cu_file"])
#        
#        kernelNames = ["computeLogLkhdWPerSpike",
#                       "computeTrapIntWPerKnot"]
#        
#        
#        src_consts = {"B" : self.params["blockSz"],
#                      "K" : int(self.base.data.K)}
#        
#        # Before compiling, make sure utils.cu is in the sys path
#        sys.path.append(self.params["cu_dir"])
#        self.gpuKernels = compileKernels(kernelSrc, kernelNames, src_consts)
    def initializeGpuKernels(self):
        kernelSrc = os.path.join(self.params["cu_dir"], self.params["cu_file"])
        
        kernelNames = ["sumNnzZPerBlock",
                       "computeWPosterior",
                       "sampleGammaRV"]
        
        src_consts = {"B" : self.params["blockSz"]}
        
        # Before compiling, make sure utils.cu is in the sys path
        sys.path.append(self.params["cu_dir"])
        sbmwGpuKernels = compile_kernels(kernelSrc, kernelNames, src_consts)
        
        # Update kernel list
        self.gpuKernels.update(sbmwGpuKernels)
        
    def initializeGpuMemory(self):
        super(StochasticBlockModelCoupledWithW,self).initializeGpuMemory()
        
        K = self.modelParams["proc_id_model", "K"]
        
        # Allocate space for current weight matrix
        self.gpuPtrs["weight_model","W"] = gpuarray.empty((K,K), dtype=np.float32)

        self.gpuPtrs["weight_model","nnz_Z_gpu"]   = gpuarray.empty((K,K), dtype=np.int32)
        self.gpuPtrs["weight_model","aW_post_gpu"] = gpuarray.empty((K,K), dtype=np.float32)
        self.gpuPtrs["weight_model","bW_post_gpu"] = gpuarray.empty((K,K), dtype=np.float32)
        
        self.gpuPtrs["weight_model","urand_KxK_gpu"] = gpuarray.empty((K,K,self.params["numThreadsPerGammaRV"]), dtype=np.float32)
        self.gpuPtrs["weight_model","nrand_KxK_gpu"] = gpuarray.empty((K,K,self.params["numThreadsPerGammaRV"]), dtype=np.float32)
        
        # Result of GPU gamma sampling
        self.gpuPtrs["weight_model","sample_status_gpu"] = gpuarray.empty((K,K), dtype=np.int32)
        
       
    def initializeModelParamsFromPrior(self):
        self.initializeGpuMemory()
        K = self.modelParams["proc_id_model", "K"]
        R = self.params["R"]
        
        # There is a multinomial distribution over each process's block identity
        # It has a beta prior
        if self.params["alpha_given"]:
            self.modelParams["graph_model","alpha"] = self.params["alpha"]
            # Make sure alpha is normalized
            self.modelParams["graph_model","alpha"] /= np.sum(self.modelParams["graph_model","alpha"])
        else:
            self.modelParams["graph_model","alpha"] = np.random.dirichlet(self.params["alpha0"]).astype(np.float32)
        
        
        # Each process 1:K is endowed with a latent block 1:R. Let Y be a vector
        # representing the block identity for each process.
        # Check if spatial locations are given
        if self.params["Y_given"]:
            if "ctype" not in self.base.data.other_data.keys():
                log.error("Cell type not specified!")
                exit()
            ctype = self.base.data.other_data["ctype"]
            log.debug("Using provided cell types as block IDs:")
            log.debug(ctype)
            assert np.size(ctype) == K
            self.modelParams["graph_model","Y"] = np.ravel(ctype)
            self.params["Y_given"] = True
        else:
            self.modelParams["graph_model","Y"] = np.zeros((K,), dtype=np.int32)
            self.params["Y_given"] = False
            # Each process 1:K is assigned to a latent block 1:R. Let Y be a vector
            # representing the block identity for each process.
            for k in np.arange(self.modelParams["proc_id_model", "K"]):
                p = np.random.rand()
                acc = 0.0
                for r in np.arange(self.params["R"]):
                    acc += self.modelParams["graph_model","alpha"][r]
                    if p <= acc:
                        self.modelParams["graph_model","Y"][k] = r
                        break
                    
        Y = self.modelParams["graph_model","Y"]
        
        # To model sparsity in the adjacency matrix we introduce a mixture with a 
        # Bernoulli random graph with sparsity rho
        self.modelParams["graph_model","Nu"] = np.random.rand(K,K) < self.params["rho"]
        
        # There is a matrix B representing the Bernoulli probability of an edge between 
        # processes on two blocks. It has a Beta prior.
        self.modelParams["graph_model","B"] = np.random.beta(self.params["b0"], self.params["b1"], (R,R)).astype(np.float32)
        
        # Sample a matrix of weight scale parameters
        self.modelParams["weight_model","beta_W"] = self.params["b_w"] * np.ones((R,R))
        
        # Sample a weight matrix given mu and the current block assignments
        W0 = np.random.gamma(self.params["a_w"],
                              1.0/self.params["b_w"], 
                              size=(K,K)).astype(np.float32)
        
        self.modelParams["weight_model","W"] = W0
        self.gpuPtrs["weight_model","W"].set(W0)
        
        # Draw a random graph from the prior
        # Sample adjacency matrix
        self.modelParams["graph_model","A"] = np.zeros((K,K), dtype=np.bool)
        for k1 in np.arange(K):
            for k2 in np.arange(K):
                if self.modelParams["graph_model","Nu"][k1,k2]:
                    self.modelParams["graph_model","A"][k1,k2] = np.random.rand() < self.modelParams["graph_model","B"][Y[k1],Y[k2]]
    
        self.gpuPtrs["graph_model","A"].set(self.modelParams["graph_model","A"])
        
        self.iter= 0
        
    def initializeModelParamsFromDict(self, paramsDB):
        self.initializeGpuMemory()
        self.modelParams["weight_model","W"] = paramsDB["weight_model","W"]
        self.gpuPtrs["weight_model","W"].set(paramsDB["weight_model","W"])
        
        self.modelParams["graph_model","A"] = paramsDB["graph_model","A"]
        self.gpuPtrs["graph_model","A"].set(self.modelParams["graph_model","A"])
        
        self.modelParams["graph_model","B"] = paramsDB["graph_model","B"] 
        
        self.modelParams["graph_model","alpha"] = paramsDB["graph_model","alpha"]
        
        self.modelParams["weight_model","beta_W"]  = paramsDB["weight_model","beta_W"]
        
        self.modelParams["graph_model","Y"] = paramsDB["graph_model","Y"]
    
    
    def computeLogProbability(self):
        """
        The log probability is the sum of individual edge log probs
        """
        A = self.modelParams["graph_model","A"]
        nnz_A = sum(sum(A))
        K = self.modelParams["proc_id_model", "K"]
        
        ll = 0.0
        
        # Log likelihood of alpha
        ll += np.sum((self.params["alpha0"]-1)*np.log(self.modelParams["graph_model","alpha"]))
        
        # Log likelihood of B
        B_flat = np.ravel(self.modelParams["graph_model","B"])
        ll += np.sum((self.params["b0"]-1)*np.log(B_flat) + (self.params["b1"]-1)*np.log(1-B_flat))
        
        # Log likelihood of Y
        for k in np.arange(K):
            ll += np.log(self.modelParams["graph_model","alpha"][self.modelParams["graph_model","Y"][k]])
        
        # Log likelihood of A
        for k1 in np.arange(K):
            for k2 in np.arange(K):
                if A[k1,k2]:
                    # Edge present implies Nu[k1,k2] = 1 
                    ll += np.log(self.params["rho"]) + np.log(self.modelParams["graph_model","B"][self.modelParams["graph_model","Y"][k1],self.modelParams["graph_model","Y"][k2]])
                elif self.modelParams["graph_model","Nu"][k1,k2]:
                    # Edge was possible but did not occur because B was too low
                    ll += np.log(self.params["rho"]) + np.log(1-self.modelParams["graph_model","B"][self.modelParams["graph_model","Y"][k1],self.modelParams["graph_model","Y"][k2]])
                else:
                    # Edge was not possible due to Nu
                    ll += np.log(1-self.params["rho"])
        
        # log likelihood of mu
        mu_flat = np.ravel(self.modelParams["graph_model","mu"])
        ll += np.sum(1/self.params["sigma_mu"]**2 * (mu_flat - self.params["mu_mu"])**2)
        
        return ll 
        
    def getConditionalEdgePr(self, ki, kj):
        """
        Return the conditional probability of edge A[ki,kj] given the model
        parameters. In this case the probability is simply rho. 
        """
        # The probability of an edge is simply the entry in B corresponding to the 
        # block identities for processes ki and kj, unless we disallow self edges
        if not self.params["allow_self_excitation"] and ki==kj:
            return 0.0
        else:
            Y = self.modelParams["graph_model","Y"]
            return self.params["rho"] * self.modelParams["graph_model","B"][Y[ki], Y[kj]]
    
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
            self.sampleNu()
            
            # Sample Y -- the block identities - given the rest of the parameters
            if not self.params["Y_given"]:
                self.sampleY()
            
            # Sample B - the Bernoulli matrix representing edge probabilities between
            # each pair of blocks
            self.sampleB()
            
            # Sample alpha - the multinomial distribution over block membership
            if not self.params["alpha_given"]:
                self.sampleAlpha()
            
            # Sample beta_W - the scale of the block-pair weight matrix
            self.sampleBetaW()
            
            # Sample W - the weight matrix
            self.sampleW()
            
        # Sample A on every iteration
        self.sampleA()
        
        self.iter += 1
    
    def sampleW(self):
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

        #startPerfTimer(perfDict, "sample_W")
        # Compute the posterior parameters for W's
        
        grid_sz = int(np.ceil(float(K)/32))
        self.gpuKernels["computeWPosterior"](np.int32(K),
                                             self.gpuPtrs["weight_model","nnz_Z_gpu"].gpudata,
                                              self.gpuPtrs["proc_id_model","Ns"].gpudata,
                                              self.gpuPtrs["graph_model","A"].gpudata,
                                              np.float32(self.params["a_w"]),
                                              np.float32(0.0),
                                              self.gpuPtrs["weight_model","aW_post_gpu"].gpudata,
                                              self.gpuPtrs["weight_model","bW_post_gpu"].gpudata,
                                              block=(32,32,1),
                                              grid=(grid_sz,grid_sz)
                                              )
        
        # Update b_w for each entry with the value from the prior
        Y = self.modelParams["graph_model","Y"]
        beta_W = self.modelParams["weight_model", "beta_W"][np.ix_(Y,Y)].astype(np.float32)
        self.gpuPtrs["weight_model","bW_post_gpu"].set(self.gpuPtrs["weight_model","bW_post_gpu"].get() + beta_W)
        
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
        
    def sampleBetaW(self):
        """
        Sample beta_W - the scale of the weights for each pair of blocks
        Since W is gamma distributed, the posterior scale will be
        gamma distributed under a Jeffrey's prior. Only condition on the 
        weights with corresponding edges since the others are free parameters. 
        Conditioning on them causes too much momentum toward bad states. 
        """
        Y = self.modelParams["graph_model","Y"]
        for r1 in np.arange(self.params["R"]):
            for r2 in np.arange(self.params["R"]):
                Wr1r2 = self.modelParams["weight_model","W"][np.ix_(Y==r1,Y==r2)].ravel()
                Ar1r2 = self.modelParams["graph_model","A"][np.ix_(Y==r1,Y==r2)].ravel()
                
                Wr1r2 = Wr1r2[Ar1r2]
                nr1r2 = np.size(Wr1r2)
                
                # Posterior is Jeffrey's distributed
                a = nr1r2 * self.params["a_w"]
                b = np.sum(Wr1r2)
                
                # If there are no edges between this pair of blocks, set the weight scale to default
                if a==0:
                    self.modelParams["weight_model", "beta_W"][r1,r2] = self.params["b_w"]  
                else:
                    self.modelParams["weight_model", "beta_W"][r1,r2] = np.random.gamma(a,1.0/b)
                        
    def sampleNu(self):
        """
        Sample Nu conditioned upon all other model params
        """
        K = self.modelParams["proc_id_model", "K"]
        A = self.modelParams["graph_model","A"]
        self.modelParams["graph_model","Nu"] = np.zeros((K,K), dtype=np.bool)
        for k1 in np.arange(K):
            for k2 in np.arange(K):
#                self.modelParams["graph_model","Nu"][A] = True
#                pNu = (1-self.modelParams["graph_model","B"][np.ix_(self.modelParams["graph_model","Y"],self.modelParams["graph_model","Y"])])*self.params["rho"] / ((1-self.modelParams["graph_model","B"][np.ix_(self.modelParams["graph_model","Y"],self.modelParams["graph_model","Y"])])*self.params["rho"] + (1-self.params["rho"]))
#                self.modelParams["graph_model","Nu"][1-A] = (np.random.rand(K,K) < pNu)[1-A]
                if A[k1,k2]:
                    self.modelParams["graph_model","Nu"][k1,k2] = True
                else:
                    # Probability the edge was possible is the probability the edge
                    # was not present due to B over the total probability. It is more likely
                    # for Nu to be 1 when the corresponding entry in B is large
                    pNu = (1-self.modelParams["graph_model","B"][self.modelParams["graph_model","Y"][k1],self.modelParams["graph_model","Y"][k2]])*self.params["rho"] / ((1-self.modelParams["graph_model","B"][self.modelParams["graph_model","Y"][k1],self.modelParams["graph_model","Y"][k2]])*self.params["rho"] + (1-self.params["rho"]))
                    
                    self.modelParams["graph_model","Nu"][k1,k2] = np.random.rand() < pNu
    
    def logPrWGivenYk(self, k, r):
        """
        Compute the log-probability of W[k,:] and W[:,k] for a specific entry Y[k]
        W is gamma distributed with individual scales for each block
        """
        logPrWin = np.zeros(self.modelParams["proc_id_model", "K"])
        logPrWout = np.zeros(self.modelParams["proc_id_model", "K"])
        
        Yr = self.modelParams["graph_model","Y"]
        Yr[k] = r 
        
        A = self.modelParams["graph_model", "A"]
        W_in = self.modelParams["weight_model", "W"][:,k][A[:,k]]
        b_w_in = self.modelParams["weight_model","beta_W"][Yr,r][A[:,k]]

        W_out = self.modelParams["weight_model", "W"][k,:][A[k,:]]
        b_w_out = self.modelParams["weight_model","beta_W"][r,Yr][A[k,:]]
        
        # Add probabilities from incoming and outgoing weights
        a_w = self.params["a_w"]        
        logPrW_in = a_w*np.log(b_w_in) + (a_w-1)*np.log(W_in) - b_w_in * W_in
        logPrW_out = a_w*np.log(b_w_out) + (a_w-1)*np.log(W_out) - b_w_out * W_out
        return np.sum(logPrW_in) + np.sum(logPrW_out)
    
    def sampleY(self):
        """
        Gibbs sample Y conditioned upon all other model params
        """
        Y = self.modelParams["graph_model","Y"]
        A = self.modelParams["graph_model","A"]
        for k in np.arange(self.modelParams["proc_id_model", "K"]):
            
            # The prior distribution is simply alpha
            ln_pYk = np.log(self.modelParams["graph_model","alpha"])
            
            # likelihood of block membership is affected by the presence and absence
            # of edges, both incoming and outgoing
            for r in np.arange(self.params["R"]):
                # Add the log prob of W given Yi=r
                ln_pYk[r] += self.logPrWGivenYk(k,r)
                
                # Block IDs of nodes we connect to
                o1 = np.bitwise_and(A[k,:], self.modelParams["graph_model","Nu"][k,:]) 
                if np.any(o1):
                    ln_pYk[r] += np.sum(np.log(self.modelParams["graph_model","B"][np.ix_([r],Y[o1])]))
                
                # Block IDs of nodes we don't connect to
                o2 = np.bitwise_and(np.logical_not(A[k,:]), self.modelParams["graph_model","Nu"][k,:])
                if np.any(o2):
                    ln_pYk[r] += np.sum(np.log(1-self.modelParams["graph_model","B"][np.ix_([r],Y[o2])]))
                
                # Block IDs of nodes that connect to us
                i1 = np.bitwise_and(A[:,k], self.modelParams["graph_model","Nu"][:,k])
                if np.any(i1):
                    ln_pYk[r] += np.sum(np.log(self.modelParams["graph_model","B"][np.ix_(Y[i1],[r])]))
    
                # Block IDs of nodes that do not connect to us
                i2 = np.bitwise_and(np.logical_not(A[:,k]), self.modelParams["graph_model","Nu"][:,k])
                if np.any(i2):
                    ln_pYk[r] += np.sum(np.log(1-self.modelParams["graph_model","B"][np.ix_(Y[i2],[r])]))
            
            try:
                Y[k] = log_sum_exp_sample(ln_pYk)
            except:
                exit()
                
    def sampleB(self):
        Y = self.modelParams["graph_model","Y"]
        A = self.modelParams["graph_model","A"]
        
        for r1 in np.arange(self.params["R"]):
            for r2 in np.arange(self.params["R"]):
                b0star = self.params["b0"]
                b1star = self.params["b1"]
                
                Ar1r2 = A[np.ix_(Y==r1, Y==r2)]
                Nur1r2 = self.modelParams["graph_model","Nu"][np.ix_(Y==r1, Y==r2)]
                if np.size(Ar1r2) > 0:
                    b0star += np.sum(Ar1r2[Nur1r2])
                    b1star += np.sum(1-Ar1r2[Nur1r2])
                
                self.modelParams["graph_model","B"][r1,r2] = np.random.beta(b0star, b1star)
                
    def sampleAlpha(self):
        A = self.modelParams["graph_model","A"]
        alpha_star = np.zeros((self.params["R"],), dtype=np.float32)
        for r in np.arange(self.params["R"]):
            alpha_star[r] = self.params["alpha0"][r] + np.sum(self.modelParams["graph_model","Y"]==r)
        
        self.modelParams["graph_model","alpha"] = np.random.dirichlet(alpha_star).astype(np.float32)
    
    def registerStatManager(self, statManager):
        """
        Register callbacks with the given StatManager
        """
        K = self.modelParams["proc_id_model","K"]
        super(StochasticBlockModelCoupledWithW,self).registerStatManager(statManager)
        
        statManager.registerSampleCallback("Y", 
                                           lambda: self.modelParams["graph_model","Y"],
                                           (K,),
                                           np.int32)
        
        statManager.registerSampleCallback("B", 
                                           lambda: self.modelParams["graph_model","B"],
                                           (self.params["R"],self.params["R"]),
                                           np.float32)
        
        statManager.registerSampleCallback("alpha", 
                                           lambda: self.modelParams["graph_model","alpha"],
                                           (self.params["R"],),
                                           np.float32)
        
        statManager.registerSampleCallback("rho", 
                                           lambda: self.params["rho"],
                                           (1,),
                                           np.float32)
        
        statManager.registerSampleCallback("beta_W", 
                                           lambda: self.modelParams["weight_model","beta_W"],
                                           (self.params["R"],self.params["R"]),
                                           np.float32)
        
        
        statManager.registerSampleCallback("W", 
                                           lambda: self.gpuPtrs["weight_model","W"].get(),
                                           (K,K),
                                           np.float32)
        