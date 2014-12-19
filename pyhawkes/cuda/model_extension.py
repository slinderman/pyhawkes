"""
Define an interface for model extensions
"""

class ModelExtension(object):
    def getInitializationOrder(self):
        """
        Get the initialization order. Sorted 0...inf. Lower is earlier
        """
        return 10
    
    def initializeModelParamsFromPrior(self):
        """
        Set the model params to a random draw from the prior
        """
        pass
    
    def initializeModelParamsFromDict(self, paramsDB):
        """
        Set the model params to the values given in paramsDB. 
        This is useful for initializing a model for prediction
        after the learning phase is over. Note that latent variables
        associated with the data cannot be loaded since they 
        correspond to the data.
        """
        pass
    
    def sampleNewProcessParams(self, newProcParams):
        """
        The number of processes may change during the learning phase.
        In order to determine whether or not to add a new process we
        may need params associated with the various model extension.
        The extension must populate the newProcParams dict with a 
        dictionary containing params corresponding to the extension.
        """
        pass
     
    def addNewProcessEventHandler(self, newProcParams):
        """
        If a process is added, update the extension with the sampled
        params.
        """
        pass
    
    def removeProcessEventHandler(self, procId):
        """
        Remove the specified process and its associated params.
        """
        pass
    
    def computeLogProbability(self):
        """
        Compute the log probability associated with the current extension
        parameters under their prior.
        """
        return 0.0
    
    def sampleModelParameters(self):
        """
        During the learning phase the Gibbs sampling loop will call each
        model extension and ask it to sample parameters conditioned upon
        the data and other param settings. 
        """
        pass
    
    def sampleLatentVariables(self):
        """
        Sample the latent variables of the model. This is called separately 
        since only latent variables are sampled in predictive models.
        """
        pass
    
    def registerStatManager(self, statManager):
        """
        Register callbacks with the statistics manager
        """
        pass
     
    def gradient(self, params, k=None):
        return None
 
    def set_params(self, params, k=None):
        pass
    
    def get_params(self, k=None):
        return None
    
    