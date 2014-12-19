"""
Class to handle stat collection
"""
import os
import numpy as np
import scipy.io
import ConfigParser
from utils import get_unique_file_name

import logging
log = logging.getLogger("global_log")


class StatManager:
    def __init__(self, configFile, name="", resultsFile=None):
        self.parseConfigurationFile(configFile)
        if not resultsFile is None:
            self.params["results_file"] = resultsFile
            
        self.name = name
        
        self.burnin_smpls = {}
        self.burnin_iter = 0
        
        self.samples = {}
        self.sample_iter = 0
        
        self.handles = {}
        self.unstructured_handles = {}
        
    def parseConfigurationFile(self, configFile):
        # Initialize defaults
        defaultParams = {}
        
        # We have the option of collecting stats during burnin
        # TODO: RENAME THESE
        defaultParams["collect_burnin_stats_intvl"] = 100
        defaultParams["results_dir"] = "."
        defaultParams["burnin"] = 10000
        defaultParams["samples"] = 1000
        defaultParams["thin"] = 10
        defaultParams["restarts"] = 1
        
        # Create a config parser object and read in the file
        cfgParser = ConfigParser(defaultParams)
        cfgParser.read(configFile)
        
        self.params = {}
        self.params["results_dir"]  = cfgParser.get("io", "results_dir")
        self.params["results_file"] = cfgParser.get("io", "results_file")
        
        self.params["samples"]      = cfgParser.getint("MCMC", "samples")
        self.params["samples_thin"] = cfgParser.getint("MCMC", "thin")
        self.params["burnin"]       = cfgParser.getint("MCMC", "burnin")
        self.params["burnin_thin"]  = cfgParser.getint("output", "collect_burnin_stats_intvl")
        self.params["restarts"]     = cfgParser.getint("MCMC", "restarts")
        
        # Update N_samples and N_burnin
        self.params["N_samples"] = self.params["samples"] * self.params["restarts"]
        self.params["N_burnin"] =  np.ceil(self.params["burnin"] / self.params["burnin_thin"]) * self.params["restarts"]  
                
        # Basic error checking
        if not os.path.exists(self.params["results_dir"]):
            log.error("Results directory does not exist!")
            raise Exception("Invalid parameter: io.results_dir %s" % self.params["results_dir"])
        
    def registerSampleCallback(self, name, callback, shape, dtype, burnin=False, postburnin=True):
        """
        Register a callback function to be called during stat collection. 
        the function should return a single sample of specified shape and dtype
        If burnin is specified then this will also be collected during burnin
        If postburnin is True this will be collected during the post-burnin phase.
        
        This overwrites existing callbacks with the same name
        """
        
        if shape == None:
            self.__registerUnstructuredSampleCallback(name, callback, burnin, postburnin)
        else:
            self.__registerArraySampleCallback(name, callback, shape, dtype, burnin, postburnin)
        
    def __registerArraySampleCallback(self, name, callback, shape, dtype, burnin, postburnin):
        """
        Register a callback function which returns a numpy array
        """
        
        # Initialize storage space
        sz = np.prod(shape)
        if burnin:
            key = name + "_burnin"
            if key not in self.burnin_smpls.keys():
                self.burnin_smpls[key] = np.empty(sz*self.params["N_burnin"], dtype=dtype)
        if postburnin:
            key = name + "_smpls"
            if key not in self.samples.keys():
                self.samples[key] = np.empty(sz*self.params["N_samples"], dtype=dtype)
        
            
        self.handles[name] = (callback, shape, sz, dtype, burnin, postburnin)
        log.debug("Registered array callback %s:\n%s", name, str(self.handles[name]))
        
    def __registerUnstructuredSampleCallback(self, name, callback, burnin, postburnin):
        """
        Register a callback which returns an unstructured sample such as a list or dict
        of variable length
        """
        if burnin:
            key = name + "_burnin"
            self.burnin_smpls[key] = []
        if postburnin:
            key = name + "_smpls"
            self.samples[key] = []
        
        
        self.unstructured_handles[name] = (callback, burnin, postburnin)
        log.debug("Registered unstructured callback %s:\n%s", name, str(self.unstructured_handles[name]))
    
    def collectBurninSamples(self):
        """
        Collect samples during burnin period
        """
#        if self.burnin_iter > self.params["N_burnin"]:
#            log.error("ERROR: Attempted to collect more than presribed number of burnin samples!")
#            return
        
        self.__collect("burnin")
    
    def collectSamples(self):
        """
        Collect samples after burnin
        """
#        if self.sample_iter > self.params["N_samples"]:
#            log.error("ERROR: Attempted to collect more than presribed number of samples!")
#            return
        
        self.__collect("postburnin")
        
        
        
    def __collect(self, phase):
        """
        Internal method to collect samples
        """
        in_burnin = (phase=="burnin")
        in_postburnin = (phase=="postburnin")
        
        if in_burnin:
            if np.mod(self.burnin_iter, self.params["burnin_thin"]) == 0:
                iter = self.burnin_iter / self.params["burnin_thin"]
                for  (name,(callback, _, sz, _, burnin, _)) in self.handles.items():
                    if burnin:
                        # Call the function to get the sample
                        sample = callback()
                        # Save the sample
                        key = name + "_burnin"
                        if (iter+1)*sz < np.size(self.burnin_smpls[key]):
                            self.burnin_smpls[key][iter*sz:(iter+1)*sz] = np.reshape(sample, (sz,), order="F")
                        else:
                            # If we exceeded the buffer size, double it
                            self.burnin_smpls[key] = np.concatenate((self.burnin_smpls[key], 
                                                                    np.empty_like(self.burnin_smpls[key])))
                            # Now try again
                            self.burnin_smpls[key][iter*sz:(iter+1)*sz] = np.reshape(sample, (sz,), order="F")
                            
                # Save unstructured samples
                for  (name,(callback, burnin, _)) in self.unstructured_handles.items():
                    if burnin:
                        # Call the function to get the sample
                        sample = callback()
                        # Save the sample
                        key = name + "_burnin"
                        self.burnin_smpls[key].append(sample)
            self.burnin_iter += 1
                
        elif in_postburnin:
            if np.mod(self.sample_iter, self.params["samples_thin"]) == 0:
                iter = self.sample_iter / self.params["samples_thin"]
                for  (name,(callback, _, sz, _, _, postburnin)) in self.handles.items():
                    if postburnin:
                        # Call the function to get the sample
                        sample = callback()
                        # Save the sample
                        key = name + "_smpls"
                        
                        if (iter+1)*sz < np.size(self.samples[key]):
                            self.samples[key][iter*sz:(iter+1)*sz] = np.reshape(sample, (sz,), order="F")
                        else:
                            # If we exceeded the buffer size, double it
                            self.samples[key] = np.concatenate((self.samples[key], 
                                                                np.empty_like(self.samples[key])))
                            # Now try again
                            self.samples[key][iter*sz:(iter+1)*sz] = np.reshape(sample, (sz,), order="F")
                            
                        
                        
                # Save unstructured samples
                for  (name,(callback, _, postburnin)) in self.unstructured_handles.items():
                    if postburnin:
                        # Call the function to get the sample
                        sample = callback()
                        # Save the sample
                        key = name + "_smpls"
                        self.samples[key].append(sample)
            self.sample_iter += 1
                
    def getSamples(self, name):
        """
        Get the list or array of samples stored under given name
        """
        key = name + "_smpls"
        return self.samples[key]
    
    def setSingleSample(self, name, value):
        """
        Set the value of a single sample. 
        """
        self.samples[name] = value
    
    def save_stats(self):
        """
        Save a dictionary of results as a mat file. Only saves the numpy array samples.
        """
        # reshape the samples
        results = {"N_samples" : self.sample_iter,
                   "N_burnin" : self.burnin_iter}
        for (name,(_,shape,_,_,burnin,postburnin)) in self.handles.items():
            
            if burnin:
                key = name + "_burnin"
                buff_sz = np.size(self.burnin_smpls[key])
                smpls_shape = shape + (int(buff_sz/np.prod(shape)),)
                smpls = np.reshape(self.burnin_smpls[key], smpls_shape, order="F")
                
                # Only save the valid samples
                results[key] = smpls[...,:self.burnin_iter]
            
            if postburnin:
                key = name + "_smpls"
                buff_sz = np.size(self.samples[key])
                smpls_shape = shape + (int(buff_sz/np.prod(shape)),)
                
                smpls = np.reshape(self.samples[key], smpls_shape, order="F")
                # Only save the valid samples
                results[key] = smpls[...,:self.sample_iter]
              
        # Save any samples which were manually added (did not have a registered callback)
        for (key,value) in self.samples.items():
            if key not in results:
                results[key] = value  
           
        # Save the results to a mat file
        unique_fname = get_unique_file_name(self.params["results_dir"], self.params["results_file"] + self.name)
        mat_file = os.path.join(self.params["results_dir"], unique_fname)
        log.info("Saving results to %s", mat_file)
        scipy.io.savemat(mat_file, results, appendmat=True, oned_as='column')
