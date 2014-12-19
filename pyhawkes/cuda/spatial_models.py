"""
Spatial models multiplicatively scale the temporal rate according to
a spatial probability distribution.
"""

import scipy.io
import logging
from ConfigParser import ConfigParser

from model_extension import ModelExtension

log = logging.getLogger("global_log")

class DefaultSpatialModel(ModelExtension):
    
    def __init__(self, baseModel, configFile):
        # Store pointer to base model
        self.base = baseModel

        # Parse config file
        self.parseConfigurationFile(configFile)

    def parseConfigurationFile(self, configFile):
        """
        Parse the configuration file to get base model parameters
        """
        pass
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
        pass

    def computeSpatialLogLkhdPerSpike(self):
        """ Compute the spatial log probability of each spike. For the
            default model, the spatial distribution is an atom with unit probability
            so the log prob is zero for all spikes
        """
        return 0.0


class WeightedKmlSpatialModel(ModelExtension):
    """ A spatial model that weights the polygons in a given KML file
        such that the total probability mass integrates to 1.

        Uses PyKML to read and parse the KML file containing the polygons
        Uses Shapely to compute the area of the polygons and determine
        whether points occurred inside them.
    """
    def __init__(self, baseModel, configFile):
        # Store pointer to base model
        self.base = baseModel

        # Initialize databases for this extension
        self.modelParams = baseModel.modelParams
        self.modelParams.addDatabase("spatial_model")
        self.gpuPtrs = baseModel.gpuPtrs
        self.gpuPtrs.addDatabase("spatial_model")

        # Parse config file
        self.parseConfigurationFile(configFile)

        # Initialize GPU kernels and memory
        self.initializeGpuKernels()

    def parseConfigurationFile(self, configFile):
        """
        Parse the configuration file to get base model parameters
        """
        # Create a config parser object and read in the file
        cfgParser = ConfigParser()
        cfgParser.read(configFile)

        self.params = {}
        #self.params["kml"] = cfgParser.get("spatial_model", "kml")
        self.params["areas_file"] = cfgParser.get("spatial_model", "areas_file")

    def load_areas_file(self):
        """ Load a .mat file containing the area of each process's region
        """
        areas = scipy.io.lodmat(self.params["areas_file"])['areas']

        # Check which polygons are in use
        import pdb; pdb.set_trace()
        if "valid_polys" in self.base.data.extra_keys:
            valid_polys = self.base.data.extra_keys["valid_polys"]

            self.modelParams["spatial_model", "areas"] = areas[valid_polys-1]
        else:
            self.modelParams["spatial_model", "areas"] = areas

        # Make sure the number of valid polygons matches the number of processes\
        if not len(self.modelParams["spatial_model", "areas"]) == self.base.data.K:
            raise Exception("Expected an area for each process!")


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
        self.load_areas_file()

    def computeSpatialLogLkhdPerSpike(self):
        """ Compute the spatial log probability of each spike. For the
            default model, the spatial distribution is an atom with unit probability
            so the log prob is zero for all spikes
        """

        # Get the process IDs

        # Prob of each spike is 1/(area of containing polygon)

        return 0.0

