import logging
import ConfigParser

from graph_models import EmptyGraphModel, CompleteGraphModel, ErdosRenyiModel, StochasticBlockModel, LatentDistanceModel
from background_models import HomogenousRateModel, GaussianProcRateModel, GlmBackgroundModel, SharedGpBkgdModel
from spatial_models import DefaultSpatialModel

from coupled_sbm_w_model import StochasticBlockModelCoupledWithW 

from cluster_model import ClusterModel
from location_model import LocationModel

log = logging.getLogger("global_log")

# Models are singleton classes. Save their instances as they are  created
model_instances = {}

def mf_construct_model_extensions(baseModel, configFile):
    """
    Initialize a model by iterating through the config file
    """
    extensions = {}
    
    cfgParser = ConfigParser()
    cfgParser.read(configFile)
    
    # Iterate over each section. If it is one of the recognized types,
    # call the constructor for that section
    for section in cfgParser.sections():
        params = dict(cfgParser.items(section))
        model = None
        if section == "cluster_model":
            model = mfConstructClusterModel(baseModel, configFile, params)
        elif section == "location_model":
            model = mfConstructLocationModel(baseModel, configFile, params)
    
        # Add this model to the extensions dictionary
        if model != None:
            if section not in extensions.keys():
                extensions[section] = model
            else:
                # If there is already an extension with this section name, append this to the list
                # This is to support multiple clusters or location models
                extensions[section] = list(extensions[section]) + [model]
                
    return extensions


def mfConstructGraphModel(graph_model, baseModel, configFile):
    """
    Return an instance of the graph model specified in parameters
    """
    if (baseModel,graph_model) in model_instances.keys():
        return model_instances[(baseModel,graph_model)]
    elif graph_model ==  "complete":
        log.info("Creating complete graph model")
        inst = CompleteGraphModel(baseModel, configFile)
    elif graph_model ==  "empty":
        log.info("Creating disconnected graph model")
        inst = EmptyGraphModel(baseModel, configFile)
    elif graph_model == "erdos_renyi":
        log.info("Creating Erdos-Renyi graph model")
        inst = ErdosRenyiModel(baseModel, configFile)
    elif graph_model == "symmetric":
        log.info("Creating Symmetric Erdos-Renyi graph model")
        inst = ErdosRenyiModel(baseModel, configFile)
    elif graph_model == "sbm":
        log.info("Creating Stochastic Block model")
        inst = StochasticBlockModel(baseModel, configFile)
    elif graph_model == "distance":
        log.info("Creating Latent Distance model")
        inst = LatentDistanceModel(baseModel, configFile)
    elif graph_model == "coupled_sbm_w":
        log.info("Creating coupled SBM+Weight prior")
        inst = StochasticBlockModelCoupledWithW(baseModel, configFile)
    else:
        log.error("Unrecognized graph model: %s", graph_model)
        exit()
        
    model_instances[(baseModel,graph_model)] = inst
    return inst
        

def mf_construct_background_rate_model(bkgd_model, baseModel, configFile):
    """
    Generic factory method to be called by base model
    """
    if (baseModel,bkgd_model) in model_instances.keys():
        return model_instances[(baseModel,bkgd_model)]
    if bkgd_model == "homogenous":
        log.info("Creating homogenous background rate model")
        inst = HomogenousRateModel(baseModel, configFile)
    elif bkgd_model == "gp":
        log.info("Creating Gaussian process weight model")
        inst = GaussianProcRateModel(baseModel, configFile)
    elif bkgd_model == "glm":
        log.info("Creating GLM background model")
        inst = GlmBackgroundModel(baseModel, configFile)
    elif bkgd_model == "shared_gp":
        log.info("Creating Shared Gaussian process weight model")
        return SharedGpBkgdModel(baseModel, configFile)
    else:
        raise Exception("Unsupported background rate model: %s", bkgd_model)
    
    model_instances[(baseModel,bkgd_model)] = inst
    return inst

def construct_spatial_model(parent_model, baseModel, configFile):
    """
    Return an instance of the spatial model specified in parameters
    """

    if parent_model ==  "default":
        log.info("Creating default spatial model")
        inst = DefaultSpatialModel(baseModel, configFile)
    else:
        raise Exception("Unsupported spatial model: %s", parent_model)

    return inst
def mfConstructClusterModel(baseModel, configFile, params):
    """
    Return an instance of the impulse response model specified in parameters
    """
    cluster_model = params["type"]
    cluster_name = params["name"]
    if (baseModel,cluster_name) in model_instances.keys():
        return model_instances[(baseModel,cluster_name)]
    elif cluster_model ==  "fixed":
        log.info("Creating cluster model with fixed number of clusters")
        return ClusterModel(baseModel, configFile,cluster_name)
    else:
        raise Exception("Unsupported cluster model: %s" % cluster_model)
    
def mfConstructLocationModel(baseModel, configFile, params):
    """
    Return an instance of the impulse response model specified in parameters
    """
    location_model = params["type"]
    location_name = params["name"]
    if (baseModel,location_name) in model_instances.keys():
        return model_instances[(baseModel,location_name)]
    elif location_model ==  "gaussian":
        log.info("Creating Gaussian location model: %s", location_name)
        return LocationModel(baseModel, configFile,location_name)
    else:
        raise Exception("Unsupported location model: %s" % location_model)    
    