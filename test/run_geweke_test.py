from base_model import *
import signal

import sys
import os
sys.path.append(os.path.join("..","utils"))
from utils import *

sys.path.append(os.path.join("..","common"))
from data_manager import *
from stat_manager import *
import copy

import logging
log = logging.getLogger("global_log")

defaultConfigFile = os.path.join("config", "chicago_proc_unknown.cfg")

def parseGlobalConfigSettings(configFile):
    defaultParams = {}
    # Data is passed in as a .mat file
    defaultParams["results_dir"] = "."
    
    # Logging parameters
    defaultParams["log_dir"] = "."
    defaultParams["print_to_console"] = True
        
    defaultParams["burnin"] = 10000
    defaultParams["samples"] = 1000
    defaultParams["thin"] = 10
    defaultParams["restarts"] = 1
    defaultParams["print_intvl"] = 1000
    
    # Create a config parser object and read in the file
    cfgParser = ConfigParser(defaultParams)
    cfgParser.read(configFile)
    
    # Create an output params dict. The config file is organized into
    # sections. Read them one at a time
    params = {}
    # Logging params
    params["log_dir"]          = cfgParser.get("log", "log_dir")
    params["log_file"]         = cfgParser.get("log", "log_file")
    params["print_to_console"] = bool(cfgParser.getint("log", "print_to_console"))
    
    # Random seed
    params["seed"]          = cfgParser.getint("base_model", "seed")
    
    # MCMC params
    params["burnin"]        = cfgParser.getint("MCMC", "burnin")
    params["samples"]       = cfgParser.getint("MCMC", "samples")
    params["thin"]          = cfgParser.getint("MCMC", "thin")
    params["restarts"]      = cfgParser.getint("MCMC", "restarts")
    
    # Output params
    params["print_intvl"]                = cfgParser.getint("output", "print_intvl")
    
    # Read in the synthetic test params
    params["K"]             = cfgParser.getint("synthetic", "K")
    params["T"]             = cfgParser.getfloat("synthetic", "T")
    
    return params

def initializeRandomness(params):
    """
    Initialize the random number generator used on GPU
    """
    if params["seed"] == -1:
        seed = np.random.randint(0,2**30)
        log.info("Random seed not specified. Using %d", seed)
        np.random.seed(seed)
    else:
        seed = params["seed"]
        log.info("Using specified seed: %d", seed)
        np.random.seed(seed)

def run_geweke_test(model, 
                    params,
                    dataManager,
                    statManager, 
                    burnin=100000, 
                    samples=10000, 
                    thin=10, 
                    print_intvl=100, 
                    perfDict=None):
    """
    Run the Gibbs Sampler 
    """
    #
    #  Start the Gibbs Sampler loop
    #
    K = params["K"]
    Tstart = 0.0
    Tstop = params["T"]
    
    start_time = time.clock()
    for iter in np.arange(burnin + samples*thin):
        
        # Save the old model params
        prevParams = copy.deepcopy(model.modelParams)
        
        # Sample new data given parameters
        (Snew,Cnew,Nnew) = model.generateData((Tstart,Tstop))
        log.info("Sampled %d new spikes.", Nnew)
        
        # Update model with the new data
        newdata = DataSet()
        newdata.loadFromArray(Nnew,K,Tstart,Tstop,Snew,Cnew)
        model.data = newdata
        model.prepareData() 
        model.initializeFromDict(prevParams)
        
#        # HACK!
#        model.extensions["impulse_model"].updateGs()

#        model = BaseModel(dataManager, options.configFile, newdata)
        
        # HACK AGAIN!
        model.extensions["parent_model"].sampleLatentVariables()
        
        # Sample model params given the new data
        model.sampleModelParametersAndLatentVars()
#        model.extensions["graph_model"].sampleModelParameters()
#        model.extensions["parent_model"].sampleLatentVariables()
#        model.extensions["weight_model"].sampleModelParameters()
#        model.extensions["bkgd_model"].sampleModelParameters()
        
        if np.mod(iter,print_intvl)==0:
            phase = "Sample" if (iter >= burnin) else "Burnin"
            phase_iter = iter-burnin if (iter >= burnin) else iter
             
            stop_time = time.clock()
            if stop_time - start_time == 0:
                log.info("%s iteration %d. Iter/s exceeds time resolution.", phase, phase_iter)
            else:
                log.info("%s iteration %d. Iter/s = %f", phase, phase_iter, float(print_intvl)/(stop_time-start_time))
            start_time = stop_time
            
        
        #  Collect samples
        if iter < burnin:
            statManager.collectBurninSamples()
        else:
            statManager.collectSamples()


def parseCommandLineArgs():
    """
    Parse command line parameters
    """
    from optparse import OptionParser
    
    parser = OptionParser()
    parser.add_option("-c", "--configFile", dest="configFile", default=defaultConfigFile,
                      help="Use this configuration file, either as filename in the config directory, or as a path")
    
    (options, args) = parser.parse_args()
    
    # If configFile not specified we default to the file at the top of this script
    # Check if configFile is a path or a filename 
    if not os.path.exists(options.configFile):
        if os.path.exists(os.path.join("config", options.configFile)):
            options.configFile = os.path.join("config", options.configFile)
        else:
            print "Invalid configuration file specified: %s" % options.configFile
            exit()
    
    # The logger is not yet initialized
    print "Using configuration file: %s" % options.configFile
    return (options, args)


def cleanup():
    """
    Cleanup and save stats on SIGINT
    """
    # Save results to .mat file
    statManager.saveStats()
    # Finally, shutdown the logger to flush to file
    logging.shutdown()
    # Make sure you exit!
    exit()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda x,y: cleanup())
    
    # Parse the command line params
    (options, _) = parseCommandLineArgs() 
    
    # Load params from config file
    params = parseGlobalConfigSettings(options.configFile)
    
    # Initialize the logger
    initializeLogger(params)
    
    # Initialize the random seed
    initializeRandomness(params)
    
    #  Load the specified data
    log.info("Initializing DataManager")
    dataManager = DataManager(options.configFile)
#    trainData = dataManager.preprocessForInference()
    
    # Create a null data set to start with
    nullData = DataSet()
    nullData.loadFromArray(2,
                           params["K"],
                           -1,0, 
                           np.array([-0.5,-0.25], dtype=np.float32),
                            np.array([0,0], dtype=np.int32))
    #  Initialize the stat manager
    statManager = StatManager(options.configFile)
    
    #  Initialize the Model
    log.info("Initializing Model")
    model = BaseModel(dataManager, options.configFile, nullData)
    
    # HACK: We need to initialize the model before registering the stat manager because
    # some statistics' shapes depend on K, and K is not know until after initialization.
    # Ideally we would be able to submit samples of different size to the stat manager
    # and it should detect at save time whether all the samples are a consistent shape and
    # can be compressed into a single array. 
    model.initializeFromPrior()
    
    # Register model callbacks with the stat manager
    model.registerStatManager(statManager)
    
    for restart in np.arange(params["restarts"]):
        # Restart the sampler with a draw from the prior
        model.initializeFromPrior()
        
        # Start the Gibbs sampler
        log.info("Running Gibbs Sampler. Restart #%d", restart)
        run_geweke_test(model, 
                        params,
                        dataManager,
                        statManager,
                        burnin=params["burnin"], 
                        samples=params["samples"], 
                        thin=params["thin"], 
                        print_intvl=params["print_intvl"], 
                        )
        
    # Save results to .mat file
    statManager.saveStats()

    # Finally, shutdown the logger to flush to file
    logging.shutdown()
