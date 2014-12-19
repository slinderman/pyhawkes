import traceback

from pyhawkes.cuda.base_model import *
from pyhawkes.utils.utils import *
from pyhawkes.utils.data_manager import *
from pyhawkes.utils.stat_manager import *
from pyhawkes.utils.sample_parser import *

import logging
log = logging.getLogger("global_log")

defaultConfigFile = None

def parse_global_config_settings(configFile):
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
    params["print_intvl"]   = cfgParser.getint("output", "print_intvl")
    if cfgParser.has_option("output","save_on_exception"):
        params["save_on_exception"] = cfgParser.getint("output", "save_on_exception")
    else:
        params["save_on_exception"] = True
    
    return params

def initialize_randomness(params):
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

def run_gibbs_sampler(model, statManager, burnin=100000, samples=10000, thin=10, print_intvl=100, perfDict=None):
    """
    Run the Gibbs Sampler 
    """
    #
    #  Start the Gibbs Sampler loop
    #
    start_time = time.clock()
    for iter in np.arange(burnin + samples*thin):
        model.sampleModelParametersAndLatentVars()
        
        if np.mod(iter,print_intvl)==0:
            phase = "Sample" if (iter >= burnin) else "Burnin"
            phase_iter = iter-burnin if (iter >= burnin) else iter
             
            stop_time = time.clock()
            if stop_time - start_time == 0:
                log.info("%s iteration %d. Iter/s exceeds time resolution.", phase, phase_iter)
            else:
                log.info("%s iteration %d. Iter/s = %f", phase, phase_iter, float(print_intvl)/(stop_time-start_time))
            start_time = stop_time
            
            # Print log likelihood
            log.info("ll: %f", model.computeLogLikelihood())
        
        #  Collect samples
        if iter < burnin:
            statManager.collectBurninSamples()
        else:
            statManager.collectSamples()

def compute_ks_statistic(trainModel, statManager):
    """
    Once the model has been trained, compute the KS statistic 
    """
    rescaled_S = trainModel.computeRescaledSpikeTimes()
    C = trainModel.modelParams["proc_id_model","C"]
    K = trainModel.modelParams["proc_id_model","K"]
    
    for k in np.arange(K):
        statManager.setSingleSample("rescaled_S_%d" % k, rescaled_S[C==k])

def compute_conditional_intensity(model, statManager):
    n_t = 10000
    # Save the conditional intensity
    t = np.linspace(model.data.Tstart, model.data.Tstop,n_t)
    ci = model.computeConditionalIntensityFunction(t)
    statManager.setSingleSample("cond_int", ci)

def get_initial_sample(sampleFile):
    """
    Get the initial sample from the sample File
    """
    if sampleFile is None:
        return None
    
    log.info("Loading samples files")
    samples = scipy.io.loadmat(options.sampleFile)
    
    # Convert the last sample to a parameter database
    init_db = parse_sample(samples,int(samples["N_samples"]-1))

    return init_db


def parse_command_line_args():
    """
    Parse command line parameters
    """
    from optparse import OptionParser
    
    parser = OptionParser()
    parser.add_option("-c", "--configFile", dest="configFile", default=defaultConfigFile,
                      help="Use this configuration file, either as filename in the config directory, or as a path")
    
    parser.add_option("-s", "--sampleFile", dest="sampleFile", default=None,
                      help="Use this sample file, either as filename in the config directory, or as a path")
    
    parser.add_option("-d", "--dataFile", dest="dataFile", default=None,
                      help="Override the data file specified in the config file")
    
    parser.add_option("-o", "--outputFile", dest="outputFile", default=None,
                      help="Override the output (results) file specified in the config file")
    
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

def run_mcmc():
    """
    Run the MCMC algorithm.
    """
    #  Initialize the Model
    log.info("Initializing Model")
    model = BaseModel(dataManager, options.configFile, trainData)
    
    # HACK: We need to initialize the model before registering the stat manager because
    # some statistics' shapes depend on K, and K is not know until after initialization.
    # Ideally we would be able to submit samples of different size to the stat manager
    # and it should detect at save time whether all the samples are a consistent shape and
    # can be compressed into a single array. 
    model.initializeFromPrior()
    
    # Register model callbacks with the stat manager
    model.registerStatManager(statManager)
    
    # Restart the sampler with a draw from the prior
    if options.sampleFile != None:
        init_db = get_initial_sample(options.sampleFile)
        model.initializeFromDict(init_db)
    else:
        # Restart the sampler with a draw from the prior
        model.initializeFromPrior()
        
    # Print log likelihood
    log.info("init ll: %f", model.computeLogLikelihood())
        
    # Start the Gibbs sampler
    run_gibbs_sampler(model,
                    statManager,
                    burnin=params["burnin"], 
                    samples=params["samples"], 
                    thin=params["thin"], 
                    print_intvl=params["print_intvl"], 
                    )
    
    log.info("Computing rescaled spike times for KS statistic")
    compute_ks_statistic(model, statManager)
    
    log.info("Computing conditional intensity")
    compute_conditional_intensity(model, statManager)

if __name__ == "__main__":
    perfDict = {}
    startPerfTimer(perfDict, "TOTAL")
        
    # Parse the command line params
    (options, _) = parse_command_line_args()
    
    # Load params from config file
    params = parse_global_config_settings(options.configFile)
    
    # Initialize the logger
    initialize_logger(params)
    
    # Initialize the random seed
    initialize_randomness(params)
    
    try:
        save_output = params["save_on_exception"]
        
        #  Load the specified data
        log.info("Initializing DataManager")
        dataManager = DataManager(options.configFile, dataFile=options.dataFile)
        trainData = dataManager.preprocess_for_inference()
        
        #  Initialize the stat manager
        statManager = StatManager(options.configFile, resultsFile=options.outputFile)
        
        # Run MCMC
        run_mcmc()
        
        save_output = True
        
    except Exception as e:
        log.error("Caught exception!")
        log.error("Message:")
        log.error(e.message)
        log.error("Stack:")
        log.error(traceback.format_exc())
        
    finally:
        # Save results to .mat file
        if save_output:
            statManager.save_stats()
        # Finally, shutdown the logger to flush to file
        logging.shutdown()
