from pyhawkes.cuda.base_model import *
from pyhawkes.utils.data_manager import *
from pyhawkes.utils.stat_manager import *
from pyhawkes.utils.sample_parser import *
from pyhawkes.utils.utils import *

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
    params["data_dir"]      = cfgParser.get("synthetic", "data_dir")
    params["data_file"]     = cfgParser.get("synthetic", "data_file")
    
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

def runGibbsSampler(trainModel, 
                    statManager, 
                    trainStatManager, 
                    burnin=100000, 
                    samples=1000, 
                    thin=10, 
                    print_intvl=100, 
                    perfDict=None):
    """
    Run the Gibbs Sampler 
    """
    #
    #  Start the Gibbs Sampler loop
    #
    start_time = time.clock()
    for iter in np.arange(burnin+samples*thin):
        trainModel.sampleModelParametersAndLatentVars()
        
        if np.mod(iter,print_intvl)==0:
            phase = "Sample" if (iter >= burnin) else "Burnin"
            phase_iter = iter-burnin if (iter >= burnin) else iter
             
            stop_time = time.clock()
            if stop_time - start_time == 0:
                log.info("Training %s iteration %d. Iter/s exceeds time resolution.", phase, phase_iter)
            else:
                log.info("Training %s iteration %d. Iter/s = %f", phase, phase_iter, float(print_intvl)/(stop_time-start_time))
            start_time = stop_time
            
        
        #  Collect samples
        if iter < burnin:
            statManager.collectBurninSamples()
        else:
            statManager.collectSamples()
            trainStatManager.collectSamples()

def computeKsStatistic(trainModel, statManager):
    """
    Once the model has been trained, compute the KS statistic 
    """
    rescaled_S = trainModel.computeRescaledSpikeTimes()
    C = trainModel.modelParams["proc_id_model","C"]
    K = trainModel.modelParams["proc_id_model","K"]
    
    for k in np.arange(K):
        statManager.setSingleSample("rescaled_S_%d" % k, rescaled_S[C==k])
    
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


def parseCommandLineArgs():
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

def saveSynthData(synthData, trainModel):
    """
    Save the synthetic data for comparison with other models
    """
    # Save the results to a mat file
    unique_fname = getUniqueFileName(params["data_dir"], params["data_file"])
    mat_file = os.path.join(params["data_dir"], unique_fname)
    log.info("Saving synthetic data to to %s", mat_file)
    
    synthDataDict = {"S":synthData.S,
                     "C":synthData.C,
                     "N":synthData.N,
                     "Tstart":synthData.Tstart,
                     "Tstop":synthData.Tstop,
                     "K":synthData.K,
                     "W":trainModel.modelParams["weight_model","W"],
                     "A":trainModel.modelParams["graph_model","A"],
                     "g_mu":trainModel.modelParams["impulse_model","g_mu"],
                     "g_tau":trainModel.modelParams["impulse_model","g_tau"]
                     }
    
    
    scipy.io.savemat(mat_file, synthDataDict, appendmat=True, oned_as='column')


if __name__ == "__main__":
    perfDict = {}
    startPerfTimer(perfDict, "TOTAL")
    
    # Parse the command line params
    (options, _) = parseCommandLineArgs() 
    
    # Load params from config file
    params = parseGlobalConfigSettings(options.configFile)
    if options.outputFile != None:
        params["data_file"] = options.outputFile
    
    # Initialize the logger
    initializeLogger(params)
    log.info("Synthetic test: K=%d T=%f", params["K"], params["T"])
    
    # Initialize the random seed
    initializeRandomness(params)
    
    #  Initialize data manager
    log.info("Initializing DataManager")
    dataManager = DataManager(options.configFile, dataFile=options.dataFile)
    
    # Create a null data set to start with
    nullData = DataSet()
    nullData.loadFromArray(2,params["K"],-1,0, np.array([-0.5,-0.25], dtype=np.float32), np.array([0,0], dtype=np.int32))
        
    #  Initialize the stat manager
    trainStatsManager = StatManager(options.configFile, name="_train")
    testStatsManager = StatManager(options.configFile, name="_test")
    
    #  Initialize the Model
    # HACK: We need to initialize the model before registering the stat manager because
    # some statistics' shapes depend on K, and K is not know until after initialization.
    # Ideally we would be able to submit samples of different size to the stat manager
    # and it should detect at save time whether all the samples are a consistent shape and
    # can be compressed into a single array. 
    log.info("Initializing training Model")
    trainModel = BaseModel(dataManager, options.configFile, nullData)
    # start the sampler with a draw from the prior
    if options.sampleFile != None:
        init_db = get_initial_sample(options.sampleFile)
        trainModel.initializeFromDict(init_db)
    else:
        # Restart the sampler with a draw from the prior
        trainModel.initializeFromPrior()
        
    
    # Register model callbacks with the stat manager
    trainModel.registerStatManager(trainStatsManager)
        
        
    # Draw a set of model parameters from the prior and
    # generate synthetic data given the training parameters from the prior
    # This can take multiple iterations if the random weight matrix is unstable    
    try:
        trainModel.initializeFromPrior()
        (S,C,N) = trainModel.generateData((0,params["T"]))
    except Exception as e:
        log.error("Failed to generate synthetic data set in 10 attempts")
        raise e
                
    synthData = DataSet()
    synthData.loadFromArray(N, params["K"], 0, params["T"], S, C)
    log.info("Synthetic dataset has %d spikes", N)
    
    saveSynthData(synthData, trainModel)

    # Initialize the test model with the synthetic data
    log.info("Initializing test Model")
    testModel = BaseModel(dataManager, options.configFile, synthData)
    testModel.initializeFromPrior()
    testModel.registerStatManager(testStatsManager)

    # Start the Gibbs sampler
    log.info("Running Gibbs Sampler.")
    runGibbsSampler(testModel,
                    testStatsManager,
                    trainStatsManager,
                    burnin=params["burnin"],
                    samples=params["samples"],
                    thin=params["thin"],
                    print_intvl=params["print_intvl"],
                    )

    # Save results to .mat file
    trainStatsManager.saveStats()
    testStatsManager.saveStats()

    # Finally, shutdown the logger to flush to file
    logging.shutdown()
