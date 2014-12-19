"""
Data manager handles loading the .mat file and setting up the data on the GPU
This could be extended if we ever moved to a distributed setup with multiple GPUs
"""
import numpy as np
import scipy.sparse as sparse
import scipy.io 

import os

import pycuda.autoinit
import pycuda.compiler as nvcc
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.curandom as curandom

from pyhawkes.utils.utils import *

# Define constant for the sparse matrix preprocessing
G_LOGISTIC_NORMAL = 0

import logging
# Get handle to global logger
log = logging.getLogger("global_log")

class GpuData:
    """
    Inner class to store pointers on the GPU
    """
    def __init__(self):
        self.Ns = None
        self.cumSumNs = None
        self.X = None

class DataSet:
    """
    Wrapper for a spike data set
    """
    def __init__(self):
        self.gpu = GpuData()
    
    def loadFromFile(self, path, sortByBlock=False):
        """
        Load the specified mat file
        """
        mat_data = scipy.io.loadmat(path, appendmat=True)
        self.N = int(mat_data["N"])
        
        if "Tstart" in mat_data.keys() and "Tstop" in mat_data.keys():
            self.Tstart = float(mat_data["Tstart"])
            self.Tstop = float(mat_data["Tstop"])
        elif "T" in mat_data.keys():
            self.Tstart = 0
            self.Tstop = float(mat_data["T"])
        else:
            log.error("Neither (Tstart,Tstop) nor T were specified in the mat file")
            exit()
            
        Sraw = np.ravel(mat_data["S"]).astype(np.float32)
        
        # Some datasets do not have process IDs
        if  "K" in mat_data.keys() and"C" in mat_data.keys():
            self.proc_ids_known = True
            self.K = int(mat_data["K"])
            Craw = (np.ravel(mat_data["C"])).astype(np.int32)
            
            # Make sure the process IDs are 0-based
            if np.max(Craw)==self.K and np.min(Craw)==1:
                # The data file is 1-indexed (i.e. generated in Matlab most likely
                Craw = Craw -1
        else:
            # Default to all spikes on the same process. This will be changed
            # during inference
            self.proc_ids_known = False
            self.K = 1
            Craw = np.zeros((self.N,), dtype=np.int32)
            
        # Some datasets have associated spatial locations for each spike
        # If so, X must be a DxN matrix where D is the dimension of the spatial data
        if "X" in mat_data.keys():
            self.isspatial = True
            Xraw = mat_data["X"].astype(np.float32)
            
            # Make sure Xraw is a DxN matrix
            if np.size(Xraw,0)==self.N:
                log.debug("Given X is NxD rather than DxN. Transposing...")
                Xraw = Xraw.T
            self.D = np.size(Xraw,0)
        else:
            self.isspatial = False
            self.X = None
            self.D = 0

        if not sortByBlock:
            (I, Ns, cumSumNs) = self.__argsortSCArray(self.K, Sraw, Craw)
        else:
            (I, Ns, cumSumNs) = self.__argsortSCArrayByBlock(self.K, Sraw, Craw)
            
#        (I, Ns, cumSumNs) = self.__argsortSCArray(self.K, , Craw)
        self.S = Sraw[I]
        self.C = Craw[I]
        if self.isspatial:
            # Slicing with I changes the view and orders as if it were NxD matrix
            self.X = np.zeros((self.D,self.N), dtype=np.float32)
            for n in np.arange(self.N):
                self.X[:,n] = Xraw[:,I[n]]
            
        self.Ns = Ns
        self.maxNs = np.max(Ns)
        self.cumSumNs = cumSumNs
        
        # Store remaining keys
        self.other_data = {}
        for key in mat_data.keys():
            if key not in ["S","K","C","T","N","X","D"]:
                self.other_data[key] = mat_data[key]
                
        self.__initializeGpuArrays()        
                
        
    def loadFromArray(self,N,K,Tstart,Tstop,S,C,X=None,D=0,other_data={},proc_ids_known=True, sortByBlock=False):
        """
        Initialize a DataSet object with the given parameters
        """
        self.N = N
        self.K = K
        self.Tstart = Tstart
        self.Tstop = Tstop
        self.other_data = other_data
        self.proc_ids_known = proc_ids_known
        self.isspatial = (X!=None)
        self.D = D
        self.X = None
        
        if N == 0:
            self.S = S
            self.C = C
            self.Ns = np.zeros(K)
            return
        
        # Make sure the process IDs are 0-based
        if np.max(C)==self.K and np.min(C)==1:
            # The data file is 1-indexed (i.e. generated in Matlab most likely
            C = C -1
        
        if not sortByBlock:
            (I, Ns, cumSumNs) = self.__argsortSCArray(self.K, S, C)
        else:
            (I, Ns, cumSumNs) = self.__argsortSCArrayByBlock(self.K, S, C)
            
        self.S = S[I]
        self.C = C[I]
        if self.isspatial:
            # Slicing with I changes the view and orders as if it were NxD matrix
            self.X = np.zeros((self.D,self.N), dtype=np.float32)
            for n in np.arange(self.N):
                self.X[:,n] = X[:,I[n]]   
                
        self.Ns = Ns
        self.maxNs = np.max(Ns)
        self.cumSumNs = cumSumNs
        
        # Set correct types
        self.S = np.float32(self.S)
        self.C = np.int32(self.C)
        self.Ns = np.int32(self.Ns)
        self.N = int(self.N)
        self.K = int(self.K)
        self.D = int(self.D)
        self.X = np.float32(self.X)
        
        self.__initializeGpuArrays()
        
    def __initializeGpuArrays(self):
        """
        Add a dictionary of GPU pointers
        """
        self.gpu.Ns = gpuarray.to_gpu(self.Ns.astype(np.int32))
        self.gpu.cumSumNs = gpuarray.to_gpu(self.cumSumNs.astype(np.int32))
        
        if self.isspatial:
#            self.gpu.X = gpuarray.empty((self.D,self.N), dtype=np.float32)
#            self.gpu.X.set(self.X.astype(np.float32))
            self.gpu.X  = gpuarray.to_gpu(self.X.astype(np.float32))

    def __argsortSCArray(self,K,S,C):
        """
        Sort an array of spikes, first by their processes, then by their spike times.
        We assume S is already sorted but C is not.
        """
        # Keep an array of spike counts
        Ns = np.zeros(K, dtype=np.int32)
        N = np.size(S)
        
        assert np.size(C) == N, "ERROR: Size of S and C do not match!"
        
        # Compute a permutation of S,C,X such that S is sorted in increasing order
        Iflat = np.argsort(S)
        
        # Compute Ns
        for k in np.arange(K):
            Ns[k] = np.count_nonzero(C==k)
            
        # Also compute the cumulative sum of Ns
        cumSumNs = np.cumsum(np.hstack(([0], Ns)), dtype=np.int32)
        
        return (Iflat, Ns, cumSumNs)

    def __argsortSCArrayByBlock(self,K,S,C):
        """
        Sort an array of spikes, first by their processes, then by their spike times.
        We assume S is already sorted but C is not.
        """
        # Keep an array of spike counts
        Ns = np.zeros(K, dtype=np.int32)
        N = np.size(S)
        
        assert np.size(C) == N, "ERROR: Size of S and C do not match!"
        
        # Initialize buffers to hold the per-process indices
        ppI = {}
        buff_sz = int(2*N/K)
        for k in np.arange(K):
             ppI[k] = np.zeros(buff_sz)
             
        for n in np.arange(N):
            cn = C[n]
            try:
                ppI[cn][Ns[cn]] = n
            except:
                # Index out of bounds -- grow buffer
                ppI[cn] = np.hstack((ppI[cn], np.zeros(buff_sz)))
                ppI[cn][Ns[cn]] = n
            Ns[cn] += 1
        
        # Flatten the permutation
        Iflat = np.zeros(N, dtype=np.int)
        off = 0
        for k in np.arange(K):
            Iflat[off:off+Ns[k]] = ppI[k][:Ns[k]]
            off += Ns[k]
                    
        # Also compute the cumulative sum of Ns
        cumSumNs = np.cumsum(np.hstack(([0], Ns)), dtype=np.int32)
        
        return (Iflat, Ns, cumSumNs)

class DataManager:
    def __init__(self, configFile, dataFile=None):
        """
        Load the data and preprocess it on the GPU.  
        """
        self.parse_config_file(configFile)
        if not dataFile is None:
            self.params["data_file"] = dataFile
        
        pprint_dict(self.params, "Data Manager Params")
        
    def preprocess_for_inference(self, sortByBlock=False):
        """
        Load all of the data 
        """        
        data = DataSet()
        mat_file = os.path.join(self.params["data_dir"], self.params["data_file"])    
        data.loadFromFile(mat_file, sortByBlock=sortByBlock)
        
        return data
    
    def preprocess_for_cross_validation(self, sortByBlock=False):
        """
        Load all of the data 
        """        
        data = DataSet()
        mat_file = os.path.join(self.params["data_dir"], self.params["xv_file"])    
        data.loadFromFile(mat_file, sortByBlock=sortByBlock)
        
        return data
        
    def preprocess_for_prediction_test(self, Tsplit=0, trainFrac=0.9, sortByBlock=False):
        """
        Load all of the data onto the GPU for parameter inference
        """
        data = DataSet()
        mat_file = os.path.join(self.params["data_dir"], self.params["data_file"])    
        data.loadFromFile(mat_file)
        
        (trainData, testData) = self.split_test_train_data(data, Tsplit, trainFrac, sortByBlock=sortByBlock)
        log.info("Train: %d spikes in time [%.2f,%.2f]", trainData.N, trainData.Tstart,trainData.Tstop)
        log.info("Test: %d spikes in time [%.2f,%.2f]", testData.N, testData.Tstart,testData.Tstop)
        
        return (trainData, testData)
    
    def parse_config_file(self, configFile):
        """
        Parse the config file for data manager params
        """
        # Initialize defaults
        defaultParams = {}
        
        # Data location
        defaultParams["data_dir"] = "."
        defaultParams["xv_file"] = "not given"
        
        # CUDA kernels are defined externally in a .cu file
        defaultParams["cu_dir"]  = os.path.join("pyhawkes", "cuda", "cpp")
        defaultParams["cu_file"] = "preprocessing_unknown_procs.cu"
        
        # Block size
        defaultParams["blockSz"] = 1024
        
        # Window the data such that only spikes within a fixed time window can
        # have an effect. It is important that this be consistent with the
        # prior on the impulse response
        defaultParams["dt_max"]   = 5.0
        defaultParams["max_hist"] = 10*1024
        
        # Create a config parser object and read in the file
        cfgParser = ConfigParser(defaultParams)
        cfgParser.read(configFile)
        
        # Create an output params dict. The config file is organized into
        # sections. Read them one at a time
        self.params = {}
        
        self.params["data_dir"]     = cfgParser.get("io", "data_dir")
        self.params["data_file"]    = cfgParser.get("io", "data_file")
        self.params["xv_file"]      = cfgParser.get("io", "xv_file")
        
        self.params["blockSz"]      = cfgParser.getint("cuda", "blockSz")
        
        self.params["cu_dir"]       = cfgParser.get("preprocessing", "cu_dir")
        self.params["cu_file"]      = cfgParser.get("preprocessing", "cu_file")
        self.params["dt_max"]       = cfgParser.getfloat("preprocessing", "dt_max")
        self.params["max_hist"]     = cfgParser.getint("preprocessing", "max_hist")
    
    def initialize_gpu_kernels(self):
        kernelSrc = os.path.join(self.params["cu_dir"], self.params["cu_file"])
        
        kernelNames = ["computeColumnSizes", 
                       "computeRowIndicesAndDs",
                       "computeDx"]
        
        src_consts = {"B" : self.params["blockSz"]}
        self.gpuKernels = compile_kernels(kernelSrc, kernelNames, srcParams=src_consts)
        
    def initialize_known_proc_gpu_kernels(self):
        kernelSrc = os.path.join(self.params["cu_dir"], self.params["cu_file"])
        
        kernelNames = ["computeColPtrs",
                       "computeDsBufferSize",
                       "computeRowAndDsOffsets",
                       "computeRowIndicesAndDs",
                       "computeColumnSizes", 
                       "computeRowIndicesAndDs"]
                       
        
        src_consts = {"B" : self.params["blockSz"]}
        self.gpuKernels = compile_kernels(kernelSrc, kernelNames, srcParams=src_consts)
    

    def split_test_train_data(self, alldata, Tsplit=0, trainFrac=0.9, sortByBlock=False):
        """
        Split the data into test and train subsets
        alldata must be a sorted Dataset
        """
        
        # First make sure the spike are sorted by time, not by block
        # Compute a permutation of S,C,X such that S is sorted in increasing order
        Iflat = np.argsort(alldata.S)
        S = alldata.S[Iflat]
        C = alldata.C[Iflat]
        X = alldata.X[:,Iflat] if alldata.X!=None else None
        
        
        if Tsplit > 0:
            # Find the index of the first spike after Tsplit
            split_ind = np.min(np.nonzero(S>Tsplit)[0])
        elif trainFrac > 0:
            split_ind = int(np.floor(trainFrac*alldata.N))
            Tsplit = (S[split_ind-1] + S[split_ind])/2.0
        else:
            log.error("Either Tsplit or trainFrac must be specified!")
            exit()
        
        
        # Create two datasets
        trainData = self.get_data_in_interval(alldata,(0,Tsplit), sortByBlock=sortByBlock)
        testData = self.get_data_in_interval(alldata,(Tsplit, alldata.T), sortByBlock=sortByBlock)
        
        return (trainData, testData)
    
    def get_data_in_interval(self, alldata, (T_start,T_stop), sortByBlock=False):
        """
        Split the data into test and train subsets
        alldata must be a sorted Dataset
        """
        
        # First make sure the spike are sorted by time, not by block
        # Compute a permutation of S,C,X such that S is sorted in increasing order
        Iflat = np.argsort(alldata.S)
        S = alldata.S[Iflat]
        C = alldata.C[Iflat]
        X = alldata.X[:,Iflat] if alldata.X!=None else None
        
        
        # Find the index of the first spike after Tsplit
        start_ind = np.min(np.nonzero(S>T_start)[0])
        stop_ind = np.max(np.nonzero(S<T_stop)[0])+1
        
        # Create two datasets
        data = DataSet()
        data.loadFromArray(stop_ind-start_ind, 
                           alldata.K,
                           T_start,
                           T_stop, 
                           S[start_ind:stop_ind], 
                           C[start_ind:stop_ind], 
                           X=X[:,start_ind:stop_ind] if X!=None else None,
                           D=alldata.D,
                           other_data=alldata.other_data,
                           proc_ids_known=alldata.proc_ids_known,
                           sortByBlock=sortByBlock)
        
        return data
        
    def compute_sparse_spike_intvl_matrices(self, dataSet1, dataSet2):
        """
        preprocess the given datasets by computing the intervals between spikes on S1
        and spikes on S2 and storing them in a sparse matrix format on the GPU.
        
        The GPU kernels require the spikes to be sorted, first in C and then in S, so 
        all the spikes on process 0 come first, and within the spikes on process 0 
        they are sorted in increasing order of S.
        """        
        # Initialize the kernels with the size of the dataset
        self.initialize_known_proc_gpu_kernels()
        
        # Temporarily copy both sets of spike times to the GPU
        S1_gpu =  gpuarray.to_gpu(dataSet1.S.astype(np.float32))
        S2_gpu =  gpuarray.to_gpu(dataSet2.S.astype(np.float32))
        
        # Now we can preprocess the interspike intervals on the GPU
        # First compute the size of each column for each matrix
        # Each spike appears in K1 matrices, so there are K1*N2 columns 
        colStartBuffer_gpu = gpuarray.empty((dataSet1.K,dataSet2.N), dtype=np.int32)
        colEndBuffer_gpu = gpuarray.empty((dataSet1.K,dataSet2.N), dtype=np.int32)
        colSizesBuffer_gpu = gpuarray.empty((dataSet1.K,dataSet2.N), dtype=np.int32)
        grid_w = int(np.ceil(float(dataSet2.N)/self.params["blockSz"]))
        status_gpu = gpuarray.zeros((dataSet1.K,grid_w),dtype=np.int32)
        self.gpuKernels["computeColumnSizes"](np.float32(self.params["dt_max"]),
                                              dataSet1.gpu.Ns.gpudata,
                                              dataSet1.gpu.cumSumNs.gpudata,
                                              S1_gpu.gpudata,
                                              np.int32(dataSet2.N),
                                              S2_gpu.gpudata,
                                              colStartBuffer_gpu.gpudata,
                                              colEndBuffer_gpu.gpudata,
                                              colSizesBuffer_gpu.gpudata,
                                              status_gpu.gpudata,
                                              block=(1024,1,1),
                                              grid=(grid_w,dataSet1.K)
                                              )
        
        # Compute the column pointers (the cumulative sum) of the
        # column sizes for each matrix. There are K1xK2 grid of matrices
        colPtrsBuffer_gpu = gpuarray.zeros((dataSet1.K,(dataSet2.N+dataSet2.K)), dtype=np.int32)
        colPtrOffsets_gpu = gpuarray.zeros((dataSet1.K,dataSet2.K), dtype=np.int32)
        self.gpuKernels["computeColPtrs"](np.int32(dataSet1.K),
                                          np.int32(dataSet2.N),
                                          dataSet2.gpu.Ns.gpudata,
                                          dataSet2.gpu.cumSumNs.gpudata,
                                          colSizesBuffer_gpu.gpudata,
                                          colPtrsBuffer_gpu.gpudata,
                                          colPtrOffsets_gpu.gpudata,
                                          block=(1,1,1),
                                          grid=(dataSet1.K,dataSet2.K)
                                          )
        
        
        # Compute the required size of the data and row buffer
        bufferSize_gpu = gpuarray.zeros(1, dtype=np.int32)
        self.gpuKernels["computeDsBufferSize"](np.int32(dataSet1.K),
                                               dataSet2.gpu.Ns.gpudata,
                                               colPtrsBuffer_gpu.gpudata,
                                               colPtrOffsets_gpu.gpudata,
                                               bufferSize_gpu.gpudata,
                                               block=(1,1,1),
                                               grid=(1,1)
                                               )
        
        bufferSize = int(bufferSize_gpu.get()[0])
        log.debug("dS has %d nonzero entries" % bufferSize)

        dsBuffer_gpu = gpuarray.empty((bufferSize,), dtype=np.float32)
        rowIndicesBuffer_gpu = gpuarray.zeros((bufferSize,), dtype=np.int32)
            
        # Compute the offsets into these buffers for each matrix
        rowAndDsOffsets_gpu = gpuarray.empty((dataSet1.K,dataSet2.K), dtype=np.int32)
        self.gpuKernels["computeRowAndDsOffsets"](np.int32(dataSet1.K),
                                                  dataSet2.gpu.Ns.gpudata,
                                                  colPtrsBuffer_gpu.gpudata,
                                                  colPtrOffsets_gpu.gpudata,
                                                  rowAndDsOffsets_gpu.gpudata,
                                                  block=(1,1,1),
                                                  grid=(1,1)
                                                  )
        
        # Now we can actually fill in row and ds buffers
        self.gpuKernels["computeRowIndicesAndDs"](np.int32(G_LOGISTIC_NORMAL),
                                                  np.int32(dataSet1.K),
                                                  dataSet1.gpu.Ns.gpudata,
                                                  dataSet1.gpu.cumSumNs.gpudata,
                                                  S1_gpu.gpudata,
                                                  np.int32(dataSet2.N),
                                                  dataSet2.gpu.cumSumNs.gpudata,
                                                  S2_gpu.gpudata,
                                                  colStartBuffer_gpu.gpudata,
                                                  colEndBuffer_gpu.gpudata,
                                                  colPtrsBuffer_gpu.gpudata,
                                                  colPtrOffsets_gpu.gpudata,
                                                  rowIndicesBuffer_gpu.gpudata,
                                                  dsBuffer_gpu.gpudata,
                                                  rowAndDsOffsets_gpu.gpudata,
                                                  block=(1024,1,1),
                                                  grid=(grid_w,dataSet1.K)
                                                  )
        
        
        # If this is a spatial dataset then also compute dX 
        dxBuffer_gpu = None
        if dataSet1.isspatial and dataSet2.isspatial:
            D = dataSet1.D
            assert dataSet2.D == D, "Error: two datasets have different spatial dimensions"
            dxBuffer_gpu = gpuarray.empty((D*bufferSize,), dtype=np.float32)
            
            # Copy the spatial data to the GPU
            X1_gpu = gpuarray.to_gpu(dataSet1.X.astype(np.float32))
            X2_gpu = gpuarray.to_gpu(dataSet2.X.astype(np.float32))
            
            self.gpuKernels["computeDx"](np.int32(D),
                                          np.int32(dataSet1.N),
                                          dataSet1.gpu.cumSumNs.gpudata,
                                          X1_gpu.gpudata,
                                          np.int32(dataSet2.N),
                                          dataSet2.gpu.cumSumNs.gpudata,
                                          X2_gpu.gpudata,
                                          rowIndicesBuffer_gpu.gpudata,
                                          colPtrsBuffer_gpu.gpudata,
                                          colPtrOffsets_gpu.gpudata,
                                          rowAndDsOffsets_gpu.gpudata,
                                          dxBuffer_gpu.gpudata,
                                          block=(1024,1,1),
                                          grid=(grid_w,dataSet1.K)
                                          )
            
        ds = dsBuffer_gpu.get()
#        assert np.all(ds < self.params["dt_max"]), "ERROR: DS contains entries equal to dt_max!"
#        assert np.all(ds > 0), "ERROR: DS contains entries equal to 0!"
        
        # Update gpuData dictionary
        gpuData = {}
        gpuData["dsBuffer_size"]        = bufferSize
        gpuData["dsBuffer_gpu"]         = dsBuffer_gpu
        gpuData["rowIndicesBuffer_gpu"] = rowIndicesBuffer_gpu
        gpuData["colPtrsBuffer_gpu"]    = colPtrsBuffer_gpu
        gpuData["rowAndDsOffsets_gpu"]  = rowAndDsOffsets_gpu
        gpuData["colPtrOffsets_gpu"]    = colPtrOffsets_gpu
        gpuData["dxBuffer_gpu"]         = dxBuffer_gpu 
        
        return gpuData
        
    def compute_sparse_spike_intvl_matrix_unknown_procs(self, S1, S2):
        """
        In the case where the process identities are unknown and to be inferred, 
        it does not make sense to have a grid of sparse matrices for each pair of 
        process identities. Instead, create a single sparse matrix for spike intervals
        
        """
        # Initialize the kernels with the size of the dataset
        self.initialize_gpu_kernels()
        
        # Temporarily copy both sets of spike times to the GPU
        N1 = len(S1)
        N2 = len(S2)
        
        # Handle the case where there are no spikes, N2=0
        if N2 == 0:
            gpuData = {}
            gpuData["dS_size"]      = 0
            gpuData["dS"]           = gpuarray.zeros(1, dtype=np.float32)
            gpuData["rowIndices"]   = gpuarray.zeros(1, dtype=np.float32)
            gpuData["colPtrs"]      = gpuarray.zeros(1, dtype=np.float32)
            return gpuData
        
        S1_gpu =  gpuarray.to_gpu(S1.astype(np.float32))
        S2_gpu =  gpuarray.to_gpu(S2.astype(np.float32))
        
        # Now we can preprocess the interspike intervals on the GPU
        # First compute the size of each column for each matrix
        # Each spike appears in K1 matrices, so there are K1*N2 columns 
        colStart_gpu = gpuarray.empty((N2,), dtype=np.int32)
        colEnd_gpu = gpuarray.empty((N2,), dtype=np.int32)
        colSizes_gpu = gpuarray.empty((N2,), dtype=np.int32)
        grid_w = int(np.ceil(float(N2)/self.params["blockSz"]))
        self.gpuKernels["computeColumnSizes"](np.float32(self.params["dt_max"]),
                                              np.int32(N1),
                                              S1_gpu.gpudata,
                                              np.int32(N2),
                                              S2_gpu.gpudata,
                                              colStart_gpu.gpudata,
                                              colEnd_gpu.gpudata,
                                              colSizes_gpu.gpudata,
                                              block=(1024,1,1),
                                              grid=(grid_w,1)
                                              )
        
        # Compute the column pointers (the cumulative sum) of the col sizes
        colSizes = colSizes_gpu.get()
        colPtrs = np.cumsum(np.hstack(([0],colSizes))).astype(np.int32)
        colPtrs_gpu = gpuarray.to_gpu(colPtrs)
        
        # Compute the required size of the data and row buffer
        bufferSize = int(colPtrs[-1])
        log.debug("dS has %d nonzero entries" % bufferSize)

        if bufferSize == 0:
            log.warning("There are no preceding parents. Potential parent matrix is empty!")
            log.debug("Setting buffer size to 1.")
            bufferSize = 1

        dS_gpu = gpuarray.empty((bufferSize,), dtype=np.float32)
        dS_gpu.fill(1.0)
        rowIndices_gpu = gpuarray.zeros((bufferSize,), dtype=np.int32)
                    
        # Now we can actually fill in row and ds buffers
        self.gpuKernels["computeRowIndicesAndDs"](np.int32(G_LOGISTIC_NORMAL),
                                                  S1_gpu.gpudata,
                                                  np.int32(N2),
                                                  S2_gpu.gpudata,
                                                  colStart_gpu.gpudata,
                                                  colEnd_gpu.gpudata,
                                                  colPtrs_gpu.gpudata,
                                                  rowIndices_gpu.gpudata,
                                                  dS_gpu.gpudata,
                                                  block=(1024,1,1),
                                                  grid=(grid_w,1)
                                                  )
        
        
        # If this is a spatial dataset then also compute dX 
#        dX_gpu = None
#        if dataSet1.isspatial and dataSet2.isspatial:
#            D = dataSet1.D
#            assert dataSet2.D == D, "Error: two datasets have different spatial dimensions"
#            dX_gpu = gpuarray.empty((D*bufferSize,), dtype=np.float32)
#            
#            # Copy the spatial data to the GPU
#            X1_gpu = gpuarray.to_gpu(dataSet1.X.astype(np.float32))
#            X2_gpu = gpuarray.to_gpu(dataSet2.X.astype(np.float32))
#            
#            self.gpuKernels["computeDx"](np.int32(D),
#                                         np.int32(N1),
#                                         X1_gpu.gpudata,
#                                         np.int32(N2),
#                                         X2_gpu.gpudata,
#                                         rowIndices_gpu.gpudata,
#                                         colPtrs_gpu.gpudata,
#                                         dX_gpu.gpudata,
#                                         block=(1024,1,1),
#                                         grid=(grid_w,1)
#                                         )
            
        ds = dS_gpu.get()
        if not np.all(ds > 0):
            log.info("Min DS: %f", np.min(ds))
            raise Exception("ERROR: DS contains nonpositive entries")
#        assert np.all(ds <= self.params["dt_max"]), "ERROR: DS contains entries greater than dt_max!"
#        assert np.all(ds < self.params["dt_max"]), "ERROR: DS contains entries equal to dt_max!"
                
        # Update gpuData dictionary
        gpuData = {}
        gpuData["dS_size"]      = bufferSize
        gpuData["dS"]           = dS_gpu
        gpuData["rowIndices"]   = rowIndices_gpu
        gpuData["colPtrs"]      = colPtrs_gpu
#        gpuData["dxBuffer_gpu"] = dX_gpu 
        
        return gpuData
