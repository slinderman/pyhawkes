"""
Helper functions for Poisson processes
"""
import numpy as np

import logging
log = logging.getLogger("global_log")

def sampleInhomogeneousPoissonProc(tt, lam):
    """
    Sample an inhomogeneous Poisson process with rate lam and times tt using
    the time-rescaling theorem. Integrate the rate from [0,t) to get the cumulative
    rate. Spikes are uniformly drawn from this cumulative rate as in a 
    homogeneous process. The spike times in the homogeneous rate can be inverted 
    to get spikes from the inhomogeneuos rate. 
    """
    # Numerically integrate (using trapezoidal quadrature) lam with respect to tt
    N_t = len(tt)
    dt = np.diff(tt)
    dlam = np.diff(lam)
    
    # Trapezoidal rule: int(x,t) ~= 0.5 * sum_n=1^N-1 (t[n+1]-t[n])*(x[n+1]+x[n])
    #                             = 0.5 * sum_n=1^N-1 dt[n+1] * (2x[n+1]-dx[n+1])
    # where dt[n+1] = t[n+1]-t[n], dx[n+1] = x[n+1]-x[n]
    trapLam = 0.5*dt*(2*lam[1:]-dlam)
    cumLam = np.ravel(np.cumsum(trapLam))
    cumLam = np.hstack((np.array([0.0]), cumLam))
    
    # Total area under lam is the last entry in cumLam
    intLam = cumLam[-1]
    
    # Spikes are drawn uniformly on interval [0,intLam] with rate 1
    # Therefore the number of spikes is Poisson with rate intLam
    N = np.random.poisson(intLam)
    # Call the transformed spike times Q
    Q = np.random.uniform(0, intLam, size=N)
    Q = np.sort(Q)
    
    # Invert the transformed spike times 
    S = np.zeros(N)
    tt_off = 0
    for (n,q) in enumerate(Q):
        while q > cumLam[tt_off]:
            tt_off += 1
            assert tt_off < N_t, "ERROR: inverted spike time exceeds time limit!"
        
        # q lies in the time between tt[tt_off-1] and tt[tt_off]. Linearly interpolate 
        # to find exact time
        q_lb = cumLam[tt_off-1]
        q_ub = cumLam[tt_off]
        q_frac = (q-q_lb)/(q_ub-q_lb)
        assert q_frac >= 0.0 and q_frac <= 1.0, "ERROR: invalid spike index"
        
        tt_lb = tt[tt_off-1]
        tt_ub = tt[tt_off] 
        S[n] = tt_lb + q_frac*(tt_ub-tt_lb)
        
    return S

def approximateFiringRate(S, xxx_todo_changeme,N_bins):
    """
    Approximate the firing rate of a set of spikes by binning into equispaced
    bins and dividing by the bin width. Smooth with a Gaussian kernel.
    
    TODO: This could be improved with equimass bins as opposed to equispaced bins.
    """
    (T_start,T_stop) = xxx_todo_changeme
    bin_edges = np.linspace(T_start,T_stop,N_bins+1)
    bin_width = (T_stop-T_start)/N_bins
    
    # Approximate firing rate. Histogram returns bin counts and edges as tuple.
    # Take only the bin counts
    fr = np.histogram(S,bin_edges, range=(T_start,T_stop))[0] / bin_width
    
    # Number of bins to eval kernel at (must be odd to have point at 0)
    N_smoothing_kernel_bins = 9
    smoothing_kernel_bin_centers = np.linspace(-(N_smoothing_kernel_bins-1)/2*bin_width, 
                                               (N_smoothing_kernel_bins-1)/2*bin_width, 
                                               N_smoothing_kernel_bins
                                               )
    # Evaluate a standard normal pdf at the bin centers. Normalize to one
    # so that we don't change the total area under the firing rate
    smoothing_kernel = np.exp(-0.5*smoothing_kernel_bin_centers**2)
    smoothing_kernel = smoothing_kernel / np.sum(smoothing_kernel)
    
    # Since the kernel is symmetric we don't have to worry about flipping the kernel left/right
    # Before smoothing, pad fr to minimize boundary effects
    l_pad = np.mean(fr[:N_smoothing_kernel_bins])
    r_pad = np.mean(fr[-N_smoothing_kernel_bins:])
    fr_pad = np.hstack((l_pad*np.ones(N_smoothing_kernel_bins),fr,r_pad*np.ones(N_smoothing_kernel_bins)))
    fr_smooth = np.convolve(fr_pad, smoothing_kernel, "same")
    
    # Drop the pad components and keep the center of the convolution
    fr_smooth = fr_smooth[N_smoothing_kernel_bins:-N_smoothing_kernel_bins]
    
    assert np.size(fr_smooth) == N_bins, "ERROR: approximation returned invalid length firing rate"
    
    return fr_smooth 