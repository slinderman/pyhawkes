"""
Estimate connectivity from covariance
"""
import numpy as np

def xcorr(S, dtmax=10):
    """
    Cross correlate each pair of columns in S at offsets up to dtmax
    """
    # import pdb; pdb.set_trace()
    (T,N) = S.shape
    H = np.zeros((N,N,dtmax))

    # Compute cross correlation at each time offset
    for dt in np.arange(dtmax):
        # print "Computing cross correlation at offset %d" % dt

        # Compute correlation in sections to conserve memory
        chunksz = 16
        for n1 in np.arange(N, step=chunksz):
            for n2 in np.arange(N, step=chunksz):
                n1c = min(n1 + chunksz, N)
                n2c = min(n2 + chunksz, N)
                # Corr coef is a bit funky. We want the upper right quadrant
                # of this matrix. The result is ((n1c-n1)+(n2c-n2)) x ((n1c-n1)+(n2c-n2))
                H[n1:n1c, n2:n2c, dt] = np.corrcoef(S[:T-dt, n1:n1c].T,
                                                    S[dt:,  n2:n2c].T)[:(n1c-n1),(n1c-n1):]

        # Set diagonal to zero at zero offset (obviously perfectly correlated)
        if dt == 0:
            H[:,:,0] = H[:,:,0]-np.diag(np.diag(H[:,:,0]))

    return H

def infer_net_from_xcorr(S, dtmax, thr=1.0):
    # Smooth S with a window of size dtmax
    S = _moving_average(S, ws=dtmax//5, axis=0)

    H = xcorr(S, dtmax)

    # Find the pairs with highest total correlation
    H_sum = np.sum(np.abs(H), axis=2)

    return H_sum

def _moving_average(a, axis=0, ws=10):
    ret = np.cumsum(a, dtype=float, axis=axis)
    ret[ws:] = ret[ws:] - ret[:-ws]
    return ret[ws - 1:] / ws