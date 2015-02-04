"""
Preprocess the fluorescence traces to get a spike train matrix.
Based on the ChaLearn Connectomics challenge starter kit
by Bisakha Ray, Javier Orlandi and Olav Stetter. I tried to clean it
up since it didn't make use of numpy functions and it was woefully
lacking in useful comments.
"""
import os
import sys
import cPickle
import gzip
import numpy as np

# Use the Agg backend in running on a server without the DISPLAY variable
if "DISPLAY" not in os.environ:
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

# Import OOPSI. Available at: https://github.com/liubenyuan/py-oopsi
# or originally in Matlab from https://github.com/jovo/oopsi
pyoopsi_path = os.path.join(os.path.expanduser("~"), "Install", "py-oopsi")
sys.path.append(pyoopsi_path)
import oopsi

def process_dataset(K=100,
                    suffix="_iNet1_Size100_CC01inh.txt",
                    dir="data/chalearn/small",
                    outfile="network1c.pkl"):

    # Get the full filenames
    fluor_file = os.path.join(dir, "fluorescence" + suffix)
    net_file   = os.path.join(dir, "network" + suffix)
    pos_file   = os.path.join(dir, "networkPositions" + suffix)

    # Parse the files
    F       = parse_fluorescence_file(fluor_file, K)

    # # Load the oopsi processed fluorescence
    # data_path = os.path.join("data", "chalearn", "small", "network1c.pkl.gz")
    # with gzip.open(data_path, 'r') as f:
    #     _, F, C, network, pos = cPickle.load(f)

    network = parse_network_file(net_file, K)
    pos     = parse_position_file(pos_file, K)

    # Discretize the fluorescence signal
    # Hardcode the bins
    # bins = np.array([-10, 0.17, 10]).reshape((1,3)).repeat(K, axis=0)
    # S, _ = discretize_fluorescence(F, edges=bins, binsui=False)
    # # S, bins = discretize_fluorescence(F, nbins=2, binsui=True)
    # S, bins = discretize_fluorescence(C, nbins=2, binsui=True)
    # # S = remove_double_spikes(S)

    # Get the spike times with oopsi
    # fast-oopsi,
    S,C = extract_spike_oopsi(F, dt=0.02)


    # Plot a segment of fluorescence traces and spikes
    start = 0
    end   = 10000
    k     = 0
    spks  = np.where(S[start:end, k])
    plt.figure()
    plt.plot(F[start:end, k], '-k')
    plt.plot(spks, F[spks,k], 'ro')
    plt.show()

    # Scatter plot the positions
    plt.figure()

    pres,posts = network.nonzero()
    for i,j in zip(pres,posts):
        if np.random.rand() < 0.25:
            plt.plot([pos[i,0], pos[j,0]],
                     [pos[i,1], pos[j,1]],
                     '-k', lw=0.5)
    plt.scatter(pos[:,0], pos[:,1], s=10, c='r', marker='o', facecolor='k')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

    # Plot the network as a function of X position
    perm = np.argsort(pos[:,0])
    plt.figure()
    plt.imshow(network[np.ix_(perm, perm)], cmap="Greys", interpolation="none")
    plt.xlabel("Postsynaptic")
    plt.ylabel("Presynaptic")
    plt.show()

    with gzip.open(os.path.join(dir, outfile + ".gz"), 'w') as f:
        cPickle.dump((S, F, C, network, pos), f, protocol=-1)


def parse_fluorescence_file(filename, K, delimiter=','):
    """
    Parse a fluorescence file to into a numpy matrix

    :param filename:        Name fo the file
    :param delimiter:       Delimeter between neurons (always ',')
    :return:                TxK matrix of fluorescence values
    """

    assert os.path.exists(filename), "File doesn't exist!"

    # Define an iterator to yield split lines
    def iter_func():
        with open(filename, 'r') as f:
            for line in f:
                line = line.rstrip().split(delimiter)
                assert len(line) == K, "Line is not of length %d!" % K
                for item in line:
                    yield np.float(item)

    F = np.fromiter(iter_func(), dtype=np.float)
    F = F.reshape((-1, K))
    return F

def parse_network_file(filename, K, vmin=0):
    """
    Parse a network file where each row is of the form I,J,W
    denoting a connection from neuron I to neuron J with weight W.
    :param filename:
    :return:
    """
    network = np.zeros((K,K))

    with open(filename, 'r') as f:
        for line in f:
            # Read the line
            i,j,w = line.rstrip().split(',')
            # Cast to ints
            i,j,w = int(i)-1, int(j)-1, int(w)

            network[i,j] = w

    network = np.clip(network, vmin, np.inf)

    return network

def parse_position_file(filename, K):
    """
    Parse a network file where each row is of the form I,J,W
    denoting a connection from neuron I to neuron J with weight W.
    :param filename:
    :return:
    """
    pos = np.zeros((K,2))

    with open(filename, 'r') as f:
        for k in xrange(K):
            line = f.readline()
            # Read the line
            x,y = line.rstrip().split(',')
            # Cast to floats
            x,y = float(x), float(y)

            pos[k,0] = x
            pos[k,1] = y

    return pos

def extract_spike_oopsi(F, dt):
    """
    Extract the spike times with OOPSI

    :param F:           Fluorescence data (each row a time bin, each column a neuron).
    :param dt:          Time bin size
    :return             The discretized signal
    """
    D = np.zeros_like(F)
    C = np.zeros_like(F)
    for k in xrange(F.shape[1]):
        print "Running oopsi on neuron ", k
        D[:,k], C[:,k] = oopsi.fast(F[:,k], dt=dt, iter_max=6)

    # Cast D to an integer matrix
    # D = D.astype(np.int)

    return D, C


def discretize_fluorescence(F,
                            nbins=2,
                            edges=None,
                            binsui=True,
                            hpfilter=True,
                            debug=False):
    """
    Discretizes the fluorescence signal so it
    can be used to compute the joint PDF. If conditioning is applied, the
    entries above the conditioning level are returned in the G vector.

    Example usage:      D = discretizeFluorescenceSignal(F)

    :param F:           Fluorescence data (each row a time bin, each column a neuron).
    :param binedges:        An array of bin edges (min length 3)
    :param nbins:       If bins is None, use nbins evenly spaced bins for each neuron
    :param binsui:      If true, pop up a UI to set the bin threshold

    :param hpfilter:    Apply a high pass filter to the fluorescence signal,
                        i.e., work with the derivative (default true).

    :return             The discretized signal
    """
    T,K = F.shape

    # Apply the high pass filter
    if hpfilter:
        Fhat = np.diff(F, axis=0)
    else:
        Fhat = F

    # Compute the range of the fluorescence for each neuron
    F_min = np.amin(Fhat, axis=0)
    F_max = np.amax(Fhat, axis=0)

    # Discretize the signal
    D = -1 * np.ones((T-1,K))

    # If global bins are not given, use neuron-specific binning evenly spaced
    # between F_min and F_max
    if edges is not None:
        bins_given = True
    else:
        edges = np.zeros((K, nbins+1))
        bins_given = False

    for k in xrange(K):
        if not bins_given:
            edges[k,:] = np.linspace(F_min[k]-1e-3, F_max[k]+1e-3, num=nbins+1)

        if binsui:
            # Histogram of the fluorescence
            fig = plt.figure()
            plt.hist(Fhat[:,k], bins=100, normed=True)
            for edge in edges[k,:]:
                plt.plot([edge, edge],
                         [0, plt.gca().get_ylim()[1]], '-r')
            plt.plot()
            plt.xlabel('F')
            plt.ylabel('p(F)')
            plt.title('Neuron %d' % k)

            # Add an event handler to get the threshold
            def onclick(event):
                edges[k,1] = event.xdata
                print "Neuron: ", k, "\tThreshold: %.3f" % edges[k,1]
                plt.close()

            fig.canvas.mpl_connect('button_press_event', onclick)
            plt.show()


        # Digitize this column
        D[:,k] = np.digitize(Fhat[:,k], edges[k,:]) - 1

    assert np.all(D >= 0) and np.all(D < nbins), "Error in digitizing!"

    # Cast D to an integer matrix
    D = D.astype(np.int)

    return D, edges

def remove_double_spikes(D):
    Dhat = D.copy()

    # Identify back to back spikes
    doubles = (Dhat[1:,:] > 0) & (D[:-1] > 0)

    # Remove the first of the pair
    Dhat = Dhat[:-1,:]
    Dhat[doubles] = 0

    return Dhat

process_dataset()