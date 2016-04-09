import os
import cPickle
import glob

path = "data/synthetic/results_K50_C5_T100000/run004"
files = glob.glob(os.path.join(path, "results.svi.itr*"))
timestamps = {}
for f in files:
    print "Getting ctime for ", f
    timestamps[os.path.basename(f)] = os.path.getctime(f)

print "Saving timestamps"
with open(os.path.join(path, "svi_timestamps.pkl"), 'w') as f:
    cPickle.dump(timestamps, f, protocol=-1)


    
