import os
import datetime
import numpy as np
import pandas as pd
import cPickle
import scipy.io

data_file = 'data.txt'
pickle_file = 'icpsr.pkl'
mat_file = 'icpsr.mat'

T_start = np.datetime64('1965-01-01T00:00')
T_stop = np.datetime64('1996-01-01T00:00')
def parse_line(line):
    """ Parse a single line into an array of fields. 
        Follow the Cookbook to determine value sizes and offsets 
        (Note that the cookbook uses 1-indexing)
    """
    # Get the time of the incident
    hominew = int(line[:5])

    injyear = 1900 + int(line[7:9])
    injmonth = int(line[9:11])
    injdate = int(line[11:13])   # 97 = blanked, 99 = missing
    injday = int(line[13])
    injhour = line[14:16]
    injmin = line[16:18]

    # The date is withheld from the restricted data that we downloaded
    # Assign a random date instead
    injmonth = 1 if injmonth > 12 else injmonth
    injdate = np.random.randint(27)+1 if injdate == 97 or injdate == 99 else injdate
    injhour = 0 if injhour.strip() == '' else int(injhour)-1
    injhour = 0 if injhour > 23 else injhour
    injmin = 0 if injmin.strip() == '' else int(injmin)
    injmin = 0 if injmin > 60 else injmin

    # Convert to a datetime object
    t = datetime.datetime(injyear, injmonth, injdate, injhour, injmin)
        
    # Get location (Neighborhood/Census tract/etc
    centract = int(line[107:111])
    comarea = int(line[111:113])
    
    # Get the causal factor 
    causfact = int(line[113:116])     # 140 = Gang
    causfact2 = int(line[116:119])
    gang = int(line[125])             # 0/1
    vrel1 = int(line[130:133])        # 723 = Same gang member
                                      # 724 = Rival gang member
        
    # Make sure we count any homicide with causal factor equal to gang
    gang_related = False
    if gang == 1 or causfact == 140 or causfact2 == 140:
        gang_related = True

    return (t, comarea, gang_related)

def parse_file():
    N = 0
    buffsz = 2**10
    bufftype = [('time','datetime64[s]'),('location','i4'),('gang','b')]
    data = np.zeros(buffsz, bufftype)
    with open(data_file) as f:
        for (i,line) in enumerate(f):
            N += 1
            if N >= len(data):
                data = np.concatenate((data, np.zeros(buffsz, bufftype)))
            data_line = parse_line(line)
            data[i] = data_line
            

    print "Parsed %d records" % N
    data = data[:N]

    # Convert into a pandas dataframe
    df = pd.DataFrame({'location' : data['location'], 
                       'gang' : data['gang']}, 
                      index = data['time'])
    df = df.sort(axis=0)
    return df
    
def extract_gang_related_incidents(df):
    """ Get the gang-related incidents for use in the Hawkes process
    """ 
    df_gangs = df[df.gang==1]
    df_gangs = df_gangs[df_gangs.location <= 77]

    print "Found %d gang-related incidents" % len(df_gangs)

    return df_gangs

def dataframe_to_mat(df):
    """ Convert the data frame to a mat file
    """
    N = len(df)

    # Absolute time of incident
    T_abs = df.index.values     
    # Relative time in fractional days
    T_rel = (T_abs - T_start) / np.timedelta64(1, 'D')
    T_stop_rel = (T_stop - T_start) / np.timedelta64(1, 'D')

    ## Only keep the unique community areas
    #locations = np.unique(df.location.values)
    #K = len(locations)
    K = 77
    C = np.zeros(N, dtype=np.int)
    Ns = np.zeros(K)
    for k in np.arange(K):
        #ks = df.location.values==locations[k] 
        ks = df.location.values == k+1
        C[ks] = k+1
        Ns[k] = np.count_nonzero(ks)
    
    mat = {'T_start' : 0,
           'T_stop' : T_stop_rel,
           'S' : T_rel,
           'C' : C,
           'N' : N,
           'K' : K,
           'Ns' : Ns,
#           'comm_areas' : locations
           }
    return mat

def run():
    # Parse the file into a DataFrame
    df = parse_file()
    
    # Extract gang related homicides
    df_gangs = extract_gang_related_incidents(df)

    # Save the data frame
    with open(pickle_file, 'w') as f:
        cPickle.dump(df, f)

    # Convert the gang-related incidents into a .mat file format
    mat = dataframe_to_mat(df_gangs)
    scipy.io.savemat(mat_file, mat)

run()
