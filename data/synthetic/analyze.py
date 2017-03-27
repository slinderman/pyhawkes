import gzip
import pickle
import os



def analyze(data_path):
    """
    Run the comparison on the given data file
    :param data_path:
    :return:
    """

    if data_path.endswith(".gz"):
        with gzip.open(data_path, 'r') as f:
            S, true_model = pickle.load(f)
    else:
        with open(data_path, 'r') as f:
            S, true_model = pickle.load(f)

    print("True model:")
    print(true_model)

    T = float(S.shape[0])
    N = S.sum(axis=0)
    print("lambda0: ", true_model.bias_model.lambda0.mean())
    print("Average event count: ", N.mean(), " +- ", N.std())
    print("Average event count: ", (N/T).mean(), " +- ", (N/T).std())


# seed = 2650533028
K = 50
C = 5
T = 100000
data_path = os.path.join("data", "synthetic", "synthetic_K%d_C%d_T%d.pkl.gz" % (K,C,T))
analyze(data_path)
