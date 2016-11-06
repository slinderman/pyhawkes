import numpy as np
from pyhawkes.models import DiscreteTimeNetworkHawkesModelSpikeAndSlab

def test_compute_rate():
    K = 1
    T = 100
    dt = 1.0
    network_hypers = {'c': np.zeros(K, dtype=np.int), 'p': 1.0, 'kappa': 10.0, 'v': 10*5.0}
    true_model = DiscreteTimeNetworkHawkesModelSpikeAndSlab(K=K, dt=dt,
                                                            network_hypers=network_hypers)
    S,R = true_model.generate(T=T)

    print("Expected number of events: ", np.trapz(R, dt * np.arange(T), axis=0))
    print("Actual number of events:   ", S.sum(axis=0))

    print("Lambda0:  ", true_model.bias_model.lambda0)
    print("W:        ", true_model.weight_model.W)
    print("")

    R_test = true_model.compute_rate()
    assert np.allclose(R, R_test)

def test_generate_statistics():
    K = 1
    T = 100
    dt = 1.0
    network_hypers = {'c': np.zeros(K, dtype=np.int), 'p': 1.0, 'kappa': 10.0, 'v': 10*5.0}
    true_model = DiscreteTimeNetworkHawkesModelSpikeAndSlab(K=K, dt=dt,
                                                            network_hypers=network_hypers)
    S,R = true_model.generate(T=T)

    E_N = np.trapz(R, dt * np.arange(T), axis=0)
    std_N = np.sqrt(E_N)
    N = S.sum(axis=0)

    assert np.all(N >= E_N - 3*std_N), "N less than 3std below mean"
    assert np.all(N <= E_N + 3*std_N), "N more than 3std above mean"

    print("Expected number of events: ", E_N)
    print("Actual number of events:   ", S.sum(axis=0))

test_compute_rate()
test_generate_statistics()