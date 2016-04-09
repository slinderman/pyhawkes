
import numpy as np
np.random.seed(1111)
np.seterr(over="raise")
import matplotlib.pyplot as plt

from pybasicbayes.util.general import ibincount
from pybasicbayes.util.text import progprint_xrange

import pyhawkes.models
reload(pyhawkes.models)


# Create the model with these parameters
K = 10
B = 3
dt = 1
dt_max = 10.
T = 100.
network_hypers = {'kappa': 1., 'p': 1., 'v': 10.}
dt_model = pyhawkes.models.\
    DiscreteTimeNetworkHawkesModelSpikeAndSlab(K=K, dt=dt, dt_max=dt_max, B=B,
                                               network_hypers=network_hypers)
assert dt_model.check_stability()

S_dt,_ = dt_model.generate(T=int(np.ceil(T/dt)), keep=False)

print "sampled dataset with ", S_dt.sum(), "events"

print "DT LL: ", dt_model.heldout_log_likelihood(S_dt)

# Convert S_test to continuous time
S_ct = dt * np.concatenate([ibincount(S) for S in S_dt.T]).astype(float)
S_ct += dt * np.random.rand(*S_ct.shape)
assert np.all(S_ct < T)
C_ct = np.concatenate([k*np.ones(S.sum()) for k,S in enumerate(S_dt.T)]).astype(int)

# Sort the data
perm = np.argsort(S_ct)
S_ct = S_ct[perm]
C_ct = C_ct[perm]

ct_model = pyhawkes.models.ContinuousTimeNetworkHawkesModel(K, dt_max=1.,
                                                            network_hypers=network_hypers)
ct_model.add_data(S_ct, C_ct, T)
# ct.resample_model()

# Hard code parameters
ct_model.bias_model.lambda0 = dt_model.bias_model.lambda0
ct_model.weight_model.A = dt_model.weight_model.A
ct_model.weight_model.W = dt_model.weight_model.W
print "CT LL: ", ct_model.heldout_log_likelihood(S_ct, C_ct, T)

# Fit the CT model
ct_lls = [ct_model.log_likelihood()]
N_samples = 100
for itr in progprint_xrange(N_samples, perline=25):
    ct_model.resample_model()
    ct_lls.append(ct_model.log_likelihood())
    assert np.all(ct_model.weight_model.A==1)

# Now fit a DT model
dt_model_test = pyhawkes.models.\
    DiscreteTimeNetworkHawkesModelSpikeAndSlab(K=K, dt=dt, dt_max=dt_max, B=B,
                                               network_hypers=network_hypers)
dt_model_test.add_data(S_dt)
dt_lls = []
for itr in progprint_xrange(N_samples, perline=25):
    dt_model_test.resample_model()
    dt_lls.append(dt_model_test.log_likelihood())
    assert np.all(dt_model_test.weight_model.A==1)


plt.figure()
plt.plot(ct_lls, 'b')
plt.plot(dt_lls, 'r')
plt.show()
