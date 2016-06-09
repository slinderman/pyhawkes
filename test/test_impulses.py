"""
Tests for impulse response functions
"""
import numpy as np

from pyhawkes.models import DiscreteTimeNetworkHawkesModelSpikeAndSlab

def test_normalization():
    dt      = 1.0
    dt_max  = 10.0
    model   = DiscreteTimeNetworkHawkesModelSpikeAndSlab(K=1, dt=dt, dt_max=dt_max)

    basis   = model.basis.basis
    volume  = dt * basis.sum(axis=0)

    import pdb; pdb.set_trace()
    assert np.allclose(volume, 1.0)

test_normalization()
