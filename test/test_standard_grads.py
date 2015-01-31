from scipy.optimize import check_grad
import numpy as np
from pyhawkes.models import DiscreteTimeStandardHawkesModel, DiscreteTimeNetworkHawkesModelSpikeAndSlab

def test_gradients():
    K = 1
    B = 3
    T = 100
    dt = 1.0
    true_model = DiscreteTimeNetworkHawkesModelSpikeAndSlab(K=K, B=B, dt=dt)
    S,R = true_model.generate(T=T)

    # Test with a standard Hawkes model
    test_model = DiscreteTimeStandardHawkesModel(K=K, B=B, dt=dt)
    test_model.add_data(S)

    # Check gradients with the initial parameters
    def objective(x):
        test_model.weights[0,:] = np.exp(x)
        return test_model.log_likelihood()

    def gradient(x):
        test_model.weights[0,:] = np.exp(x)
        return test_model.compute_gradient(0)

    print "Checking initial gradient: "
    print gradient(np.log(test_model.weights[0,:]))
    check_grad(objective, gradient,
               np.log(test_model.weights[0,:]))

    print "Checking gradient at true model parameters: "
    test_model.initialize_with_gibbs_model(true_model)

    print gradient(np.log(test_model.weights[0,:]))
    check_grad(objective, gradient,
               np.log(test_model.weights[0,:]))




test_gradients()

