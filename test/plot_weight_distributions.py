import numpy as np
from scipy.stats import gamma
import matplotlib.pyplot as plt

def plot_weight_distributions(kappa_delta, v_delta,
                              kappa_1, v_1,
                              kappa_0, v_0):

    w = np.linspace(0,0.75, num=1000)
    plt.figure()
    plt.plot(w, gamma(kappa_delta, scale=1.0/v_delta).pdf(w), '-k')
    plt.plot(w, gamma(kappa_0, scale=1.0/v_0).pdf(w), '-b')
    plt.plot(w, gamma(kappa_1, scale=1.0/v_1).pdf(w), '-r')
    plt.xlabel('w')
    plt.ylabel('p(w)')
    plt.show()

def plot_synthetic_weight_distributions():
    # Delta function is
    kappa_delta = 0.01
    v_delta     = 100.0

    # K=20, C=5
    kappa_1 = 10.0
    v_1      = kappa_1 * 5.0

    kappa_0 = 10.0
    v_0     = kappa_0 * 10.0

    plot_weight_distributions(kappa_delta, v_delta,
                              kappa_1, v_1,
                              kappa_0, v_0)

plot_synthetic_weight_distributions()