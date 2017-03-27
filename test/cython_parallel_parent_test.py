import numpy as np

from pyhawkes.internals.parent_updates import mf_update_Z

def test_parallel_parent_updates():
    """
    Just run the parent updates in a parallel loop and watch htop
    to make sure that all the cores are being used.
    :return:
    """

    T = 10000
    K = 100
    B = 3
    S = np.random.poisson(2.0, size=((T,K))).astype(np.int)

    EZ0 = np.zeros((T,K))
    EZ  = np.zeros((T,K,K,B))

    exp_E_log_lambda0 = np.random.gamma(1.0, 1.0, size=(K))
    exp_E_log_W       = np.random.gamma(1.0, 1.0, size=(K,K))
    exp_E_log_g       = np.random.gamma(1.0, 1.0, size=(K,K,B))
    F                 = np.random.gamma(1.0, 1.0, size=(T,K,B))

    for itr in range(1000):
        if itr % 10 == 0:
            print("Iteration\t", itr)

        mf_update_Z(EZ0,
                    EZ,
                    S,
                    exp_E_log_lambda0,
                    exp_E_log_W,
                    exp_E_log_g,
                    F)

        # It looks like the big problem is that check_EZ
        # is done on a single core (slow!)
        # check_EZ(EZ0, EZ, S)

def check_EZ(EZ0, EZ, S):
    """
    Check that Z adds up to the correct amount
    :return:
    """
    EZsum = EZ0 + EZ.sum(axis=(1,3))
    # assert np.allclose(self.S, Zsum), "_check_Z failed. Zsum does not add up to S!"
    if not np.allclose(S, EZsum):
        print("_check_Z failed. Zsum does not add up to S!")
        import pdb; pdb.set_trace()

test_parallel_parent_updates()