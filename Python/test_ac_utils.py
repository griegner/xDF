import ac_utils
import numpy as np
from test_data import generate_X


def test_ac_fft():
    rng = np.random.default_rng(1)  # initialize random number generator
    n_timepoints = 1200

    for i in range(1, 10, 2):
        ar_rho = i / 10

        X = generate_X(rng, n_timepoints=n_timepoints, ar_rho=ar_rho, corr_rho=0)
        X_ac, _ = ac_utils.ac_fft(X, n_timepoints)

        assert np.allclose(X_ac[:, 1], ar_rho, atol=0.06)


def test_xc_fft():
    rng1 = np.random.default_rng(1)
    rng2 = np.random.default_rng(10)
    n_timepoints = 1200

    for i in range(1, 10, 2):
        corr_rho = i / 10  # set cross correlation parameter

        X1 = generate_X(rng1, n_timepoints=n_timepoints, ar_rho=0.4, corr_rho=corr_rho)
        X2 = generate_X(rng2, n_timepoints=n_timepoints, ar_rho=0.8, corr_rho=corr_rho)
        X = np.vstack((X1, X2))
        X_xc, _ = ac_utils.xc_fft(X, n_timepoints)

        assert np.isclose(X_xc[0, 1, n_timepoints - 1], corr_rho, atol=0.05)  # zero lag
        assert np.isclose(X_xc[2, 3, n_timepoints - 1], corr_rho, atol=0.05)
