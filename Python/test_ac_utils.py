import ac_utils
import numpy as np
from test_data import generate_X


def test_ac_fft():
    rng = np.random.default_rng(1)  # initialize random number generator
    n_timepoints = 1200

    for i in range(1, 10, 2):
        ar_rho = i / 10

        X = generate_X(rng, n_timepoints=n_timepoints, ar_rho=ar_rho, corr_rho=0)
        ac, _ = ac_utils.ac_fft(X, n_timepoints)

        assert np.allclose(ac[:, 1], ar_rho, atol=0.06)
