import matman
import numpy as np
from test_data import generate_X


def test_corr_mat():

    rng = np.random.default_rng(1)  # initialize random number generator
    n_timepoints = 1200

    for i in range(1, 10, 2):
        corr_rho = i / 10  # set cross correlation parameter
        rho_to_z = np.arctanh(corr_rho) * np.sqrt(n_timepoints - 3)

        X = generate_X(rng, n_timepoints=n_timepoints, ar_rho=0, corr_rho=corr_rho)
        r_corr, z_corr = matman.corr_mat(X, n_timepoints=n_timepoints)

        # difference btw estimated and true parameters
        assert np.allclose(r_corr[0, 1], corr_rho, atol=0.05)
        assert np.allclose(z_corr[0, 1], rho_to_z, atol=1.0)
