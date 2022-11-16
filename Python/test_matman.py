import matman
import numpy as np
from test_data import generate_X


def test_corr_mat():

    rng = np.random.default_rng(12)  # initialize random number generator
    corr_rho = 0.5  # set cross correlation parameter
    n_timepoints = 1200
    rho_to_z = np.arctanh(corr_rho) * np.sqrt(n_timepoints - 3)
    X = generate_X(rng, n_timepoints=n_timepoints, corr_rho=corr_rho)

    r_corr, z_corr = matman.corr_mat(X, n_timepoints=n_timepoints)

    assert np.allclose(r_corr[0, 1], corr_rho, atol=0.001)
    assert np.allclose(z_corr[0, 1], rho_to_z, atol=0.01)
