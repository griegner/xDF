import matman
import numpy as np


def _sim_ar1(x, rho=None):
    """simulate AR(1) process"""
    x_ar1 = [x[0]]
    x_ar1.extend(rho * x_ar1[-1] + x_t for x_t in x[1:])
    return np.array(x_ar1)


def _sim_corr(x, y, rho=None):
    """simulate correlation between timeseries"""
    return (rho * x) + (y * np.sqrt(1 - rho**2))


def _generate_X(n_timepoints=1200, ar_rho=0.5, corr_rho=0.5, rng=None):
    """generate X matrix (2 regions x n_timepoints) for testing"""
    x = rng.standard_normal(n_timepoints)  # iid (0, 1)
    y = rng.standard_normal(n_timepoints)
    y_corr = _sim_corr(x, y, corr_rho)  # cross correlation
    x_ar1 = _sim_ar1(x, ar_rho)  # serial correlation
    y_corr_ar1 = _sim_ar1(y_corr, ar_rho)
    return np.array([x_ar1, y_corr_ar1])


def test_corr_mat():

    rng = np.random.default_rng(12)  # initialize random number generator
    corr_rho = 0.5  # set correlation parameter
    n_timepoints = 1200
    rho_to_z = np.arctanh(corr_rho) * np.sqrt(n_timepoints - 3)
    X = _generate_X(n_timepoints=n_timepoints, rng=rng)

    r_corr, z_corr = matman.corr_mat(X, n_timepoints=n_timepoints)

    assert np.allclose(r_corr[0, 1], corr_rho, atol=0.001)
    assert np.allclose(z_corr[0, 1], rho_to_z, atol=0.01)
