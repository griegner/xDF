import numpy as np


def _sim_ar1(x, rho=None):
    """simulate AR(1) process"""
    x_ar1 = [x[0]]
    x_ar1.extend(rho * x_ar1[-1] + x_t for x_t in x[1:])
    return np.array(x_ar1)


def _sim_corr(x, y, rho=None):
    """simulate correlation between timeseries"""
    return (rho * x) + (y * np.sqrt(1 - rho**2))


def generate_X(rng, n_timepoints=1200, ar_rho=0.5, corr_rho=0.5):
    """Generate X matrix for testing

    Parameters
    ----------
    rng : np.random.default_rng()
        Random number generator
    n_timepoints : int, optional
        Number of samples in X, by default 1200
    ar_rho : float, optional
        Autocorrelation (AR1) parameter, by default 0.5
    corr_rho : float, optional
        Cross correlation parameter, by default 0.5

    Returns
    -------
    X: ndarray (2 regions x n_timepoints)
        Simulated X matrix with specified degree of cross and serial correlation
    """
    x = rng.standard_normal(n_timepoints)  # iid (0, 1)
    y = rng.standard_normal(n_timepoints)
    y_corr = _sim_corr(x, y, corr_rho)  # cross correlation
    x_ar1 = _sim_ar1(x, ar_rho)  # serial correlation
    y_corr_ar1 = _sim_ar1(y_corr, ar_rho)
    return np.array([x_ar1, y_corr_ar1])
