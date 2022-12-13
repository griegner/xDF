import matman
import numpy as np
import pytest
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
        assert np.isclose(r_corr[0, 1], corr_rho, atol=0.05)
        assert np.isclose(z_corr[0, 1], rho_to_z, atol=1.0)


def test_ac_sum():

    # create 2x3 array
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    arr_expected = np.array([[[2, 4, 6], [5, 7, 9]], [[5, 7, 9], [8, 10, 12]]])
    arr_sum = matman.ac_sum(arr, n_lags=3)
    assert np.array_equal(arr_sum, arr_expected)

    # create 3x4 array
    arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    arr_lag_1 = np.array([[4, 8, 12], [8, 12, 16], [12, 16, 20]])
    arr_sum = matman.ac_sum(arr, n_lags=4)
    assert np.array_equal(arr_sum[:, :, 1], arr_lag_1)

    # test (n_lags, n_regions) assertion
    with pytest.raises(AssertionError):
        matman.ac_sum(arr, n_lags=3)


def test_ac_prod():

    # create 2x3 array
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    arr_expected = np.array([[[1, 4, 9], [4, 10, 18]], [[4, 10, 18], [16, 25, 36]]])
    arr_prod = matman.ac_prod(arr, n_lags=3)
    assert np.array_equal(arr_prod, arr_expected)

    # create 3x4 array
    arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    arr_lag_1 = np.array([[4, 12, 20], [12, 36, 60], [20, 60, 100]])
    arr_prod = matman.ac_prod(arr, n_lags=4)
    assert np.array_equal(arr_prod[:, :, 1], arr_lag_1)

    # test (n_lags, n_regions) assertion
    with pytest.raises(AssertionError):
        matman.ac_prod(arr, n_lags=3)
