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

    for ac in range(1, 10, 2):
        ar_rho = ac / 10

        for xc in range(1, 10, 2):
            corr_rho = xc / 10

            X1 = generate_X(rng1, n_timepoints, ar_rho, corr_rho)
            X2 = generate_X(rng2, n_timepoints, ar_rho, corr_rho)
            X = np.vstack((X1, X2))
            X_xc, _ = ac_utils.xc_fft(X, n_timepoints)

            # test *cross*correlations at lag 0
            assert np.isclose(X_xc[0, 1, n_timepoints - 1], corr_rho, atol=0.1)
            assert np.isclose(X_xc[2, 3, n_timepoints - 1], corr_rho, atol=0.1)

            # compare to numpy corrcoef(X) function
            assert np.allclose(X_xc[:, :, n_timepoints - 1], np.corrcoef(X), atol=0.001)

            # test *auto*correlations at lag 1 (AR1)
            assert np.allclose(
                np.diag(X_xc[:, :, n_timepoints]), np.full(4, ar_rho), atol=0.06
            )


def test_tukey_taper():

    n_lags = 25
    breakpoint = 20
    tukey_multiplier = (1 + np.cos(np.arange(1, breakpoint) * np.pi / breakpoint)) / 2

    # sample ACF
    expected_ac = np.linspace(0.9, 0.1, n_lags)
    expected_ac_tapered = np.zeros(expected_ac.shape)
    expected_ac_tapered[: breakpoint - 1] = (
        tukey_multiplier * expected_ac[: breakpoint - 1]
    )

    # test 1d array
    ac_tapered = ac_utils.tukey_taper(expected_ac, n_lags=n_lags, breakpoint=breakpoint)
    assert np.array_equal(ac_tapered, expected_ac_tapered)

    # test 2d array
    expected_ac = np.vstack((expected_ac, expected_ac))
    expected_ac_tapered = np.vstack((expected_ac_tapered, expected_ac_tapered))
    ac_tapered = ac_utils.tukey_taper(expected_ac, n_lags=n_lags, breakpoint=breakpoint)
    assert np.array_equal(ac_tapered, expected_ac_tapered)

    # test 3d array
    expected_xc = np.dstack((expected_ac, expected_ac)).swapaxes(2, 1)
    expected_xc_tapered = np.dstack(
        (expected_ac_tapered, expected_ac_tapered)
    ).swapaxes(2, 1)
    xc_tapered = ac_utils.tukey_taper(expected_xc, n_lags=n_lags, breakpoint=breakpoint)
    assert np.array_equal(xc_tapered, expected_xc_tapered)


def test_truncate():

    n_lags = 25
    breakpoint = 10

    # sample ACF
    expected_ac = np.linspace(0.9, 0.1, n_lags)
    expected_ac_truncated = np.zeros(expected_ac.shape)
    expected_ac_truncated[:breakpoint] = expected_ac[:breakpoint]

    # test 1d array
    ac_truncated = ac_utils.truncate(expected_ac, breakpoint=breakpoint)
    assert np.array_equal(ac_truncated, expected_ac_truncated)

    # test 2d array
    expected_ac = np.vstack((expected_ac, expected_ac))
    expected_ac_truncated = np.vstack((expected_ac_truncated, expected_ac_truncated))
    ac_truncated = ac_utils.truncate(expected_ac, breakpoint=breakpoint)
    assert np.array_equal(ac_truncated, expected_ac_truncated)

    # test 3d array
    expected_xc = np.dstack((expected_ac, expected_ac)).swapaxes(2, 1)
    expected_xc_truncated = np.dstack(
        (expected_ac_truncated, expected_ac_truncated)
    ).swapaxes(2, 1)
    xc_truncated = ac_utils.truncate(expected_xc, breakpoint=breakpoint)
    assert np.array_equal(xc_truncated, expected_xc_truncated)


def test_adaptive_truncate():
    rng = np.random.default_rng(1)
    n_timepoints = 1200

    X = generate_X(rng, n_timepoints=n_timepoints, ar_rho=0.5, corr_rho=0)
    X_ac, _ = ac_utils.ac_fft(X, n_timepoints)
    X_ac, breakpoints = ac_utils.adaptive_truncate(X_ac, n_timepoints)

    assert list(breakpoints) == [4, 5]
    assert X_ac[0, 4] == X_ac[0, 5] == 0
