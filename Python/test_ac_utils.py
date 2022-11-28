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
