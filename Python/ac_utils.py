#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 16:53:44 2018

@author: sorooshafyouni
University of Oxford, 2019
srafyouni@gmail.com
"""
import numpy as np


def _nextpow2(n_fft):
    """The computational complexity of an FFT calculation is lower for a power of 2,
    by `O(n log(n))`.

    Parameters
    ----------
    n_fft : int
        Number of samples as input to FFT.

    Returns
    -------
    int
        The length of the FFT output. If larger than the input, the FFT output is padded with zeros.
    """
    return 1 if n_fft == 0 else int(2 ** np.ceil(np.log2(n_fft)))


def autocorr(x, t=1):
    """dumb autocorrelation on a 1D array,
    almost identical to matlab autocorr()"""
    x = x.copy()
    x = x - np.tile(np.mean(x), np.size(x))
    AC = np.zeros(t)
    for l in np.arange(t):
        AC[l] = np.corrcoef(np.array([x[: len(x) - l], x[l:]]))[0, 1]
    return AC


def ac_fft(X, n_timepoints):
    """Approximates the autocorrelation functions of `n_regions` over `n_timepoints` using FFT.

    Convolution theorem: convolution of each timeseries with itself using pointwise multiplication
    in the frequency domain - the complex conjugate of the discrete Fourier Transform.

    Parameters
    ----------
    X : array_like (n_regions x n_timepoints)
        An array containing the time series of each regions.
    n_timepoints : int
        Number of samples in X.

    Returns
    -------
    X_ac : array_like (n_regions x n_lags)
        The full-lag autocorrelation function (ACF) for each region.
    ci : list [lower, upper]
        95% confidence intervals of X_ac.
    """

    assert X.shape[1] == n_timepoints, "X should be in (n_regions x n_timepoints) form."

    X_demean = X - X.mean(axis=1).reshape(-1, 1)  # demean along n_timepoints

    n_timepoints_fft = _nextpow2(2 * n_timepoints - 1)  # zero-pad the hell out!

    X_fft = np.fft.fft(X_demean, n=n_timepoints_fft, axis=1)  # frequency domain

    # convolution theorem: using the complex conjugate of X_fft
    X_cov = np.real(np.fft.ifft(X_fft * np.conj(X_fft), axis=1))  # time domain
    X_cov = X_cov[:, :n_timepoints]  # remove zero-padding

    X_var = np.sum(X_demean**2, axis=1)
    X_ac = X_cov / X_var.reshape(-1, 1)  # covariances to correlations

    bnd = (np.sqrt(2) * 1.3859) / np.sqrt(n_timepoints)  # assumes normality for X_ac
    ci = [-bnd, bnd]

    return X_ac, ci


def xc_fft(X, n_timepoints):
    """Approximates the pairwise cross-correlations of `n_regions` over `n_timepoints` using FFT.

    Calculates a 3D array of (n_regions x n_regions x [n_timepoints-1]*2+1)
    1) diagonals: *auto*correlations across positive and negative lags of `n_timepoints`
    2) off diagnonals: *cross*correlations between pairs of `n_regions` across positive and negative lags of `n_timepoints`

    Parameters
    ----------
    X : array_like (n_regions x n_timepoints)
        An array containing the time series of each regions.
    n_timepoints : int
        Number of samples in X.

    Returns
    -------
    X_xc : array_like (n_regions x n_regions x [n_lags-1]*2+1)
        The full-lag cross-correlations between each pair of regions.
    lag_idx : 1D array with the indices of the all lags (axis=2 of X_xc)
    """

    assert X.shape[1] == n_timepoints, "X should be in (n_regions x n_timepoints) form."

    n_regions = X.shape[0]

    X_demean = X - X.mean(axis=1).reshape(-1, 1)  # demean along n_timepoints

    n_timepoints_fft = _nextpow2(2 * n_timepoints - 1)  # zero-pad the hell out!

    X_fft = np.fft.fft(X_demean, n=n_timepoints_fft, axis=1)  # frequency domain

    n_lags = (n_timepoints - 1) * 2 + 1
    X_xc = np.zeros([n_regions, n_regions, n_lags])  # initialize X_xc 3d array

    triu_i, triu_j = np.triu_indices(n_regions, 0)  # upper triangle including diagonal

    for i, j in zip(triu_i, triu_j):  # loop around edges.

        X_cov = np.real(np.fft.ifft(X_fft[i, :] * np.conj(X_fft[j, :]), axis=0))
        X_cov = np.concatenate([X_cov[-n_timepoints + 1 :], X_cov[:n_timepoints]])[::-1]

        X_var = np.sqrt(np.sum(X_demean[i, :] ** 2) * np.sum(X_demean[j, :] ** 2))

        X_cor = X_cov / X_var
        X_xc[i, j, :] = X_cor  # upper triangle
        X_xc[j, i, :] = X_cor  # lower triangle, overwrite the diagonal

        del X_cov, X_var, X_cor

    lag_idx = np.arange(-(n_timepoints - 1), n_timepoints)  # index of lags axis=2

    return X_xc, lag_idx


def ACL(X, n_timepoints):
    """
    Calculates Autocorrelation Length of time series ts
    SA, Ox, 2019
    """
    return np.sum(ac_fft(X, n_timepoints) ** 2, axis=1)

    ######################## AC Reg Functions #################################


def tukey_taper(ac, n_lags, breakpoint):
    """Multiply the ACF by a scaling window for `n_lags` <= `breakpoint`, and zero `n_lags` > `breakpoint`.

    Parameters
    ----------
    ac : array_like (1d, 2d, or 3d)
        The full-lag autocorrelation function (ACF).
    n_lags : int
        The number of lags used to calculate the ACF.
    breakpoint : int
        The index of the breakpoint.

    Returns
    -------
    ac_tapered: array_like (1d, 2d, or 3d)
        The tapered ACF, zeroed after `breakpoint` along the `n_lags` axis.
    """
    assert n_lags in ac.shape, "`n_lags` not in `ac`"

    breakpoint = int(np.round(breakpoint))
    tukey_multiplier = (1 + np.cos(np.arange(1, breakpoint) * np.pi / breakpoint)) / 2
    ac_tapered = np.zeros(ac.shape)

    if len(ac.shape) == 2:
        assert ac.shape[1] == n_lags, "ac should be in (n_regions x n_lags) form."
        ac_tapered[:, : breakpoint - 1] = tukey_multiplier * ac[:, : breakpoint - 1]

    elif len(ac.shape) == 3:
        ac_tapered[:, :, : breakpoint - 1] = (
            tukey_multiplier * ac[:, :, : breakpoint - 1]
        )

    elif len(ac.shape) == 1:
        ac_tapered[: breakpoint - 1] = tukey_multiplier * ac[: breakpoint - 1]

    return ac_tapered


def truncate(ac, breakpoint):
    """Zero the ACF at `n_lags` >= `breakpoint`.

    Parameters
    ----------
    ac : array_like (1d, 2d, or 3d)
        The full-lag autocorrelation function (ACF).

    breakpoint : int
        The index of the breakpoint.

    Returns
    -------
    ac_truncated : array_like (1d, 2d, or 3d)
        The ACF zeroed after `breakpoint` along the `n_lags` axis.
    """
    mask = np.zeros(ac.shape)
    if len(ac.shape) == 2:
        mask[:, :breakpoint] = 1

    elif len(ac.shape) == 3:
        mask[:, :, :breakpoint] = 1

    elif len(ac.shape) == 1:
        mask[:breakpoint] = 1

    return mask * ac


def adaptive_truncate(X_ac, n_lags):
    """An adaptive truncation method to regularize ACF estimates by zeroing `n_lags` >= `breakpoint`,
    where the breakpoint is the smallest lag where the null hypothesis is not rejected at uncorrected alpha = 5%.
    This is based on approximate normality of the ACF and sampling variance of 1/n_timepoints.

    Parameters
    ----------
    X_ac : array_like (n_regions x n_timepoints)
        The full-lag autocorrelation function (ACF) for each region.
    n_lags : int
        The number of lags used to calculate the ACF.

    Returns
    -------
    X_ac_truncated : array_like (n_regions x n_timepoints)
        The ACF zeroed after `breakpoint` for each region.
    breakpoints : array_like (n_regions)
        The index of the breakpoint for each region.
    """
    assert X_ac.shape[1] == n_lags, "X_ac should be in (n_regions x n_lags) form."

    # assumes normality for AC, 95% CI
    bnd = (np.sqrt(2) * 1.3859) / np.sqrt(n_lags)

    mask = np.abs(X_ac) > bnd
    breakpoints = np.argmin(mask, axis=1)

    # set all values after the breakpoint to zero
    for region, breakpoint in enumerate(breakpoints):
        mask[region, breakpoint + 1 :] = False

    return X_ac * mask, breakpoints
