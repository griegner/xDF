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
    X_ac : array_like (n_regions x n_timepoints)
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
    X_xc : array_like (n_regions x n_regions x [n_timepoints-1]*2+1)
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


def tukeytaperme(ac, n_timepoints, M, verbose=True):
    """
    performs single Tukey tapering for given length of window, M, and initial
    value, intv. intv should only be used on crosscorrelation matrices.

    SA, Ox, 2018
    """
    ac = ac.copy()
    # ----Checks:
    if n_timepoints not in ac.shape:
        raise ValueError("tukeytaperme::: There is something wrong, mate!")
        # print('Oi')
    # ----

    M = int(np.round(M))

    tukeymultiplier = (1 + np.cos(np.arange(1, M) * np.pi / M)) / 2
    tt_ts = np.zeros(ac.shape)

    if len(ac.shape) == 2:
        if ac.shape[1] != n_timepoints:
            ac = ac.T
        if verbose:
            print("tukeytaperme::: The input is 2D.")
        N = ac.shape[0]
        tt_ts[:, 0 : M - 1] = np.tile(tukeymultiplier, [N, 1]) * ac[:, 0 : M - 1]

    elif len(ac.shape) == 3:
        if verbose:
            print("tukeytaperme::: The input is 3D.")
        N = ac.shape[0]
        tt_ts[:, :, 0 : M - 1] = (
            np.tile(tukeymultiplier, [N, N, 1]) * ac[:, :, 0 : M - 1]
        )

    elif len(ac.shape) == 1:
        if verbose:
            print("tukeytaperme::: The input is 1D.")
        tt_ts[: M - 1] = tukeymultiplier * ac[: M - 1]

    return tt_ts


def curbtaperme(corr, max_breakpoint):
    """Zero the correlation estimates at lags k >= `max_breakpoint`.

    Parameters
    ----------
    corr : array_like (1d, 2d, or 3d)

    max_breakpoint : int
        Zero all lags k >= `max_breakpoint`.

    Returns
    -------
    msk * corr : array_like (1d, 2d, or 3d)
        The autocorrelations zeroed along the kth axis.
    """
    msk = np.zeros(corr.shape)
    if len(corr.shape) == 2:
        msk[:, :max_breakpoint] = 1

    elif len(corr.shape) == 3:
        msk[:, :, :max_breakpoint] = 1

    elif len(corr.shape) == 1:
        msk[:max_breakpoint] = 1

    return msk * corr


def adaptive_truncation(X_ac, n_timepoints):
    """An adaptive truncation method to regularize ACF estimates by zeroing lags k >= M,
    where M is the smallest lag where the null hypothesis is not rejected at uncorrected alpha = 5%.
    This is based on approximate normality of the ACF and sampling variance of 1/N.

    Parameters
    ----------
    X_ac : array_like (n_regions x n_timepoints)
        The full-lag autocorrelation function (ACF) for each region.
    n_timepoints : int
        The number of samples in X_ac.

    Returns
    -------
    X_ac : array_like (n_regions x n_timepoints)
        The truncated ACF for each region.
    breakpoints : array_like (n_regions)
        The index of the breakpoint for each region.
    """
    assert (
        X_ac.shape[1] == n_timepoints
    ), "X should be in (n_regions x n_timepoints) form."

    # assumes normality for AC, 95% CI
    bnd = (np.sqrt(2) * 1.3859) / np.sqrt(n_timepoints)

    mask = np.abs(X_ac) > bnd
    breakpoints = np.argmin(mask, axis=1)

    # set all values after the breakpoint to zero
    for region, breakpoint in enumerate(breakpoints):
        mask[region, breakpoint + 1 :] = False

    return X_ac * mask, breakpoints
