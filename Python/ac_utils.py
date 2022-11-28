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

    triu_i, triu_j = np.triu_indices(n_regions, 1)  # upper triangle above diagonal

    for i, j in zip(triu_i, triu_j):  # loop around edges.

        X_cov = np.real(np.fft.ifft(X_fft[i, :] * np.conj(X_fft[j, :]), axis=0))
        X_cov = np.concatenate([X_cov[-n_timepoints + 1 :], X_cov[:n_timepoints]])[::-1]

        X_var = np.sqrt(np.sum(X_demean[i, :] ** 2) * np.sum(X_demean[j, :] ** 2))

        X_xc[i, j, :] = X_cov / X_var
        del X_cov, X_var

    X_xc = X_xc + np.transpose(X_xc, (1, 0, 2))  # add lower triangle
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


def curbtaperme(ac, n_timepoints, M, verbose=True):
    """
    Curb the autocorrelations, according to Anderson 1984
    multi-dimensional, and therefore is fine!
    SA, Ox, 2018
    """
    ac = ac.copy()
    M = int(round(M))
    msk = np.zeros(ac.shape)
    if len(ac.shape) == 2:
        if verbose:
            print("curbtaperme::: The input is 2D.")
        msk[:, 0:M] = 1

    elif len(ac.shape) == 3:
        if verbose:
            print("curbtaperme::: The input is 3D.")
        msk[:, :, 0:M] = 1

    elif len(ac.shape) == 1:
        if verbose:
            print("curbtaperme::: The input is 1D.")
        msk[:M] = 1

    return msk * ac


def shrinkme(ac, n_timepoints):
    """
    Shrinks the *early* bucnhes of autocorr coefficients beyond the CI.
    Yo! this should be transformed to the matrix form, those fors at the top
    are bleak!

    SA, Ox, 2018
    """
    ac = ac.copy()

    if ac.shape[1] != n_timepoints:
        ac = ac.T

    bnd = (np.sqrt(2) * 1.3859) / np.sqrt(n_timepoints)  # assumes normality for AC

    N = ac.shape[0]
    msk = np.zeros(ac.shape)
    BreakPoint = np.zeros(N)
    for i in np.arange(N):
        TheFirstFalse = np.where(
            np.abs(ac[i, :]) < bnd
        )  # finds the break point -- intercept
        if (
            np.size(TheFirstFalse) == 0
        ):  # if you coulnd't find a break point, then continue = the row will remain zero
            continue
        else:
            BreakPoint_tmp = TheFirstFalse[0][0]
        msk[i, :BreakPoint_tmp] = 1
        BreakPoint[i] = BreakPoint_tmp
    return ac * msk, BreakPoint
