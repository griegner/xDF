#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 13:31:32 2019

@author: sorooshafyouni
University of Oxford, 2019
"""

import os
import sys

import ac_utils
import matman
import numpy as np
import scipy.stats as sp


def calc_xdf(
    X,
    n_timepoints,
    method="truncate",
    methodparam="adaptive",
    verbose=True,
    truncate_variance=True,
):

    X = X.copy()  # Make sure you are not messing around with the original time series
    assert X.shape[1] == n_timepoints, "X should be in (n_regions x n_timepoints) form"

    n_regions = X.shape[0]

    X_std = np.std(X, axis=1, ddof=1)
    X = X / X_std.reshape(-1, 1)  # standardise
    print("calc_xdf::: Time series standardised by their standard deviations.")

    ##### Estimate xC and AC ---------------------------------------------------

    # Corr----------------------------------------------------------------------
    rho, znaive = matman.corr_mat(X, n_timepoints)

    # Autocorr------------------------------------------------------------------
    X_ac, ci = ac_utils.ac_fft(X, n_timepoints)
    # The last element of ACF is rubbish, the first one is 1
    X_ac = X_ac[:, 1 : n_timepoints - 1]
    n_lags = n_timepoints - 2

    # Cross-corr----------------------------------------------------------------
    X_xc, lag_idx = ac_utils.xc_fft(X, n_timepoints)

    X_xc_pos = X_xc[:, :, 1 : n_timepoints - 1]
    X_xc_pos = np.flip(X_xc_pos, axis=2)  # positive-lag xcorrs
    X_xc_neg = X_xc[:, :, n_timepoints:-1]  # negative-lag xcorrs

    ##### Start of Regularisation-----------------------------------------------
    breakpoint = np.sqrt(n_timepoints) if methodparam == "" else methodparam

    if method.lower() == "tukey":
        if verbose:
            print(
                f"calc_xdf::: AC Regularisation: Tukey tapering of M = {int(np.round(breakpoint))}"
            )
        X_ac = ac_utils.tukey_taper(X_ac, n_lags, breakpoint)
        X_xc_pos = ac_utils.tukey_taper(X_xc_pos, n_lags, breakpoint)
        X_xc_neg = ac_utils.tukey_taper(X_xc_neg, n_lags, breakpoint)

    elif method.lower() == "truncate":
        if type(methodparam) == str:  # Adaptive Truncation
            assert (
                methodparam.lower() == "adaptive"
            ), "What?! Choose adaptive as the option, or pass an integer for truncation"
            if verbose:
                print("calc_xdf::: AC Regularisation: Adaptive Truncation")
            X_ac, breakpoints = ac_utils.adaptive_truncate(X_ac, n_lags)
            # truncate the cross-correlations, by the breaking point found from the ACF. (choose the largest of two)
            for i in np.arange(n_regions):
                for j in np.arange(n_regions):  # iterate through every pair of regions
                    max_bp = np.max([breakpoints[i], breakpoints[j]])
                    X_xc_pos[i, j, :] = ac_utils.truncate(X_xc_pos[i, j, :], max_bp)
                    X_xc_neg[i, j, :] = ac_utils.truncate(X_xc_neg[i, j, :], max_bp)
        elif type(methodparam) == int:  # Npne-Adaptive Truncation
            if verbose:
                print(
                    f"calc_xdf::: AC Regularisation: Non-adaptive Truncation on M = {str(methodparam)}"
                )

            X_ac = ac_utils.truncate(X_ac, methodparam)
            X_xc_pos = ac_utils.truncate(X_xc_pos, methodparam)
            X_xc_neg = ac_utils.truncate(X_xc_neg, methodparam)

        else:
            raise ValueError(
                "calc_xdf::: methodparam for truncation method should be either str or int."
            )

    ##### Start of the Monster Equation--------------------------------------

    # Equation!--------------------------------------------------------------

    # fmt: off
    wgtm3 = np.tile(np.arange(n_lags, 0, -1), (n_regions, n_regions, 1))
    # reference equation (2)
    var_rho_hat = (
        (n_timepoints - 1) * (1 - rho**2) ** 2
        + rho**2 * np.sum(wgtm3 * (matman.ac_sum(X_ac**2, n_lags) + X_xc_pos**2 + X_xc_neg**2), axis=2)
        - 2 * rho * np.sum(wgtm3 * (matman.ac_sum(X_ac, n_lags) * (X_xc_pos + X_xc_neg)), axis=2)
        + 2 * np.sum(wgtm3 * (matman.ac_prod(X_ac, n_lags) + (X_xc_pos * X_xc_neg)), axis=2)
                ) / (n_timepoints**2)   
    # fmt: on

    ##### Truncate to Theoritical Variance ------------------------------------
    # Assuming that the variance can *only* get larger in presence of autocorrelation.
    var_rho = (1 - rho**2) ** 2 / n_timepoints
    var_rho[range(n_regions), range(n_regions)] = 0  # set the diagonal to zero

    idx_ex = np.where(var_rho_hat < var_rho)
    n_idx_ex = (np.shape(idx_ex)[1]) / 2

    if n_idx_ex > 0 and truncate_variance:
        if verbose:
            print("Variance truncation is ON.")
        var_rho_hat[idx_ex] = var_rho[idx_ex]
        n_edges = n_regions * (n_regions - 1) / 2
        if verbose:
            print(
                f"calc_xdf::: {str(n_idx_ex)} ({str(round(n_idx_ex / n_edges * 100, 3))}%) edges had variance smaller than the textbook variance!"
            )

    elif verbose:
        print("calc_xdf::: NO truncation to the theoritical variance.")

    rf = np.arctanh(rho)
    # delta method; make sure the n_regions is correct! So they cancel out.
    sf = var_rho_hat / ((1 - rho**2) ** 2)
    rzf = rf / np.sqrt(sf)
    f_pval = 2 * sp.norm.cdf(-abs(rzf))  # both tails

    # diagonal is rubbish;
    var_rho_hat[range(n_regions), range(n_regions)] = 0
    # NaN screws up everything, so get rid of the diag, but be careful here.
    f_pval[range(n_regions), range(n_regions)] = 0
    rzf[range(n_regions), range(n_regions)] = 0

    return {
        "p": f_pval,
        "z": rzf,
        "znaive": znaive,
        "v": var_rho_hat,
        "TV": var_rho,
        "TVExIdx": idx_ex,
    }
