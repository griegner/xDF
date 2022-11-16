#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 13:31:32 2019

@author: sorooshafyouni
University of Oxford, 2019
"""

import os
import sys

import numpy as np
import scipy.stats as sp

import ac_utils
import matman


def calc_xdf(
    X,
    n_timepoints,
    method="truncate",
    methodparam="adaptive",
    verbose=True,
    TV=True,
    copy=True,
):

    # if not verbose: blockPrint()

    if copy:  # Make sure you are not messing around with the original time series
        X = X.copy()

    if X.shape[1] != n_timepoints:
        if verbose:
            print(
                "calc_xdf::: Input should be in (n_regions x n_timepoints) form, the matrix was transposed."
            )
        X = X.T

    n_regions = X.shape[0]

    X_std = np.std(X, axis=1, ddof=1)
    X = X / X_std.reshape(-1, 1)  # standardise
    print("calc_xdf::: Time series standardised by their standard deviations.")

    ##### Estimate xC and AC ---------------------------------------------------

    # Corr----------------------------------------------------------------------
    rho, znaive = matman.corr_mat(X, n_timepoints)

    # Autocorr------------------------------------------------------------------
    [ac, CI] = ac_utils.ac_fft(X, n_timepoints)
    ac = ac[
        :, 1 : n_timepoints - 1
    ]  # The last element of ACF is rubbish, the first one is 1
    nLg = n_timepoints - 2

    # Cross-corr----------------------------------------------------------------
    [xcf, lid] = ac_utils.xc_fft(X, n_timepoints)

    xc_p = xcf[:, :, 1 : n_timepoints - 1]
    xc_p = np.flip(xc_p, axis=2)  # positive-lag xcorrs
    xc_n = xcf[:, :, n_timepoints:-1]  # negative-lag xcorrs

    ##### Start of Regularisation-----------------------------------------------
    if method.lower() == "tukey":
        M = np.sqrt(n_timepoints) if methodparam == "" else methodparam
        if verbose:
            print(
                f"calc_xdf::: AC Regularisation: Tukey tapering of M = {int(np.round(M))}"
            )

        ac = ac_utils.tukeytaperme(ac, nLg, M)
        xc_p = ac_utils.tukeytaperme(xc_p, nLg, M)
        xc_n = ac_utils.tukeytaperme(xc_n, nLg, M)

        # print(np.round(ac[0,0:50],4))

    elif method.lower() == "truncate":
        if type(methodparam) == str:  # Adaptive Truncation
            if methodparam.lower() != "adaptive":
                raise ValueError(
                    "What?! Choose adaptive as the option, or pass an integer for truncation"
                )
            if verbose:
                print("calc_xdf::: AC Regularisation: Adaptive Truncation")
            [ac, bp] = ac_utils.shrinkme(ac, nLg)
            # truncate the cross-correlations, by the breaking point found from the ACF. (choose the largest of two)
            for i in np.arange(n_regions):
                for j in np.arange(n_regions):
                    maxBP = np.max([bp[i], bp[j]])
                    xc_p[i, j, :] = ac_utils.curbtaperme(
                        xc_p[i, j, :], nLg, maxBP, verbose=False
                    )
                    xc_n[i, j, :] = ac_utils.curbtaperme(
                        xc_n[i, j, :], nLg, maxBP, verbose=False
                    )
        elif type(methodparam) == int:  # Npne-Adaptive Truncation
            if verbose:
                print(
                    f"calc_xdf::: AC Regularisation: Non-adaptive Truncation on M = {str(methodparam)}"
                )

            ac = ac_utils.curbtaperme(ac, nLg, methodparam)
            xc_p = ac_utils.curbtaperme(xc_p, nLg, methodparam)
            xc_n = ac_utils.curbtaperme(xc_n, nLg, methodparam)

        else:
            raise ValueError(
                "calc_xdf::: methodparam for truncation method should be either str or int."
            )

    ##### Start of the Monster Equation----------------------------------------
    wgt = np.arange(nLg, 0, -1)
    wgtm2 = np.tile((np.tile(wgt, [n_regions, 1])), [n_regions, 1])
    wgtm3 = np.reshape(
        wgtm2, [n_regions, n_regions, np.size(wgt)]
    )  # this is shit, eats all the memory!
    """
     VarHatRho = (Tp*(1-rho.^2).^2 ...
     +   rho.^2 .* sum(wgtm3 .* (SumMat(ac.^2,nLg)  +  xc_p.^2 + xc_n.^2),3)...         %1 2 4
     -   2.*rho .* sum(wgtm3 .* (SumMat(ac,nLg)    .* (xc_p    + xc_n))  ,3)...         % 5 6 7 8
     +   2      .* sum(wgtm3 .* (ProdMat(ac,nLg)    + (xc_p   .* xc_n))  ,3))./(T^2);   % 3 9 
    """

    Tp = n_timepoints - 1
    # Da Equation!--------------------------------------------------------------
    VarHatRho = (
        Tp * (1 - rho**2) ** 2
        + rho**2
        * np.sum(wgtm3 * (matman.SumMat(ac**2, nLg) + xc_p**2 + xc_n**2), axis=2)
        - 2 * rho * np.sum(wgtm3 * (matman.SumMat(ac, nLg) * (xc_p + xc_n)), axis=2)
        + 2 * np.sum(wgtm3 * (matman.ProdMat(ac, nLg) + (xc_p * xc_n)), axis=2)
    ) / (n_timepoints**2)

    ##### Truncate to Theoritical Variance --------------------------------------
    TV_val = (1 - rho**2) ** 2 / n_timepoints
    TV_val[range(n_regions), range(n_regions)] = 0

    idx_ex = np.where(VarHatRho < TV_val)
    NumTVEx = (np.shape(idx_ex)[1]) / 2
    # print(NumTVEx)

    if NumTVEx > 0 and TV:
        if verbose:
            print("Variance truncation is ON.")
        # Assuming that the variance can *only* get larger in presence of autocorrelation.
        VarHatRho[idx_ex] = TV_val[idx_ex]
        FGE = n_regions * (n_regions - 1) / 2
        if verbose:
            print(
                f"calc_xdf::: {str(NumTVEx)} ({str(round(NumTVEx / FGE * 100, 3))}%) edges had variance smaller than the textbook variance!"
            )

    elif verbose:
        print("calc_xdf::: NO truncation to the theoritical variance.")

    # Sanity Check:
    #        for ii in np.arange(NumTVEx):
    #            print( str( idx_ex[0][ii]+1 ) + '  ' + str( idx_ex[1][ii]+1 ) )

    # -------------------------------------------------------------------------
    #####Start of Statistical Inference -------------------------------------------

    # Well, these are all Matlab and pretty useless -- copy pasted them just in case though...
    # Pearson's turf -- We don't really wanna go there, eh?
    # rz      = rho./sqrt((ASAt));     %abs(ASAt), because it is possible to get negative ASAt!
    # r_pval  = 2 * normcdf(-abs(rz)); %both tails
    # r_pval(1:nn+1:end) = 0;          %NaN screws up everything, so get rid of the diag, but becareful here.

    # Our turf--------------------------------

    rf = np.arctanh(rho)
    sf = VarHatRho / (
        (1 - rho**2) ** 2
    )  # delta method; make sure the N is correct! So they cancel out.
    rzf = rf / np.sqrt(sf)
    f_pval = 2 * sp.norm.cdf(-abs(rzf))  # both tails

    # diagonal is rubbish;
    VarHatRho[range(n_regions), range(n_regions)] = 0
    f_pval[
        range(n_regions), range(n_regions)
    ] = 0  # NaN screws up everything, so get rid of the diag, but be careful here.
    rzf[range(n_regions), range(n_regions)] = 0

    return {
        "p": f_pval,
        "z": rzf,
        "znaive": znaive,
        "v": VarHatRho,
        "TV": TV_val,
        "TVExIdx": idx_ex,
    }


def blockPrint():
    """disable verbose"""
    sys.stdout = open(os.devnull, "w")


def enablePrint():
    """enable verbose"""
    sys.stdout = sys.__stdout__
