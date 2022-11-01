#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 16:41:29 2019


Many of these functions were copy pasted from bctpy package:
    https://github.com/aestrivex/bctpy
under GNU V3.0:
    https://github.com/aestrivex/bctpy/blob/master/LICENSE
    


@author: sorooshafyouni
University of Oxford, 2019
"""

import numpy as np
import scipy.stats as sp
import statsmodels.stats.multitest as smmt


def OLSRes(YOrig, RG, n_timepoints, copy=True):
    """
    Or how to deconfound stuff!
    For regressing out stuff from your time series, quickly and nicely!
    SA,Ox,2019
    """
    if copy:
        YOrig = YOrig.copy()

    if YOrig.shape[0] != n_timepoints or RG.shape[0] != n_timepoints:
        raise ValueError("The Y and the X should be (n_timepoints x n_regions) format.")

    # demean anyways!
    mRG = np.mean(RG, axis=0)
    RG = RG - np.tile(mRG, (n_timepoints, 1))
    # B       = np.linalg.solve(RG,YOrig) # more stable than pinv
    invRG = np.linalg.pinv(RG)
    B = np.dot(invRG, YOrig)
    Yhat = np.dot(RG, B)  # find the \hat{Y}
    return YOrig - Yhat  # return the residuals - i.e. cleaned time series


def issymmetric(W):
    """Check whether a matrix is symmetric"""
    return (W.transpose() == W).all()


def SumMat(Y0, n_timepoints, copy=True):
    """
    Parameters
    ----------
    Y0 : a 2D matrix of size (n_timepoints x N)

    Returns
    -------
    SM : 3D matrix, obtained from element-wise summation of each row with other
         rows.

    SA, Ox, 2019
    """

    if copy:
        Y0 = Y0.copy()

    if Y0.shape[0] != n_timepoints:
        print(
            "SumMat::: Input should be in (n_timepoints x N) form, the matrix was transposed."
        )
        Y0 = Y0.T

    N = Y0.shape[1]
    Idx = np.triu_indices(N)
    # F = (N*(N-1))/2
    SM = np.empty([N, N, n_timepoints])
    for i in np.arange(0, np.size(Idx[0]) - 1):
        xx = Idx[0][i]
        yy = Idx[1][i]
        SM[xx, yy, :] = Y0[:, xx] + Y0[:, yy]
        SM[yy, xx, :] = Y0[:, yy] + Y0[:, xx]

    return SM


def ProdMat(Y0, n_timepoints, copy=True):
    """
    Parameters
    ----------
    Y0 : a 2D matrix of size (n_timepoints x N)

    Returns
    -------
    SM : 3D matrix, obtained from element-wise multiplication of each row with
         other rows.

    SA, Ox, 2019
    """

    if copy:
        Y0 = Y0.copy()

    if Y0.shape[0] != n_timepoints:
        print(
            "ProdMat::: Input should be in (n_timepoints x N) form, the matrix was transposed."
        )
        Y0 = Y0.T

    N = Y0.shape[1]
    Idx = np.triu_indices(N)
    # F = (N*(N-1))/2
    SM = np.empty([N, N, n_timepoints])
    for i in np.arange(0, np.size(Idx[0]) - 1):
        xx = Idx[0][i]
        yy = Idx[1][i]
        SM[xx, yy, :] = Y0[:, xx] * Y0[:, yy]
        SM[yy, xx, :] = Y0[:, yy] * Y0[:, xx]

    return SM


def corr_mat(
    X,
    n_timepoints,
    zero_diagonal=True,
    copy=True,
):
    """Compute a pairwise Pearson correlation matrix between regions of X.

    Parameters
    ----------
    X : array_like (n_regions x n_timepoints)
        An array containing the time series of each regions.
    n_timepoints : int
        Number of samples in X.
    zero_diagonal : bool, optional
        The diagonal of the correlation matrix is set to zero, by default True.
    copy : bool, optional
        Copy X before computing correlations, by default True.

    Returns
    -------
    rho: ndarray (n_regions x n_regions)
        Pairwise correlation coefficients.
    Znaive: ndarray (n_regions x n_regions)
        Fisher's z-transformed correlation coefficients (naive approach).
    """

    if copy:
        X = X.copy()

    if X.shape[1] != n_timepoints:
        print(
            "xDF::: Input should be in (n_regions x n_timepoints) form, the matrix was transposed."
        )
        X = X.T

    n_regions = X.shape[0]

    r_mat = np.corrcoef(X)

    if zero_diagonal:
        np.fill_diagonal(r_mat, 0)

    z_mat = np.arctanh(r_mat) * np.sqrt(n_timepoints - 3)  # check this

    return np.round(r_mat, 7), np.round(z_mat, 7)


def stat_threshold(Z, mce="fdr_bh", a_level=0.05, side="two", copy=True):
    """
    Threshold z maps

    Parameters
    ----------

    mce: multiple comparison error correction method, should be
    among of the options below. [defualt: 'fdr_bh'].
    The options are from statsmodels packages:

        `b`, `bonferroni` : one-step correction
        `s`, `sidak` : one-step correction
        `hs`, `holm-sidak` : step down method using Sidak adjustments
        `h`, `holm` : step-down method using Bonferroni adjustments
        `sh`, `simes-hochberg` : step-up method  (independent)
        `hommel` : closed method based on Simes tests (non-negative)
        `fdr_i`, `fdr_bh` : Benjamini/Hochberg  (non-negative)
        `fdr_n`, `fdr_by` : Benjamini/Yekutieli (negative)
        'fdr_tsbh' : two stage fdr correction (Benjamini/Hochberg)
        'fdr_tsbky' : two stage fdr correction (Benjamini/Krieger/Yekutieli)
        'fdr_gbs' : adaptive step-down fdr correction (Gavrilov, Benjamini, Sarkar)
    """

    if copy:
        Z = Z.copy()

    sideflag = 1 if side == "one" else 2
    Idx = np.triu_indices(Z.shape[0], 1)
    Zv = Z[Idx]

    Pv = sp.norm.cdf(-np.abs(Zv)) * sideflag

    [Hv, adjpvalsv] = smmt.multipletests(Pv, method=mce)[:2]
    adj_pvals = np.zeros(Z.shape)
    Zt = np.zeros(Z.shape)

    Zv[np.invert(Hv)] = 0
    Zt[Idx] = Zv
    Zt = Zt + Zt.T

    adj_pvals[Idx] = adjpvalsv
    adj_pvals = adj_pvals + adj_pvals.T

    adj_pvals[range(Z.shape[0]), range(Z.shape[0])] = 0

    return Zt, binarize(Zt), adj_pvals


def RemoveNeg(Mats, copy=True):
    """quickly remove negative values"""
    if copy:
        Mats = Mats.copy()
    Mats[Mats < 0] = 0
    return Mats


class MatManParamError(RuntimeError):
    pass


def threshold_absolute(W, thr, copy=True):
    """
    This function thresholds the connectivity matrix by absolute weight
    magnitude. All weights below the given threshold, and all weights
    on the main diagonal (self-self connections) are set to 0.

    If copy is not set, this function will *modify W in place.*

    Parameters
    ----------
    W : np.ndarray
        weighted connectivity matrix
    thr : float
        absolute weight threshold
    copy : bool
        if True, returns a copy of the matrix. Otherwise, modifies the matrix
        in place. Default value=True.

    Returns
    -------
    W : np.ndarray
        thresholded connectivity matrix
    """
    if copy:
        W = W.copy()
    np.fill_diagonal(W, 0)  # clear diagonal
    W[W < thr] = 0  # apply threshold
    return W


def threshold_proportional(W, p, copy=True):
    """
    This function "thresholds" the connectivity matrix by preserving a
    proportion p (0<p<1) of the strongest weights. All other weights, and
    all weights on the main diagonal (self-self connections) are set to 0.

    If copy is not set, this function will *modify W in place.*

    Parameters
    ----------
    W : np.ndarray
        weighted connectivity matrix
    p : float
        proportional weight threshold (0<p<1)
    copy : bool
        if True, returns a copy of the matrix. Otherwise, modifies the matrix
        in place. Default value=True.

    Returns
    -------
    W : np.ndarray
        thresholded connectivity matrix

    Notes
    -----
    The proportion of elements set to 0 is a fraction of all elements
    in the matrix, whether or not they are already 0. That is, this function
    has the following behavior:

    >> x = np.random.random((10,10))
    >> x_25 = threshold_proportional(x, .25)
    >> np.size(np.where(x_25)) #note this double counts each nonzero element
    46
    >> x_125 = threshold_proportional(x, .125)
    >> np.size(np.where(x_125))
    22
    >> x_test = threshold_proportional(x_25, .5)
    >> np.size(np.where(x_test))
    46

    That is, the 50% thresholding of x_25 does nothing because >=50% of the
    elements in x_25 are aleady <=0. This behavior is the same as in BCT. Be
    careful with matrices that are both signed and sparse.
    """

    if p > 1 or p < 0:
        raise MatManParamError("Threshold must be in range [0,1]")
    if copy:
        W = W.copy()
    n = len(W)  # number of nodes
    np.fill_diagonal(W, 0)  # clear diagonal

    if np.allclose(W, W.T):  # if symmetric matrix
        W[np.tril_indices(n)] = 0  # ensure symmetry is preserved
        ud = 2  # halve number of removed links
    else:
        ud = 1

    ind = np.where(W)  # find all links

    I = np.argsort(W[ind])[::-1]  # sort indices by magnitude

    en = int(round((n**2 - n) * p / ud))  # number of links to be preserved

    W[(ind[0][I][en:], ind[1][I][en:])] = 0  # apply threshold

    if ud == 2:  # if symmetric matrix
        W[:, :] = W + W.T  # reconstruct symmetry

    return W


def weight_conversion(W, wcm, copy=True):
    """
    W_bin = weight_conversion(W, 'binarize');
    W_nrm = weight_conversion(W, 'normalize');
    L = weight_conversion(W, 'lengths');

    This function may either binarize an input weighted connection matrix,
    normalize an input weighted connection matrix or convert an input
    weighted connection matrix to a weighted connection-length matrix.

    Binarization converts all present connection weights to 1.

    Normalization scales all weight magnitudes to the range [0,1] and
    should be done prior to computing some weighted measures, such as the
    weighted clustering coefficient.

    Conversion of connection weights to connection lengths is needed
    prior to computation of weighted distance-based measures, such as
    distance and betweenness centrality. In a weighted connection network,
    higher weights are naturally interpreted as shorter lengths. The
    connection-lengths matrix here is defined as the inverse of the
    connection-weights matrix.

    If copy is not set, this function will *modify W in place.*

    Parameters
    ----------
    W : NxN np.ndarray
        weighted connectivity matrix
    wcm : str
        weight conversion command.
        'binarize' : binarize weights
        'normalize' : normalize weights
        'lengths' : convert weights to lengths (invert matrix)
    copy : bool
        if True, returns a copy of the matrix. Otherwise, modifies the matrix
        in place. Default value=True.

    Returns
    -------
    W : NxN np.ndarray
        connectivity matrix with specified changes

    Notes
    -----
    This function is included for compatibility with BCT. But there are
    other functions binarize(), normalize() and invert() which are simpler to
    call directly.
    """
    if wcm == "binarize":
        return binarize(W, copy)
    elif wcm == "normalize":
        return normalize(W, copy)
    elif wcm == "lengths":
        return invert(W, copy)
    else:
        raise NotImplementedError("Unknown weight conversion command.")


def binarize(W, copy=True):
    """
    Binarizes an input weighted connection matrix.  If copy is not set, this
    function will *modify W in place.*

    Parameters
    ----------
    W : NxN np.ndarray
        weighted connectivity matrix
    copy : bool
        if True, returns a copy of the matrix. Otherwise, modifies the matrix
        in place. Default value=True.

    Returns
    -------
    W : NxN np.ndarray
        binary connectivity matrix
    """
    if copy:
        W = W.copy()
    W[W != 0] = 1
    return W


def normalize(W, copy=True):
    """
    Normalizes an input weighted connection matrix.  If copy is not set, this
    function will *modify W in place.*

    Parameters
    ----------
    W : np.ndarray
        weighted connectivity matrix
    copy : bool
        if True, returns a copy of the matrix. Otherwise, modifies the matrix
        in place. Default value=True.

    Returns
    -------
    W : np.ndarray
        normalized connectivity matrix
    """
    if copy:
        W = W.copy()
    W /= np.max(np.abs(W))
    return W


def invert(W, copy=True):
    """
    Inverts elementwise the weights in an input connection matrix.
    In other words, change the from the matrix of internode strengths to the
    matrix of internode distances.

    If copy is not set, this function will *modify W in place.*

    Parameters
    ----------
    W : np.ndarray
        weighted connectivity matrix
    copy : bool
        if True, returns a copy of the matrix. Otherwise, modifies the matrix
        in place. Default value=True.

    Returns
    -------
    W : np.ndarray
        inverted connectivity matrix
    """
    if copy:
        W = W.copy()
    E = np.where(W)
    W[E] = 1.0 / W[E]
    return W


def autofix(W, copy=True):
    """
    Fix a bunch of common problems. More specifically, remove Inf and NaN,
    ensure exact binariness and symmetry (i.e. remove floating point
    instability), and zero diagonal.


    Parameters
    ----------
    W : np.ndarray
        weighted connectivity matrix
    copy : bool
        if True, returns a copy of the matrix. Otherwise, modifies the matrix
        in place. Default value=True.

    Returns
    -------
    W : np.ndarray
        connectivity matrix with fixes applied
    """
    if copy:
        W = W.copy()

    # zero diagonal
    np.fill_diagonal(W, 0)

    # remove np.inf and np.nan
    W[np.logical_or(np.where(np.isinf(W)), np.where(np.isnan(W)))] = 0

    # ensure exact binarity
    u = np.unique(W)
    if np.all(np.logical_or(np.abs(u) < 1e-8, np.abs(u - 1) < 1e-8)):
        W = np.around(W, decimal=5)

    # ensure exact symmetry
    if np.allclose(W, W.T):
        W = np.around(W, decimals=5)

    return W


def density_und(CIJ):
    """
    Density is the fraction of present connections to possible connections.

    Parameters
    ----------
    CIJ : NxN np.ndarray
        undirected (weighted/binary) connection matrix

    Returns
    -------
    kden : float
        density
    N : int
        number of vertices
    k : int
        number of edges

    Notes
    -----
    Assumes CIJ is undirected and has no self-connections.
            Weight information is discarded.
    """
    n = len(CIJ)
    k = np.size(np.where(np.triu(CIJ).flatten()))
    kden = k / ((n**2 - n) / 2)
    return kden, n, k
