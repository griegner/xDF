#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 11:56:37 2019

@author: sorooshafyouni
University of Oxford, 2019
"""

import numpy as np
import scipy.io

from matman import stat_threshold
from xdf import calc_xdf


def sim_ar1(x, rho=None):
    """simulate AR(1) process"""
    x_ar1 = [x[0]]
    x_ar1.extend(rho * x_ar1[-1] + x_t for x_t in x[1:])
    return np.array(x_ar1)


def sim_corr(x, y, rho=None):
    """simulate correlation between timeseries"""
    return (rho * x) + (y * np.sqrt(1 - rho**2))


n_timepoints = 1200
ar_rho = 0.5
corr_rho = 0.8

rng = np.random.default_rng(12)

# iid (0, std=1)
x, y = rng.standard_normal(n_timepoints), rng.standard_normal(n_timepoints)
y_corr = sim_corr(x, y, corr_rho)  # cross correlation
x_ar1, y_corr_ar1 = sim_ar1(x, ar_rho), sim_ar1(y_corr, ar_rho)  # serial correlation

assert np.isclose(
    scipy.stats.pearsonr(x_ar1, y_corr_ar1)[0], corr_rho, atol=0.05
), "pearsonr != corr_rho"

mts = np.array([x_ar1, y_corr_ar1])

print("+++++++ xDF without regularisation::: +++++++++++++++++++++++++++++++")
xDFOut_TVOn = calc_xdf(mts, n_timepoints, method="", truncate_variance=True)

Z = stat_threshold(xDFOut_TVOn["z"], mce="b")[0]
print(len(np.where(Z != 0)[0]) / 2)

print("+++++++ xDF without truncation::: +++++++++++++++++++++++++++++++++++")
xDFOut_tna = calc_xdf(
    mts, n_timepoints, method="truncate", methodparam=n_timepoints // 4, verbose=True
)


Z = stat_threshold(xDFOut_tna["z"], mce="b")[0]
print(len(np.where(Z != 0)[0]) / 2)

print("+++++++ xDF with ADAPTIVE truncation::: ++++++++++++++++++++++++++++++")
xDFOut_ta = calc_xdf(
    mts, n_timepoints, method="truncate", methodparam="adaptive", verbose=True
)
Z = stat_threshold(xDFOut_ta["z"], mce="fdr_bh")[0]
print(len(np.where(Z != 0)[0]) / 2)

print("+++++++ xDF without tapering::: +++++++++++++++++++++++++++++++++++++")
xDFOut_tt = calc_xdf(
    mts, n_timepoints, method="tukey", methodparam=np.sqrt(n_timepoints), verbose=True
)

Z = stat_threshold(xDFOut_tt["z"], mce="b")[0]
print(len(np.where(Z != 0)[0]) / 2)
