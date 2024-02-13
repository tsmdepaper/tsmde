
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mosum
import rpy2 # to run piecewise MOSUM R code in python

import sys
import os

current_file_path = os.path.dirname(os.path.realpath(__file__))
parent_file_path  = os.path.dirname(current_file_path)

sys.path.append(current_file_path)
sys.path.append(parent_file_path)

from scipy.stats import chi2

import library.auxilliary as aux
from library.helpers import *
from library.ll import fit_local_linear, fit_linear

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 11
})

piecewise_linear_mosum_R_filepath = ""

assert len(piecewise_linear_mosum_R_filepath) > 0, "Please provide a filepath to the piecewise linear MOSUM R code, see the installation in the readme for details"


def R_setup():
    from rpy2.robjects.vectors import StrVector
    import rpy2.robjects.packages as rpackages

    utils = rpackages.importr('utils')
    packnames = ('tidyverse', 'RcppArmadillo')
    
    names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
    if len(names_to_install) > 0:
        utils.install_packages(StrVector(names_to_install))   

def sim_linear(T, n, seed = 1):

    np.random.seed(seed)
    
    mu_t = lambda t: 2*t

    x = np.array([mu_t(t) + np.random.randn(1) for t in np.linspace(0, 1, T)])

    binned_data = []
    for ti in np.arange(0, T, n):
        binned_data.append(x[ti:ti+n, :])

    if binned_data[-1].shape[0] < n:
        binned_data = binned_data[:-1]

    return x, binned_data


def sim_piecewise_linear(T, n, seed = 1):
    
    np.random.seed(seed)
    
    cps = [1/3, 2/3]
    def mu_t(t):
        if t < cps[0]:
            return 0
        elif t < cps[1]:
            return 10*(t - cps[0])
        else:
            return 10*(cps[1] - cps[0])

    x = np.array([mu_t(t) + np.random.randn(1) for t in np.linspace(0, 1, T)])

    binned_data = []
    for ti in np.arange(0, T, n):
        binned_data.append(x[ti:ti+n, :])

    if binned_data[-1].shape[0] < n:
        binned_data = binned_data[:-1]

    return x, binned_data

def sim_abrupt_cp(T, n, seed = 1):
        
    np.random.seed(seed)
    
    cps = [1/2]
    def mu_t(t):
        if t < cps[0]:
            return 0
        elif t > cps[0]:
            return 2

    x = np.array([mu_t(t) + np.random.randn(1) for t in np.linspace(0, 1, T)])

    binned_data = []
    for ti in np.arange(0, T, n):
        binned_data.append(x[ti:ti+n, :])

    if binned_data[-1].shape[0] < n:
        binned_data = binned_data[:-1]

    return x, binned_data

def run_linear(T, n, ax1, ax2, seed=1):

    x, data = sim_linear(T, n, seed=seed)
    tseq = np.linspace(0, 1, len(data))[:, None]

    # fit ours
    alpha, dthetat, detector, Sig_dth = fit_linear(tseq, data, f=lambda x: x, lam=0)
    
    # fit regression
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression(fit_intercept=False).fit(np.linspace(0, 1, len(x))[:, None], x.flatten())

    # plot
    for axi in [ax1, ax2]:
        axi.set_xlim(0, 1)
        axi.set_xlabel("$t$")

    ax1.plot(np.linspace(0, 1, len(x)), x, linewidth=0.25, color="k")
    ax1.plot(np.linspace(0, 1, len(x)), reg.predict(np.linspace(0, 1, len(x))[:, None]), linewidth=2, color="blue", linestyle="--")
    ax1.set_title("Linear Regression", fontsize=15)

    ax1.text(x=-0.4, y=0.5, s='Linear\nTrend', fontsize=16, rotation=0, va='center', ha='center')
    
    ax2.plot(np.linspace(0, 1, len(x)), x, linewidth=0.25, color="k")
    ax2.plot(tseq, alpha[0]*tseq, linewidth=2, color="red", linestyle="--")
    ax2.set_title("TSM-DE", fontsize=15)


def run_piecewise_linear_R(x):
    
    # setup vector in R environment
    xr = rpy2.robjects.FloatVector(x.flatten())
    rpy2.robjects.globalenv['xr'] = xr

    MOSUM_cps = rpy2.robjects.r('source("'+piecewise_linear_mosum_R_filepath+'")' + \
        '''
        T = length(xr)

        i = 3
        G = c(as.integer(0.1*T), as.integer(0.1*T))
        while (G[length(G)] < T/log(T, base=10)){
            G[i] = G[i-1] + G[i-2]
            i = i + 1
        }
                                
        G = G[2:(length(G))]
                          
        MOSUM_linear(xr, G_vec = G)
        ''')
    
    return np.array(MOSUM_cps)

def run_piecewise_linear(T, n, ax1, ax2, seed=1):

    np.random.seed(seed)

    x, data = sim_piecewise_linear(T, n, seed=seed)
    tseq = np.linspace(0, 1, len(data))[:, None]

    # fit ours
    alpha, dthetat, detector, Sig_dth = fit_local_linear(0.1, tseq, data, f=lambda x: x, lam=0.025)
    cps_start, cps_end = aux.detector_to_change_region(detector, tseq, thresh=chi2.ppf(0.95, 1), eps = 0.025, small_peak_eps = 0.02)
    cps_all = np.concatenate((cps_start, cps_end))

    # fit piecewise mosum
    MOSUM_cps = run_piecewise_linear_R(x)

    # plot
    for axi in [ax1, ax2]:    
        axi.set_xlim(0, 1)
        axi.set_xlabel("$t$")

    ax1.plot(np.linspace(0, 1, len(x)), x, linewidth=0.25, color="k")
    for c in MOSUM_cps:
        ax1.axvline(np.linspace(0, 1, len(x))[c], color="blue", linewidth=2)
    ax1.set_title("Piecewise MOSUM", fontsize=15)
    ax1.text(x=-0.4, y=0.5, s='Piecewise\nLinear\nChange', fontsize=16, rotation=0, va='center', ha='center')


    ax2.plot(np.linspace(0, 1, len(x)), x, linewidth=0.25, color="k")
    for c in cps_all:
        ax2.axvline(c, color="red", linewidth=2)

    ax2.set_title("TSM-DE", fontsize=15)
        
def run_abrupt_cp(T, n, ax1, ax2, seed=1):

    np.random.seed(seed)

    x, data = sim_abrupt_cp(T, n, seed=seed)
    tseq = np.linspace(0, 1, len(data))[:, None]

    # fit ours
    alpha, dthetat, detector, Sig_dth = fit_local_linear(0.075, tseq, data, f=lambda x: x, lam=0.025)
    cps = aux.detector_to_changepoint(detector, tseq, thresh=chi2.ppf(0.95, 1))

    # fit MOSUM
    mod = mosum.mosum(x.flatten(), G = int(len(x)/6))
    mosum_cpts = mod.cpts

    # plot
    for axi in [ax1, ax2]:    
        axi.set_xlim(0, 1)
        axi.set_xlabel("$t$")

    ax1.plot(np.linspace(0, 1, len(x)), x, linewidth=0.25, color="k")
    for c in mosum_cpts:
        ax1.axvline(np.linspace(0, 1, len(x))[c], color="blue", linewidth=2, label = "MOSUM")
    ax1.set_title("MOSUM", fontsize=15)
    ax1.text(x=-0.4, y=0.5, s='Abrupt\nChange', fontsize=16, rotation=0, va='center', ha='center')

    ax2.plot(np.linspace(0, 1, len(x)), x, linewidth=0.25, color="k")
    for c in cps:
        ax2.axvline(c, color="red", linewidth=2, label = "TSM-DE")

    ax2.set_title("TSM-DE", fontsize=15)


if __name__ == "__main__":

    R_setup()

    fig, ax = plt.subplots(3, 2, figsize=(6, 5))
    run_linear(500, 3, ax[0, 0], ax[0, 1])
    run_piecewise_linear(500, 3, ax[1, 0], ax[1, 1])
    run_abrupt_cp(500, 3, ax[2, 0], ax[2, 1])
    fig.tight_layout()
