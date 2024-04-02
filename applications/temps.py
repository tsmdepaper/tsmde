import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rpy2

import sys
import os

current_file_path = os.path.dirname(os.path.realpath(__file__))
parent_file_path  = os.path.dirname(current_file_path)

sys.path.append(current_file_path)
sys.path.append(parent_file_path)

import library.auxilliary as aux
from library.helpers import *
from library.ll import fit_local_linear

from scipy.stats import chi2

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 16
})
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

fname = ""
assert len(fname) > 0, "Please provide a filename to the temperature data, see the readme for details"

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

def run_pwl_mosum(x):
    
    R_setup()

    # setup vector in R environment
    xr = rpy2.robjects.FloatVector(x.flatten())
    rpy2.robjects.globalenv['xr'] = xr

    MOSUM_cps = rpy2.robjects.r('source("'+piecewise_linear_mosum_R_filepath+'")' + \
        '''
        T = length(xr)

        i = 3
        G = c(as.integer(0.15*T), as.integer(0.15*T))
        while (G[length(G)] < T/log(T, base=10)){
            G[i] = G[i-1] + G[i-2]
            i = i + 1
        }
                                
        G = G[2:(length(G))]
        print(G)
        MOSUM_linear(xr, G_vec = G)
        ''')
    
    return np.array(MOSUM_cps)


def fit_pwl_mosum(x, change_type=["mean", "var", "both"][0]):

    T = len(x)
    x = x.flatten()

    mod_mean_cpts = run_pwl_mosum(x)
    return mod_mean_cpts.tolist()

if __name__ == "__main__":

    # Load data
    df = pd.read_csv(fname)

    # Convert to datetime
    df["Month"] = df["Year"].astype(str).str[-2:]
    df["Date"]  = pd.to_datetime("01/" + df["Month"] + "/" + df["Year"].astype(str).str[:-2], format="%d/%m/%Y")

    # bin data
    n = 5
    data = []
    for ti in np.arange(0, len(df), n):
        data.append(df["Anomaly"].iloc[ti:ti+n].values[:, None])

    # remove final bin if it is not full
    if data[-1].shape[0] < n:
        data = data[:-1]

    tseq = np.linspace(0, 1, len(data))[:, None]

    # Fit method
    alpha, dthetat, detector, Sig_dth = fit_local_linear(b = 0.1, tseq=tseq, data=data, 
                                                         f=lambda x: x, lam = 0.2)
    
    # detector change regions/change periods
    sig_level = 0.01
    thresh = chi2.ppf(1 - sig_level, df=1)
    cps_start, cps_end = aux.detector_to_change_region(detector, tseq, thresh)
    
    # remove final endpoint because it hasn't actually passed back through the threshold
    cps_end = cps_end[:-1] 

    # concat changepoints
    est_cpts_all = cps_start + cps_end
    
    
    # Plot detector

    fig, ax = plt.subplots(2, 1, figsize=(7, 4))
    ax[0].plot(df["Date"], df["Anomaly"], color="#404040", linewidth=0.5)
    ax[0].set_ylabel("Temperature \n Anomaly ($^\circ$C)", va="center")

    ax[0].set_xlim(df["Date"].min(), df["Date"].max())
    ax[0].set_xticks([])

    t_sub_n = df["Date"][np.arange(0, len(df), n)][:-1]
    ax[1].plot(t_sub_n, np.log1p(detector), color="black", linewidth=2)
    ax[1].set_ylabel("$\\log(1 + D(t))$", labelpad=20)
    ax[1].axhline(np.log1p(thresh), color="red", linestyle="--", linewidth=1, label = "$\\log(1 + \\chi^2_{1, 0.99})$")
    ax[1].set_xlim(df["Date"].min(), df["Date"].max())
    ax[1].set_xlabel("Year")

    for c in est_cpts_all:
        t_c = t_sub_n.iloc[np.argmin(np.abs(tseq - c))]
        ax[0].axvline(t_c, color="red")
        ax[1].axvline(t_c, color="red", label = "Change Region Markers")

    # Set the position of the y-axis labels
    ax[0].yaxis.set_label_coords(-0.075, 0.5)
    ax[1].yaxis.set_label_coords(-0.075, 0.5)

    handles, labels = ax[1].get_legend_handles_labels()
    legend_dict = dict(zip(labels, handles))  
    
    # Add a legend at the bottom of the plot
    ax[1].legend(legend_dict.values(), legend_dict.keys(), loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3)

    fig.tight_layout()
    plt.show()

    ## Run Piecewise Linear MOSUM
    mosum_cpts = fit_pwl_mosum(df["Anomaly"].values[:, None])

    # closest to tseq
    tseq_big   = np.linspace(0, 1, len(df))[:, None]
    mosum_cpts = [tseq_big[c] for c in mosum_cpts]

    # print output of piecewise linear MOSUM detected changes
    print("Piecewise Linear MOSUM Detected Changes")
    t_sub_big = df["Date"]
    for c in mosum_cpts:
        t_c = t_sub_big.iloc[np.argmin(np.abs(tseq_big - c))]
        print(t_c)
