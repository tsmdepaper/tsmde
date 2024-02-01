import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
import os

parent_dir = os.path.dirname(os.path.abspath('.'))
sys.path.append(parent_dir)

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

if __name__ == "__main__":

    fname = ""
    assert len(fname) > 0, "Please provide a filename to the temperature data, see the readme for details"

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

