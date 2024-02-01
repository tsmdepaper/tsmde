import numpy as np
import matplotlib.pyplot as plt

import sys
import os

parent_dir = os.path.dirname(os.path.abspath('.'))
sys.path.append(parent_dir)

from scipy.stats import chi2

from library import sim
from library.helpers import *
from library.ll import fit_local_linear, fit_with_search, fit_linear
import library.plotting as pt
import library.auxilliary as aux


plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 22
})
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


def sim_crazy(T, n, mu_t, seed = 1):

    np.random.seed(seed)

    x = np.array([mu_t(t) + np.random.randn(1) for t in np.linspace(0, 1, T)])

    binned_data = []
    for ti in np.arange(0, T, n):
        binned_data.append(x[ti:ti+n, :])

    if binned_data[-1].shape[0] < n:
        binned_data = binned_data[:-1]

    return x, binned_data


def run_crazy(T, n, ax1, ax2, seed=1):


    mu_t = lambda t: np.sin(2*np.pi*t) 
    mu_dt = lambda t: 2*np.pi*np.cos(2*np.pi*t) 

    x, data = sim_crazy(T, n, mu_t, seed=seed)

    tseq = np.linspace(0, 1, len(data))[:, None]

    # fit ours
    alpha, dthetat, detector, Sig_dth = fit_local_linear(0.15, tseq, data, f=lambda x: x, lam=0)

    # fig, ax = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
    ax1.plot(tseq, mu_t(tseq), color="lime", linewidth=3, label="$\\theta^{\\star}(t) = \sin(2\pi t)$")
    ax1.set_ylabel("$x$")
    
    
    pt.plot_data(data, ax=ax1)
    ax2.plot(tseq, mu_dt(tseq), color="red", linestyle= "dotted", label="$\\partial_t \\theta^{\\star}(t) = 2\pi \cos(2\pi t)$", linewidth=3)
    ax2.plot(tseq, dthetat, color='blue', linestyle = "dashed", label = "$\\partial_t \\widehat{\\theta}(t)$", alpha=0.85, linewidth=3)

    ax2.set_ylabel("$\partial_t \\theta(t)$")
    ax2.set_xlabel("$t$")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    # Concatenate the handles and labels from both axes
    handles = handles1 + handles2
    labels = labels1 + labels2
    legend_dict = dict(zip(labels, handles))
    ax2.legend(legend_dict.values(), legend_dict.keys(), bbox_to_anchor=(0.5, -0.2), loc="upper center", ncol=len(legend_dict))
    for axi in [ax1, ax2]:
        axi.yaxis.set_label_coords(-0.065, 0.5)




def mean_and_var_change(T, n, ax1, ax2, seed=1):

    tseq = np.linspace(0, 1, T)[:, None]
    data, true_cpts, _, _ = sim.simulate_both_hard(n, T, seed=seed)

    # fit local linear model
    alpha, dthetat, detector, Sig_dth = fit_local_linear(0.02, tseq, data, 
                                                         f=lambda x: np.array([x, x**2]),
                                                         lam = 0.2)
    
    # get cps
    thresh = chi2.ppf(0.99, 2)
    cps = aux.detector_to_changepoint(detector, tseq, thresh, eps=0.01)


    pt.plot_data(data, ax=ax1)
    ax1.set_ylabel("$x$")

    for axi in [ax1, ax2]:
        for c in true_cpts:
            axi.axvline(c, color='b', linestyle='dotted', label = "True CPs", linewidth=3.5)
        for c in cps:
            axi.axvline(c, color='r', linestyle='solid', label = "Estimated CPs", linewidth=2.25)

    ax2.plot(tseq, np.log1p(detector), color='black')
    ax2.axhline(np.log1p(thresh), linestyle='--', color='r', label = "$\log (1 + \chi^2_{2, 0.99})$")
    ax2.set_ylabel("$\log (1+D(t))$")
    ax2.set_xlabel("$t$")

    for axi in [ax1, ax2]:
        axi.yaxis.set_label_coords(-0.065, 0.5)

    handles, labels = ax2.get_legend_handles_labels()
    legend_dict = dict(zip(labels, handles))  
    ax2.legend(legend_dict.values(), legend_dict.keys(), bbox_to_anchor=(0.5, -0.2), loc="upper center", ncol=len(legend_dict))



if __name__ == "__main__":


    fig, ax = plt.subplots(2, 2, figsize=(22, 4.5), sharex=True)
    mean_and_var_change(600, 10, ax1=ax[0, 0], ax2=ax[1, 0])
    run_crazy(5000, 10, ax1=ax[0, 1], ax2=ax[1, 1])
