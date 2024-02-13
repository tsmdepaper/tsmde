import autograd.numpy as np
import sys
import os
import rpy2

current_file_path = os.path.dirname(os.path.realpath(__file__))
parent_file_path  = os.path.dirname(current_file_path)

sys.path.append(current_file_path)
sys.path.append(parent_file_path)

from scipy.stats import chi2

from library.basis import fit_basis_with_search
from library.ll import fit_with_search as fit_local_linear_with_search
from library.auxilliary import detector_to_change_region
from library.metrics import covering_metric
from library.sim import *

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

# We create a huge time series and then condense it into 20 time point 'chunks'

def sim_pwl_mean(n, T, seed = 1):
    
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

    tseq = np.linspace(0, 1, len(binned_data))[:, None]
    true_cps = [round(c*T) for c in cps]

    return tseq, binned_data, x, true_cps

def sim_pwl_var(n, T, seed = 1):
    
    np.random.seed(seed)
    
    cps = [1/3, 2/3]

    def sigma_t(t):
        if t < cps[0]:
            return 1
        elif t < cps[1]:
            return 1 + 2*(t - cps[0])/(cps[1] - cps[0])
        else: 
            return 3

    x = np.array([np.random.randn(1)*sigma_t(t) for t in np.linspace(0, 1, T)])

    binned_data = []
    for ti in np.arange(0, T, n):
        binned_data.append(x[ti:ti+n, :])

    if binned_data[-1].shape[0] < n:
        binned_data = binned_data[:-1]

    tseq = np.linspace(0, 1, len(binned_data))[:, None]

    true_cps = [round(c*T) for c in cps]

    return tseq, binned_data, x, true_cps


def sim_pwl_both(n, T, seed = 1):
    
    np.random.seed(seed)
    
    cps_mean = [1/5, 2/5]

    def mu_t(t):
        if t < cps_mean[0]:
            return 0
        elif t < cps_mean[1]:
            return 10*(t - cps_mean[0])
        else:
            return 10*(cps_mean[1] - cps_mean[0])
        
    cps_var = [3/5, 4/5]

    def sigma_t(t):
        if t < cps_var[0]:
            return 1
        elif t > cps_var[1]:
            return 3
        elif t < cps_var[1]:
            return 1 + 2*(t - cps_var[0])/(cps_var[1] - cps_var[0])

    x = np.array([np.random.randn(1)*sigma_t(t) + mu_t(t) for t in np.linspace(0, 1, T)])

    binned_data = []
    for ti in np.arange(0, T, n):
        binned_data.append(x[ti:ti+n, :])

    if binned_data[-1].shape[0] < n:
        binned_data = binned_data[:-1]

    tseq = np.linspace(0, 1, len(binned_data))[:, None]

    true_cps = [round(c*T) for c in cps_mean + cps_var]

    return tseq, binned_data, x, true_cps

def fit_our_local_linear(tseq, data, change_type=["mean", "var", "both"][0]):
    
    if change_type == "mean":
        f = lambda x: x
        bs   = [0.05, 0.075, 0.1]
    elif change_type == "var":
        f = lambda x: np.array([x, x**2])
        bs   = [0.075, 0.1, 0.125]
    elif change_type == "both":
        f = lambda x: np.array([x, x**2])
        bs   = [0.075, 0.1, 0.125]

    T = len(tseq)
    n = data[0].shape[0]
    d = len(f(data[0][0, :]))

    lams = [0, 0.5, 1, 2]
    inner_nw_bw = [1/(10*T), 1/(20*T)]
    alpha, dthetaf, detector_output, Sig_dth = fit_local_linear_with_search(tseq, data, f, 
                                                        lams = lams, bs = bs, 
                                                        inner_nw = True, inner_nw_bws=inner_nw_bw,
                                                        n_splits = n, perc_train = 1-1/n,
                                                        outer_nw = False, outer_nw_bws = None,
                                                        asymptotics = True, verbose = False)
                                         

    # dont need to remove edges for LL but make sure these bits are gone for consistency with other methods
    detector_output[:int(0.05*len(tseq))] = np.nan
    detector_output[int(0.95*len(tseq)):] = np.nan

    # Get changepoints
    sig_level = 0.01
    thresh = chi2.ppf(1-sig_level, df = d)
    est_cpts0, est_cpts1 = detector_to_change_region(detector_output, tseq, thresh, eps = 0.025, small_peak_eps=0.015)
    est_cpts = est_cpts0 + est_cpts1
    
    # convert to indices
    c0s      = [abs(tseq - c).argmin() for c in est_cpts]
    est_cpts = [round(np.array([c0*n, (c0+1)*n]).mean()) for c0 in c0s]

    return est_cpts


def fit_our_basis(tseq, data, change_type=["mean", "var", "both"][0]):

    if change_type == "mean":
        f = lambda x: x
    elif change_type == "var":
        f = lambda x: np.array([x, x**2])
    elif change_type == "both":
        f = lambda x: np.array([x, x**2])

    T = len(tseq)
    n = data[0].shape[0]
    d = len(f(data[0][0, :]))

    
    lams = [0, 0.5, 1, 2]
    bws = [0.005, 0.01, 0.02]
    inner_nw_bws = [1/(10*T), 1/(20*T)]
    alpha, dthetaf, detector, Sig_dth = fit_basis_with_search(tseq, data, f = f, b = 50,
                            lams = lams, bws = bws, inner_nw_bws = inner_nw_bws,
                            n_splits = n, perc_train = 1-1/n,
                        inner_nw = True, asymptotics=True, verbose=False
                    )
    
    # Get detector
    detector_output = detector(tseq)

    # remove edges
    detector_output[:int(0.05*len(tseq))] = np.nan
    detector_output[int(0.95*len(tseq)):] = np.nan

    # Get changepoints
    sig_level = 0.01
    thresh = chi2.ppf(1-sig_level, df = d)
    est_cpts0, est_cpts1 = detector_to_change_region(detector_output, tseq, thresh, eps = 0.025, small_peak_eps=0.015)
    est_cpts = est_cpts0 + est_cpts1

    # convert to indices
    c0s      = [abs(tseq - c).argmin() for c in est_cpts]
    est_cpts = [round(np.array([c0*n, (c0+1)*n]).mean()) for c0 in c0s]

    return est_cpts

def run_pwl_mosum(x):
    
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


def fit_pwl_mosum(x, change_type=["mean", "var", "both"][0]):

    T = len(x)
    x = x.flatten()

    if change_type == "mean":
        mod_mean_cpts = run_pwl_mosum(x)
        return mod_mean_cpts.tolist()

    elif change_type == "var":
        mod_var_cpts = run_pwl_mosum((x - x.mean())**2)
        return mod_var_cpts.tolist()

    elif change_type == "both":

        mod_mean_cpts = run_pwl_mosum(x)
        mod_var_cpts  = run_pwl_mosum((x - x.mean())**2)

        # Combine any duplicate changepoints into the mean of the two within radius epsilon
        eps = 0.04

        new_cpts = []
        mean_delete_inds = []
        var_delete_inds  = []
        for i, cp_mean in enumerate(mod_mean_cpts):
            for j, cp_var in enumerate(mod_var_cpts):
                if abs(cp_mean - cp_var) < (eps*T):
                    mean_delete_inds.append(i)
                    var_delete_inds.append(j)
                    new_cpts.append(int((cp_mean + cp_var)/2))
        
        mod_mean_cpts = np.delete(mod_mean_cpts, mean_delete_inds)
        mod_var_cpts  = np.delete(mod_var_cpts, var_delete_inds)

        # combine them all
        mosum_cpts = np.array(np.concatenate([mod_mean_cpts, mod_var_cpts, new_cpts]), dtype=int)
        mosum_cpts.sort()              
    
        return mosum_cpts.tolist()

def trial(seed, n, full_T, change_type):

    if change_type == "mean":
        tseq, data, x, true_cps = sim_pwl_mean(n, full_T,seed = seed)
    elif change_type == "var":
        tseq, data, x, true_cps = sim_pwl_var(n, full_T, seed = seed)
    elif change_type == "both":
        tseq, data, x, true_cps = sim_pwl_both(n, full_T, seed = seed)
    
    basis_cpts = fit_our_basis(tseq, data, change_type=change_type)
    ll_cpts    = fit_our_local_linear(tseq, data, change_type=change_type)
    mosum_cpts = fit_pwl_mosum(x, change_type=change_type)

    error = np.zeros(3)

    error[0] = covering_metric(mosum_cpts, true_cps, full_T) # mosum
    error[1] = covering_metric(basis_cpts, true_cps, full_T) # basis
    error[2] = covering_metric(ll_cpts,    true_cps, full_T) # sliding window linear

    return error

if __name__ == "__main__":

    # run for a single seed
    seed = 0

    print(f"seed = {seed}")

    R_setup()

    mean_error = trial(seed, full_T = 5000, n = 5, change_type="mean")
    var_error  = trial(seed, full_T = 5000, n = 5, change_type="var")
    both_error = trial(seed, full_T = 5000, n = 5, change_type="both")

