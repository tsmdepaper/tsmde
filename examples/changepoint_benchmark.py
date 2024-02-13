import autograd.numpy as np
import mosum
import ruptures as rpt
import sys
import os

current_file_path = os.path.dirname(os.path.realpath(__file__))
parent_file_path  = os.path.dirname(current_file_path)

sys.path.append(current_file_path)
sys.path.append(parent_file_path)

from scipy.stats import chi2

from library.basis import fit_basis_with_search
from library.ll import fit_with_search as fit_local_linear_with_search
from library.auxilliary import detector_to_changepoint
from library.metrics import covering_metric
from library.sim import *


# We create a huge time series and then condense it into 20 time point 'chunks'

def sim_mean_change(n, T, change_size = 2, seed = 1):
    np.random.seed(seed)

    # Simulate mean change
    true_cps = [int(T/2)]
    x = np.array([np.random.normal(0, 1, (1,)) for i in range(true_cps[0])] +
                 [np.random.normal(change_size, 1, (1,)) for i in range(true_cps[0], T)])
        
    binned_data = []
    for ti in np.arange(0, T, n):
        binned_data.append(x[ti:ti+n, :])

    if binned_data[-1].shape[0] < n:
        binned_data = binned_data[:-1]

    tseq = np.linspace(0, 1, len(binned_data))[:, None]

    return tseq, binned_data, x, true_cps

def sim_var_change(n, T, change_size = 2, seed = 1):
    np.random.seed(seed)

    # Simulate variance change
    true_cps = [int(T/2)]
    x = np.array([np.random.normal(0, 1, (1,)) for i in range(true_cps[0])] +
                 [np.random.normal(0, 1+change_size, (1,)) for i in range(true_cps[0], T)])
        
    binned_data = []
    for ti in np.arange(0, T, n):
        binned_data.append(x[ti:ti+n, :])

    if binned_data[-1].shape[0] < n:
        binned_data = binned_data[:-1]

    tseq = np.linspace(0, 1, len(binned_data))[:, None]

    return tseq, binned_data, x, true_cps

def sim_both_change(n, T, change_size = 2, seed = 1):

    np.random.seed(seed)

    # Simulate mean + variance change
    true_cps = [int(T/3), int(2*T/3)] 
    x = np.array([np.random.normal(0, 1, (1,)) for i in range(true_cps[0])] +
                [np.random.normal(change_size, 1, (1,)) for i in range(true_cps[0], true_cps[1])] +
                [np.random.normal(change_size, 1+change_size, (1,)) for i in range(true_cps[1], T)])
        
    binned_data = []
    for ti in np.arange(0, T, n):
        binned_data.append(x[ti:ti+n, :])

    if binned_data[-1].shape[0] < n:
        binned_data = binned_data[:-1]
    
    tseq = np.linspace(0, 1, len(binned_data))[:, None]

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
    est_cpts = detector_to_changepoint(detector_output, tseq, thresh, eps = 0.025, small_peak_eps=0.015)

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
    est_cpts = detector_to_changepoint(detector_output, tseq, thresh, eps = 0.025, small_peak_eps=0.015)

    # convert to indices
    c0s      = [abs(tseq - c).argmin() for c in est_cpts]
    est_cpts = [round(np.array([c0*n, (c0+1)*n]).mean()) for c0 in c0s]

    return est_cpts

def fit_mosum(x, change_type=["mean", "var", "both"][0]):

    T = len(x)
    x = x.flatten()

    if change_type == "mean":
        
        mod_mean = mosum.mosum(x, G = int(T/6))

        return mod_mean.cpts.tolist()

    elif change_type == "var":
        
        mod_var = mosum.mosum((x - x.mean())**2, G = int(T/6))

        return mod_var.cpts.tolist()

    elif change_type == "both":

        mod_mean = mosum.mosum(x, G = int(T/6))
        mod_var  = mosum.mosum((x - x.mean())**2, G = int(T/6))

        # Combine any duplicate changepoints into the mean of the two within radius epsilon
        eps = 0.04
        mean_cpts = mod_mean.cpts
        var_cpts  = mod_var.cpts

        new_cpts = []
        mean_delete_inds = []
        var_delete_inds  = []
        for i, cp_mean in enumerate(mod_mean.cpts):
            for j, cp_var in enumerate(mod_var.cpts):
                if abs(cp_mean - cp_var) < (eps*T):
                    mean_delete_inds.append(i)
                    var_delete_inds.append(j)
                    new_cpts.append(int((cp_mean + cp_var)/2))
        
        mean_cpts = np.delete(mean_cpts, mean_delete_inds)
        var_cpts  = np.delete(var_cpts, var_delete_inds)

        # combine them all
        mosum_cpts = np.array(np.concatenate([mean_cpts, var_cpts, new_cpts]), dtype=int)
        mosum_cpts.sort()              
    
        return mosum_cpts.tolist()

def fit_pelt(ts):

    algo = rpt.Pelt("rbf").fit(ts)
    rpt_cpts = algo.predict(pen = 2*np.log(len(ts)))
    rpt_cpts.pop() # last changepoint in ruptures is always the end of the series so remove it

    return rpt_cpts

def trial(seed, n, full_T, change_type):

    if change_type == "mean":
        tseq, data, x, true_cps = sim_mean_change(n, full_T, change_size = 2, seed = seed)
    elif change_type == "var":
        tseq, data, x, true_cps = sim_var_change(n, full_T, change_size = 2, seed = seed)
    elif change_type == "both":
        tseq, data, x, true_cps = sim_both_change(n, full_T, change_size = 2, seed = seed)
    
    basis_cpts = fit_our_basis(tseq, data, change_type=change_type)
    ll_cpts    = fit_our_local_linear(tseq, data, change_type=change_type)
    mosum_cpts = fit_mosum(x, change_type=change_type)
    pelt_cpts  = fit_pelt(x)

    error = np.zeros(4)

    error[0] = covering_metric(mosum_cpts, true_cps, full_T) # mosum
    error[1] = covering_metric(basis_cpts, true_cps, full_T) # basis
    error[2] = covering_metric(ll_cpts,    true_cps, full_T) # sliding window linear
    error[3] = covering_metric(pelt_cpts,  true_cps, full_T) # pelt

    return error

if __name__ == "__main__":

    # run for a single seed
    seed = 0

    print(f"seed = {seed}")

    mean_error = trial(seed, full_T = 5000, n = 5, change_type="mean")
    var_error  = trial(seed, full_T = 5000, n = 5, change_type="var")
    both_error = trial(seed, full_T = 5000, n = 5, change_type="both")

