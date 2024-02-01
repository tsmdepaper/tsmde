##
#
# 
# Benchmarks 
# - mean change detection
# - variance change detection
# - mean and variance change detection 
#

# - piecewise linear change detection?
# - RDPG change detection?
# - linear regression
import sys
import os

sys.path.append("/home/danny/OneDrive/Work/TimeSM/library/")
sys.path.append("/home/danny/OneDrive/Work/TimeSM/")

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

sys.path.append(script_dir)
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, "library"))


import time

# import matplotlib.pyplot as plt
import autograd.numpy as np
# import mosum
# import rpy2

from scipy.stats import chi2
# from tqdm.auto import tqdm

import plotting as pt
from basis import fit_basis_with_search
from ll import fit_with_search as fit_local_linear_with_search
from auxilliary import detector_to_changepoint, detector_to_change_region
from metrics import covering_metric
from sim import *

# def R_setup():
#     import rpy2
#     from rpy2.robjects.packages import importr
#     from rpy2.robjects.vectors import StrVector
#     import rpy2.robjects.packages as rpackages

#     utils = rpackages.importr('utils')
#     packnames = ('tidyverse', 'RcppArmadillo', 'cpop')
    
#     names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
#     if len(names_to_install) > 0:
#         utils.install_packages(StrVector(names_to_install))   


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

# def run_piecewise_linear_R(x):
    
#     # setup vector in R environment
#     xr = rpy2.robjects.FloatVector(x.flatten())
#     rpy2.robjects.globalenv['xr'] = xr

#     MOSUM_cps = rpy2.robjects.r('''
#         source("/home/danny/OneDrive/Work/TimeSM/PiecewiseMOSUMR/MOSUM_linear.R")

#         T = length(xr)

#         i = 3
#         G = c(as.integer(0.1*T), as.integer(0.1*T))
#         while (G[length(G)] < T/log(T, base=10)){
#             G[i] = G[i-1] + G[i-2]
#             i = i + 1
#         }
#         print(length(xr))
#         G = G[2:(length(G))]
#         print(G)
                          
#         MOSUM_linear(xr, G_vec = G)
#         ''')
    
#     return np.array(MOSUM_cps)

# def run_cpop_R(x):
    
#     # setup vector in R environment
#     xr = rpy2.robjects.FloatVector(x.flatten())
#     rpy2.robjects.globalenv['xr'] = xr

#     cpop_cps = rpy2.robjects.r('''
#         library(cpop)
#         res <- cpop(xr)
#         changepoints(res)$location
#         ''')
    
#     return np.array([int(c) for c in np.array(cpop_cps)])




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

    t = time.perf_counter()

    lams = [0, 0.5, 1, 2]
    inner_nw_bw = [1/(10*T), 1/(20*T)]
    alpha, dthetaf, detector_output, Sig_dth = fit_local_linear_with_search(tseq, data, f, 
                                                        lams = lams, bs = bs, 
                                                        inner_nw = True, inner_nw_bws=inner_nw_bw,
                                                        outer_nw = False, outer_nw_bws = None,
                                                        n_splits = n, perc_train = 1-1/n,
                                                        asymptotics = True, verbose = True)
                                                            
    # record time
    time_out = time.perf_counter() - t

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
    est_cpts.sort()

    return est_cpts, time_out


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

    t = time.perf_counter()
    
    lams = [0, 0.5, 1, 2]
    bws = [0.005, 0.01, 0.02]
    inner_nw_bws = [1/(10*T), 1/(20*T)]
    alpha, dthetaf, detector, Sig_dth = fit_basis_with_search(tseq, data, f = f, b = 50,
                            lams = lams, bws = bws, inner_nw_bws = inner_nw_bws,
                            n_splits = n, perc_train = 1-1/n,
                        inner_nw = True, asymptotics=True, verbose=True
                    )
    
    # Get detector
    detector_output = detector(tseq)

    # record time
    time_out = time.perf_counter() - t

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
    est_cpts.sort()

    return est_cpts, time_out

# def fit_pwl_cpop(x, change_type=["mean", "var", "both"][0]):

#     T = len(x)
#     x = x.flatten()

#     if change_type == "mean":
        
#         t = time.perf_counter()
#         cpts = run_cpop_R(x)
#         t_out =  time.perf_counter() - t

#         return cpts.tolist(), t_out

#     elif change_type == "var":
        
#         t = time.perf_counter()
#         cpts = run_cpop_R((x - x.mean())**2)
#         t_out =  time.perf_counter() - t

#         return cpts.tolist(), t_out

#     elif change_type == "both":

#         t = time.perf_counter()
#         mean_cpts = run_cpop_R(x)
#         var_cpts  = run_cpop_R((x - x.mean())**2)
#         t_out =  time.perf_counter() - t

#         # Combine any duplicate changepoints into the mean of the two within radius epsilon
#         eps = 0.04

#         new_cpts = []
#         mean_delete_inds = []
#         var_delete_inds  = []
#         for i, cp_mean in enumerate(mean_cpts):
#             for j, cp_var in enumerate(var_cpts):
#                 if abs(cp_mean - cp_var) < (eps*T):
#                     mean_delete_inds.append(i)
#                     var_delete_inds.append(j)
#                     new_cpts.append(int((cp_mean + cp_var)/2))
        
#         mean_cpts = np.delete(mean_cpts, mean_delete_inds)
#         var_cpts  = np.delete(var_cpts, var_delete_inds)

#         # combine them all
#         cpop_cpts = np.array(np.concatenate([mean_cpts, var_cpts, new_cpts]), dtype=int)
#         cpop_cpts.sort()              
    
#         return cpop_cpts.tolist(), t_out
    
# def fit_pwl_mosum(x, change_type=["mean", "var", "both"][0]):

#     T = len(x)
#     x = x.flatten()

#     if change_type == "mean":
        
#         t = time.perf_counter()
#         cpts = run_piecewise_linear_R(x)
#         t_out =  time.perf_counter() - t

#         return cpts.tolist(), t_out

#     elif change_type == "var":
        
#         t = time.perf_counter()
#         cpts = run_piecewise_linear_R((x - x.mean())**2)
#         t_out =  time.perf_counter() - t

#         return cpts.tolist(), t_out

#     elif change_type == "both":

#         t = time.perf_counter()
#         mean_cpts = run_piecewise_linear_R(x)
#         var_cpts  = run_piecewise_linear_R((x - x.mean())**2)
#         t_out =  time.perf_counter() - t

#         # Combine any duplicate changepoints into the mean of the two within radius epsilon
#         eps = 0.04

#         new_cpts = []
#         mean_delete_inds = []
#         var_delete_inds  = []
#         for i, cp_mean in enumerate(mean_cpts):
#             for j, cp_var in enumerate(var_cpts):
#                 if abs(cp_mean - cp_var) < (eps*T):
#                     mean_delete_inds.append(i)
#                     var_delete_inds.append(j)
#                     new_cpts.append(int((cp_mean + cp_var)/2))
        
#         mean_cpts = np.delete(mean_cpts, mean_delete_inds)
#         var_cpts  = np.delete(var_cpts, var_delete_inds)

#         # combine them all
#         mosum_cpts = np.array(np.concatenate([mean_cpts, var_cpts, new_cpts]), dtype=int)
#         mosum_cpts.sort()              
    
#         return mosum_cpts.tolist(), t_out

def trial(seed, n, full_T, change_size, change_type):

    if change_type == "mean":
        tseq, data, x, true_cps = sim_pwl_mean(n, full_T, seed = seed)
    elif change_type == "var":
        tseq, data, x, true_cps = sim_pwl_var(n, full_T, seed = seed)
    elif change_type == "both":
        tseq, data, x, true_cps = sim_pwl_both(n, full_T, seed = seed)

    
    basis_cpts, basis_time = fit_our_basis(tseq, data, change_type=change_type)
    ll_cpts, ll_time       = fit_our_local_linear(tseq, data, change_type=change_type)
    # mosum_cpts, mosum_time = fit_pwl_mosum(x, change_type=change_type)
    # cpop_cpts, cpop_time = fit_pwl_cpop(x, change_type=change_type)

    error = np.zeros(2)
    # time = np.zeros(3)

    # error[0] = covering_metric(mosum_cpts, true_cps, full_T) # mosum
    # error[1] = covering_metric(cpop_cpts,  true_cps, full_T) # mosum
    error[0] = covering_metric(basis_cpts, true_cps, full_T) # basis
    error[1] = covering_metric(ll_cpts,    true_cps, full_T) # local linear
    

    return error



if __name__ == "__main__":

    # import rpy2
    # R_setup()

    # input a vector of seeds
    seed = int(sys.argv[1])
    # exec(f"seeds={seeds}")

    # seed=0
    # seeds = [14]
    # print(f"seeds = {seeds}")

    # for seed in seeds:

    mean_error = trial(seed, full_T = 5000, n = 5, change_type="mean", change_size=2)
    var_error  = trial(seed, full_T = 5000, n = 5, change_type="var",  change_size=2)
    both_error = trial(seed, full_T = 5000, n = 5, change_type="both", change_size=2)


    # save results
    np.save(f'hpc_outputs/mean_pwl_benchmark_seed_{seed}_oursonly_newvar.npy', mean_error)
    np.save(f'hpc_outputs/var_pwl_benchmark_seed_{seed}_oursonly_newvar.npy',  var_error)
    np.save(f'hpc_outputs/both_pwl_benchmark_seed_{seed}_oursonly_newvar.npy', both_error)


