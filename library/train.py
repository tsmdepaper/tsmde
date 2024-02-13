import autograd.numpy as np

from scipy.optimize import minimize
from tqdm.auto import tqdm
from autograd import elementwise_grad as grad

import os
import sys

current_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_file_path)

from helpers import *
from sm import *

# from helpers import g_abs as gf
# from helpers import dg_abs as dgf

from helpers import g_quad as gf
from helpers import dg_quad as dgf


# No expectations use the NW estimator, all regular
def get_data_vars_vanilla(tseq, data, f, verbose=False):

    T = len(data)
    n = data[0].shape[0]
    d = len(f(data[0][0, :]))

    fX = np.empty((T, n, d))
    g  = np.empty(T)
    dg = np.empty(T)
    for ti, t in enumerate(tseq.flatten()):
        for i in range(n):
            fX[ti, i, :] = f(data[ti][i]).flatten()
        g[ti]  = gf(t)
        dg[ti] = dgf(t)

    Ef = np.empty((T, d))
    for ti, t in enumerate(tseq):
        Ef[ti, :] = fX[ti, :, :].mean(0)

    Ephis = np.empty((T, d, d))
    for ti, t in enumerate(tseq):
        phi = fX[ti, :, :] - Ef[ti, :]
        Ephis[ti, :, :] = ((phi.T @ phi)/n)

    return fX, g, dg, None, Ef, Ephis


# in this function, all expectations use the NW estimator
def get_data_vars_all_NW(tseq, data, f, inner_nw_bw, outer_nw_bw, verbose=False):

    T = len(data)
    n = data[0].shape[0]
    d = len(f(data[0][0, :]))

    fX = np.empty((T, n, d))
    g  = np.empty(T)
    dg = np.empty(T)
    for ti, t in enumerate(tseq.flatten()):
        for i in range(n):
            fX[ti, i, :] = f(data[ti][i]).flatten()
        g[ti]  = gf(t)
        dg[ti] = dgf(t)

    if inner_nw_bw == outer_nw_bw:
        weights_inner = rbf2(tseq, tseq, inner_nw_bw)
        weights_outer = weights_inner        
    else:
        weights_inner = rbf2(tseq, tseq, inner_nw_bw)
        weights_outer = rbf2(tseq, tseq, outer_nw_bw)

    # inner expectation
    Ef_inner = E_nw_f(fX, weights_inner, verbose=verbose)

    # outer expectations
    Ef_outer = E_nw_f(fX, weights_outer, verbose=verbose)

    # this is where the inner expectation is calculated, so use Ef_inner
    Ephis = E_nw_phi(fX, Ef_inner, weights_outer, verbose=verbose) 

    return fX, g, dg, Ef_inner, Ef_outer, Ephis

# in this function, only Ef that is involved in the calculation of Ephi uses the NW estimator
def get_data_vars_t1_NW(tseq, data, f, inner_nw_bw, verbose=False):

    T = len(data)
    n = data[0].shape[0]
    d = len(f(data[0][0, :]))

    fX = np.empty((T, n, d))
    g  = np.empty(T)
    dg = np.empty(T)
    for ti, t in enumerate(tseq.flatten()):
        for i in range(n):
            fX[ti, i, :] = f(data[ti][i]).flatten()
        g[ti]  = gf(t)
        dg[ti] = dgf(t)

    # inner expectation is NW weighted
    weights = rbf2(tseq, tseq, inner_nw_bw)
    Ef_inner = E_nw_f(fX, weights, verbose=verbose)

    # outer expectation is regular
    Ef_outer = np.empty((T, d))
    for ti, t in enumerate(tseq):
        Ef_outer[ti, :] = fX[ti, :, :].mean(0)

    # inner expectation is used  here
    Ephis = np.empty((T, d, d))
    for ti, t in enumerate(tseq):
        phi = fX[ti, :, :] - Ef_inner[ti, :]
        Ephis[ti, :, :] = ((phi.T @ phi)/n)

    return fX, g, dg, Ef_inner, Ef_outer, Ephis

# function that takes data, which is a list of nxd numpy arrays,
# and concatenates data above and below (up to a window size) to each time point to the first axis (sample size)
def do_sliding_window(data, window_size, eps = 0):
    T = len(data)
    n = data[0].shape[0]
    # d = data[0].shape[1]

    # assert window_size % 2 == 0, "window size must be even"
    # window_half = int(window_size/2)

    # bool to show positions where sliding window data is applicable
    data_bool = np.zeros(T, dtype=bool)

    data_new = []
    for ti in range(T):
        
        # get indices of data
        t_start = max(0, ti - window_size)
        t_end   = min(T, ti + window_size + 1)

        # append data where possible
        if (t_end - t_start - 1) >= int(window_size*2):
            new_data = np.vstack([data[t] + np.random.normal(0, eps, (n,1)) for t in range(t_start, t_end)])
            data_new.append(new_data)
            data_bool[ti] = True

    return data_new, data_bool



def train(
        tseq, data, f,
        model_t, model_tt,
        init,
        lam = 0.1,
        penalty = "l1",
        sliding_window = False,
        window_size = 0, # sliding window on data
        inner_nw = False,    # NW expectation on inner expectation
        inner_nw_bw = 0.001,
        outer_nw = False, # NW expectation on all expectations (not just inner one)
        outer_nw_bw = 0.001,
        verbose = False, maxiter=1000
    ):

    if sliding_window and window_size > 0:
        data, dbool = do_sliding_window(data, window_size)
        tseq = tseq[dbool]
        # tseq = np.linspace(0, 1, len(data))[:, None]

    if outer_nw and not inner_nw:
        inner_nw_bw = outer_nw_bw

    if outer_nw and outer_nw_bw > 0:
        _, g, dg, _, Ef, Ephis = get_data_vars_all_NW(tseq, data, f, inner_nw_bw, outer_nw_bw, verbose)
    elif inner_nw and inner_nw_bw > 0:
        _, g, dg, _, Ef, Ephis = get_data_vars_t1_NW(tseq, data, f, inner_nw_bw, verbose)
    else:
        _, g, dg, _, Ef, Ephis = get_data_vars_vanilla(tseq, data, f, verbose)

    if penalty == "l1":
        penalty_fn = lambda par, t: l1_penalty(par, t, model_t)
    else:
        penalty_fn = penalty
    
    # set up objective function and penalty function
    def obj(par):
        t1 = timesm_obj(par, Ef, Ephis, tseq, model_t, model_tt, g, dg)
        tpel = penalty_fn(par, tseq)
        return t1 + lam*tpel

    # take gradient with autograd for speed
    obj_grad = lambda x: grad(obj)(x)

    if verbose:
        options={'disp': True, "maxiter": maxiter, "iprint": 101}
    else:
        options={"maxiter": maxiter}
        
    opt  = minimize(obj, init, method="L-BFGS-B", jac = obj_grad, options=options)

    return opt.x 