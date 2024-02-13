# Local Linear:
# ----------------
# select b = B/T for some b in (0, 0.5)
# for T time points and [0, B] we estimate theta(t) = alpha*t + beta,
# then for [1, B+1] we estimate theta(t) = alpha*t + beta, etc.
# in this case, partial_t theta(t) = alpha, so it is just a constant
#
# for each window, we can treat this as estimating a constant change over time for a regular problem
# with T=B, and then we can use the same code as before 
#
# for each window, the asymptotics are the same, we estimate Sigma and use it to calculate the chi^2 variable
# the threshold is the same across all windows

import autograd.numpy as np

from autograd import elementwise_grad as grad
from tqdm.auto import tqdm

import os
import sys

current_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_file_path)

from library.asymptotic import Sigma_dthetat, do_phi_reshape, mdot_v, Mddot_v
from library.helpers import rbf2, E_nw_f, E_nw_phi, E_nw_dd_looped, cholesky_inv
from library.hyperparam import create_splits, val_split


# redefine the get_data_vars function here because need to change g and dg functions to be boundary 
# 0 at non-{0,1} points

# function that takes a time sequence and returns a function that is zero at the beginning and end
# of the time sequence, and quadratic in between
def get_gf(tseq):
    T = len(tseq)
    def gf(t):
        return -(t - tseq[0])*(t - tseq[-1])
    return gf

# derivative of this function
def get_dgf(tseq):
    def dgf(t):
        return -(2*t - tseq[0] - tseq[-1])
    return dgf

# No expectations use the NW estimator, all regular
def get_data_vars_vanilla(tseq, data, f, verbose=False):

    gf  = get_gf(tseq)
    dgf = get_dgf(tseq)

    T = len(data)
    n = data[0].shape[0]
    d = len(f(data[0][0]))

    fX = np.empty((T, n, d))
    g  = np.empty(T)
    dg = np.empty(T)
    for ti, t in enumerate(tseq.flatten()):
        for i in range(n):
            fX[ti, i, :] = f(data[ti][i, :]).flatten()
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

    gf  = get_gf(tseq)
    dgf = get_dgf(tseq)

    T = len(data)
    n = data[0].shape[0]
    d = len(f(data[0][0]))

    fX = np.empty((T, n, d))
    g  = np.empty(T)
    dg = np.empty(T)
    for ti, t in enumerate(tseq.flatten()):
        for i in range(n):
            fX[ti, i, :] = f(data[ti][i, :]).flatten()
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

    gf  = get_gf(tseq)
    dgf = get_dgf(tseq)

    T = len(data)
    n = data[0].shape[0]
    d = len(f(data[0][0]))

    fX = np.empty((T, n, d))
    g  = np.empty(T)
    dg = np.empty(T)
    for ti, t in enumerate(tseq.flatten()):
        for i in range(n):
            fX[ti, i, :] = f(data[ti][i, :]).flatten()
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

# also need to redefine the Sigma_alpha function
def Sigma_alpha(tseq, data, f, dphi, d2phi, 
                inner_nw = False,    # NW expectation on inner expectation
                inner_nw_bw = 0.001,
                outer_nw = False, # NW expectation on all expectations (not just inner one)
                outer_nw_bw = 0.001,
                verbose=False
            ):
    
    if verbose:
        progress = lambda x, y: tqdm(x, total=y)
    else:
        progress = lambda x, y: x

    if outer_nw and not inner_nw:
        inner_nw_bw = outer_nw_bw

    if outer_nw and outer_nw_bw > 0:
        fX, g, dg, Ef_inner, _, Ephis = get_data_vars_all_NW(tseq, data, f, inner_nw_bw, outer_nw_bw, verbose)
    elif inner_nw and inner_nw_bw > 0:
        fX, g, dg, Ef_inner, _, Ephis = get_data_vars_t1_NW(tseq, data, f, inner_nw_bw, verbose)
    else:
        fX, g, dg, _, Ef_inner, Ephis = get_data_vars_vanilla(tseq, data, f, verbose)

    # some parameters we'll need
    b = 1 # dphi(np.atleast_2d(tseq[0])).shape[1]
    T, n, d = fX.shape

    f_0 = (fX - Ef_inner[:, None, :]).reshape(-1, d) 
    ffT  = np.matmul(f_0[:, :, None], f_0[:, None, :]).reshape(T, n, d, d)

    if outer_nw and outer_nw_bw > 0:
        weights = rbf2(tseq, tseq, outer_nw_bw)
        EffT = E_nw_dd_looped(ffT, weights)
    else:
        EffT = ffT.mean(1)
    
    EM   = np.zeros((b*d, b*d))
    EmmT = np.zeros((b*d, b*d))

    # outer loop - expectation over time / uniform distribution (tseq is uniform samples / linspaced)
    for ti, t in progress(enumerate(tseq), len(tseq)):

        # get items that depend on time
        gt  = g[ti]
        dgt = dg[ti]
        dphit  = dphi(np.atleast_2d(t))[:, None]
        d2phit = d2phi(np.atleast_2d(t))[:, None]

        # reshape phi and dphi to be (bd x d) matrices
        dphit  = do_phi_reshape(dphit, d, b)
        d2phit = do_phi_reshape(d2phit, d, b)

        EmmT += mdot_v(gt, dgt, dphit, d2phit, EffT[ti, :])
        EM   += Mddot_v(gt, dphit, Ephis[ti, :, :])

    # calculate and return the asymptotic variance
    EMinv = cholesky_inv(EM, 1e-4) 
    return EMinv @ EmmT @ EMinv

# since there is no penalty, we can use the closed form solution
def closed_form_alpha(dphi, d2phi,
                      Ef, Ephis, g, dg,
                      lam = 0.1):

    t1 = np.sum((dphi[:, :, None] * dphi[:, None, :] * Ephis) * g[:, None, None], axis=0)
    t2 = np.sum((dg[:, None] * dphi + g[:, None] * d2phi) * Ef, axis=0)

    tpel = lam*np.eye(t1.shape[0])*(dphi[:, :, None] * dphi[:, None, :]).sum(0)[0][0]

    alpha_hat = -np.linalg.inv(t1 + tpel) @ t2

    return alpha_hat

def numerical_alpha_with_l1(tseq,
                      Ef, Ephis, g, dg,
                      lam = 0.1):
    
    init = np.random.randn(Ef.shape[1])

    model_t  = lambda par, t: par*np.ones((len(t), Ef.shape[1]))
    model_tt = lambda par, t: np.zeros((len(t), Ef.shape[1]))

    from sm import timesm_obj
    def obj(par):
        tsm  = timesm_obj(par, Ef, Ephis, tseq, model_t, model_tt, g, dg)
        tpel = lam*np.abs(model_t(par, tseq)).mean()
        return tsm + tpel

    from scipy.optimize import minimize
    obj_grad = grad(obj)
    res = minimize(obj, init, jac=obj_grad, method='L-BFGS-B', options={'disp': False, 'maxiter': 1000})

    return res.x


def fit_linear(tseq, data, f, lam = 0.1,
               inner_nw = True, inner_nw_bw = 0.001,
               outer_nw = False, outer_nw_bw = 0.001,
               asymptotics = True, verbose=True
               ):

    d = len(f(data[0][0]))

    linear_dphi  = lambda t: np.ones((len(t), 1))
    linear_d2phi = lambda t: np.zeros((len(t), 1))
    linear_model_t = lambda par, t: par*np.ones((len(t), d))
    
    if outer_nw and not inner_nw:
        inner_nw_bw = outer_nw_bw

    if outer_nw and outer_nw_bw > 0:
        fX, g, dg, _, Ef, Ephis = get_data_vars_all_NW(tseq, data, f, inner_nw_bw, outer_nw_bw, verbose)
    elif inner_nw and inner_nw_bw > 0:
        fX, g, dg, _, Ef, Ephis = get_data_vars_t1_NW(tseq, data, f, inner_nw_bw, verbose)
    else:
        fX, g, dg, _, Ef, Ephis = get_data_vars_vanilla(tseq, data, f, verbose)

    # alpha_hat = closed_form_alpha(linear_dphi(tseq), linear_d2phi(tseq),
    #                   Ef, Ephis, g, dg, lam)
        
    alpha_hat = numerical_alpha_with_l1(tseq, Ef, Ephis, g, dg, lam)
    

    if asymptotics:
        Sig_alp = Sigma_alpha(tseq, data, f,
                            dphi  = linear_dphi, 
                            d2phi = linear_d2phi,
                            inner_nw = inner_nw, inner_nw_bw = inner_nw_bw,
                            outer_nw = outer_nw, outer_nw_bw = outer_nw_bw,
                            verbose=verbose
                            )
        Sig_alp /= (len(data[0]))

        # Sig_dth = Sigma_dthetat(tseq, Sig_alp, dphi = lambda t: drbf2(t, centroids, bw))
        Sig_dthf = lambda t: Sigma_dthetat(t, Sig_alp, dphi = linear_dphi)

        def detector(tseq):
            dthetat = linear_model_t(alpha_hat, tseq)
            Sig_dth_d = Sigma_dthetat(tseq, Sig_alp, dphi = linear_dphi)
            chi2_var = np.empty(len(tseq))
            for ti, t in enumerate(tseq):
                cov = np.atleast_2d(Sig_dth_d[ti, :, :])
                chi2_var[ti]  = dthetat[ti, :].T @ cholesky_inv(cov, 1e-6) @ dthetat[ti, :]

            return chi2_var

        return alpha_hat, lambda t: linear_model_t(alpha_hat, t), detector, Sig_dthf
    else:
        return alpha_hat, lambda t: linear_model_t(alpha_hat, t), None, None

def fit_local_linear(b, tseq, data, f, lam = 0.1,
               inner_nw = True, inner_nw_bw = 0.001,
               outer_nw = False, outer_nw_bw = 0.001,
               asymptotics = True, verbose=False):

    if verbose:
        progress = lambda x, y: tqdm(x, total=y)
    else:
        progress = lambda x, y: x

    T = len(data)
    d = len(f(data[0][0]))
    B = int(b*T)

    # transform lambda to be on par with other methods
    lam *= (B/T)

    alphas    = np.empty((T-2*B, d))
    # dthetas   = []
    detectors = []
    Sig_dths  = []

    for ti in progress(range(B, T-B), T-2*B):

        t_start = int(ti - B)
        t_end   = int(ti + B)
        data_sub = data[t_start:t_end]
        tseq_sub = tseq[t_start:t_end, :]

    
        alphas[ti-B, :], _, detector, Sig_dthf = fit_linear(tseq_sub, data_sub, f, lam = lam,
                                                            inner_nw = inner_nw, outer_nw = outer_nw,
                                                            inner_nw_bw = inner_nw_bw, outer_nw_bw = outer_nw_bw,
                                                            asymptotics = asymptotics, verbose=False)

        if asymptotics:
            detectors.append(detector(tseq_sub))
            Sig_dths.append(Sig_dthf(tseq_sub))
    

    # dthetat = np.array(alpha_hat)
    dthetat = np.vstack([np.full((B, d), np.nan), alphas, np.full((B, d), np.nan)])

    if asymptotics:
        detector = np.array([dt.mean() for dt in detectors])[:, None]
        detector = np.vstack([np.full((B, 1), np.nan), detector, np.full((B, 1), np.nan)])

        Sig_dth = np.array([dt.mean(0) for dt in Sig_dths])
        Sig_dth = np.vstack([np.full((B, d, d), np.nan), Sig_dth, np.full((B, d, d), np.nan)])

        return alphas, dthetat, detector, Sig_dth

    else:
        return alphas, dthetat, None, None

def timesm_for_search_1(tseq, data, f, dthetat, d2thetat):

    _, g, dg, _, Ef, Ephis = get_data_vars_vanilla(tseq, data, f, False)

    t1 = (Ephis * dthetat[:, :, None] * dthetat[:, None, :] * g[:, None, None]).sum((1,2))
    t2 = ((dg[:, None] * dthetat + g[:, None]*d2thetat) * Ef).sum(1)

    return (t1 + 2*t2).mean()

def timesm_for_search_2(b, tseq, data, f, dthetat, d2thetat, 
                        inner_nw = True, inner_nw_bw = 0.001,
                        outer_nw = False, outer_nw_bw = 0.001):

    T = len(data)
    B = int(b*T)

    obj_vals = []

    for ti in range(B, T-B):

        t_start = int(ti - B)
        t_end   = int(ti + B)
        data_sub = data[t_start:t_end]
        tseq_sub = tseq[t_start:t_end, :]

        dthetat_sub  = dthetat[t_start:t_end, :]
        d2thetat_sub = d2thetat[t_start:t_end, :]

        if outer_nw and not inner_nw:
            inner_nw_bw = outer_nw_bw

        # if outer_nw and outer_nw_bw > 0:
        #     _, g, dg, _, Ef, Ephis = get_data_vars_all_NW(tseq_sub, data_sub, f, inner_nw_bw, outer_nw_bw, False)
        # elif inner_nw and inner_nw_bw > 0:
        #     _, g, dg, _, Ef, Ephis = get_data_vars_t1_NW(tseq_sub, data_sub, f, inner_nw_bw, False)
        # else:
        _, g, dg, _, Ef, Ephis = get_data_vars_vanilla(tseq_sub, data_sub, f, False)

        t1 = (Ephis * dthetat_sub[:, :, None] * dthetat_sub[:, None, :] * g[:, None, None]).sum((1,2))
        t2 = ((dg[:, None] * dthetat_sub + g[:, None]*d2thetat_sub) * Ef).sum(1)

        obj_vals.append((t1 + 2*t2).mean())

    return np.array(obj_vals).mean()

def search(tseq, data, f, 
           lams, bs, inner_nw_bws, outer_nw_bws,
           outer_nw = False, inner_nw = True, 
           perc_train = 0.9, n_splits = 5,
           verbose = True):

    n = data[0].shape[0]
    T = len(data)
    d = len(f(data[0][0]))

    if not outer_nw:
        outer_nw_bws = [None]

    if not inner_nw:
        inner_nw_bws = [None]

    grid = np.zeros((n_splits, len(outer_nw_bws), len(inner_nw_bws), len(lams), len(bs)))

    # Cross-validation splits
    for split_i in range(n_splits):

        # get split and train/val set
        val_inds = create_splits(n, perc_train, n_splits)
        val_ind  = val_inds[split_i, :]
        train_data, val_data = val_split(data, val_ind)

        # if not outer_nw and not inner_nw:
        #     _, g_val, dg_val, _, Ef_val, Ephis_val = get_data_vars_vanilla(tseq, val_data, f, verbose=False)
        
        # weighted NW expectation (outer expectation)
        for outer_nw_bw_i, outer_nw_bw in enumerate(outer_nw_bws):

            # weighted NW expectation (inner expectation)
            for inner_nw_bw_i, inner_nw_bw in enumerate(inner_nw_bws):

                # if outer_nw:
                #     _, g_val, dg_val, _, Ef_val, Ephis_val = get_data_vars_all_NW(tseq, val_data, f, 
                #                                                             inner_nw_bw=inner_nw_bw,
                #                                                             outer_nw_bw=outer_nw_bw,
                #                                                                 verbose=False)
                # elif inner_nw:
                #     _, g_val, dg_val, _, Ef_val, Ephis_val = get_data_vars_t1_NW(tseq, val_data, f, 
                #                                                             inner_nw_bw=inner_nw_bw,
                #                                                             verbose=False)

                # regularisation parameter lambda
                for lam_i, lam in enumerate(lams):

                    # any model hypeparameters
                    for b_i, b in enumerate(bs):

                        B = int(b*T)

                        # get estimate of dthetat and second deriv (ll cant just use alpha directly)
                        alpha_hat, dthetat, _, _ = fit_local_linear(b, tseq, train_data, f = f, lam = lam, 
                                                                    inner_nw = inner_nw, inner_nw_bw = inner_nw_bw,
                                                                    outer_nw = outer_nw, outer_nw_bw = outer_nw_bw, 
                                                                    asymptotics=False, verbose=False)


                        # specific for the linear model
                        dthetat  = np.array(alpha_hat)
                        d2thetat = np.zeros(dthetat.shape)

                        # pad first B and last B values with 0
                        dthetat  = np.vstack([np.zeros((B, d)), dthetat, np.zeros((B, d))])
                        d2thetat = np.vstack([np.zeros((B, d)), d2thetat, np.zeros((B, d))])
                        
                        # record out-of-sample objective function error
                        # grid[split_i, outer_nw_bw_i, inner_nw_bw_i, lam_i, b_i] = timesm_for_search(Ef_val, Ephis_val, dthetat, d2thetat, g_val, dg_val)

                        # grid[split_i, outer_nw_bw_i, inner_nw_bw_i, lam_i, b_i] = timesm_for_search_2(
                        #                                                             b, tseq, val_data, f, dthetat, d2thetat, 
                        #                                                             inner_nw = inner_nw, inner_nw_bw = inner_nw_bw,
                        #                                                             outer_nw = outer_nw, outer_nw_bw = outer_nw_bw
                        #                                                         )

                        grid[split_i, outer_nw_bw_i, inner_nw_bw_i, lam_i, b_i] = timesm_for_search_1(tseq, val_data, f, dthetat, d2thetat)

                        # print progress
                        if verbose:
                            print(f"SPLIT {split_i+1}/{n_splits}  | outer_nw_bw: {outer_nw_bw_i+1}/{len(outer_nw_bws)}, inner_nw_bw: {inner_nw_bw_i+1}/{len(inner_nw_bws)}, lam: {lam_i+1}/{len(lams)}, B: {b_i+1}/{len(bs)}")
    
    # take mean over cross validated splits
    mean_grid = grid.mean(0)

    # output best value of each param
    best_inds         = np.unravel_index(np.argmin(mean_grid), mean_grid.shape)
    best_outer_nw_bw  = outer_nw_bws[best_inds[0]]
    best_inner_nw_bw  = inner_nw_bws[best_inds[1]]
    best_lam          = lams[best_inds[2]]
    best_b            = bs[best_inds[3]]

    if verbose:
        print(f"\n\nSummary of Hyperparameter search:")
        print(f"________________________________")
        
        if outer_nw:
            print(f"Best Outer NW BW = {round(best_outer_nw_bw, 6)} ({best_inds[0]}), Mean Value: {round(mean_grid[best_inds[0], :, :, :].mean() , 3)}")
        if inner_nw:
            print(f"Best Inner NW BW = {round(best_inner_nw_bw, 6)} ({best_inds[1]}), Mean Value: {round(mean_grid[:, best_inds[1], :, :].mean() , 3)}")
        print(f"Best lambda = {round(best_lam, 3)} ({best_inds[2]}), Mean Value: {round(mean_grid[:, :, best_inds[2] :].mean(), 3)}")
        print(f"Best B = {best_b} ({best_inds[3]}), Mean Value: {round(mean_grid[:, :, :, best_inds[3]].mean(), 3)}")
        

    return best_outer_nw_bw, best_inner_nw_bw, best_lam, best_b, mean_grid

def fit_with_search(tseq, data, f, 
                    lams = [0, 1e-3, 5e-3, 1e-2],
                    bs   = None, 
                    inner_nw_bws = None, 
                    outer_nw_bws = None,
                    outer_nw = False, inner_nw = False,
                    asymptotics = True, 
                    perc_train = 0.9, n_splits = 1,
                    verbose = False, **kwargs
                ):
    
    defaultKwargs = {'maxiter': 1000, "search_maxiter": 1000}
    kwargs = { **defaultKwargs, **kwargs }

    # get variables
    T = len(tseq)
    d = len(f(data[0][0]))

    # default values to search over
    if bs is None:
        bs = [0.01, 0.02, 0.05, 0.1]

    if inner_nw_bws is None:
        inner_nw_bws = [1/(20*T), 1/(10*T), 1/(5*T)]
    
    if outer_nw_bws is None:
        outer_nw_bws = [1/(20*T), 1/(10*T), 1/(5*T)]

    # use hyperparam grid search to output parameters
    outer_nw_bw, inner_nw_bw, lam, B, grid = search(tseq, data, f, 
           lams, bs, inner_nw_bws, outer_nw_bws,
           outer_nw = outer_nw, inner_nw = inner_nw, 
           perc_train = perc_train, n_splits = n_splits,
           verbose = verbose)

    # train and output alpha_hat based on best values from grid search
    alphas, dthetas, detector, Sig_dth = fit_local_linear(B, tseq, data, f, lam = lam, 
                                   inner_nw = inner_nw, inner_nw_bw = inner_nw_bw,
                                   outer_nw = outer_nw, outer_nw_bw = outer_nw_bw,
                                   asymptotics = asymptotics, verbose=False)

    return alphas, dthetas, detector, Sig_dth
