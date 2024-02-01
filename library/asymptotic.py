import numpy as np
from train import get_data_vars_t1_NW, get_data_vars_all_NW, get_data_vars_vanilla
from train import do_sliding_window
from helpers import rbf2, E_nw_dd_looped, regularised_inv, cholesky_inv

from tqdm.auto import tqdm

# reshape phi as explained in paper appendix
def do_phi_reshape(phi, p, b):

    phi_tilde = np.zeros((p*b, p))
    for i in range(p):
        phi_tilde[i*b:(i+1)*b, i] = phi.flatten()

    return phi_tilde

# functions for \dot{m} and \ddot{M} for a given sample (x / i) and a given time (t)
def mdot(gt, dgt, dphit, d2phit, fx):
    return 2 * np.dot(dgt * dphit + gt * d2phit, fx)

def Mddot(gt, dphit, fx, Efx):
    return 2 * gt * dphit @ (fx - Efx) @ (fx - Efx).T @ dphit.T

def mdot_v(gt, dgt, dphit, d2phit, EffT):
    t1 = dgt**2 * dphit @ EffT @ dphit.T
    t2 = dgt*gt * d2phit @ EffT @ dphit.T
    t3 = dgt*gt * dphit @ EffT @ d2phit.T
    t4 = gt**2  * d2phit @ EffT @ d2phit.T
    return 4*(t1 + t2 + t3 + t4)

def Mddot_v(gt, dphit, Ephi):
    return 2 * gt * dphit @ Ephi @ dphit.T

# calculate the asymptotic variance of Sigma_alpha, just the inner part of the matrix (bp x bp)
def Sigma_alpha_loop(tseq, data, f, dphi, d2phi, nw=False, nw_bw = 0.01, verbose=False):

    fX, g, dg, Ef, Ephis = get_data_vars_vanilla(tseq, data, f, nw, nw_bw, verbose)

    # some parameters we'll need
    b = dphi(np.atleast_2d(tseq[0])).shape[1]
    T, n, d = fX.shape

    # there are two terms in the asymptotic variance, set them up first
    EM   = np.zeros((b*d, b*d))
    EmmT = np.zeros((b*d, b*d))

    # outer loop - expectation over time / uniform distribution (tseq is uniform samples)
    for ti, t in enumerate(tseq):

        # get items that depend on time
        gt  = g[ti]
        dgt = dg[ti]
        dphit  = dphi(np.atleast_2d(t))[:, None]
        d2phit = d2phi(np.atleast_2d(t))[:, None]

        # reshape phi and dphi to be (bd x d) matrices
        dphit = do_phi_reshape(dphit, d, b)
        d2phit = do_phi_reshape(d2phit, d, b)

        # inner loop - expectation for a given t over samples in X
        for i in range(n):
            
            fx  = fX[ti, i, :].reshape(d, 1) # ensures fx is a column vector

            mdot_xt = mdot(gt, dgt, dphit, d2phit, fx - Ef[ti, :])
            Mdot_xt = Mddot(gt, dphit, fx, Ef[ti, :])

            EmmT += mdot_xt @ mdot_xt.T 
            EM   += Mdot_xt
    
    # divide by samples to give expectation
    EmmT /= (n)
    EM   /= (n)

    # calculate and return the asymptotic variance
    return cholesky_inv(EM, 1e-4) @ EmmT @ cholesky_inv(EM, 1e-4)

# calculate the asymptotic variance of Sigma_alpha, just the inner part of the matrix (bp x bp)
def Sigma_alpha(tseq, data, f, dphi, d2phi, 
                inner_nw = False,    # NW expectation on inner expectation
                inner_nw_bw = 0.001,
                outer_nw = False, # NW expectation on all expectations (not just inner one)
                outer_nw_bw = 0.001,
                sliding_window = False, window_size = 0,
                verbose=False
            ):
    
    if verbose:
        progress = lambda x, y: tqdm(x, total=y)
    else:
        progress = lambda x, y: x

    if sliding_window and window_size > 0:
        data, dbool = do_sliding_window(data, window_size)
        tseq = tseq[dbool]
        # tseq = np.linspace(0, 1, len(data))[:, None]

    if outer_nw and not inner_nw:
        inner_nw_bw = outer_nw_bw

    if outer_nw and outer_nw_bw > 0:
        fX, g, dg, Ef_inner, _, Ephis = get_data_vars_all_NW(tseq, data, f, inner_nw_bw, outer_nw_bw, verbose)
    elif inner_nw and inner_nw_bw > 0:
        fX, g, dg, Ef_inner, _, Ephis = get_data_vars_t1_NW(tseq, data, f, inner_nw_bw, verbose)
    else:
        fX, g, dg, _, Ef_inner, Ephis = get_data_vars_vanilla(tseq, data, f, verbose)

    # some parameters we'll need
    b = dphi(np.atleast_2d(tseq[0])).shape[1]
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


def Sigma_dthetat(tseq, Sigma_alpha_eval, dphi):

    # same parameters as above
    T = len(tseq)
    b = dphi(np.atleast_2d(tseq[0])).shape[1]
    p = int(Sigma_alpha_eval.shape[0] // b)

    # there is a different matrix for each t, there is no expectation for this t
    Sigma_dthetat = np.zeros((T, p, p))

    # loop over time
    for ti, t in enumerate(tseq):
        
        dphit = dphi(np.atleast_2d(t))
        dphit = do_phi_reshape(dphit, p, b)

        Sigma_dthetat[ti, :, :] = dphit.T @ Sigma_alpha_eval @ dphit
    
    return Sigma_dthetat
