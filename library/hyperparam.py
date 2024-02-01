import autograd.numpy as np
from autograd import elementwise_grad as grad
from scipy.optimize import minimize

from helpers import *
from sm import *
from train import get_data_vars_t1_NW, get_data_vars_all_NW, get_data_vars_vanilla
from train import do_sliding_window

def create_splits(n, perc_train, n_splits):
    
    assert n > 1, "n must be greater than 1"

    n_val = int(np.ceil(n*(1-perc_train)))

    if n*(1-perc_train) < 0.99:
        print(f"perc_train = {perc_train}, {perc_train*100}% of n={n} is {round(n*(1-perc_train), 4)}. Setting n_val = 1.")

    if n % n_val != 0:
        all_inds = np.arange(n - (n % n_val)).reshape(-1, n_val)
    else:
        all_inds = np.arange(n).reshape(-1, n_val)
    return all_inds[:n_splits, :]


def val_split(data, val_ind):
    train_ind  = np.delete(np.arange(data[0].shape[0]), val_ind)
    train_data = [d[train_ind, :] for d in data]
    val_data   = [d[val_ind, :] for d in data]
    return train_data, val_data

# train function almost copied exactly from train.py except it has data vars as input to save computation in search fn
def train_for_search(
        tseq, model_t, model_tt,
        g, dg, Ef, Ephis,
        init,
        lam = 0.1,
        penalty = "l1",
        verbose = False, maxiter=1000
    ):

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

def search(tseq, data, f, 
           model_t, model_tt,
           init,
           lams, model_hparams, inner_nw_bws, outer_nw_bws, window_sizes,
           outer_nw = False, inner_nw = True, sliding_window = False,
           perc_train = 0.9, n_splits = 5,
           maxiter = 1000, verbose = True):

    n, d = data[0].shape
    T = len(data)

    if not outer_nw:
        outer_nw_bws = [None]

    if not inner_nw:
        inner_nw_bws = [None]

    if not sliding_window:
        window_sizes = [0]

    grid = np.zeros((n_splits, len(outer_nw_bws), len(inner_nw_bws), len(window_sizes), len(lams), len(model_hparams)))

    # Cross-validation splits
    for split_i in range(n_splits):

        # sliding window size
        for window_size_i, window_size in enumerate(window_sizes):
            
            if sliding_window:
                newdata, dbool = do_sliding_window(data, window_size)
                newtseq = tseq[dbool]
            else:
                newdata = data
                newtseq = tseq

            # get split and train/val set
            val_inds = create_splits(n + window_size*2, perc_train, n_splits)
            val_ind  = val_inds[split_i, :]
            train_data, val_data = val_split(newdata, val_ind)

            if not outer_nw and not inner_nw:
                
                if verbose:
                    print("----------------------------------------------")

                _, g, dg, _, Ef, Ephis = get_data_vars_vanilla(newtseq, train_data, f, verbose=False)
                _, g_val, dg_val, _, Ef_val, Ephis_val = get_data_vars_vanilla(newtseq, val_data, f, verbose=False)
            

            # weighted NW expectation (outer expectation)
            for outer_nw_bw_i, outer_nw_bw in enumerate(outer_nw_bws):

                # weighted NW expectation (inner expectation)
                for inner_nw_bw_i, inner_nw_bw in enumerate(inner_nw_bws):

                    
                    # get data vars as soon as possible as this is the slowest part
                    if outer_nw:

                        if verbose:
                            print("----------------------------------------------")
                        
                        _, g, dg, _, Ef, Ephis = get_data_vars_all_NW(newtseq, train_data, f, 
                                                                inner_nw_bw=inner_nw_bw, 
                                                                outer_nw_bw=outer_nw_bw,
                                                                verbose=False)
                        _, g_val, dg_val, _, Ef_val, Ephis_val = get_data_vars_all_NW(newtseq, val_data, f, 
                                                                                inner_nw_bw=inner_nw_bw,
                                                                                outer_nw_bw=outer_nw_bw,
                                                                                    verbose=False)
                    elif inner_nw:
                        
                        if verbose:
                            print("----------------------------------------------")
                        
                        _, g, dg, _, Ef, Ephis = get_data_vars_t1_NW(newtseq, train_data, f, 
                                                                inner_nw_bw=inner_nw_bw, 
                                                                verbose=False)
                        _, g_val, dg_val, _, Ef_val, Ephis_val = get_data_vars_t1_NW(newtseq, val_data, f, 
                                                                                inner_nw_bw=inner_nw_bw,
                                                                                verbose=False)
                    

                    # regularisation parameter lambda
                    for lam_i, lam in enumerate(lams):

                        # any model hypeparameters
                        for model_param_i, model_param in enumerate(model_hparams):

                            model_t_i  = lambda par, t: model_t(par, t, model_param)
                            model_tt_i = lambda par, t: model_tt(par, t, model_param)

                            # get param estimate
                            alpha_hat = train_for_search(
                                            newtseq, model_t_i, model_tt_i,
                                            g, dg, Ef, Ephis,
                                            init,
                                            lam = lam,
                                            penalty = "l1",
                                            verbose = False, maxiter=maxiter
                                        )
                            
                            # record out-of-sample objective function error
                            grid[split_i, outer_nw_bw_i, inner_nw_bw_i, window_size_i, lam_i, model_param_i] = timesm_obj(alpha_hat, Ef_val, Ephis_val, newtseq, model_t_i, model_tt_i, g_val, dg_val)

                            # print progress
                            if verbose:
                                print(f"SPLIT {split_i+1}/{n_splits}   |  window size: {window_size_i+1}/{len(window_sizes)}, outer_nw_bw: {outer_nw_bw_i+1}/{len(outer_nw_bws)}, inner_nw_bw: {inner_nw_bw_i+1}/{len(inner_nw_bws)}, lam: {lam_i+1}/{len(lams)}, model_hparam: {model_param_i+1}/{len(model_hparams)}")
        
    # take mean over cross validated splits
    mean_grid = grid.mean(0)

    # output best value of each param
    best_inds         = np.unravel_index(np.argmin(mean_grid), mean_grid.shape)
    best_outer_nw_bw  = outer_nw_bws[best_inds[0]]
    best_inner_nw_bw  = inner_nw_bws[best_inds[1]]
    best_window_size  = window_sizes[best_inds[2]]
    best_lam          = lams[best_inds[3]]
    best_model_hparam = model_hparams[best_inds[4]]

    if verbose:
        print(f"\n\nSummary of Hyperparameter search:")
        print(f"________________________________")
        
        if outer_nw:
            print(f"Best Outer NW BW = {round(best_outer_nw_bw, 6)} ({best_inds[0]}), Mean Value: {round(mean_grid[best_inds[0], :, :, :, :].mean() , 3)}")
        if inner_nw:
            print(f"Best Inner NW BW = {round(best_inner_nw_bw, 6)} ({best_inds[1]}), Mean Value: {round(mean_grid[:, best_inds[1], :, :, :].mean() , 3)}")
        if sliding_window:
            print(f"Best window size = {best_window_size} ({best_inds[2]}), Mean Value: {round(mean_grid[:, :, best_inds[2], :, :].mean() , 3)}")
        print(f"Best lambda = {round(best_lam, 3)} ({best_inds[3]}), Mean Value: {round(mean_grid[:, :, :, best_inds[3] :].mean() , 3)}")
        print(f"Best model hparam = {best_model_hparam} ({best_inds[4]}), Mean Value: {round(mean_grid[:, :, :, :, best_inds[4]].mean() , 3)}")
        

    return best_outer_nw_bw, best_inner_nw_bw, best_window_size, best_lam, best_model_hparam, mean_grid

