from scipy.stats import chi2

from helpers import *
from asymptotic import *
from train import train
from hyperparam import search

def fit_basis(tseq, data, f, b = 50, lam = 0.1, 
              bw = 0.1, 
              inner_nw = True, inner_nw_bw = 0.001,
              outer_nw = True, outer_nw_bw = 0.001,
              sliding_window = False, window_size = 0, 
              asymptotics = True,
              verbose = False, **kwargs
            ):

    defaultKwargs = {'maxiter': 1000}
    kwargs = { **defaultKwargs, **kwargs }

    p = len(f(data[0][0, :]))
    centroids = np.linspace(0, 1, b)[:, None]
    def model_t(par, t):
        par = par.reshape(b, p)
        dphi_eval = drbf2(t, centroids, bw).T
        return (par.T @ dphi_eval).T

    def model_tt(par, t):
        par = par.reshape(b, p)
        d2phi_eval = d2rbf2(t, centroids, bw).T
        return (par.T @ d2phi_eval).T

    init = np.random.randn(b, p)

    alpha_hat = train(tseq, data, f, model_t, model_tt, init, 
            lam = lam, 
            inner_nw = inner_nw, inner_nw_bw = inner_nw_bw,
            outer_nw = outer_nw, outer_nw_bw = outer_nw_bw,
            sliding_window = sliding_window, window_size=window_size,
            verbose = verbose, maxiter = kwargs["maxiter"]
         )
    
    if asymptotics:
        Sig_alp = Sigma_alpha(tseq, data, f,
                            dphi = lambda t: drbf2(t, centroids, bw), 
                            d2phi = lambda t: d2rbf2(t, centroids, bw),
                            sliding_window = sliding_window, window_size = window_size,            
                            inner_nw = inner_nw, inner_nw_bw = inner_nw_bw,
                            outer_nw = outer_nw, outer_nw_bw = outer_nw_bw,
                            verbose=verbose
                            )
        Sig_alp /= (len(data[0]))

        # Sig_dth = Sigma_dthetat(tseq, Sig_alp, dphi = lambda t: drbf2(t, centroids, bw))
        Sig_dthf = lambda t: Sigma_dthetat(t, Sig_alp, dphi = lambda t: drbf2(t, centroids, bw))

        def detector(t):
            dthetat = model_t(alpha_hat, t)
            Sig_dth_d = Sigma_dthetat(t, Sig_alp, dphi = lambda t: drbf2(t, centroids, bw))
            chi2_var = np.empty(len(t))
            for ti, t in enumerate(t):
                cov = np.atleast_2d(Sig_dth_d[ti, :, :])
                chi2_var[ti]  = dthetat[ti, :].T @ cholesky_inv(cov, 1e-6) @ dthetat[ti, :]

            return chi2_var

        return alpha_hat, lambda t: model_t(alpha_hat, t), detector, Sig_dthf
    else:
        return alpha_hat, lambda t: model_t(alpha_hat, t), None, None
        
    
def fit_basis_with_search(tseq, data, f, b = 50, 
                    lams = [0, 0.1, 1, 2, 5], 
                    bws  = None, 
                    inner_nw_bws = None, 
                    outer_nw_bws = None,
                    window_sizes = [0],
                    sliding_window = False, 
                    outer_nw = False, inner_nw = True,
                    asymptotics = True, 
                    perc_train = 0.9, n_splits = 1,
                    verbose = False, **kwargs
                ):
    
    defaultKwargs = {'maxiter': 1000, "search_maxiter": 1000}
    kwargs = { **defaultKwargs, **kwargs }

    # get variables
    T = len(tseq)
    p = len(f(data[0][0, :]))

    # default values to search over
    if bws is None:
        bws = [1/(20*T), 1/(10*T), 1/(5*T)]

    if inner_nw_bws is None:
        inner_nw_bws = [1/(20*T), 1/(10*T), 1/(5*T)]
    
    if outer_nw_bws is None:
        outer_nw_bws = [1/(20*T), 1/(10*T), 1/(5*T)]

    
    # define model functions with extra argument (3rd arg is model hparam)
    centroids = np.linspace(0, 1, b)[:, None]
    def model_t(par, t, bw):
        par = par.reshape(b, p)
        dphi_eval = drbf2(t, centroids, bw).T
        return (par.T @ dphi_eval).T

    def model_tt(par, t, bw):
        par = par.reshape(b, p)
        d2phi_eval = d2rbf2(t, centroids, bw).T
        return (par.T @ d2phi_eval).T

    # use hyperparam grid search to output parameters
    init = np.random.randn(b, p)
    outer_nw_bw, inner_nw_bw, window_size, lam, bw, grid = search(tseq, data, f, 
           model_t, model_tt,
           init,
           lams, bws, inner_nw_bws, outer_nw_bws, window_sizes,
           sliding_window = sliding_window, outer_nw = outer_nw, inner_nw = inner_nw,
           perc_train = perc_train, n_splits = n_splits,
           maxiter = kwargs["search_maxiter"], verbose=verbose)

    # train and output alpha_hat based on best values from grid search
    alpha_hat = train(tseq, data, f, 
            lambda par, t: model_t(par, t, bw), 
            lambda par, t: model_tt(par, t, bw), 
            init = init, lam = lam, 
            sliding_window = sliding_window, window_size = window_size,
            inner_nw = inner_nw, inner_nw_bw = inner_nw_bw,
            outer_nw = outer_nw, outer_nw_bw = outer_nw_bw,
            verbose = False, maxiter = kwargs["maxiter"]
         )
    
    if asymptotics:
        # get asymptotic variance estimate
        Sig_alp = Sigma_alpha(tseq, data, f,
                            dphi = lambda t: drbf2(t, centroids, bw), 
                            d2phi = lambda t: d2rbf2(t, centroids, bw),
                            inner_nw = inner_nw, inner_nw_bw = inner_nw_bw,
                            outer_nw = outer_nw, outer_nw_bw = outer_nw_bw,
                            sliding_window = sliding_window, window_size = window_size,
                            verbose = verbose
                            )
        Sig_alp /= (len(data[0]))
        Sig_dth = Sigma_dthetat(tseq, Sig_alp, dphi = lambda t: drbf2(t, centroids, bw))

        # function that aliases the model to only argument t
        dtheta_f = lambda t: model_t(alpha_hat, t, bw)

        # function which gives chi^2 estimate based on asymptotic variance
        def detector(t):
            dthetat = dtheta_f(t)
            Sig_dth_d = Sigma_dthetat(t, Sig_alp, dphi = lambda t: drbf2(t, centroids, bw))
            chi2_var = np.empty(len(t))
            for ti, _ in enumerate(t):
                cov = np.atleast_2d(Sig_dth_d[ti, :, :])
                chi2_var[ti] = dthetat[ti, :].T @ cholesky_inv(cov, 1e-6) @ dthetat[ti, :]

            return chi2_var

        return alpha_hat, dtheta_f, detector, Sig_dth
    
    else:
        return alpha_hat, lambda t: model_t(alpha_hat, t, bw), None, None
        
        
    
    
    