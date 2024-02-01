import autograd.numpy as np
from scipy.stats import multivariate_normal
from scipy.interpolate import interp1d

# Simulate smooth changepoint
def sim_changepoints_both(n, d, T, 
                          pos_start_mean = [0.2, 0.8], 
                          pos_end_mean = [0.25, 0.85],
                          pos_start_var = [0.4, 0.6], 
                          pos_end_var = [0.45, 0.65],
                          mus = [0, 5, 0], 
                          sigmas = [1, 5, 1],
                          return_logqt=False,
                          return_natural_params=False,
                          return_thetat=False):
    """
    n: sample size
    d: dimension
    T: number of time points
    pos_start_mean: positions of initial change in mean
    pos_end_mean: positions that the change _ends_ in the mean. for example, 
                  if pos_start_mean = [0.1, 0.4] and pos_end_mean = [0.2, 0.5], 
                  the first change will gradually happen over the period 0.1 - 0.2
                  and the second over the period 0.4-0.5
    pos_start_var: same as above but for variance
    pos_end_var:   same as above but for variance
    mus: values of mu that change. mu starts at 0, then changes to these values in order
         according to how many pos_start and pos_ends they have. e.g. mus = [5, 1] means
         mu will start at 0, change to 5 after the first set of pos_starts, then change to 1
         for the second set
    sigmas: same as above but for variance (sigma^2)
    return_logqt: return as an output of the function the finite differenced partial_t log q_t
    return_thetat: return as outputs of the function the finite differenced
                   mu_t, sigma^2_t, partial_t mu_t, and partial_t sigma^2_t
    """

    # time scale is between 0 and 1
    tseq = np.linspace(0, 1, T)
    
    # Set up mean generation (across t \in [0, 1]) as a function so we can
    # finite difference later.
    def dovars(tseq):

        # Initialise variables
        T = len(tseq)
        muts = np.empty((T, d))
        muts[:] = np.nan

        sigmats = np.empty((T, d))
        sigmats[:] = np.nan

        # Loop over each change point for mean
        for i in range(len(pos_start_mean) + 1):

            # Set up boolean indexing based on whether we are the first changepoint,
            # a middle one, or the last one
            #
            # note that we leave time points inbetween the changes as blank 
            # (i.e. between pos_start and pos_end)
            if i == 0:
                boo_s = tseq < pos_start_mean[i]
                boo_e = tseq < pos_end_mean[i]
            elif i == (len(pos_start_mean)):
                boo_s = tseq >= pos_start_mean[i-1]
                boo_e = tseq >= pos_end_mean[i-1]
            else:
                boo_s = np.bitwise_and(tseq < pos_start_mean[i], tseq >= pos_start_mean[i-1])
                boo_e = np.bitwise_and(tseq < pos_end_mean[i], tseq >= pos_end_mean[i-1])

            # For the specified interval, save mu
            boo = np.bitwise_and(boo_s, boo_e)
            if isinstance(mus[i], int) or isinstance(mus[i], float) or  len(mus[i]) == 1:
                muts[boo, :] = np.ones(d)*mus[i]
            else:
                muts[boo, :] = mus[i]
        
        # Interpolate between pos_start and pos_end using scipy.interpolate.interp1d
        # here I have used cubic interpolation to make it slightly smoother
        for j in range(d):
            y = muts[:, j]
            nans, x = np.isnan(y), lambda z: z.nonzero()[0]

            x2 = x(~nans)
            y2 = muts[~nans, j]
            f = interp1d(x2, y2, kind="cubic")
            muts[nans, j] = f(x(nans))

        # Variance, follow same procedure as above
        for i in range(len(pos_start_var) + 1):
            if i == 0:
                boo_s = tseq < pos_start_var[i]
                boo_e = tseq < pos_end_var[i]
            elif i == (len(pos_start_var)):
                boo_s = tseq >= pos_start_var[i-1]
                boo_e = tseq >= pos_end_var[i-1]
            else:
                boo_s = np.bitwise_and(tseq < pos_start_var[i], tseq >= pos_start_var[i-1])
                boo_e = np.bitwise_and(tseq < pos_end_var[i], tseq >= pos_end_var[i-1])

            boo = np.bitwise_and(boo_s, boo_e)
            if isinstance(sigmas[i], int) or isinstance(sigmas[i], float) or len(sigmas[i]) == 1:
                sigmats[boo, :] = np.ones(d)*sigmas[i]
            else:
                sigmats[boo, :] = sigmas[i]
        
        for j in range(d):
            y = sigmats[:, j]
            nans, x = np.isnan(y), lambda z: z.nonzero()[0]

            x2 = x(~nans)
            y2 = sigmats[~nans, j]
            f = interp1d(x2, y2, kind="cubic")
            sigmats[nans, j] = f(x(nans))
            
        return muts, sigmats
    
    # Get variables by just running the function above
    muts, sigmats = dovars(tseq)

    # Simulate data from MVN using these mus and sigmas
    data = []
    for ti, t in enumerate(tseq):
        Sigma = np.diag(sigmats[ti, :])
        data.append(np.random.multivariate_normal(muts[ti, :], Sigma, n))



    # Only returns either log q_t or theta_ts
    if return_logqt:
        # Do finite differencing to get partial_t log q_t
        
        # Function to do FD on
        def dologqt(tseq):
            mus, sigmas = dovars(tseq)
            logqt  = np.zeros((len(tseq), 100))
            xp     = np.linspace(-4, 10, 100)
            for ti, t in enumerate(tseq):
                logqt[ti, :] = multivariate_normal.logpdf(xp, mus[ti,:], np.diag(sigmas[ti, :]))
            return logqt
    
        logqt  = dologqt(tseq)
        
        # Do the FD on a finer sequence of time points to improve accuracy
        tseq_fine = np.linspace(0, 1, 2000)
        dlogqt = (dologqt(tseq_fine + 1e-3) - dologqt(tseq_fine-1e-3))/2e-3

        return data, logqt, dlogqt
    
    if return_thetat:

        # Same as above

        tseq_fine = np.linspace(0, 1, 2000)
        
        domu    = lambda x: dovars(x)[0]
        dosigma = lambda x: dovars(x)[1]

        dmut    = (domu(tseq_fine + 1e-3) - domu(tseq_fine-1e-3))/2e-3
        dsigmat = (dosigma(tseq_fine + 1e-3) - dosigma(tseq_fine-1e-3))/2e-3

        return data, muts, sigmats, dmut, dsigmat
    
    if return_natural_params:

        tseq_fine = np.linspace(0, 1, 2000)

        def do_mu_over_sigma(tseq):
            mu, sigma2 = dovars(tseq)
            return mu/sigma2
        
        def do_one_over_sigma(tseq):
            _, sigma2 = dovars(tseq)
            return -1/sigma2
        
        dnat1 = (do_mu_over_sigma(tseq_fine + 1e-3) - do_mu_over_sigma(tseq_fine-1e-3))/2e-3
        dnat2 = (do_one_over_sigma(tseq_fine + 1e-3) - do_one_over_sigma(tseq_fine-1e-3))/2e-3

        return data, dnat1, dnat2
    
    else:
        return data


# Simulate multi-d evolution over time
def quadratic(n, T, d, sigma2 = 1):
    
    tseq = np.linspace(0, 1, T)
    Sigma = np.eye(d)*sigma2

    beta2 = np.ones(d)*3
    beta1 = np.ones(d)*4
    beta1[np.arange(len(beta1))%2==1] *= -1
    beta0 = 0

    def muf(t): 
        return beta0 + beta1*t + beta2*t**2

    def dmuf(t): 
        return beta1 + 2*beta2*t

    data = []
    mus  = np.empty((T, d))
    dmus = np.empty((T, d))
    for ti, t in enumerate(tseq):
        data.append(np.random.multivariate_normal(muf(t), Sigma, n))
        mus[ti, :] = muf(t)
        dmus[ti, :] = dmuf(t)

    return data, tseq, mus, dmus

def circle(n, T, d, sigma2=1):
    def muf(t):
        return np.array([np.sin(t*(2*np.pi)), np.cos(t*(2*np.pi))])

    def dmuf(t):
        return (2*np.pi)*np.array([np.cos(t*(2*np.pi)), -np.sin(t*(2*np.pi))])

    tseq = np.linspace(0, 1, T)
    Sigma = np.eye(d)*sigma2

    data = []
    mus  = np.empty((T, d))
    dmus = np.empty((T, d))
    for ti, t in enumerate(tseq):
        data.append(np.random.multivariate_normal(muf(t), Sigma, n))
        mus[ti, :] = muf(t)
        dmus[ti, :] = dmuf(t)

    
    return data, tseq, mus, dmus



def simulate_mean_easy(n, T, seed = None):

    np.random.seed(seed)


    sigmas = [1, 1]
    pos_start_var = [0.4]
    pos_end_var   = [0.45]

    mus            = [10, 15]
    pos_start_mean =    [0.5]
    pos_end_mean   =    [0.55]

    true_changepoints = (np.array(pos_start_mean) + np.array(pos_end_mean))/2

    data, dnat1, dnat2 = sim_changepoints_both(n, 1, T, 
                                 mus = mus, sigmas = sigmas,
                                 pos_start_mean = pos_start_mean,
                                 pos_end_mean   = pos_end_mean,
                                 pos_start_var  = pos_start_var,
                                 pos_end_var    = pos_end_var, 
                                 return_natural_params=True)
    

    return data, true_changepoints, dnat1, dnat2

def simulate_mean_medium(n, T, seed = None):

    np.random.seed(seed)


    sigmas = [1, 1]
    pos_start_var = [0.4]
    pos_end_var = [0.45]

    mus            = [0, 2,   -2,    4,    0]
    pos_start_mean =    [0.2,  0.4,  0.6,  0.75]
    pos_end_mean   =    [0.22, 0.43, 0.61, 0.8]

    true_changepoints = (np.array(pos_start_mean) + np.array(pos_end_mean))/2

    data, dnat1, dnat2 = sim_changepoints_both(n, 1, T, 
                                 mus = mus, sigmas = sigmas,
                                 pos_start_mean = pos_start_mean,
                                 pos_end_mean   = pos_end_mean,
                                 pos_start_var  = pos_start_var,
                                 pos_end_var    = pos_end_var, 
                                 return_natural_params=True)
    

    return data, true_changepoints, dnat1, dnat2

 
def simulate_mean_hard(n, T, seed = None):

    np.random.seed(seed)


    sigmas = [1, 1]
    pos_start_var = [0.4]
    pos_end_var = [0.45]

    mus      = [0,    1,   -3,    2,    4,   -2,    6,    -2,    4,    -2,    1,    4]
    pos_start_mean = [0.2,  0.25, 0.3,  0.35, 0.4,  0.43,  0.45, 0.6,  0.65,  0.7,  0.75]
    pos_end_mean   = [0.22, 0.27, 0.32, 0.38, 0.41, 0.44,  0.46,  0.61, 0.68,  0.72, 0.8]

    true_changepoints = (np.array(pos_start_mean) + np.array(pos_end_mean))/2

    data, dnat1, dnat2 = sim_changepoints_both(n, 1, T, 
                                 mus = mus, sigmas = sigmas,
                                 pos_start_mean = pos_start_mean,
                                 pos_end_mean   = pos_end_mean,
                                 pos_start_var  = pos_start_var,
                                 pos_end_var    = pos_end_var, 
                                 return_natural_params=True)
    

    return data, true_changepoints, dnat1, dnat2

def simulate_var_easy(n, T, seed = None):

    np.random.seed(seed)


    mus = [0, 0]
    pos_start_mean = [0.4]
    pos_end_mean = [0.45]

    sigmas        = [1, 4]
    pos_start_var =    [0.5]
    pos_end_var   =    [0.55]

    true_changepoints = (np.array(pos_start_var) + np.array(pos_end_var))/2

    data, dnat1, dnat2 = sim_changepoints_both(n, 1, T, 
                                 mus = mus, sigmas = sigmas,
                                 pos_start_mean = pos_start_mean,
                                 pos_end_mean   = pos_end_mean,
                                 pos_start_var  = pos_start_var,
                                 pos_end_var    = pos_end_var, 
                                 return_natural_params=True)
    

    return data, true_changepoints, dnat1, dnat2


def simulate_var_medium(n, T, seed = None):

    np.random.seed(seed)


    mus = [0, 0]
    pos_start_mean = [0.4]
    pos_end_mean = [0.45]

    sigmas        = [1, 4,    2,    1]
    pos_start_var =    [0.3,  0.4,  0.7]
    pos_end_var   =    [0.35, 0.43, 0.71]

    true_changepoints = (np.array(pos_start_var) + np.array(pos_end_var))/2

    data, dnat1, dnat2 = sim_changepoints_both(n, 1, T, 
                                 mus = mus, sigmas = sigmas,
                                 pos_start_mean = pos_start_mean,
                                 pos_end_mean   = pos_end_mean,
                                 pos_start_var  = pos_start_var,
                                 pos_end_var    = pos_end_var, 
                                 return_natural_params=True)
    

    return data, true_changepoints, dnat1, dnat2


def simulate_var_hard(n, T, seed = None):

    np.random.seed(seed)


    mus = [0, 0]
    pos_start_mean = [0.4]
    pos_end_mean   = [0.45]

    sigmas        = [1,    3,    0.5,  2,    7,    3,    6,     1]
    pos_start_var =       [0.20, 0.35, 0.40, 0.55, 0.60, 0.73,  0.78]
    pos_end_var   =       [0.22, 0.37, 0.47, 0.58, 0.61, 0.74,  0.80]

    true_changepoints = (np.array(pos_start_var) + np.array(pos_end_var))/2

    data, dnat1, dnat2 = sim_changepoints_both(n, 1, T, 
                                 mus = mus, sigmas = sigmas,
                                 pos_start_mean = pos_start_mean,
                                 pos_end_mean   = pos_end_mean,
                                 pos_start_var  = pos_start_var,
                                 pos_end_var    = pos_end_var, 
                                 return_natural_params=True)
    

    return data, true_changepoints, dnat1, dnat2



def simulate_both_easy(n, T, seed = None):

    np.random.seed(seed)

    sigmas        = [1, 5]
    pos_start_var =    [0.6]
    pos_end_var   =    [0.65]

    mus            = [0, 2]
    pos_start_mean =    [0.3]
    pos_end_mean   =    [0.35]

    mean_changepoints = (np.array(pos_start_mean) + np.array(pos_end_mean))/2
    var_changepoints  = (np.array(pos_start_var) + np.array(pos_end_var))/2
    true_changepoints = np.hstack((mean_changepoints, var_changepoints))

    data, dnat1, dnat2 = sim_changepoints_both(n, 1, T, 
                                 mus = mus, sigmas = sigmas,
                                 pos_start_mean = pos_start_mean,
                                 pos_end_mean   = pos_end_mean,
                                 pos_start_var  = pos_start_var,
                                 pos_end_var    = pos_end_var, 
                                 return_natural_params=True)
    

    return data, true_changepoints, dnat1, dnat2

def simulate_both_medium(n, T, seed = None):

    np.random.seed(seed)

    sigmas        = [1, 5,    3]
    pos_start_var =    [0.4,  0.7]
    pos_end_var   =    [0.45, 0.75]

    mus            = [0, 3,    1]
    pos_start_mean =    [0.3,  0.5]
    pos_end_mean   =    [0.35, 0.55]

    mean_changepoints = (np.array(pos_start_mean) + np.array(pos_end_mean))/2
    var_changepoints  = (np.array(pos_start_var) + np.array(pos_end_var))/2
    true_changepoints = np.hstack((mean_changepoints, var_changepoints))

    data, dnat1, dnat2 = sim_changepoints_both(n, 1, T, 
                                 mus = mus, sigmas = sigmas,
                                 pos_start_mean = pos_start_mean,
                                 pos_end_mean   = pos_end_mean,
                                 pos_start_var  = pos_start_var,
                                 pos_end_var    = pos_end_var, 
                                 return_natural_params=True)
    

    return data, true_changepoints, dnat1, dnat2


def simulate_both_hard(n, T, seed = None):

    np.random.seed(seed)

    sigmas        = [1, 5,    0.5,  3]
    pos_start_var =    [0.4,  0.56, 0.7]
    pos_end_var   =    [0.45, 0.60, 0.75]

    mus            = [0, 5,    3,    1,    0,    2]
    pos_start_mean =    [0.2,  0.33, 0.5,  0.65, 0.8]
    pos_end_mean   =    [0.22, 0.35, 0.55, 0.68, 0.83]

    mean_changepoints = (np.array(pos_start_mean) + np.array(pos_end_mean))/2
    var_changepoints  = (np.array(pos_start_var) + np.array(pos_end_var))/2
    true_changepoints = np.hstack((mean_changepoints, var_changepoints))

    data, dnat1, dnat2 = sim_changepoints_both(n, 1, T, 
                                 mus = mus, sigmas = sigmas,
                                 pos_start_mean = pos_start_mean,
                                 pos_end_mean   = pos_end_mean,
                                 pos_start_var  = pos_start_var,
                                 pos_end_var    = pos_end_var, 
                                 return_natural_params=True)
    

    return data, true_changepoints, dnat1, dnat2


