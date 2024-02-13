import autograd.numpy as np

import os
import sys

current_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_file_path)

# convert detector to indices of timepoints
def detector_to_changepoint_dist(detector, tseq, thresh, dist_between_changes):

    # initialise stuff
    changepoints = np.empty(0)
    some_t_left  = True

    detector = abs(detector)

    if (detector > thresh).sum() == 0:
        return changepoints
    
    # get times where the detector is above the threshold (candidate time points for the changepoint)
    t_above = tseq[detector > thresh]
    detector_above = detector[detector > thresh]    

    # loop until there are no candidate times left
    while some_t_left:
        
        # take the max time on current sequence
        t_above_max = t_above[detector_above.argmax()]

        # this is the first changepoint
        changepoints = np.append(changepoints, t_above_max)

        # remove any other time points that are a distance of dist_between_changes away from the current changepoint
        cond = np.bitwise_or(
            t_above < (t_above_max - dist_between_changes),
            t_above > (t_above_max + dist_between_changes)
        ) 

        t_above = t_above[cond]
        detector_above = detector_above[cond]

        # keep looping until no times left
        some_t_left = len(t_above) != 0
    
    return changepoints

# convert detector to indices of timepoints
def detector_to_changepoint(detector, tseq, thresh, eps = 0.02, remove_small_peaks=True, small_peak_eps = 0.01):

    # initialise stuff
    changepoints = np.empty(0)
    detector = abs(detector).flatten()

    if (detector > thresh).sum() == 0:
        return changepoints
        
    a = detector > thresh
    start_and_end = np.where(np.diff(a))[0]

    if a[0]:
        start_and_end = np.append(0, start_and_end)

    if a[-1]:
        start_and_end = np.append(start_and_end, len(a)-1)

    starts0 = start_and_end[range(0, len(start_and_end), 2)]
    ends0   = start_and_end[range(1, len(start_and_end), 2)]

    # changepoints within +- epsilon% of T of each other are considered the same
    eps_t = int(eps * len(tseq))
    to_replace1 = abs(ends0[1:] - starts0[:-1]) <= eps_t
    to_replace2 = abs(ends0[:-1] - starts0[1:]) <= eps_t
    to_replace_ind = np.where(np.bitwise_or(to_replace1, to_replace2))[0]

    starts = starts0.copy(); ends = ends0.copy()

    for i in range(len(to_replace_ind)):
        starts[to_replace_ind[i]+1] = starts[to_replace_ind[i]]

    starts = np.delete(starts, to_replace_ind)
    ends   = np.delete(ends, to_replace_ind)
    


    # this is for debugging to make sure the epsilon thing works
    # plt.plot(np.fmin(detector, 50), color="black")
    # plt.axhline(thresh, color = "blue")
    # for s in starts0:
    #     plt.axvline(s, color="green")
    
    # for e in ends0:
    #     plt.axvline(e, color="red")

    # for s in starts:
    #     plt.axvline(s, color="green", linewidth=3, linestyle="--")
    
    # for e in ends:
    #     plt.axvline(e, color="red", linewidth=3, linestyle="--")


    # starts = starts2.copy(); ends = ends2.copy()
    
    full_bool = np.zeros(len(detector), dtype=bool)

    # loop over start and end points, get changepoints (max value within this region)
    for i in range(len(starts)):
        boo = np.zeros(len(detector), dtype=bool)
        boo[(starts[i]+1):(ends[i]+1)] = True

        full_bool[(starts[i]+1):(ends[i]+1)] = True

        if remove_small_peaks and np.sum(boo) < small_peak_eps * len(detector):
            continue
        
        detector2 = detector.copy()
        detector2[~boo] = -np.inf

        # ind = np.where(detector == detector[(starts[i]+1):(ends[i]+1)].max())[0][0]
        changepoints = np.append(changepoints, tseq[detector2.argmax()])

    return changepoints

def detector_to_change_region(detector, tseq, thresh, eps = 0.02, remove_small_peaks=True, small_peak_eps = 0.01):
    
    # initialise stuff
    detector = abs(detector).flatten()

    if (detector > thresh).sum() == 0:
        return [], []

    # initial boolean array above threshold
    intial_bool = detector > thresh

    # get start and end points of boolean array
    start_and_end = np.where(np.diff(intial_bool))[0]

    if intial_bool[0]:
        start_and_end = np.append(0, start_and_end)

    if intial_bool[-1]:
        start_and_end = np.append(start_and_end, len(intial_bool)-1)

    starts0 = start_and_end[range(0, len(start_and_end), 2)]
    ends0   = start_and_end[range(1, len(start_and_end), 2)]

    # changepoints within +- epsilon% of T of each other are considered the same and those peaks are removed from the start/ends
    eps_t = int(eps * len(tseq))
    to_replace1 = abs(ends0[1:] - starts0[:-1]) <= eps_t
    to_replace2 = abs(ends0[:-1] - starts0[1:]) <= eps_t
    to_replace_ind = np.where(np.bitwise_or(to_replace1, to_replace2))[0]

    starts = starts0.copy(); ends = ends0.copy()

    for i in range(len(to_replace_ind)):
        starts[to_replace_ind[i]+1] = starts[to_replace_ind[i]]

    starts = np.delete(starts, to_replace_ind)
    ends   = np.delete(ends, to_replace_ind)

    # loop over start and end points, create new boolean array above threshold
    full_bool = np.zeros(len(detector), dtype=bool)
    for i in range(len(starts)):
        boo = np.zeros(len(detector), dtype=bool)
        boo[(starts[i]+1):(ends[i]+1)] = True

        if remove_small_peaks and np.sum(boo) < small_peak_eps * len(detector):
            continue
        else:
            full_bool[(starts[i]+1):(ends[i]+1)] = True

    # get start and end points again but with reduced version
    start_and_end_full = np.where(np.diff(full_bool))[0]

    # if full_bool[0]:
    #     start_and_end_full = np.append(0, start_and_end_full)

    # if full_bool[-1]:
    #     start_and_end_full = np.append(start_and_end_full, len(full_bool)-1)

    starts_full = start_and_end_full[range(0, len(start_and_end_full), 2)]
    ends_full   = start_and_end_full[range(1, len(start_and_end_full), 2)]

    starts_out = [tseq.flatten()[s] for s in starts_full]
    ends_out   = [tseq.flatten()[e] for e in ends_full]

    return starts_out, ends_out
        
        

# function that automatically finds a threshold which gives n number of changepoints
def find_threshold(detector, n, tseq, change_diff=0.1, threshold=0.8, max_iter=1000, verbose=False, eps=0.02):
        
    # initialise stuff
    num_changes = 0
    iter = 0
    thresh = threshold
    flag1 = False
    flag2 = False
    max_thresh = 1.0    
    max_num_changes = 0
    
    # loop until we get the right number of changepoints
    while (num_changes != n) and (iter < max_iter):
        
        # get the changepoints
        # changes = np.abs(dthetat) > thresh
        num_changes = len(detector_to_changepoint(detector, tseq, thresh, eps=eps))
        
        # if we have too many changepoints, increase the threshold
        if num_changes > n:
            thresh += change_diff
            flag1 = True

        # if we have too few changepoints, decrease the threshold
        elif num_changes < n:
            thresh -= change_diff
            flag2 = True
        
        # increment the iteration counter
        iter += 1
        
        # print out the number of changepoints if we want
        if verbose:
            print("Iteration: %d, Threshold: %f, Num. Changes: %d, n: %d" % (iter, thresh, num_changes, n))
            print((num_changes != n))

        # if we never reach desired number, take the threshold that gives the most changepoints
        if num_changes > max_num_changes:
            max_thresh = thresh
            max_num_changes = num_changes

        # If we pass above and below the changepoint, reduce the search size by half
        if flag1 and flag2:
            change_diff /= 2
            flag1 = False
            flag2 = False
    
    # if we've reached the maximum number of iterations, print out a warning
    if iter == max_iter:
        print("Warning: Maximum number of iterations reached.")
        thresh = max_thresh
    
    # return the threshold
    return thresh


# convert detector and tseq to subset version
def detector_subset(detector, tseq, t_low=0.1, t_high=0.9):
    
    tseq = tseq.flatten()

    cond = np.bitwise_and(tseq > t_low, tseq < t_high)
    detector2 = detector[cond, :]
    tseq2     = tseq[cond]

    # if abs(detector2).max() == 0:
        # return detector2, tseq2
    
    # detector2  /= abs(detector2).max()

    return detector2, tseq2


# 'score' function S(\t_0, \mu) = \int mu(x) \partial_t \log q_t(x) | t = t_0 dx
def S(t0, x, dlogpt):
    return dlogpt(t0, x).mean()

def KL(x1, t1, t2, dlogpt):
    ts = np.linspace(t2, t1, 5)
    return np.array([S(t, x1, dlogpt) for t in ts]).mean()