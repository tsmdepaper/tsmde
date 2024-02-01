import matplotlib.pyplot as plt
import autograd.numpy as np
import pandas as pd

# from scipy import integrate
from sklearn.metrics import confusion_matrix, auc

from auxilliary import detector_to_changepoint, detector_to_changepoint_dist

# reorganise an array based on a secondary array for how close each element in a is to a0
def find_nearest(a, a0):
    idx = np.abs(a - a0).argmin()
    return a.flat[idx]

# convert from in [0,1] to in [0, T]
def convert_to_time(tseq, t):
    return int(t*len(tseq))

# covering metric
# detected_cps/true_cps - timepoints (in [0, T])
def covering_metric(detected_cps, true_cps, T):
    
    if len(detected_cps) == 0 and len(true_cps) > 0:
        return 0
    elif len(detected_cps) == 0 and len(true_cps) == 0:
        return 1

    if isinstance(detected_cps[0], float) and all([t < 1 for t in detected_cps]):
        tseq = np.linspace(0, 1, T)
        detected_cps = [convert_to_time(tseq, t) for t in detected_cps]

    if isinstance(true_cps[0], float) and all([t < 1 for t in true_cps]):
        tseq = np.linspace(0, 1, T)
        true_cps = [convert_to_time(tseq, t) for t in true_cps]


    detected_cps.sort()
    true_cps.sort()
    
    cps = [0] + true_cps + [T]
    est_cps = [0] + detected_cps + [T]
    
    true_part = []
    est_part  = []

    for ii in range(len(cps)-1):
        true_part.append(range(cps[ii]+1, cps[ii+1]))
    
    for ii in range(len(est_cps)-1):
        est_part.append(range(est_cps[ii]+1, est_cps[ii+1]))
    
    
    out = 0
    for ii in range(len(cps)-1):
        max_inter = 0
        for jj in range(len(est_cps)-1):
            intersects = np.intersect1d(list(true_part[ii]), list(est_part[jj]))
            unions     = np.union1d(list(true_part[ii]), list(est_part[jj]))
            max_inter  = max(max_inter, len(intersects)/len(unions) )
        
        out += max_inter * len(true_part[ii])
    
    return out/T

# given detector, return classification AUC
def get_auc(detector_est, tseq_est, true_changepoints, dist_between_changepoints, plot=False, ax = None):
    
    num_thresholds = 500

    tcs_lower = true_changepoints - dist_between_changepoints
    tcs_upper = true_changepoints + dist_between_changepoints
    correct_changepoints = np.zeros(len(tseq_est), dtype=bool)
    for i in range(len(true_changepoints)):
        correct_changepoints = np.bitwise_or(
            correct_changepoints,
            np.bitwise_and(tseq_est > tcs_lower[i], tseq_est < tcs_upper[i])
        )
    
    thresholds  = np.linspace(0.999, 0, num_thresholds)

    fps = []; fns = []
    tps = []; tns = []
    threshs = []
    current_thresh = 0.00001
    no_detected_changes = np.inf

    while no_detected_changes != 0:
        
        detected_changepoints = abs(detector_est) > current_thresh
        tn, fp, fn, tp = confusion_matrix(
            correct_changepoints, 
            detected_changepoints
        ).ravel()

        threshs.append(current_thresh)
        tns.append(tn)
        fps.append(fp)
        fns.append(fn)
        tps.append(tp)

        current_thresh += 0.005

        changepoints = detector_to_changepoint_dist(detector_est, tseq_est, current_thresh, dist_between_changepoints)
        no_detected_changes = len(changepoints)

    fps = np.array(fps); fns = np.array(fns)
    tps = np.array(tps); tns = np.array(tns)

    ratios = (fps/(fps+tns)) / (tps/(tps+fns) )
    best_thresh = thresholds[ratios.argmax()]

    out =  auc(fps/(fps+tns), tps/(tps+fns))

    if plot:

        if ax is None:
            fig, ax = plt.subplots(1, 1)

        ax.plot(fps/(fps+tns), tps/(tps+fns), color="red", label = "Ours")
        ax.axline((0, 0), slope=1, color="black", linestyle="--");

        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend()

        # ax.set_title(f"Classification AUC: {round(out, 4)}");
        # fig.tight_layout()

    return out, best_thresh

# given detector, return MSE between itself and the true detector
def mse(detector_est, detector_true):

    # requires interpolation as true detector is always higher resolution
    # (because in finite differencing we needed a really high res)
    if len(detector_true) % len(detector_est) == 0:

        num_to_interp =  int(len(detector_true) / len(detector_est)) - 1  
        
        a = [detector_est]
        for i in range(num_to_interp):
            a.append(np.ones(len(detector_est))*np.nan)

        detector_est_interp = np.vstack(a).flatten(order="F")

        line = pd.Series(detector_est_interp).interpolate().values

        # MSE between normalised detectors
        out = ((line - detector_true)**2).sum()**0.5
    
    else:
        out = np.nan
        
    return out


def detector_to_cm(detector, tseq, thresh, true_changepoints):
    cps_est = detector_to_changepoint(detector, tseq, thresh)
    cps_est = [find_nearest(tseq, c) for c in cps_est]
    cps_est = [np.where(c == tseq)[0][0] for c in cps_est]

    cps_true = [find_nearest(tseq, c) for c in true_changepoints]
    cps_true = [np.where(c == tseq)[0][0] for c in cps_true]

    return covering_metric(cps_est, cps_true, len(tseq))

def get_covering_metric(detector, tseq, true_changepoints, plot = False, ax=None):

    cms = []
    threshs = []
    current_thresh = 0.000001
    no_detected_changes = np.inf
    while no_detected_changes != 0:
        
        cps_est  = detector_to_changepoint(detector, tseq, current_thresh)
        cps_est = [find_nearest(tseq, c) for c in cps_est]
        cps_est = [np.where(c == tseq)[0][0] for c in cps_est]

        cps_true = [find_nearest(tseq, c) for c in true_changepoints]
        cps_true = [np.where(c == tseq)[0][0] for c in cps_true]

        cms.append(covering_metric(cps_est, cps_true, len(tseq)))
        threshs.append(current_thresh)

        current_thresh += 0.005
        no_detected_changes = len(cps_est)

    cm_mean = np.array(cms).mean()
    
    if plot:
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        ax.plot(threshs, cms, color="red", label = "Ours")

        ax.set_xlabel("Threshold Value")
        ax.set_ylabel("Covering Metric")
        ax.legend()

    return cm_mean
