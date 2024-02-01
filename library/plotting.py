import matplotlib.pyplot as plt
import autograd.numpy as np
import torch

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 13
})

def do_shading(t, x, changes, ax):
    
    change_locs = np.where(changes)[0]
    
    if len(change_locs) == 0:
        return None

    diffs = np.hstack((0, np.diff(change_locs)))
    nonones = np.where(diffs > 1)[0]
    len_nonones = len(nonones)

    if len(nonones) == 0:
        nonones = np.array([len(change_locs)])

    pos = np.zeros((2, len_nonones+1), dtype=int)
    for j in range(len_nonones + 1):
        if j == 0:
            pos[:, j] = np.array([change_locs[0], change_locs[nonones[j]-1]+1])
        elif j == len_nonones:
            pos[:, j] = np.array([change_locs[nonones[-1]], change_locs[-1]+1])
        else:
            pos[:, j] = np.array([change_locs[nonones[j-1]], change_locs[nonones[j]-1]+1])

    for i in range(pos.shape[1]):
        ax.fill_between(t[pos[0, i]:pos[1, i]], x.min(), x[pos[0, i]:pos[1, i]], alpha=0.5)

def plot_changes(t, x, dthetat, threshold = 0.8, d=1, 
                 num_changes = None, change_diff = 0.1):

    """
    x is (n x d x T)
    """

    changes = np.abs(dthetat) > threshold

    fig, ax = plt.subplots(d+1, 1, figsize=(7, 5))
    
    for i in range(d):
        xi = x[:, i, :].flatten()
        ts = np.tile(t.flatten(), (x.shape[0], 1)).flatten()
        ax[i].scatter(ts, xi, c="k", s=0.5)

    do_shading(t, x.min(0).flatten(), changes, ax[0])

    ax[0].set_title("Data")
   
    ax[d].plot(t, dthetat, lw=2.5)
    ax[d].set_title("$\partial_t {\\theta}_t$")
    

    if num_changes is not None:
        if len(dthetat.shape) == 1:

            t_mask = np.empty(len(t), dtype=bool)
            t_mask[:] = True

            # maxs = np.empty(0, dtype=int)
            # ts = np.empty(0)

            print(f"Changepoints at:")
            for i in range(num_changes):
                l = abs(dthetat)[t_mask]
                
                m = np.nanmax(l) 
                idx = np.where(abs(dthetat) == m)[0][0]

                t_mask[np.bitwise_and(t < (t[idx] + change_diff/2), t >= (t[idx] - change_diff/2) )] = False
                # maxs = np.append(maxs, idx)
                # ts   = np.append(ts, t[idx])
                
                ax[0].vlines(t[idx], x.min(), x.max())
                print(f"t={round(t[idx], 3)} (index {idx})")
                

    ax[0].set_xlim(0, 1)
    ax[d].set_xlim(0, 1)
    ax[d].hlines(0, 0, 1, linestyles="dashed", colors="k")

    fig.tight_layout()
    return fig, ax

def plot_data_and_detector(data, detector, tseq, 
                           cps = None, true_changepoints = None, thresh = None, est = "Detector"):

    x = np.stack(data, 2)
    fig, ax = plt.subplots(detector.shape[1]+1, 1, figsize=(7, 4))

    xi = x[:, 0, :].flatten()
    ts = np.tile(np.linspace(0, 1, len(data)), (x.shape[0], 1)).flatten()
    
    ax[0].scatter(ts, xi, c="darkslategrey", s=0.5)
    ax[0].set_xlim(0, 1)
    # ax[0].set_ylim(xi.min()-1, xi.max()+1)
    ax[0].set_ylabel("Data")
    
    ax[1].plot(tseq, detector[:, 0], color="red", label="detector")
    ax[1].set_xlim(0, 1)
    # ax[1].set_ylim(detector.min()-detector.std(), detector.max()+detector.std())
    ax[1].set_xlabel("$t$")

    ax[1].set_ylabel(est)

    if cps is not None:
        for c in cps:
            ax[0].axvline(c, color="black", linestyle="--")
            ax[1].axvline(c, color="black", linestyle="--")
    
    if true_changepoints is not None:
        for c in true_changepoints:
            ax[0].axvline(c, color="blue", linestyle="--", alpha=0.4)
            ax[1].axvline(c, color="blue", linestyle="--", alpha=0.4)
    
    if thresh is not None:
        ax[1].hlines(thresh, 0, 1, linestyle="--", color="purple", label="threshold", alpha=0.5)


    if detector.shape[1] > 1:
        ax[2].vlines(cps, -9999, 9999, color="black", linestyle="--")
        ax[2].vlines(true_changepoints, -9999, 9999, color="blue", linestyle="--", alpha=0.4, label = "true CPs")
        ax[2].plot(tseq, abs(detector[:, 1]), color="red", label="detector")
        ax[2].hlines(thresh, 0, 1, linestyle="--", color="purple", label="threshold", alpha=0.5)
        ax[2].set_xlim(0, 1)
        ax[2].set_ylim(0, abs(detector).max()+abs(detector).std())
        ax[2].set_xlabel("$t$")
        ax[2].set_ylabel("$\partial_t ( - 1 / 2 \sigma_t^2)$")

    ax[-1].legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    return ax


def plot_data_and_regions(data, detector, tseq, 
                          starts = None, ends = None,
                          thresh = None, est = "Detector"):

    x = np.stack(data, 2)
    fig, ax = plt.subplots(detector.shape[1]+1, 1, figsize=(7, 4))

    xi = x[:, 0, :].flatten()
    ts = np.tile(np.linspace(0, 1, len(data)), (x.shape[0], 1)).flatten()
    
    ax[0].scatter(ts, xi, c="darkslategrey", s=0.5)
    ax[0].set_xlim(0, 1)
    # ax[0].set_ylim(xi.min()-1, xi.max()+1)
    ax[0].set_ylabel("Data")
    
    ax[1].plot(tseq, detector[:, 0], color="red", label="detector")
    ax[1].set_xlim(0, 1)
    # ax[1].set_ylim(detector.min()-detector.std(), detector.max()+detector.std())
    ax[1].set_xlabel("$t$")

    ax[1].set_ylabel(est)

    if starts is not None:
        for c in starts:
            ax[0].axvline(c, color="green", linestyle="--")
            ax[1].axvline(c, color="green", linestyle="--")

    if starts is not None:
        for c in ends:
            ax[0].axvline(c, color="red", linestyle="--")
            ax[1].axvline(c, color="red", linestyle="--")

    if thresh is not None:
        ax[1].hlines(thresh, 0, 1, linestyle="--", color="purple", label="threshold", alpha=0.5)

    ax[-1].legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    return ax

def plot_detector_with_true(detector, tseq, detector_true, tseq_true):

    d = detector.shape[1]

    ymins = np.vstack([detector.min(0), detector_true.min(0)]).min(0)
    ymaxs = np.vstack([detector.max(0), detector_true.max(0)]).max(0)

    fig, ax = plt.subplots(d, 1, figsize=(7, 2.5*d))
    ax = np.atleast_1d(ax)

    for i in range(d):
        axi = ax[i]
        axi.plot(tseq_true, detector_true[:, i], color = "blue", label = "Truth")
        axi.plot(tseq,      detector[:, i],      color = "red",  label = "Estimated")
        axi.set_xlim(0, 1)
        axi.set_ylim(ymins[i]-0.1, ymaxs[i]+0.1)
        axi.set_xlabel("$t$")
        axi.set_ylabel(f"Detector (dim {i})")

    axi.legend()
    fig.suptitle("Estimated vs. True Detector")

    return ax
    

def plot_data(data, ax=None):

    if ax is None:
        ax = plt.gca()
    
    if torch.is_tensor(data[0]):
        data = [d.detach().cpu().numpy() for d in data]

    n = data[0].shape[0]
    T = len(data)
    tseq = np.linspace(0, 1, T)

    data2 = np.stack(data, 1)

    for i in range(n):
        ax.scatter(tseq, data2[i, :, 0], color="k", alpha=0.5, s=2)
    ax.set_xlim(0, 1)
