import torch
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from tqdm.auto import tqdm
from scipy.stats import chi2

import sys
import os

current_file_path = os.path.dirname(os.path.realpath(__file__))
parent_file_path  = os.path.dirname(current_file_path)

sys.path.append(current_file_path)
sys.path.append(parent_file_path)

import library.auxilliary as aux
import library.ll as ll
from concept_drift_models import CNNNet



device = "cuda" if torch.cuda.is_available() else "cpu"  # device is a global variable

# input fname - filepath to the 'list_attr_celeba.txt' attributes file
def get_labels(fname):

    label_df = pd.read_csv(fname, sep="\s+", skiprows=1)
    label_df = label_df.loc[:, ["Smiling", "Eyeglasses"]]
    label_df[label_df == -1] = 0

    return label_df.index.values, label_df["Smiling"].values, label_df["Eyeglasses"].values

# get filenames and labels filtered by smiling/glasses
def get_fnames(fname):

    fnames, smiles, glasses = get_labels(fname)

    # keep 5000 non-smiling/non-glasses
    # keep 5000 smiling/non-glasses
    # keep 5000 non-smiling/glasses
    # keep 5000 smiling/glasses

    # and additional 5000 smiling/non-glasses and 5000 non-smiling/glasses for later

    # get indices of each
    non_smiling_non_glasses = np.where(np.bitwise_and(smiles == 0, glasses == 0))[0]
    smiling_non_glasses     = np.where(np.bitwise_and(smiles == 1, glasses == 0))[0]
    non_smiling_glasses     = np.where(np.bitwise_and(smiles == 0, glasses == 1))[0]
    smiling_glasses         = np.where(np.bitwise_and(smiles == 1, glasses == 1))[0]

    # randomly sample 5000 from each
    non_smiling_non_glasses = np.random.choice(non_smiling_non_glasses, size=5000*2, replace=False)
    smiling_non_glasses     = np.random.choice(smiling_non_glasses, size=5000*2, replace=False)
    non_smiling_glasses     = np.random.choice(non_smiling_glasses, size=5000, replace=False)
    smiling_glasses         = np.random.choice(smiling_glasses, size=5000, replace=False)

    non_smiling_non_glasses0 = non_smiling_non_glasses[:5000]
    non_smiling_non_glasses1 = non_smiling_non_glasses[5000:]

    smiling_non_glasses0 = smiling_non_glasses[:5000]
    smiling_non_glasses1 = smiling_non_glasses[5000:]

    # get the indices of the chosen images
    non_glasses_indices_main = np.concatenate([non_smiling_non_glasses0, smiling_non_glasses0])
    non_glasses_indices_main = np.sort(non_glasses_indices_main)

    non_glasses_indices_extra = np.concatenate([non_smiling_non_glasses1, smiling_non_glasses1])
    non_glasses_indices_extra = np.sort(non_glasses_indices_extra)

    glasses_indices = np.concatenate([non_smiling_glasses, smiling_glasses])
    glasses_indices = np.sort(glasses_indices)

    return fnames[non_glasses_indices_main], fnames[glasses_indices], fnames[non_glasses_indices_extra], smiles[non_glasses_indices_main], smiles[glasses_indices], smiles[non_glasses_indices_extra]
    

def load_smiling_model(fname = "smiling_model.pth"):

    # load the model
    model = CNNNet()
    model.load_state_dict(torch.load(fname))
    model.eval()
    model = model.to(device)

    return model

def probability_of_glasses(t, cp_start = 0.4, cp_end = 0.6):
    if t < cp_start:
        return 0
    elif t >= cp_end:
        return 0.62
    elif t >= cp_start and t < cp_end:
        return 0.62 * (t - cp_start) / (cp_end - cp_start)

# require folderpath - where images are, e.g. /home/username/Downloads/img_align_celeba/
# require attrpath - where the attributes file is, e.g. /home/username/Downloads/list_attr_celeba.txt
def get_data(folderpath, attrpath, T, n, model):

    # set up image variables
    image_size = 64
    preprocess = transforms.Compose([
                            transforms.Resize(image_size),
                            transforms.CenterCrop(image_size),
                            transforms.ToTensor(),
                        ])
    
    # get filenames for glasses/non-glasses wearers
    _, glasses_fnames, non_glasses_fnames, _, _, _ = get_fnames(attrpath)
    
    # point those filenames to the image folder
    glasses_fnames     = folderpath + glasses_fnames
    non_glasses_fnames = folderpath + non_glasses_fnames
    
    # index for sampling which will increase when we sample each one
    count_glasses     = 0
    count_non_glasses = 0

    image_data = [[] for _ in range(T)]
    data = [np.empty((n, 16)) for _ in range(T)]
    tseq = np.linspace(0, 1, T)[:, None]
    for ti, t in tqdm(enumerate(tseq), total=T):

        p = probability_of_glasses(t)
 
        for i in range(n):
            
            u = np.random.uniform()

            # sample glasses
            if u < p:
                fname = glasses_fnames[count_glasses]
                
                count_glasses += 1
                if count_glasses == len(glasses_fnames):
                    print("resetting count for glasses")
                    count_glasses = 0

            # sample non-glasses
            else:
                fname = non_glasses_fnames[count_non_glasses]

                count_non_glasses += 1
                if count_non_glasses == len(non_glasses_fnames):
                    print("resetting count for non-glasses")
                    count_non_glasses = 0

            # load image
            with Image.open(fname) as img:
                img_t = preprocess(img)
                img_t = img_t.to(device)

                image_data[ti].append(img)#.resize((64, 64)))
                data[ti][i, :] = model.fea(img_t[None, :, :, :]).cpu().detach().numpy()

    return tseq, data, image_data

def plot_example(data, num_per_time = 3):

    times = [0.2, 0.4, 0.6, 0.8]

    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(num_per_time+1, len(times)*3 - 1, 
                           width_ratios=np.array([1, 1, 0.5]*len(times))[:-1],
                           height_ratios=[0.000000] + [1]*num_per_time)
    
    fig = plt.figure(figsize=(10, num_per_time*1.33))

    count = 0    

    for time_j, j in enumerate(np.arange(0, len(times)*3, 3)):
        t = times[time_j]
        ti = int(t * len(data))

        for i in range(num_per_time):
            ax = plt.subplot(gs[i+1, j])
            ax.imshow(data[ti][i])
            ax.axis("off")

        for i in range(num_per_time):
            ax = plt.subplot(gs[i+1, j+1])
            ax.imshow(data[ti][i + num_per_time])
            ax.axis("off")

        ax = plt.subplot(gs[0, j:j+2])
        ax.axis("off")
        ax.set_title(f"t = {t}", fontsize=24)
        count += 1

    for time_j, j in enumerate(np.arange(2, len(times)*3 - 2, 3)):
        for i in range(num_per_time):
            ax = plt.subplot(gs[i, j])
            ax.grid(False)
            ax.axis("off")

    fig.tight_layout()
    fig.show()
    

if __name__ == "__main__":

    np.random.seed(1)
    torch.manual_seed(1)

    # experiment variables
    T = 500
    n = 20

    # load the model
    image_size = 64
    model = load_smiling_model()

    # get data
    folderpath = ""
    attrpath = ""
    
    assert len(folderpath) > 0 and len(attrpath) > 0, "Please specify folderpath and attrpath, the paths to the image folder and attributes file respectively"

    tseq, data, image_data = get_data(folderpath, attrpath, T, n, model)

    # example images
    plot_example(image_data, 3)
    
    # plot some of the net outputs
    fig, ax = plt.subplots(3, 1, figsize=(8, 8), squeeze=False); count = 0
    for i in range(3):
        for j in range(1):

            dt = np.stack([dt[:, count] for dt in data], 0).flatten()

            ax[i, j].plot(np.linspace(0, 1, len(dt)), dt)
            ax[i, j].set_title(f"dim = {count}")
            count += 1
    
    # fit method
    bwh = 1/(n**4)
    
    # small coding hack - apply f(x) first to data and use f(x) = x in implementation to simplify code
    alpha, dthetat, detector, Sig = ll.fit_with_search(tseq=tseq, data=data, f = lambda x: x, 
                                                       lams = [0.0025, 5e-3,  0.0075],
                                                       bs   = [0.075, 0.1, 0.125],
                                                       inner_nw=True, inner_nw_bws=[bwh],
                                                       verbose=True)
    
    
    # threshold and get change points beginning and end
    sig_level = 0.01
    thresh = chi2.ppf(1 - sig_level, len(data[0][0, :]))
    cps_start, cps_end = aux.detector_to_change_region(detector, tseq, thresh)

    # plot
    fig, ax = plt.subplots(2, 1, figsize=(7, 3));

    ax[0].plot(tseq, [probability_of_glasses(t) for t in tseq], color = "black")
    ax[0].set_ylabel("$\\pi(t)$", fontsize=16)
    ax[0].set_xlim(0, 1)
    ax[0].set_xticklabels([])

    ax[1].plot(tseq, detector, color = "black")
    ax[1].axhline(thresh, color="red", linestyle="--", label = "$\chi^2_{16, 0.99}$")
    for c in cps_start:    
        ax[1].axvline(c, color="red")
    for c in cps_end:
        ax[1].axvline(c, color="red", label = "Change Period Markers")
    ax[1].set_ylabel("$D(t)$", fontsize=16)
    ax[1].set_xlabel("$t$", fontsize=16)
    ax[1].set_xlim(0, 1)

    for axi in [ax[0], ax[1]]:
        axi.yaxis.set_label_coords(-0.075, 0.5)
    
    handles, labels = ax[1].get_legend_handles_labels()
    legend_dict = dict(zip(labels, handles))
    ax[1].legend(legend_dict.values(), legend_dict.keys(), bbox_to_anchor=(0.5, -0.3), 
                 loc="upper center", ncol=len(legend_dict), fontsize=16)
    fig.show()
