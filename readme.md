# TSM-DE Code Implementation

This is a repository for the code used in the paper "Time Score Matching for Parameter Change Estimation and Changepoint Detection".

## Installation

Clone the github repository:
```
git clone https://github.com/tsmdepaper/tsmde
cd tsmde
```
and reproduce the results of the paper by running each file individually, explained below. We recommend creating a virtual environment with something such as anaconda, with Python version 3.11.4, e.g. in bash
```
conda create -n tsmde python=3.11.4 ipython
conda activate tsmde
```
and installing required packages given with the `requirements.txt` file
```
pip install -r requirements.txt
``````
to ensure every package is installed correctly for this repo. 

You also need to have a working R installation to allow the `motivation_plots.py` file that uses `rpy2` to work, for which you also require the implementation from [1], at https://github.com/joonpyo-kim/movingsumlin. Therefore for this file you also need to run
```
git clone https://github.com/joonpyo-kim/movingsumlin
```
Additionally, it might be necessary to modify the `sourceCpp("mosumCwald.cpp")` line in `MOSUM_linear.R`, replacing `"mosumCwald.cpp"` with the full path to this file as it is saved in your local directory.

## File Directory

The file directory is given as
```
.
├── applications
│   ├── concept_drift_models.py
│   ├── concept_drift.py
│   ├── smiling_model.pth
│   ├── SNP.py
│   └── temps.py
├── examples
│   ├── changepoint_benchmark.py
│   ├── changepoint_benchmarks_pwl.py
│   ├── illustrative_examples.py
│   └── motivation_plots.py
├── library
│   ├── ...
└── readme.md
```

### Library

The `library` folder contains packaged code for the TSM-DE method. This is split into separate files, each containing a different routine. For example, `asymptotic` contains code to calculate the asymptotic variance / detector of the TSM-DE estimator. The `train` function in `train.py` is the main function for training the TSM-DE model, and requires specification of two important functions: 

```
def model_t(par, t):
    ...

def model_tt(par, t):
    ...
```
corresponding to the first derivative $\partial_t \theta_t$ and the second derivative $\partial_t^2 \theta_t$ of the model parameters, these are required by the TSM-DE objective function.

The two main methods given by the RBF feature and the sliding window model are in the files `basis` and `ll`, named for the RBF basis and local linear (sliding window) method. These provide aliases to the `train` function and are more straightforward to use. For example

```python
from library.basis import fit_basis
from library.sim import simulate_mean_easy
data, true_changepoints, _, _ = simulate_mean_easy(n = 5, T=500)

tseq = np.linspace(0, 1, 500)[:, None]
alpha, dthetaf, detector, Sig_dthf = fit_basis(tseq, data, f=lambda x: x,             
            b = 50, lam = 0.1, bw = 0.001
          )

dthetat   = dthetaf(tseq)
detectort = detector(tseq)
```

which will give the estimates $\partial_t \hat{\theta}_t$ as `dthetat` and $D(t)$ as `detectort`. Then running
```python
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 1, figsize=(10, 6))
ax[0].plot(tseq, dthetat)
ax[1].plot(tseq, detectort)
```
will plot these values. This is a basic example of the method for a mean change scenario. To test different examples of changepoint detection, you can modify the data simulation function. All different toy examples are given in `library/sim.py`. For a full explanation of the methods and the parameters involved, see the paper. A similar approach can be used for the sliding window method, see the `fit_local_linear` function `library/ll.py` for details.

### Examples

These are the numerical examples found in the paper in Figures 1 and 2, and results from Table 1.

To reproduce Figure 1, you first must modify the `piecewise_linear_mosum_R_filepath` variable at the top of the `motivation_plots.py` file to point towards the `MOSUM_linear.R` file in your installation of the implementation from [1]. Then run `motivation_plots.py`. This will produce the plots in Figure 1 of the paper.

To reproduce Figure 2, run `illustrative_examples.py`. This will produce the plots in Figure 2 of the paper.

To reproduce Table 1, run `changepoint_benchmark.py`. This will produce one trial (one seed), for which seeds 0 - 63 are averaged and reported in the paper. This will take a few minutes to run. To reproduce the table completely, run the file 64 times, each time changing the seed to be 0, 1, ..., 63, and average the results.

### Applications

Section 6.1 contains the temperature anomalies data example, obtained from https://www.ncei.noaa.gov/access/monitoring/global-temperature-anomalies/. Download the Monthly temperature anomalies data for the Northern hemisphere across Land and Ocean. Modify the `fname` variable in `temps.py` to point to this .csv file. Then run `temps.py` to reproduce Figure 3 of the paper.

Section 6.2 contains the S&P Data example, obtained from Wharton Research Data Services. You need to request the data, and we unforunately cannot provide it here. Once you have the data, modify the `fname` variable in `SNP.py` to point to the .csv file. Then run `SNP.py` to reproduce Figure 4 of the paper.

Section 6.3 contains the concept drift example, which is a simulated example. We have provided the pre-trained classification model in `smiling_model.pth`, which is a PyTorch model. However, you need to download the CelebA dataset, obtained from https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html. 
The files to download are (a) the folder `CelebA/Img/img_align_celeba`, and (b) the annotations text file `CelebA/Anno/list_attr_celeb.txt`.
Then, modify the `folderpath` and `attrpath` variables in `concept_drift.py` to point to the folder containing the images and the text file containing attribute labels. Then run `concept_drift.py` to reproduce Figure 5 of the paper.


[1] Kim, J., Oh, H.-S., and Cho, H. Moving sum procedure for change point detection under piecewise linearity. arXiv preprint arXiv:2208.04900, 2022

