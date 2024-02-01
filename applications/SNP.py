import matplotlib.pyplot as plt
import autograd.numpy as np
import pandas as pd
import yfinance as yf # for getting stock volumes

import sys
import os

parent_dir = os.path.dirname(os.path.abspath('.'))
sys.path.append(parent_dir)

from library.ll import fit_with_search
from library.auxilliary import detector_to_changepoint
from library.sim import *

from scipy.stats import chi2
from tqdm.auto import tqdm


plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 12
})
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# mapper to go from company name in dataset to stock ticker
mapper = {
        "AMERICAN EXPRESS CO": "AXP",
        "AFLAC INC": "AFL",
        "AMERICAN INTERNATIONAL GROUP": "AIG",
        "BANK OF NEW YORK MELLON CORP": "BK",
        "JPMORGAN CHASE & CO": "JPM",
        "AON PLC": "AON",
        "COMERICA INC": "CMA",
        "CITIGROUP INC": "C",
        "FIFTH THIRD BANCORP": "FITB",
        "REGIONS FINANCIAL CORP": "RF",
        "M & T BANK CORP": "MTB",
        "U S BANCORP": "USB",
        "FRANKLIN RESOURCES INC": "BEN",
        "ARTHUR J GALLAGHER & CO": "AJG",
        "HUNTINGTON BANCSHARES": "HBAN",
        "LINCOLN NATIONAL CORP": "LNC",
        "LOEWS CORP": "L",
        "MARSH & MCLENNAN COS": "MMC",
        "S&P GLOBAL INC": "SPGI",
        "BANK OF AMERICA CORP": "BAC",
        "NORTHERN TRUST CORP": "NTRS",
        "WELLS FARGO & CO": "WFC",
        "PNC FINANCIAL SVCS GROUP INC": "PNC",
        "RAYMOND JAMES FINANCIAL CORP": "RJF",
        "KEYCORP": "KEY",
        "STATE STREET CORP": "STT",
        "GLOBE LIFE INC": "GL",
        "ZIONS BANCORPORATION NA": "ZION",
        "TRUIST FINANCIAL CORP": "TFC",
        "MORGAN STANLEY": "MS",
        "PRICE (T. ROWE) GROUP": "TROW",
        "UNUM GROUP": "UNM",
        "PROGRESSIVE CORP-OHIO": "PGR",
        "SCHWAB (CHARLES) CORP": "SCHW",
        "BERKLEY (W R) CORP": "WRB",
        "CINCINNATI FINANCIAL CORP": "CINF",
        "PEOPLE'S UNITED FINL INC": "PBCT",
        "SVB FINANCIAL GROUP": "SIVBQ",
        "CHUBB LTD": "CB",
        "ALLSTATE CORP": "ALL",
        "INVESCO LTD": "IVZ",
        "CAPITAL ONE FINANCIAL CORP": "COF",
        "EVEREST RE GROUP LTD": "EG",
        "HARTFORD FINANCIAL SERVICES": "HIG",
        "GOLDMAN SACHS GROUP INC": "GS",
        "BLACKROCK INC": "BLK"
    }

# convert company name to stock ticker
def get_ticker_symbol(company_name):
    return mapper[company_name]

def get_company_name(ticker_symbol):
    inverted_mapper =  {v: k for k, v in mapper.items()}
    return inverted_mapper[ticker_symbol]

def get_volume(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)
    return data["Volume"]

# feature function
def f(x):
    dimensionality = len(x)
    result = []

    for i in range(dimensionality):
        for j in range(dimensionality):
            if i < j:
                result.append(x[i] * x[j])

    return np.array(result).flatten()

    
if __name__ == "__main__":

    np.random.seed(1)
    d = 5

    fname = ""
    assert len(fname) > 0, "Please provide a filename to the S&P data, see the readme for details"

    df = pd.read_csv(fname, index_col=0)

    # convert datadate to datetime, format YYYY-MM-DD
    df["datadate"] = pd.to_datetime(df["datadate"], format="%Y%m%d")

    # take d highest volume companies 
    company_tickers       = [get_ticker_symbol(name) for name in df.columns[1:]]
    volumes               = get_volume(company_tickers, df.datadate.min().strftime("%Y-%m-%d"),  df.datadate.max().strftime("%Y-%m-%d"))
    top_d_volume_tickers  = volumes.sum(0).sort_values(ascending=False).index[:d].values
    subset                = [get_company_name(ticker) for ticker in top_d_volume_tickers]

    # subset to only these companies
    df_sub = df[["datadate"] + list(subset)] 

    # convert to datetime
    T = df_sub.shape[0]
    n = 10

    # convert to data format and bin
    binned_data = []
    times = []
    for ti in np.arange(0, T, n):
        binned_data.append(df_sub.iloc[ti:(ti+n), 1:].values)
        times.append(df_sub.iloc[ti:(ti+n), 0].mean())

    # remove last bin if it's not full
    if binned_data[-1].shape[0] < n:
        binned_data = binned_data[:-1]
        times = times[:-1]

    tseq = np.linspace(0, 1, len(binned_data))[:, None]

    alpha, dthetat, detector, _ = fit_with_search(tseq, binned_data, f,
                                                  lams = [5],
                                                  bs = [0.05, 0.075, 0.1],
                                                  inner_nw = True, inner_nw_bws = [1/(20*T), 1/(40*T)],        
                                                  verbose=True                                          
                                                  )
    sig_level = 0.01
    thresh = chi2.ppf(1-sig_level, df=len(f(binned_data[0][0, :])))

    # detect changepoints
    changepoints = detector_to_changepoint(detector, tseq, thresh, eps=0.01)

    company_labels = top_d_volume_tickers

    # Plot correlations in dthetat

    x_ticks = [times[0], times[-1]]
    x_labels = [times[0].year, times[-1].year]

    fd = len(f(binned_data[0][0, :]))
    titles = []
    for i in range(d):
        for j in range(d):
            if i < j:
                titles.append(f"{company_labels[i]} x {company_labels[j]}")

    fig, ax = plt.subplots(2, int(fd/2), figsize=(10, 3.5))
    
    count = 0
    for row_i in range(2):

        
        for col_i in range(int(fd/2)):
            ax[row_i, col_i].plot(times, dthetat[:, count], color="black")
            ax[row_i, col_i].set_xlim(x_ticks[0] - pd.DateOffset(years=1, months=6), x_ticks[-1] + pd.DateOffset(years=1, months=6))

            ax[row_i, col_i].set_title(f"{titles[count]}", fontsize=21)
            
            if row_i == 0:
                ax[row_i, col_i].set_xticks([])
            
            if row_i == 1:
                ax[row_i, col_i].set_xticks(x_ticks)  # Set x-tick locations
                ax[row_i, col_i].set_xticklabels(x_labels) 
                ax[row_i, col_i].set_xlabel("Year", fontsize=19)

            for c in changepoints:
                t_c = np.abs(tseq - c).argmin()
                ax[row_i, col_i].axvline(times[t_c], color="r", label = "Detected CP", linewidth=1)
            
            count += 1

        ax[row_i, 0].set_ylabel("$\\partial_t \\hat{\\theta}(t)$", fontsize=19)
    
    fig.tight_layout()

    # Plot detector and CPs
    fig, ax = plt.subplots(1, 1, figsize=(7, 2.5))

    ax.plot(times, np.log1p(detector), color="black")
    ax.axhline(np.log1p(thresh), color="r", linestyle="--", label = "$\\log(1+\\chi^2_{1, 0.99})$")

    for c in changepoints:
        t_c = np.abs(tseq - c).argmin()
        ax.axvline(times[t_c], color="r", label = "Estimated CPs")

    ax.set_ylabel("$\\log (1 + D(t))$", fontsize=16)
    ax.set_xlabel("Year", fontsize=16)

    handles, labels = ax.get_legend_handles_labels()
    legend_dict = dict(zip(labels, handles))  
    
    # plot legend underneath
    ax.legend(legend_dict.values(), legend_dict.keys(), loc='upper center', 
              bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize=16)

    fig.tight_layout()













    # rolling_corrs = pd.DataFrame(index=df_sub.index)

    # for i in range(len(subset)):
    #     for j in range(i+1, len(subset)):
    #         colname = f'corr_{subset[i]}_{subset[j]}'
    #         rolling_corrs[colname] = df_sub[subset[i]].rolling(window=60).corr(df_sub[subset[j]])

    # plt.plot(rolling_corrs.values.mean(1))