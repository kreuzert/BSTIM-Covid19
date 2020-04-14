import pymc3 as pm
import pandas as pd
import matplotlib
import numpy as np
import pickle as pkl
import datetime
from BaseModel import BaseModel
import isoweek
from matplotlib import rc
from shared_utils import *
from pymc3.stats import quantiles
from matplotlib import pyplot as plt
# from pandas import register_matplotlib_converters
# register_matplotlib_converters() # the fk python
def temporal_contribution(save_plot=False):

    plt.style.use('ggplot')

    with open('../data/counties/counties.pkl', "rb") as f:
        county_info = pkl.load(f)

    #
    # t_start, t_start_b, t_end = pd.Timestamp(2011,1,1), pd.Timestamp(2013,1,1), pd.Timestamp(2018, 1, 1)
    #
    # _t = np.linspace(datetime.timedelta(days=7).total_seconds(), datetime.timedelta(days=30).total_seconds(), 101)
    # t = _t.astype("timedelta64[s]")
    # tticks = [datetime.timedelta(days=d).total_seconds() for d in [7, 14, 21, 28]]
    # tticks_l = [7, 14, 21, 28]
    #
    # t_year = pd.timedelta_range("0D","365D", freq="1D") + t_start
    # t_all_cr = pd.timedelta_range("0D","{}D".format((t_end-t_start).days), freq="7D") + t_start
    # t_all_b = pd.timedelta_range("0D","{}D".format((t_end-t_start_b).days), freq="7D") + t_start_b
    #
    # x = np.linspace(-75.0, 75.0, 101)
    # σs = [6.25, 12.5, 25.0, 50]
    # time_plot = np.arange(1,54)
    # cmap = matplotlib.cm.RdBu_r

    C1 = "#D55E00"
    C2 = "#E69F00"
    C3 = C2  # "#808080"

    fig = plt.figure(figsize=(12, 9))
    #fig.suptitle("Learned interaction kernels and temporal contributions", fontsize=20)
    grid = plt.GridSpec(3, len(diseases), top=0.93, bottom=0.12,
                        left=0.11, right=0.97, hspace=0.28, wspace=0.30)

    i = 0
    disease = "covid19"
    use_age = True
    use_eastwest = True
    prediction_region = "germany"


    data = load_daily_data(disease, prediction_region, county_info)
    # data_train, target_train, data_test, target_test = split_data(data)
    __, target_train, _, _ = split_data(
        data, train_start=pd.Timestamp(
            2020, 1, 28), test_start=pd.Timestamp(
            2020, 3, 30), post_test=pd.Timestamp(
            2020, 3, 31)) # plots for the training period!
    tspan = (target_train.index[0], target_train.index[-1])
    model = BaseModel(tspan,
                    county_info,
                    ["../data/ia_effect_samples/{}_{}.pkl".format(disease,
                                                                    i) for i in range(100)],
                    include_demographics=use_age)

    features = model.evaluate_features(
        target_train.index, target_train.columns)

    trend_features = features["temporal_trend"].swaplevel(0, 1).loc["09162"]
    periodic_features = features["temporal_seasonal"].swaplevel(
        0, 1).loc["09162"]
    #t_all = t_all_b if disease == "borreliosis" else t_all_cr

    trace = load_trace(disease, use_age, use_eastwest)
    trend_params = pm.trace_to_dataframe(trace, varnames=["W_t_t"])
    periodic_params = pm.trace_to_dataframe(trace, varnames=["W_t_s"])

    TT = trend_params.values.dot(trend_features.values.T)
    TP = periodic_params.values.dot(periodic_features.values.T)
    TTP = TT + TP
    TT_quantiles = quantiles(TT, (25, 75))
    TP_quantiles = quantiles(TP, (25, 75))
    TTP_quantiles = quantiles(TTP, (25, 75))

    dates = [pd.Timestamp(day) for day in target_train.index.values]
    days = [ (day - min(dates)).days for day in dates]


    # Temporal periodic effect
    ax_p = fig.add_subplot(grid[0, i])

    ax_p.fill_between(days, np.exp(TP_quantiles[25]), np.exp(
        TP_quantiles[75]), alpha=0.5, zorder=1, facecolor=C1)
    ax_p.plot(days, np.exp(TP.mean(axis=0)),
                "-", color=C1, lw=2, zorder=5)
    ax_p.plot(days, np.exp(
        TP_quantiles[25]), "-", color=C2, lw=2, zorder=3)
    ax_p.plot(days, np.exp(
        TP_quantiles[75]), "-", color=C2, lw=2, zorder=3)
    ax_p.plot(days, np.exp(TP[:25, :].T),
                "--", color=C3, lw=1, alpha=0.5, zorder=2)

    ax_p.tick_params(axis="x", rotation=45)

    # Temporal trend effect
    ax_t = fig.add_subplot(grid[1, i], sharex=ax_p)

    ax_t.fill_between(days, np.exp(TT_quantiles[25]), np.exp(
        TT_quantiles[75]), alpha=0.5, zorder=1, facecolor=C1)
    ax_t.plot(days, np.exp(TT.mean(axis=0)),
                "-", color=C1, lw=2, zorder=5)
    ax_t.plot(days, np.exp(
        TT_quantiles[25]), "-", color=C2, lw=2, zorder=3)
    ax_t.plot(days, np.exp(
        TT_quantiles[75]), "-", color=C2, lw=2, zorder=3)
    ax_t.plot(days, np.exp(TT[:25, :].T),
                "--", color=C3, lw=1, alpha=0.5, zorder=2)

    ax_t.tick_params(axis="x", rotation=45)

    # Temporal trend+periodic effect
    ax_tp = fig.add_subplot(grid[2, i], sharex=ax_p)

    ax_tp.fill_between(days, np.exp(TTP_quantiles[25]), np.exp(
        TTP_quantiles[75]), alpha=0.5, zorder=1, facecolor=C1)
    ax_tp.plot(days, np.exp(TTP.mean(axis=0)),
                    "-", color=C1, lw=2, zorder=5)
    ax_tp.plot(days, np.exp(
        TTP_quantiles[25]), "-", color=C2, lw=2, zorder=3)
    ax_tp.plot(days, np.exp(
        TTP_quantiles[75]), "-", color=C2, lw=2, zorder=3)
    ax_tp.plot(days, np.exp(TTP[:25, :].T),
                    "--", color=C3, lw=1, alpha=0.5, zorder=2)

    ax_tp.tick_params(axis="x", rotation=45)

    # ax_p.set_xticks(np.arange(1,54))
    # ax_p.set_xticklabels(tticks_l_year)

    ax_p.set_title("campylob." if disease ==
                "campylobacter" else disease, fontsize=22)
    ax_tp.set_xlabel("time [days]", fontsize=22)

    if i == 0:
        ax_p.set_ylabel("periodic\ncontribution", fontsize=22)
        ax_t.set_ylabel("trend\ncontribution", fontsize=22)
        ax_tp.set_ylabel("combined\ncontribution", fontsize=22)
    # elif i==2:
    ax_t.set_xlim(days[0], days[-1])
    # ax_t.set_xlim((isoweek.Week(2013, 1).wednesday(),
    #                isoweek.Week(2016, 1).wednesday()))
    # ax_t.set_xticks([isoweek.Week(i, 1).wednesday()
    #                  for i in range(2013, 2017)])
    # ax_t.set_xticks()
    # ax_t.set_xticklabels(range(2013, 2017))

    ax_t.tick_params(labelbottom=False, labelleft=True, labelsize=18, length=6)
    ax_p.tick_params(labelbottom=False, labelleft=True, labelsize=18, length=6)
    ax_tp.tick_params(labelbottom=True, labelleft=True, labelsize=18, length=6)

    # fig.text(0, 1 + 0.025, r"$\textbf{" + str(i + 1) + r"A}$",
    #          fontsize=22, transform=ax_p.transAxes, usetex=True)
    # fig.text(0, 1 + 0.025, r"$\textbf{" + str(i + 1) + r"B}$",
    #          fontsize=22, transform=ax_t.transAxes, usetex=True)
    # fig.text(0, 1 + 0.025, r"$\textbf{" + str(i + 1) + r"C}$",
    #          fontsize=22, transform=ax_tp.transAxes, usetex=True)

    if save_plot:
        fig.savefig("../figures/temporal_contribution.pdf")

    return fig
