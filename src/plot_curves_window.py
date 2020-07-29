import matplotlib
from config import *
from plot_utils import *
from shared_utils import *
import pickle as pkl
import numpy as np
from collections import OrderedDict
from matplotlib import pyplot as plt
from pymc3.stats import quantiles
import os

# Load only one county
def curves(start, county, save_plot=False, update=True):
    '''
    start -- start_id (28.01.2020 -> 0, 29.01.2020 -> 1, ...)
    update -- whether to update all plots
    '''

    with open('../data/counties/counties.pkl', "rb") as f:
        counties = pkl.load(f)

    start = int(start)
    n_weeks = 3
    start_day = pd.Timestamp('2020-01-28') + pd.Timedelta(days=start)
    year = str(start_day)[:4]
    month = str(start_day)[5:7]
    day = str(start_day)[8:10]

    countyByName = make_county_dict()
    plot_county_names = {"covid19": [county]}

    if not update:
        if os.path.exists("../figures/{}_{}_{}/curve_trend_{}.png".format(
            year, month, day,countyByName[county]
            )):
            return

    fig = plt.figure(figsize=(12, 6))
    grid = plt.GridSpec(
        1,
        1,
        top=0.9,
        bottom=0.2,
        left=0.07,
        right=0.97,
        hspace=0.25,
        wspace=0.15,
        )

    disease = "covid19"
    prediction_region = "germany"
    data = load_daily_data_n_weeks(
        start, n_weeks, disease, prediction_region, counties
        )

    start_day = pd.Timestamp('2020-01-28') + pd.Timedelta(days=start)
    day_0 = start_day + pd.Timedelta(days=n_weeks*7+5)
    day_p5 = day_0 + pd.Timedelta(days=5)

    _, target, _, _ = split_data(
        data,
        train_start=start_day,
        test_start=day_0,
        post_test=day_p5)

    county_ids = target.columns
    county_id = countyByName[county]
        
    # Load our prediction samples
    n_days = (day_p5 - start_day).days 

    res = load_pred_window(start, n_weeks)
    prediction_samples = np.reshape(res['y'], (res['y'].shape[0], -1, 412)) 
    prediction_samples = prediction_samples[:,:n_days,:]

    # Extended Index over all dates.
    ext_index = pd.DatetimeIndex(
        [d for d in target.index] + \
        [d for d in pd.date_range(
            target.index[-1]+timedelta(1),
            day_p5-timedelta(1)
            )
        ]
    )

    prediction_quantiles = quantiles(prediction_samples, (5, 25, 75, 95)) 

    prediction_mean = pd.DataFrame(
        data=np.mean(
            prediction_samples,
            axis=0),
        index=ext_index,
        columns=target.columns
        )
    prediction_q25 = pd.DataFrame(
        data=prediction_quantiles[25],
        index=ext_index,
        columns=target.columns
        )
    prediction_q75 = pd.DataFrame(
        data=prediction_quantiles[75],
        index=ext_index,
        columns=target.columns
        )
    prediction_q5 = pd.DataFrame(
        data=prediction_quantiles[5],
        index=ext_index,
        columns=target.columns
        )
    prediction_q95 = pd.DataFrame(
        data=prediction_quantiles[95],
        index=ext_index,
        columns=target.columns
        )
    
    # Colors for curves.
    C1 = "#D55E00"
    C2 = "#E69F00"
        
    ax = fig.add_subplot(grid[0, 0])
    county_id = countyByName[plot_county_names[disease][0]]
    dates = [pd.Timestamp(day) for day in ext_index]
    days = [ (day - min(dates)).days for day in dates]

    # Plot predictions.
    p_pred = ax.plot_date(
        dates,
        prediction_mean[county_id],
        "-",
        color=C1,
        linewidth=2.0,
        zorder=4)

    # Plot quantiles.
    p_quant = ax.fill_between(
        dates,
        prediction_q25[county_id],
        prediction_q75[county_id],
        facecolor=C2,
        alpha=0.5,
        zorder=1
        )
    ax.plot_date(
        dates,
        prediction_q25[county_id],
        ":",
        color=C2,
        linewidth=2.0,
        zorder=3
        )
    ax.plot_date(
        dates,
        prediction_q75[county_id],
        ":",
        color=C2,
        linewidth=2.0,
        zorder=3
        )
    p_quant2 = ax.fill_between(
        dates,
        prediction_q5[county_id],
        prediction_q95[county_id],
        facecolor=C2,
        alpha=0.25,
        zorder=0
        )
    ax.plot_date(
        dates, 
        prediction_q5[county_id], 
        ":",
        color=C2, 
        alpha=0.5, 
        linewidth=2.0, 
        zorder=1
        )
    ax.plot_date(
        dates, 
        prediction_q95[county_id], 
        ":",
        color=C2, 
        alpha=0.5, 
        linewidth=2.0, 
        zorder=1
        )

    # Plot ground truth.
    p_real = ax.plot_date(dates[:-5], target[county_id], "k.")

    # Plot markers for now- and forecast.
    ax.axvline(dates[-5],ls='-', lw=2, c='cornflowerblue')
    ax.axvline(dates[-10],ls='--', lw=2, c='cornflowerblue')
    fontsize_bluebox = 18
    fig.text(
        0.67,0.86,"Nowcast",fontsize=fontsize_bluebox,
        bbox=dict(facecolor='cornflowerblue')
        )
    fig.text(
        0.828,0.86,"Forecast",fontsize=fontsize_bluebox,
        bbox=dict(facecolor='cornflowerblue')
        )

    # Set ticks and labels.
    ax.tick_params(
        axis="both", direction='out',
        size=6, labelsize=16, length=6
        )
    ticks = [
        start_day+pd.Timedelta(days=i) for i in [0,5,10,15,20,25,30,35,40]
        ]
    labels = [
        "{}.{}.{}".format(
            str(d)[8:10], 
            str(d)[5:7], 
            str(d)[:4]
            ) 
            for d in ticks
        ]
    plt.xticks(ticks,labels)        
    plt.setp(ax.get_xticklabels(), rotation=45)

    # Set axis limits.
    ax.set_xlim([start_day,day_p5-pd.Timedelta(days=1)])
    ylimmax = max(3*(target[county_id]).max(),10)
    ax.set_ylim([-(1/30)*ylimmax,ylimmax])
    ax.autoscale(True)

    # Set the legend.
    ax.legend(
        [p_real[0], p_pred[0], p_quant, p_quant2],
        ["Fallzahlen", "Vorhersage", "25\%-75\%-Quantil", "5\%-95\%-Quantil"],
        fontsize=16, 
        loc="upper left"
        )
    
    # Calculate and plot probability of increase.
    i_county =  county_ids.get_loc(county_id)
    trace = load_trace_window(disease, start, n_weeks)
    trend_params = pm.trace_to_dataframe(trace, varnames=["W_t_t"]).values
    trend_w2 = np.reshape(trend_params, newshape=(1000,412,2))[:,i_county,1]
    prob2 = np.mean(trend_w2>0)
    
    fontsize_probtext = 14
    if prob2 >=0.5:
        fig.text(
            0.865, 0.685, 
            "Die Fallzahlen \n werden mit einer \n Wahrscheinlichkeit \
                 \n von {:2.1f}\% steigen.".format(prob2*100), 
            fontsize=fontsize_probtext,
            bbox=dict(facecolor='white')
            )
    else:
        fig.text(
            0.865, 0.685, 
            "Die Fallzahlen \n werden mit einer \n Wahrscheinlichkeit \
                 \n von {:2.1f}\% fallen.".format(100-prob2*100), 
            fontsize=fontsize_probtext,
            bbox=dict(facecolor='white')
            )
        

    if save_plot:
        day_folder_path = "../figures/{}_{}_{}".format(year, month, day)
        if not os.path.isdir(day_folder_path):
            os.mkdir(day_folder_path)
        plt.savefig(
            "../figures/{}_{}_{}/curve_{}.png".format(
                year, month, day,countyByName[county]
                ), 
            dpi=200
            )

    plt.close()
    return fig