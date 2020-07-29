import matplotlib
from config import *
from plot_utils import *
from shared_utils import *
import pickle as pkl
import numpy as np
from collections import OrderedDict
from matplotlib import pyplot as plt
from pymc3.stats import quantiles

def curves(start, n_weeks=3, save_plot=False):

    with open('../data/counties/counties.pkl', "rb") as f:
        counties = pkl.load(f)

    start = int(start)
    n_weeks = int(n_weeks)
    model_i = 35
    
    # colors for curves
    C1 = "#D55E00"
    C2 = "#E69F00"
    C3 = "#0073CF"

    # quantiles we want to plot
    qs = [0.25, 0.50, 0.75]

    fig = plt.figure(figsize=(6, 8))
    grid = plt.GridSpec(1,1)

    i = 0
    disease = "covid19"
    prediction_region = "germany"
    data = load_daily_data_n_weeks(
        start, n_weeks, disease, prediction_region, counties
        )

    start_day = pd.Timestamp('2020-01-28') + pd.Timedelta(days=start)
    i_start_day = 0
    day_0 = start_day + pd.Timedelta(days=n_weeks*7+5)
    day_m5 = day_0 - pd.Timedelta(days=5)
    day_p5 = day_0 + pd.Timedelta(days=5)

    _, target, _, _ = split_data(
        data,
        train_start=start_day,
        test_start=day_0,
        post_test=day_p5)

    county_ids = target.columns

    res = load_pred_window(start, n_weeks)
    n_days = (day_p5 - start_day).days
    prediction_samples = np.reshape(res['y'], (res['y'].shape[0], -1, 412))
    prediction_samples = prediction_samples[:,i_start_day:i_start_day+n_days,:]
    ext_index = pd.DatetimeIndex(
        [d for d in target.index] + \
        [d for d in pd.date_range(
                        target.index[-1]+timedelta(1),
                        day_p5-timedelta(1)
                    )])

    # TODO: figure out where quantiles comes from and if its pymc3, how to replace it
    prediction_quantiles = quantiles(prediction_samples, (5, 25, 75, 95)) 

    prediction_mean = pd.DataFrame(
        data=np.mean(
            prediction_samples,
            axis=0),
        index=ext_index,
        columns=target.columns)
        
    map_ax = fig.add_subplot(grid[0, i])
  
    # Calculate predicted case per 100000 people.
    map_vals = prediction_mean.iloc[-10]
    ik= 0
    for key, _ in counties.items():
        n_people = counties[key]['demographics'][('total',2018)]
        map_vals[ik] = (map_vals[ik] / n_people) * 100000
        ik = ik+1

    # plot the chloropleth map
    plot_counties(map_ax,
                counties,
                map_vals.to_dict(),
                edgecolors=None,
                xlim=xlim,
                ylim=ylim,
                contourcolor="black",
                background=False,
                xticks=False,
                yticks=False,
                grid=False,
                frame=True,
                ylabel=False,
                xlabel=False,
                lw=2)
    map_ax.set_rasterized(True)
    
    fig.text(
        0.71,
        0.17,
        "Neuinfektionen \n pro 100.000 \n Einwohner", 
        fontsize=14, 
        color=[0.3,0.3,0.3]
        )

    if save_plot:
        year = str(start_day)[:4]
        month = str(start_day)[5:7]
        day = str(start_day)[8:10]
        day_folder_path = "../figures/{}_{}_{}".format(year, month, day)
        if not os.path.isdir(day_folder_path):
            os.mkdir(day_folder_path)

        plt.savefig(
            "../figures/{}_{}_{}/map.png".format(year, month, day), 
            dpi=300
            )

    plt.close()
    return fig


