from shared_utils import *
from BaseModel import BaseModel
from config import *
import pymc3 as pm
import pickle as pkl
import pandas as pd
import os
import sys

start = int(os.environ["SGE_DATE_ID"])
number_of_weeks = 3
model_i = 35 

start_date = pd.Timestamp("2020-01-28") + pd.Timedelta(days=start)
year = str(start_date)[:4]
month = str(start_date)[5:7]
day = str(start_date)[8:10]

#NOTE: for jureca, extend to the number of available cores (chains and cores!)
num_samples = 250
num_chains = 4
num_cores = num_chains

# Whether to sample parameters and predictions or only predictions.
SAMPLE_PARAMS = True

disease = "covid19"
prediction_region = "germany"

filename_params = "../data/mcmc_samples_backup/parameters_{}_{}".\
                                                format(disease,start)
filename_pred = "../data/mcmc_samples_backup/predictions_{}_{}.pkl".\
                                                format(disease, start)
filename_pred_trend = "../data/mcmc_samples_backup/predictions_\
                                trend_{}_{}.pkl".format(disease, start)
filename_model = "../data/mcmc_samples_backup/model_{}_{}.pkl".\
                                                format(disease, start)

# Load data
with open('../data/counties/counties.pkl', "rb") as f:
    county_info = pkl.load(f)

days_into_future = 5
data = load_daily_data_n_weeks(
    start, 
    number_of_weeks, 
    disease, 
    prediction_region, 
    county_info, 
    pad=days_into_future
    )

first_day = data.index.min()
last_day = data.index.max()

data_train, target_train, data_test, target_test = split_data(
    data,
    train_start=first_day,
    test_start=last_day - pd.Timedelta(days=days_into_future+4),
    post_test=last_day + pd.Timedelta(days=1)
)

tspan = (target_train.index[0], target_train.index[-1])

print("training for {} in {} with final model from {} to {}\n \
     Will create files {}, {} and {}".format(
        disease, 
        prediction_region, 
        *tspan, 
        filename_params, 
        filename_pred, 
        filename_model
    ))

model = BaseModel(
    tspan,
    county_info,
    ["../data/ia_effect_samples/{}_{}_{}/{}_{}.pkl".\
    format(year, month, day,disease, i) for i in range(100)],
    include_ia=True,
    include_report_delay=False,
    include_demographics=True,
    trend_poly_order=1,
    periodic_poly_order=4
)

if SAMPLE_PARAMS:
    print("Sampling parameters on the training set.")
    trace = model.sample_parameters(
        target_train,
        samples=num_samples,
        tune=100,
        target_accept=0.95,
        max_treedepth=15,
        chains=num_chains,
        cores=num_cores,
        window=True
        )

    with open(filename_model, "wb") as f:
    	pkl.dump(model.model, f)

    with model.model:
        pm.save_trace(trace, filename_params, overwrite=True)
else:
    print("Load parameters.")
    trace = load_trace_window(disease, start, number_of_weeks) 

print("Sampling predictions on the training and test set.")

pred = model.sample_predictions(
    target_train.index, 
    target_train.columns, 
    trace, 
    target_test.index,
    average_all=False,
    window=True
    )

pred_trend = model.sample_predictions(
    target_train.index, 
    target_train.columns, 
    trace, 
    target_test.index, 
    average_all=True,
    window=True
    )

with open(filename_pred, 'wb') as f:
    pkl.dump(pred, f)

with open(filename_pred_trend, "wb") as f:
    pkl.dump(pred_trend, f)
