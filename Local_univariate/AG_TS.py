import pandas as pd
import numpy as np
import os
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import matplotlib.pyplot as plt
from load_function import load_timeseries, calculate_metrics, find_base_path
import time

def to_timeseriesdataframe(df):
    df = df.copy()
    
    ts_df = TimeSeriesDataFrame.from_data_frame(
        df=df,
        id_column='site',
        timestamp_column='date'
    )
    
    return ts_df

def evaluate_autogluon(df, prediction_length, covariates=True, multiple_ts=False):
    preset = 'high_quality'

    sites = list(df['site'].unique())

    print(f'running AutoGluon model with preset={preset} and prediction length {prediction_length}')

    ts_df = to_timeseriesdataframe(df)

    # Train/test split
    train_data, test_data = ts_df.train_test_split(prediction_length=prediction_length)
    
    start_time = time.time()
    
    if covariates:
        known_cov_names = [col for col in ts_df.columns if col not in ['site', 'date', 'value']]
    else:
        known_cov_names = []
    
    try:
        predictor = TimeSeriesPredictor(
            prediction_length=prediction_length,
            freq='D',
            cache_predictions=False,
            eval_metric="MAPE",
            target="value", 
            known_covariates_names=known_cov_names,
        ).fit(train_data, presets=preset)
    
    except Exception as e:
        print(f"error occurred: {e}")
        print("continuing with next model")
        site_results = []
        site_results.append({
                    'Model': 'Autogluon',
                    'MAPE': 0,
                    'MSE': 0,
                    'MAE': 0,
                    'Time': 0,
                    'Site': 0  } 
            )
        
        return site_results


    # Need to loop over the sites so we can split the multi index and get metrics per site.
    covar_df = pd.DataFrame()
    y_true = pd.DataFrame()
    for pb in sites:
        mask = test_data.index.get_level_values('item_id') == pb
        
        pb_test_data = test_data.loc[mask]
        y_true = pd.concat([y_true, pb_test_data[-prediction_length:]])
        covar_df = pd.concat([covar_df, pb_test_data[-prediction_length:]])
        covar_df = covar_df.drop(columns=['value'])

    if covariates and known_cov_names:
        y_pred = predictor.predict(train_data, known_covariates=covar_df)
    else:
        y_pred = predictor.predict(train_data)
    
    total_time = time.time() - start_time
    
    site_results = []
    if multiple_ts:
        # check how y_pred works when predicting multiple ts
        for site in sites:
            mask = y_pred.index.get_level_values('item_id') == site
            site_predictions, site_true = y_pred.loc[mask], y_true.loc[mask]
            MAPE, MSE, MAE = calculate_metrics(site_true['value'], site_predictions['mean'])
            site_results.append({
                    'Model': 'Autogluon',
                    'MAPE': MAPE,
                    'MSE': MSE,
                    'MAE': MAE,
                    'Time': total_time,
                    'Site': site  } 
            )

    else:
        MAPE, MSE, MAE = calculate_metrics(y_true['value'], y_pred['mean'])
        site_results.append({
                'Model': 'Autogluon',
                'MAPE': MAPE,
                'MSE': MSE,
                'MAE': MAE,
                'Time': total_time,
                'Site': site  } 
            )
    
    y_pred.to_csv('./newcsv_aarland4.csv')
    return site_results

# prediction_length = 3
# np.random.seed(42)
# base_path = find_base_path()
# files = os.listdir(base_path + '2_Hydraulic head data/Sensor data')

# df = pd.DataFrame()

# for i in range(3):
#     temp_df = load_timeseries(files[i])
#     temp_df['value'] = temp_df['value'].diff()
#     temp_df = temp_df.dropna(axis=0)
#     df = pd.concat([df, temp_df])