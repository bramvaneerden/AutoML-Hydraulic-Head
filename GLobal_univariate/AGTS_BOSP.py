import pandas as pd
import numpy as np
import os
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

import time
from datetime import datetime
from autogluon.timeseries.splitter import ExpandingWindowSplitter
from sklearn.preprocessing import MinMaxScaler
import random
import torch
from autogluon.timeseries.models import presets

def calculate_metrics(y_true, y_pred):
    MAPE = mean_absolute_percentage_error(y_true, y_pred)
    MSE = mean_squared_error(y_true, y_pred)
    MAE = mean_absolute_error(y_true, y_pred)

    return MAPE, MSE, MAE

def to_timeseriesdataframe(df):
    df = df.copy()
    
    ts_df = TimeSeriesDataFrame.from_data_frame(
        df=df,
        id_column='site',
        timestamp_column='date'
    )
    
    return ts_df

def set_all_seeds(seed):
    """Set all seeds for reproducibility"""
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        
def evaluate_autogluon(df, prediction_length, random_seed, target, strat, num_val_windows=1, covariates=True):
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    preset = 'high_quality'

    sites = list(df['site'].unique())

    print(f'running AutoGluon model with preset={preset} and prediction length {prediction_length} for {num_val_windows} windows')
    print(f'using random seed: {random_seed}')
    
    df = df.copy(deep=True)
    
    if covariates == False:
        df = df.drop(columns=['precipitation', 'evaporation'])
    ts_df = to_timeseriesdataframe(df)

    train_data, test_data = ts_df.train_test_split(prediction_length=prediction_length*num_val_windows)
    
    if covariates:
        known_cov_names = [col for col in ts_df.columns if col not in ['site', 'date', 'value']]
    else:
        known_cov_names = []
    
    try:
        set_all_seeds(seed=random_seed)
        
        predictor = TimeSeriesPredictor(
            prediction_length=prediction_length,
            freq='D',
            cache_predictions=False,  
            eval_metric="MAE",
            target="value", 
            known_covariates_names=known_cov_names,
            verbosity=2
        ).fit(
            train_data, 
            random_seed=random_seed,
            presets=preset,
            excluded_model_types=[m for m in presets.MODEL_TYPES.keys() if m not in ['DirectTabular', 'TemporalFusionTransformer', 'DeepAR']],  
        )
    
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
    
    leaderboard = predictor.leaderboard()
    # leaderboard.to_csv(f'./model_info/leaderboard_{target}_{strat}_{datetime.now().strftime("%m_%d_%Y_%H_%M_%S")}.csv')
    
    splitter = ExpandingWindowSplitter(prediction_length=prediction_length, num_val_windows=num_val_windows)

    site_results = []
    all_pred = pd.DataFrame()
    results_per_window = []
    # average over results, dump y_pred after concat
    for window_idx, (train_split, val_split) in enumerate(splitter.split(test_data)):    
        y_pred = predictor.predict(train_split, known_covariates=val_split[[col for col in val_split.columns if col not in ['site', 'date', 'value']]], model='DirectTabular')
        y_pred_scaled = y_pred.copy()
        all_pred = pd.concat([all_pred, y_pred])
        y_true = val_split.groupby(level='item_id').tail(prediction_length)
        
        mask = y_pred.index.get_level_values('item_id') == target
        site_predictions = y_pred.loc[mask]
        site_true_unscaled = y_true.loc[mask]
        
        MAPE, MSE, MAE = calculate_metrics(site_true_unscaled['value'], site_predictions['mean'])
        
        # print(f"Window {window_idx} - MAPE: {MAPE:.4f}, MSE: {MSE:.4f}, MAE: {MAE:.4f}")
        results_per_window.append({
            'window_idx': window_idx,
            'MAPE': MAPE,
            'MSE': MSE,
            'MAE': MAE
        })
        
        site_results.append({
                'MAPE': MAPE,
                'MSE': MSE,
                'MAE': MAE } 
        )
    results_df = pd.DataFrame(site_results)
    agg_metrics = [{
        'MAPE_mean': results_df['MAPE'].mean(),
        'MAPE_std': results_df['MAPE'].std(),
        'MSE_mean': results_df['MSE'].mean(),
        'MSE_std': results_df['MSE'].std(),
        'MAE_mean': results_df['MAE'].mean(),
        'MAE_std': results_df['MAE'].std(),
        'target': target,
        'strategy': strat,
        'model': 'AutoGluon',
        'seed': random_seed,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }]
    
    all_pred['seed'] = random_seed
    all_pred['strategy'] = strat
    # if results_df['MAE'].mean() < 2.5:
    all_pred.to_csv(f'./BOSP_preds/{target}_all_pred_strat_{strat}_seed_{random_seed}_{datetime.now().strftime("%m_%d_%Y_%H_%M_%S")}.csv')
    
    return agg_metrics