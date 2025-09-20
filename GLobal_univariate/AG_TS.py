import pandas as pd
import numpy as np
import os
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import matplotlib.pyplot as plt
from load_function import load_timeseries, calculate_metrics, find_base_path
import time
from datetime import datetime
from autogluon.timeseries.splitter import ExpandingWindowSplitter
from sklearn.preprocessing import MinMaxScaler
import random
import torch

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
        
def evaluate_autogluon(df, prediction_length, random_seed, target, strat, num_val_windows=1, covariates=True, evaluate_global=False):
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
        ).fit(
            train_data, 
            # hyperparameters=models, 
            random_seed=random_seed,
            presets=preset,
            excluded_model_types=["AutoETS", "AutoARIMA", "DynamicOptimizedTheta", "SeasonalNaive", "AutoETS", "SeasonalNaive", "RecursiveTabular"],  
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
    # leaderboard.to_csv(f'./BO_model_info/leaderboard_{target}_{strat}_{datetime.now().strftime("%m_%d_%Y_%H_%M_%S")}.csv')
    
    splitter = ExpandingWindowSplitter(prediction_length=prediction_length, num_val_windows=num_val_windows)

    site_results = []
    all_pred = pd.DataFrame()
    results_per_window = []
    
    # average over results, dump y_pred after concat
    for window_idx, (train_split, val_split) in enumerate(splitter.split(test_data)):    
        y_pred = predictor.predict(train_split, known_covariates=val_split[[col for col in val_split.columns if col not in ['site', 'date', 'value']]])
        all_pred = pd.concat([all_pred, y_pred])
        y_true = val_split.groupby(level='item_id').tail(prediction_length)
        
        if evaluate_global:
            for site in sites:
                mask = y_pred.index.get_level_values('item_id') == site
                site_predictions = y_pred.loc[mask]
                site_true_unscaled = y_true.loc[mask]
                
                MAPE, MSE, MAE = calculate_metrics(site_true_unscaled['value'], site_predictions['mean'])
                site_results.append({
                        'Site': site,
                        'MAPE': MAPE,
                        'MSE': MSE,
                        'MAE': MAE,
                        'Seed': random_seed} 
                )
                
        else:
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
    if evaluate_global:
        PRED_DIR = 'global_pred'
        METRIC_DIR = 'global_metric'
        for dir in [PRED_DIR,  METRIC_DIR]:
            if not os.path.exists(dir):
                os.makedirs(dir)
        results_df = pd.DataFrame(site_results)
        all_pred['seed'] = random_seed
        all_pred['strategy'] = 9
        all_pred.to_csv(f'./{PRED_DIR}/{target}_all_pred_strat_{strat}_seed_{random_seed}_{datetime.now().strftime("%m_%d_%Y_%H_%M_%S")}.csv')
    
        # results_df.to_csv(f'./{METRIC_DIR}/global_results_df_seed_{random_seed}_{datetime.now().strftime("%m_%d_%Y_%H_%M_%S")}.csv')
        return site_results
    else: 
        PRED_DIR = 'AGTS_pred'
        METRIC_DIR = 'AGTS_metric'
        for dir in [PRED_DIR,  METRIC_DIR]:
            if not os.path.exists(dir):
                os.makedirs(dir)
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
        all_pred.to_csv(f'./{PRED_DIR}/{target}_all_pred_strat_{strat}_seed_{random_seed}_{datetime.now().strftime("%m_%d_%Y_%H_%M_%S")}.csv')
        # results_df.to_csv(f'./{METRIC_DIR}/results_df_seed_{random_seed}_{datetime.now().strftime("%m_%d_%Y_%H_%M_%S")}.csv')

    
        return agg_metrics