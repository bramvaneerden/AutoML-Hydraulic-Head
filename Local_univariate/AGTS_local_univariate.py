import numpy as np
import pandas as pd
from load_function import load_timeseries, calculate_metrics, find_base_path
import time
import os
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from autogluon.timeseries.splitter import ExpandingWindowSplitter
import random
import torch
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime

def set_all_seeds(seed: int):
    """Set all seeds for reproducibility"""
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def to_timeseriesdataframe(df: pd.DataFrame) -> TimeSeriesDataFrame:
    """Convert regular dataframe to AutoGluon TimeSeriesDataFrame format"""
    df = df.copy()
    ts_df = TimeSeriesDataFrame.from_data_frame(
        df=df,
        id_column='site',
        timestamp_column='date'
    )
    return ts_df


# AutoGluon backtesting
def evaluate_autogluon_backtest(
    df: pd.DataFrame,
    target: str,
    strat: int,
    random_seed: int,
    prediction_length: int = 3,
    num_val_windows: int = 10,
    covariates: bool = True,
    evaluate_global: bool = False,
    preset: str = 'medium_quality',
):
    """
    Runs AutoGluon once, then performs multi-window backtesting with ExpandingWindowSplitter.
    Returns aggregated metrics across windows for the target site.
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    sites = list(df['site'].unique())
    print(
        f"Running AutoGluon with preset={preset}, "
        f"prediction_length={prediction_length}, num_val_windows={num_val_windows}, seed={random_seed}"
    )

    df = df.copy(deep=True)
    if not covariates:
        df = df.drop(columns=['precipitation', 'evaporation'], errors='ignore')

    ts_df = to_timeseriesdataframe(df)

    # Hold out final (prediction_length * num_val_windows) timestamps for backtesting windows
    train_data, test_data = ts_df.train_test_split(
        prediction_length=prediction_length * num_val_windows
    )

    if covariates:
        known_cov_names = [c for c in ts_df.columns if c not in ['site', 'date', 'value']]
    else:
        known_cov_names = []

    # Fit predictor once on the fixed train_data
    set_all_seeds(random_seed)
    start_time = time.time()
    predictor = TimeSeriesPredictor(
        prediction_length=prediction_length,
        freq='D',
        cache_predictions=False,
        eval_metric="MAE",
        target="value",
        known_covariates_names=known_cov_names,
    ).fit(
        train_data,
        random_seed=random_seed,
        presets=preset, 
        excluded_model_types=[
            "AutoETS", "AutoARIMA", "DynamicOptimizedTheta",
            "SeasonalNaive", "RecursiveTabular", "ETS", "Theta", "NPTS"
        ],
    )
    fit_time = time.time() - start_time

    try:
        lb = predictor.leaderboard(silent=True)
        os.makedirs('./AG_model_info', exist_ok=True)
        lb.to_csv(f'./AG_model_info/leaderboard_{target}_{strat}_{datetime.now().strftime("%m_%d_%Y_%H_%M_%S")}.csv', index=False)
    except Exception as e:
        print(f"Could not save leaderboard (non-fatal): {e}")

    # Backtesting across windows
    splitter = ExpandingWindowSplitter(prediction_length=prediction_length, num_val_windows=num_val_windows)

    site_results = []
    all_pred = pd.DataFrame()

    for window_idx, (train_split, val_split) in enumerate(splitter.split(test_data)):
        # Predict next horizon, given history up to current window
        if covariates:
            known_cov_val = val_split[[col for col in val_split.columns if col not in ['site', 'date', 'value']]]
        else:
            known_cov_val = None

        y_pred = predictor.predict(train_split, known_covariates=known_cov_val)
        all_pred = pd.concat([all_pred, y_pred])

        # Evaluate either per-site or for the specified target site
        y_true = val_split.groupby(level='item_id').tail(prediction_length)

        if evaluate_global:
            for site in sites:
                mask = y_pred.index.get_level_values('item_id') == site
                site_pred = y_pred.loc[mask]
                site_true = y_true.loc[mask]
                MAPE, MSE, MAE = calculate_metrics(site_true['value'], site_pred['mean'])
                site_results.append({
                    'Site': site,
                    'MAPE': MAPE,
                    'MSE': MSE,
                    'MAE': MAE,
                    'Seed': random_seed,
                    'Window': window_idx
                })
        else:
            mask = y_pred.index.get_level_values('item_id') == target
            site_pred = y_pred.loc[mask]
            site_true = y_true.loc[mask]
            MAPE, MSE, MAE = calculate_metrics(site_true['value'], site_pred['mean'])
            site_results.append({
                'MAPE': MAPE,
                'MSE': MSE,
                'MAE': MAE,
                'Seed': random_seed,
                'Window': window_idx
            })

    # Save predictions
    try:
        os.makedirs('./AG_pred', exist_ok=True)
        all_pred['seed'] = random_seed
        all_pred['strategy'] = strat
        all_pred.to_csv(
            f'./AG_pred/{target}_all_pred_strat_{strat}_seed_{random_seed}_{datetime.now().strftime("%m_%d_%Y_%H_%M_%S")}.csv'
        )
    except Exception as e:
        print(f"Could not save predictions (non-fatal): {e}")

    # Aggregate across windows
    results_df = pd.DataFrame(site_results)
    if evaluate_global:
        # If global, return per-site rows
        return results_df, fit_time
    else:
        agg_metrics = {
            'MAPE_mean': results_df['MAPE'].mean(),
            'MAPE_std': results_df['MAPE'].std(),
            'MSE_mean': results_df['MSE'].mean(),
            'MSE_std': results_df['MSE'].std(),
            'MAE_mean': results_df['MAE'].mean(),
            'MAE_std': results_df['MAE'].std(),
            'target': target,
            'model': 'AutoGluon',
            'seed': random_seed,
            'n_windows': num_val_windows,
            'prediction_length': prediction_length,
            'fit_time_sec': round(fit_time, 2),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        return pd.DataFrame([agg_metrics]), fit_time


def run_autogluon_for_file(
    file: str,
    seeds = (123,124,125),  
    prediction_length: int = 3,
    num_val_windows: int = 10,
    covariates: bool = True,
    preset: str = 'medium_quality'
):
    """Run the AutoGluon backtest for a single file."""
    all_results = []
    try:
        df = load_timeseries(file)
        sites = list(df['site'].unique())
        if not sites:
            print(f"No sites found in {file}, skipping.")
            return []

        target = sites[0]
        strat = 0

        for seed in seeds:
            print(f"\nRunning AutoGluon for {file} | target={target} | seed={seed}")
            set_all_seeds(seed)
            metrics_df, fit_time = evaluate_autogluon_backtest(
                df=df,
                target=target,
                strat=strat,
                random_seed=seed,
                prediction_length=prediction_length,
                num_val_windows=num_val_windows,
                covariates=covariates,
                evaluate_global=False,
                preset=preset,
            )
            metrics_df['file'] = file
            all_results.append(metrics_df)

    except Exception as e:
        print(f"Error loading/processing {file}: {e}")

    if len(all_results):
        return pd.concat(all_results, ignore_index=True)
    return []

def main():
    """Main: ONLY runs AutoGluon with separate multi-window backtesting."""
    print("Starting AutoGluon-only multi-window backtesting...")
    output_dir = './thesis_output_AG/'
    os.makedirs(output_dir, exist_ok=True)

    base_path = find_base_path()
    files = os.listdir(os.path.join(base_path, '2_Hydraulic head data/Sensor data'))

    seeds = [123,124,125] 
    n_windows = 10 
    prediction_length = 3    
    preset = 'medium_quality'

    print(f"Found {len(files)} files")
    print(f"Seeds: {seeds} | Windows: {n_windows} | Horizon: {prediction_length} | Preset: {preset}")

    all_results = []

    for i, file in enumerate(files):
        print(f"\n{'='*60}\nProcessing {i+1}/{len(files)}: {file}\n{'='*60}")
        res_df = run_autogluon_for_file(
            file=file,
            seeds=seeds,
            prediction_length=prediction_length,
            num_val_windows=n_windows,
            covariates=True,
            preset=preset
        )
        if isinstance(res_df, pd.DataFrame) and not res_df.empty:
            res_df.to_csv(f'{output_dir}/results_{file}_autogluon.csv', index=False)
            print(f"Saved AG results for {file}")
            all_results.append(res_df)

    if all_results:
        all_results_df = pd.concat(all_results, ignore_index=True)
        all_results_df.to_csv(f'{output_dir}/all_results_autogluon.csv', index=False)
        print(f"\nSaved combined AG results -> {output_dir}/all_results_autogluon.csv")

        # Small summary
        summary = (
            all_results_df
            .groupby('file')[['MAE_mean', 'MAPE_mean', 'MSE_mean']]
            .mean()
            .round(4)
        )
        summary.to_csv(f'{output_dir}/summary_autogluon.csv')
        print("Saved summary -> summary_autogluon.csv")
        print(summary)
        return all_results_df, summary

    print("No results generated.")
    return None, None

if __name__ == "__main__":
    results_df, summary_table = main()
