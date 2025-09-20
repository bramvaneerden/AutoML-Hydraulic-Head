import os
import time
import random
import warnings
import numpy as np
import pandas as pd
from contextlib import redirect_stdout, redirect_stderr

from sklearn.linear_model import LinearRegression
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from load_function import load_timeseries, calculate_metrics, find_base_path

warnings.filterwarnings('ignore')

def set_all_seeds(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def naive_forecast(df, prediction_length, covariates=True, diff=False):
    if diff:
        df = df.copy()
        df['value'] = df['value'].diff().dropna()
    train = df[:-prediction_length]
    y_true = df[-prediction_length:]
    start = time.time()
    last_value = train['value'].iloc[-1]
    y_pred = np.full(prediction_length, last_value)
    elapsed = time.time() - start
    MAPE, MSE, MAE = calculate_metrics(y_true['value'], y_pred)
    return MAPE, MSE, MAE, elapsed

def seasonal_naive_forecast(df, prediction_length, covariates=True, diff=False, season_length=7):
    if diff:
        df = df.copy()
        df['value'] = df['value'].diff().dropna()
    train = df[:-prediction_length]
    y_true = df[-prediction_length:]
    start = time.time()
    if len(train) >= season_length:
        y_pred = np.array([train['value'].iloc[len(train)-season_length + (i % season_length)]
                           for i in range(prediction_length)])
    else:
        y_pred = np.full(prediction_length, train['value'].iloc[-1])
    elapsed = time.time() - start
    MAPE, MSE, MAE = calculate_metrics(y_true['value'], y_pred)
    return MAPE, MSE, MAE, elapsed

def linear_trend_forecast(df, prediction_length, covariates=True, diff=False):
    if diff:
        df = df.copy()
        df['value'] = df['value'].diff().dropna()
    train = df[:-prediction_length]
    y_true = df[-prediction_length:]
    start = time.time()
    X = np.arange(len(train)).reshape(-1, 1)
    y = train['value'].values
    model = LinearRegression().fit(X, y)
    X_pred = np.arange(len(train), len(train) + prediction_length).reshape(-1, 1)
    y_pred = model.predict(X_pred)
    elapsed = time.time() - start
    MAPE, MSE, MAE = calculate_metrics(y_true['value'], y_pred)
    return MAPE, MSE, MAE, elapsed

def exponential_smoothing_forecast(df, prediction_length, covariates=True, diff=False):
    if diff:
        df = df.copy()
        df['value'] = df['value'].diff().dropna()
    train = df[:-prediction_length]
    y_true = df[-prediction_length:]
    start = time.time()
    try:
        model = ExponentialSmoothing(train['value'], trend='add', seasonal=None,
                                     initialization_method='estimated')
        fit = model.fit()
        y_pred = fit.forecast(prediction_length)
    except Exception:
        last_value = train['value'].iloc[-1]
        y_pred = np.full(prediction_length, last_value)
    elapsed = time.time() - start
    MAPE, MSE, MAE = calculate_metrics(y_true['value'], y_pred)
    return MAPE, MSE, MAE, elapsed

def evaluate_SARIMA(df, prediction_length, covariates=True, diff=False):
    if diff:
        df = df.copy()
        df['value'] = df['value'].diff().dropna()
    train = df[:-prediction_length]
    y_true = df[-prediction_length:]
    start = time.time()
    exog_train = train[['precipitation', 'evaporation']] if covariates else None
    exog_test  = y_true[['precipitation', 'evaporation']] if covariates else None
    model = SARIMAX(train['value'], exog=exog_train, order=(2, 0, 0))
    try:
        with open(os.devnull, 'w') as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
            fit = model.fit(disp=0)
        y_pred = fit.predict(start=len(train), end=len(train) + len(y_true) - 1, exog=exog_test)
    except Exception:
        return np.nan, np.nan, np.nan, time.time() - start
    elapsed = time.time() - start
    MAPE, MSE, MAE = calculate_metrics(y_true['value'], y_pred)
    return MAPE, MSE, MAE, elapsed

def evaluate_ARIMA(df, prediction_length, covariates=True, diff=False):
    if diff:
        df = df.copy()
        df['value'] = df['value'].diff().dropna()
    train = df[:-prediction_length]
    y_true = df[-prediction_length:]
    start = time.time()
    exog_train = train[['precipitation', 'evaporation']] if covariates else None
    exog_test  = y_true[['precipitation', 'evaporation']] if covariates else None
    model = ARIMA(train['value'], exog=exog_train, order=(2, 0, 0))
    try:
        with open(os.devnull, 'w') as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
            fit = model.fit()
        y_pred = fit.predict(start=len(train), end=len(train) + len(y_true) - 1, exog=exog_test)
    except Exception:
        return np.nan, np.nan, np.nan, time.time() - start
    elapsed = time.time() - start
    MAPE, MSE, MAE = calculate_metrics(y_true['value'], y_pred)
    return MAPE, MSE, MAE, elapsed

def simple_autoregressive(df, prediction_length, covariates=True, diff=False, lags=2):
    if diff:
        df = df.copy()
        df['value'] = df['value'].diff().dropna()
    train = df[:-prediction_length]
    y_true = df[-prediction_length:]
    start = time.time()
    exog_train = train[['precipitation', 'evaporation']] if covariates else None
    exog_oos   = y_true[['precipitation', 'evaporation']] if covariates else None
    model = AutoReg(train['value'], lags=lags, exog=exog_train)
    try:
        with open(os.devnull, 'w') as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
            fit = model.fit()
        y_pred = fit.predict(start=len(train), end=len(train) + len(y_true) - 1, exog_oos=exog_oos)
    except Exception:
        return np.nan, np.nan, np.nan, time.time() - start
    elapsed = time.time() - start
    MAPE, MSE, MAE = calculate_metrics(y_true['value'], y_pred)
    return MAPE, MSE, MAE, elapsed

def multi_window_backtest(df, model_func, n_windows=10, prediction_length=3, covariates=True):
    results = []
    min_train = 30
    total_len = len(df)
    if total_len < min_train + n_windows * prediction_length:
        n_windows = max(1, (total_len - min_train) // prediction_length)

    test_end = total_len
    for i in range(n_windows):
        test_start = test_end - prediction_length
        train_end = test_start
        if train_end < min_train:
            break
        window_df = df.iloc[:test_end].copy()
        try:
            mape, mse, mae, t = model_func(window_df, prediction_length, covariates=covariates, diff=False)
        except Exception as e:
            mape = mse = mae = t = np.nan
        results.append({
            'window': i + 1,
            'MAPE': mape, 'MSE': mse, 'MAE': mae, 'Time': t,
            'train_size': train_end, 'test_start': test_start, 'test_end': test_end
        })
        test_end = test_start
    return results

def run_experiments_for_file(file, seeds=(123, 124, 125), n_windows=10, prediction_length=3, use_covariates=True,verbose=True):
    models = {
        'Naive': naive_forecast,
        'Seasonal_Naive': seasonal_naive_forecast,
        'Linear_Trend': linear_trend_forecast,
        'Exponential_Smoothing': exponential_smoothing_forecast,
        'AutoRegressive': simple_autoregressive,
        'ARIMA': evaluate_ARIMA,
        'SARIMA': evaluate_SARIMA,
    }

    all_results = []
    for seed in seeds:
        set_all_seeds(seed)
        try:
            df = load_timeseries(file)
        except Exception as e:
            print(f"Load error {file}: {e}")
            continue

        for model_name, model_func in models.items():
            if verbose:
                print(f"Running {model_name} for {file} for seed {seed}")
            window_results = multi_window_backtest(
                df, model_func, n_windows=n_windows,
                prediction_length=prediction_length, covariates=use_covariates
            )
            for r in window_results:
                r.update({'model': model_name, 'file': file, 'seed': seed})
            all_results.extend(window_results)
    return all_results

def create_summary_table(results_df: pd.DataFrame) -> pd.DataFrame:
    summary = results_df.groupby('model').agg(
        MAE_mean=('MAE', 'mean'), MAE_std=('MAE', 'std'),
        MAPE_mean=('MAPE', 'mean'), MAPE_std=('MAPE', 'std'),
        MSE_mean=('MSE', 'mean'), MSE_std=('MSE', 'std'),
        Time_mean=('Time', 'mean'), Time_std=('Time', 'std'),
        N_Windows=('MAE', 'count')
    ).round(4)
    out = pd.DataFrame(index=summary.index)
    out['MAE (mean±std)']  = summary['MAE_mean'].astype(str)  + ' ± ' + summary['MAE_std'].astype(str)
    out['MAPE (mean±std)'] = summary['MAPE_mean'].astype(str) + ' ± ' + summary['MAPE_std'].astype(str)
    out['MSE (mean±std)']  = summary['MSE_mean'].astype(str)  + ' ± ' + summary['MSE_std'].astype(str)
    out['Time (mean±std)'] = summary['Time_mean'].astype(str) + ' ± ' + summary['Time_std'].astype(str)
    out['N_Windows'] = summary['N_Windows'].astype(int)
    return out

def main():
    output_dir = './output_stat_models/'
    os.makedirs(output_dir, exist_ok=True)

    base_path = find_base_path()
    data_dir = (base_path + '\\2_Hydraulic head data\\Sensor data')
    files = [f for f in os.listdir(data_dir) if not f.startswith('.')]

    seeds = (123, 124, 125)
    n_windows = 10
    prediction_length = 3
    use_covariates = True

    all_results = []
    for i, file in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {file}")
        try:
            file_results = run_experiments_for_file(
                file=file,
                seeds=seeds,
                n_windows=n_windows,
                prediction_length=prediction_length,
                use_covariates=use_covariates,
                verbose=True
            )
            all_results.extend(file_results)
            if file_results:
                pd.DataFrame(file_results).to_csv(
                    f'{output_dir}/results_{os.path.basename(file)}.csv', index=False
                )
        except Exception as e:
            print(f"Error {file}: {e}")

    if not all_results:
        print("No results generated.")
        return None, None

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f'{output_dir}/all_results.csv', index=False)

    summary = create_summary_table(results_df)
    summary.to_csv(f'{output_dir}/summary.csv')

    print("\nSummary:")
    print(summary)
    return results_df, summary

if __name__ == "__main__":
    results_df, summary_table = main()
