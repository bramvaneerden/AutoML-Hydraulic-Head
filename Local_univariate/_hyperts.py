from hyperts import make_experiment
from hyperts.datasets import load_network_traffic

from sklearn.model_selection import train_test_split

from load_function import load_timeseries, calculate_metrics, find_base_path
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from load_function import load_timeseries, calculate_metrics
import time

def evaluate_hyperts(df, prediction_length,num_windows, mode, seed=123):
            
    site = df['site'].unique()[0]
    data = df[['date' , 'value','precipitation', 'evaporation']] 
    train_data = data[:-(prediction_length * num_windows)]
    test_data = data[-(prediction_length * num_windows):]
    print(f'running Hyperts model with preset={mode} and prediction length {prediction_length}')
    # start_time = time.time()
    try:
        model = make_experiment(train_data.copy(),
                                task='univariate-forecast',
                                target='value',  
                                mode='dl',
                                eval_size=0.2,
                                tf_gpu_usage_strategy=1,
                                timestamp='date',
                                covariates=['precipitation', 'evaporation'],
                                early_stopping_time_limit = 300,
                                dl_forecast_window=24*prediction_length,
                                reward_metric='mae',
                                verbose=1,
                                random_state=seed,
                                ).run() #
    except Exception as e:
        print(f"error occurred: {e}")
        print("continuing with next model")

    
    metrics = []
    X_test, y_test = model.split_X_y(test_data.copy())
    y_pred = model.predict(X_test)
    model.plot(forecast=y_pred, actual=test_data, interactive=True)
        
    MAPE, MSE, MAE = calculate_metrics(test_data['value'], y_pred['value'])
    metrics.append({'site':site,
                    'MAPE':MAPE,
                    'MSE':MSE,
                    'MAE':MAE,
                    'seed':seed
                    })
    
    # total_time = time.time() - start_time
    
    return metrics, y_pred


def main():
    output_dir = './output_hyperts/'
    os.makedirs(output_dir, exist_ok=True)

    base_path = find_base_path()
    data_dir = base_path + '\\2_Hydraulic head data + Sensor data'
    files = [f for f in os.listdir(data_dir) if not f.startswith('.')]

    seeds = (123, 124, 125)
    num_windows = 10
    prediction_length = 3
    use_covariates = True

    all_results = []
    for seed in seeds:
        for i, file in enumerate(files, 1):
            print(f"[{i}/{len(files)}] {file}")
            try:
                file_results, predictions = evaluate_hyperts(
                    df=load_timeseries(file),
                    prediction_length=3,
                    num_windows = 10,
                    mode='dl',
                    seed=seed
                )

                all_results.extend(file_results)
                if predictions is not None:
                    predictions.to_csv(
                        f'{output_dir}/predictions_{os.path.basename(file)}_seed_{seed}.csv', index=False
                    )
            except Exception as e:
                print(f"Error {file}: {e}")

    if not all_results:
        print("No results generated.")
        return None, None

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f'{output_dir}/all_results.csv', index=False)


if __name__ == "__main__":
    main()
