from autots import AutoTS, load_daily
import os
import pandas as pd
import numpy as np
from load_function import load_timeseries, calculate_metrics, find_base_path
import time
import pickle

def evaluate_autots(df, prediction_length,num_windows,seed, covariates=True ):

        
    # site = df['site'].iloc[0]
                
    df.index = pd.to_datetime(df['date'])
    df = df.drop(columns=['date'])
    # df['date'] = pd.to_datetime(df['date'])
    train = df[:-(prediction_length * num_windows)]
    y_true = df[-(prediction_length * num_windows):]
    
    # target.index = pd.to_datetime(target['date'])
    # train = target[:-prediction_length]
    # y_true = target[-prediction_length:]
    cols = [col for col in df.columns if col not in ['site', 'precipitation', 'evaporation']]
    train_target = train[cols]
    if covariates:
        train_regressors = train[['precipitation', 'evaporation']]
        test_regressors = y_true[['precipitation', 'evaporation']]
    else:
        train_regressors = None
        test_regressors = None
    
    long = False

    print(train_target.head())
    print(f'running AutoTS with prediction length {prediction_length}')
    start = time.time()
    model = AutoTS(
        forecast_length=prediction_length,
        frequency="infer",
        prediction_interval=0.9,
        ensemble=None,
        model_list="fast",
        transformer_list="fast",
        drop_most_recent=1,
        max_generations=5,
        num_validations=0,
        validation_method="backwards",
        verbose=0
    )
    model = model.fit(
                train_target,
                future_regressor=train_regressors
            )
    try:
        if covariates:
            model = model.fit(
                train_target,
                date_col='date' if long else None,
                value_col='value' if long else None,
                # id_col='site' if long else None,
                future_regressor=train_regressors
            )
        else:
            model = model.fit(train_target)
    except Exception as e:
        print(f"error occurred: {e}")
        print("continuing with next model")
        
    total_time = time.time() - start
    try:
        if covariates:
            output = model.predict(future_regressor=test_regressors)
        else:
            output = model.predict()
    except Exception as e:
        print(f"error occurred: {e}")
        print("continuing with next model")
        site_results = []
        site_results.append({
                    'Model': 'AutoTS',
                    'MAPE': 0,
                    'MSE': 0,
                    'MAE': 0,
                    'Time': 0,
                    'Site': 0  } 
            )
        
        return site_results 
    results = []
    for timeseries in list(output.forecast.columns):
        MAPE, MSE, MAE = calculate_metrics(y_true[timeseries], output.forecast[timeseries])
        results.append({
                'Model': 'AutoTS',
                'MAPE': MAPE,
                'MSE': MSE,
                'MAE': MAE,
                'Time': total_time,
                'Site': timeseries              
        })
    # with open('Autots_aarland_5_uni.pkl', 'wb') as f:
    #     pickle.dump(output, f)
    return results

def main():
    output_dir = './thesis_output_autots/'
    os.makedirs(output_dir, exist_ok=True)

    base_path = find_base_path()
    data_dir = os.path.join(base_path, '2_Hydraulic head data', 'Sensor data')
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
                file_results, predictions = evaluate_autots(
                    df=load_timeseries(file),
                    prediction_length=3,
                    num_windows = 10,
                    mode='dl',
                    seed=seed
                )

                all_results.extend(file_results)
                if predictions:
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