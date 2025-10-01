import matplotlib.pyplot as plt
import numpy as np
import torch

from darts.timeseries import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import (
    TFTModel,
    RNNModel,
    TiDEModel
)

# logging.disable(logging.CRITICAL)

from load_function import load_timeseries, find_base_path

import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

def preload_timeseries(files):
    ts = {}
    for fname in files:
        df = load_timeseries(fname).copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        site_id = df['site'].unique().tolist()[0]
        ts[site_id] = df

    max_start = max(df['date'].min() for df in ts.values())
    return {f: df.loc[df['date'] >= max_start] for f, df in ts.items()}

def load_model(model):
    
    # try:
    #     model.reset_model()
    # except Exception as e:
    #     print(f"error occurred: {e}")
        
    early_stopper = EarlyStopping(
    monitor="val_loss",
    patience=20,
    min_delta=0.001,
    mode="min",
    )
    add_encoders = {
    'datetime_attribute': {
        'past': ['day_of_week',  'day_of_year'],
        'future': ['day_of_week', 'day_of_year']},
    # 'transformer': Scaler(),
    }
    
    if model == 'DeepVAR':
        return RNNModel(
                input_chunk_length=10,
                output_chunk_length=3,
                model="LSTM",
                hidden_dim=40,
                n_rnn_layers=2,
                n_epochs=100,
                dropout=0.1,
                batch_size=64,
                pl_trainer_kwargs={"callbacks": [early_stopper]},
                # add_encoders=add_encoders # encoders also added in autogluon,
                # likelihood=LaplaceLikelihood()
                
            ) 
    elif model == 'TiDe':
        return TiDEModel(
                input_chunk_length=64,
                output_chunk_length=3,
                num_encoder_layers=2,
                num_decoder_layers=2,
                decoder_output_dim=16,
                hidden_size=64,
                temporal_decoder_hidden=64,
                dropout=0.2,
                use_layer_norm=True,
                batch_size=256,
                n_epochs=100,
                temporal_hidden_size_past=4,
                temporal_hidden_size_future=4,
                pl_trainer_kwargs={"callbacks": [early_stopper]},
            )
        
    else:
        return TFTModel(
                input_chunk_length=64,
                output_chunk_length=3,
                hidden_size=32,
                lstm_layers=1,
                num_attention_heads=4,
                dropout=0.1,
                n_epochs=100,
                pl_trainer_kwargs={"callbacks": [early_stopper]},
                optimizer_kwargs = {'lr':1e-3},
                batch_size=64,
                # add_encoders=add_encoders # encoders also added in autogluon
            )
        
def train_models(ts, covar,seed, target, window_size=3, n_windows=10,use_ensemble=True):
    
    test_size = window_size * n_windows
    val_size = max(90,test_size  * 2) # context length + prediction length (must be divisible by 3)
    total = test_size + val_size
    # gotta match this to autogluons transforms for each model
    train_set, val_set, test_set = ts[:-total],ts[-total:-test_size], ts[-test_size:]
    scaler = Scaler()
    train_scaled = scaler.fit_transform(train_set)
    val_scaled   = scaler.transform(val_set)
    
    train_covar, val_covar= covar[:-total],covar[-total:-test_size]
    
    covar_scaler    = Scaler()
    train_covar_s   = covar_scaler.fit_transform(train_covar)
    val_covar_s     = covar_scaler.transform(val_covar)

    ts_transformed = scaler.transform(ts)
    covar_transformed = covar_scaler.transform(covar)
    model_list = ['DeepVAR', 'TFT', 'TiDe']
    # model_list = ['DeepVAR']
    models = {}

    for model in model_list:
        print(f"training {model}:")
        model_init = load_model(model)
        model_init.fit(
            series=train_scaled,
            future_covariates=train_covar_s ,
            val_series = val_scaled,
            val_future_covariates = val_covar_s 
        )

        models[model] = model_init
    
    # ensemble
    # get model predictions on validation set
    all_preds = {}
    n_val_windows = len(val_set) // window_size
    start = len(ts) - val_size -1
    for model in models.keys():
        preds = pd.DataFrame()
        
        for i in range(n_val_windows):
            
            cutoff = ts_transformed.time_index[start + i*window_size]
            p = scaler.inverse_transform(models[model].predict(n=window_size,
                                series=ts_transformed[:cutoff],
                                future_covariates=covar_transformed)).to_dataframe()
            preds = pd.concat([preds, p])
        # break
        all_preds[model] = preds
    
    # ensemble 
    # errors per col
    val_gt =  val_set.to_dataframe()
    test_gt = test_set.to_dataframe()
    
    budget = 100
    best_error = np.inf
    ensemble = []
    remaining = set(all_preds.keys())

    while budget > 0:
        # best_temp_err = np.inf
        best_temp_model = None

        for option in remaining:
            current_members = ensemble + [option]
            
            #avg predictions across ensemble members
            stacked = [all_preds[m] for m in current_members]
            avg_preds = pd.concat(stacked).groupby(level=0).mean()
            
            # MAE per column, then take mean over columns
            mae = np.mean([mean_absolute_error(val_gt[col], avg_preds[col]) for col in val_gt.columns])

            if mae < best_error:
                best_error = mae
                best_temp_model = option
        
        if best_temp_model is None:
            break  # no improvement possible
        ensemble.append(best_temp_model)
        budget -= 1    
        
    start = len(ts) - test_size -1
    
    model_test_predictions = {}
    for model in model_list:
        test_preds = pd.DataFrame()
        for i in range(n_windows):
            cutoff = ts_transformed.time_index[start + i*window_size]
                    
            # forecast 3 steps ahead
            p = scaler.inverse_transform(models[model].predict(n=window_size,
                                series=ts_transformed[:cutoff],
                                future_covariates=covar_transformed)).to_dataframe()
            
            test_preds = pd.concat([test_preds, p])
        model_test_predictions[model] = test_preds
        
    if use_ensemble:
        # for model in ensemble:
        stacked = [model_test_predictions[m] for m in ensemble]
        model_test_predictions['ensemble'] = pd.concat(stacked).groupby(level=0).mean()
    
    [model_test_predictions[model].to_csv(f"./multivar/predictions_{target}_model_{model}_seed_{seed}.csv") for model in model_test_predictions.keys()]
    return model_test_predictions, test_gt

        
def main():
    os.makedirs('multivar', exist_ok=True)
    base_path = find_base_path()
    files = os.listdir(base_path + '\\2_Hydraulic head data/Sensor data')
    prediction__length = 3
    
    print("preloading timeseries..")
    preloaded = preload_timeseries(files=files)
    # site_names = list(set([f.split('_')[0] for f in files]))
    site_names = list(set([f[:-6] for f in preloaded.keys()])) 
    
    
    for target in site_names[1:]:
        concat_df = pd.DataFrame()
        target_timeseries = [f for f in list(preloaded.keys()) if target in f]
        for file in target_timeseries: # fix for all names
            concat_df = pd.concat([concat_df, preloaded[file]])
        # break
    

        value_wide = concat_df.pivot(index='date', columns='site', values='value')
        weather_df = concat_df[['date', 'precipitation', 'evaporation']].drop_duplicates(subset='date')
        wide_df = pd.merge(value_wide.reset_index(), weather_df, on='date').dropna()
        
        ts = TimeSeries.from_dataframe(wide_df, 
                                time_col="date",
                                value_cols=[col for col in wide_df.columns if col not in ["precipitation", "evaporation", "date"]],
                                fill_missing_dates=True ,freq='D')
        
        covar = TimeSeries.from_dataframe(wide_df, 
                                time_col="date",
                                value_cols=["precipitation", "evaporation"]	,
                                fill_missing_dates=True ,freq='D')
        
        errors = []
        print("Starting evaluations:")
        for seed in [123,124,125]: # ,124,125
            torch.manual_seed(seed)
            np.random.seed(seed)
            model_test_predictions, test_gt = train_models(ts=ts, covar=covar, seed=seed,target=target, window_size=3,n_windows=40,use_ensemble=True)
            
            for model in model_test_predictions.keys():
                mae = np.mean([mean_absolute_error(test_gt[col], model_test_predictions[model][col]) for col in test_gt.columns])
                errors.append({
                    "Model": model,
                    "MAE": mae 
                } )
                print(f"Model {model} has MAE: {mae}")
        print(f"\n average errors per model runs: \n")
        df = pd.DataFrame(errors)
        print(df.groupby('Model')['MAE'].mean())
        
        df.to_csv(f'./multivar/model_errors_{target}.csv')
        # break
if __name__ == "__main__":
    main()