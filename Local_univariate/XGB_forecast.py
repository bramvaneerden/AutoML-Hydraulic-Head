import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from load_function import load_timeseries, calculate_metrics
import time
import os
from xgboost import XGBRegressor

def evaluate_XGBoost(df, prediction_length, covariates=True):
    lags = 2
    print(f'running XGBoost model with lags={lags} and prediction length {prediction_length}')
    
    # Create lag features for the target
    for i in range(1, lags+1):
        df[f'value_lag_{i}'] = df['value'].shift(i)
    
    train = df.iloc[lags:-prediction_length].copy()
    test = df.iloc[-prediction_length:].copy()
    
    if covariates:
        reg_cols = ['precipitation', 'evaporation']
    else:
        reg_cols = []
        
    feature_cols = reg_cols + [f'value_lag_{i}' for i in range(1, lags+1)]
    
    X_train = train[feature_cols]
    y_train = train['value']
    
    start_time = time.time()
    model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"error occurred: {e}")
        print("continuing with next model")
        return 0, 0, 0, 0
    
    y_pred = []
    y_true = test['value'].tolist()
    
    for i in range(prediction_length):
        if i == 0:
            X_test = test.iloc[[i]][feature_cols]
        else:
            for j in range(lags, 0, -1):
                if j == 1:
                    test.iloc[i, test.columns.get_loc(f'value_lag_{j}')] = y_pred[i-1]
                else:
                    test.iloc[i, test.columns.get_loc(f'value_lag_{j}')] = test.iloc[i-1, test.columns.get_loc(f'value_lag_{j-1}')]
            X_test = test.iloc[[i]][feature_cols]
        
        prediction = model.predict(X_test)[0] 
        y_pred.append(prediction)
    
    total_time = time.time() - start_time
    MAPE, MSE, MAE = calculate_metrics(y_true, y_pred)
    
    return MAPE, MSE, MAE, total_time
