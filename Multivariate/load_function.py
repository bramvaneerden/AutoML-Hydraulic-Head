import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np
import pandas as pd
import random
from pathlib import Path

def find_base_path():
    return str(Path(__file__).parent.parent)

def load_clean_df(hh_df):
    hh_df = hh_df[['Date and time (UTC+00:00)', 'Monitoring point name','Water level referenced to Vertical reference datum (cm)']]
    hh_df = hh_df.rename(columns={'Date and time (UTC+00:00)': 'date', 'Monitoring point name': 'site', 'Water level referenced to Vertical reference datum (cm)': 'value'}) # , inplace=True

    hh_df['date'] = pd.to_datetime(hh_df['date'])
    hh_df = hh_df.set_index('date') # , inplace=True

    hh_df = hh_df.groupby('site')['value'].resample('D').mean()
    
    hh_df = hh_df.reset_index()
    hh_df['date'] = pd.to_datetime(hh_df['date'])
    
    if hh_df['value'].isnull().sum() > 0: 
        hh_df['value'] = hh_df['value'].interpolate(method='linear') # , inplace=True
    
    return hh_df

def load_timeseries(file):
    base_path = find_base_path()

    knmi_df = pd.read_csv(base_path+ '\\3_Local rainfall and potential evaporation\\KNMI_data.csv')
    knmi_df['date'] = pd.to_datetime(knmi_df['date'])
    knmi_df.set_index('date', inplace=True)
    
    df = pd.read_csv(base_path + "\\2_Hydraulic head data\\Sensor data\\" + file)
    
    df = load_clean_df(df)
    df =  pd.merge(df, knmi_df, on=['date'], how='inner')

    return df

def calculate_metrics(y_true, y_pred):
    """calculate the three metrics were using

    Args:
        y_true (list or array): ground truth values
        y_pred (list or array): predicted values

    Returns:
        tuple(MAPE, MSE, MAE): MAPE, MSE, MAE
    """
    MAPE = mean_absolute_percentage_error(y_true, y_pred)
    MSE = mean_squared_error(y_true, y_pred)
    MAE = mean_absolute_error(y_true, y_pred)

    return MAPE, MSE, MAE
