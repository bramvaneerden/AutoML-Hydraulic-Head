
import numpy as np
import os
import random
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import matplotlib.pyplot as plt
from load_function import load_timeseries, find_base_path

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
import re
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputClassifier

def preload_timeseries(files):
    ts = {}
    print("Starting preload.")
    for fname in files:
        df = load_timeseries(fname).copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        site_id = df['site'].unique().tolist()[0]
        ts[site_id] = df

    max_start = max(df['date'].min() for df in ts.values())
    print("Preload finished. \n")
    return {f: df.loc[df['date'] >= max_start] for f, df in ts.items()}

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

def create_peak_targets(values, lookback=3, min_increase=0.8, peak_window=3, min_peak_distance=2):
    """
    Improved peak detection with better local maxima identification and peak suppression.
    
    Args:
        values: Time series values
        lookback: Period to calculate rate of change
        min_increase: Minimum rate of change to consider
        peak_window: Window size to look for local maxima
        min_peak_distance: Minimum distance between consecutive peaks
    """
    targets = []
    values = np.array(values)
    last_peak_idx = -float('inf')
    
    for i in range(len(values)):
        if i < lookback:
            targets.append(0.0)
            continue
            
        # Rate of change over lookback period
        rate_change = (values[i] - values[i-lookback]) / lookback
        
        # Check for local maximum
        start_idx = max(0, i - peak_window//2)
        end_idx = min(len(values), i + peak_window//2 + 1)
        local_window = values[start_idx:end_idx]
        local_max_idx = np.argmax(local_window) + start_idx
        is_local_max = (local_max_idx == i)
        
        #current value should be higher than recent past
        # sometimes previous function dont catch exceptions where the local max in the window is not higher than the past values
        recent_past = values[max(0, i-2):i]
        is_higher_than_recent = len(recent_past) == 0 or values[i] > np.mean(recent_past)
        
        #  minimum distance from last peak
        sufficient_distance = (i - last_peak_idx) >= min_peak_distance
        
        # Combined conditions
        is_peak = (rate_change > min_increase and 
                  is_local_max and 
                  is_higher_than_recent and 
                  sufficient_distance)
        
        if is_peak:
            targets.append(1.0)
            last_peak_idx = i
        else:
            targets.append(0.0)
    
    return targets

class PeakPredictor:
    def __init__(self, prediction_length=3):
        self.prediction_length = prediction_length
        self.model = None 
        self.site_encoder = LabelEncoder()
        self.feature_names = None
        
    def create_features(self, df):
        """Create features for peak prediction"""
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        
        # Detect column names
        if 'site' in df.columns:
            col = 'site'
            tcol = 'date'
        else:
            col = 'item_id'
            tcol = 'timestamp'
            df = df.reset_index()
        
        df = df.copy().sort_values([col, tcol])
        
        features_list = []
        
        for site in df[col].unique():
            site_df = df[df[col] == site].copy()
            site_df = site_df.sort_values(tcol).reset_index(drop=True)
            
            # lag features for value
            for lag in [1, 2, 3, 5, 7]:
                site_df[f'value_lag_{lag}'] = site_df['value'].shift(lag)
            
            # Rolling statistics
            for window in [3, 7, 14]:
                site_df[f'value_rolling_mean_{window}'] = site_df['value'].rolling(window).mean()
                site_df[f'value_rolling_std_{window}'] = site_df['value'].rolling(window).std()
                site_df[f'value_rolling_min_{window}'] = site_df['value'].rolling(window).min()
                site_df[f'value_rolling_max_{window}'] = site_df['value'].rolling(window).max()
            
            # Rate of change features
            site_df['value_diff_1'] = site_df['value'].diff(1)
            site_df['value_diff_3'] = site_df['value'].diff(3)
            site_df['value_pct_change_1'] = site_df['value'].pct_change(1)
            site_df['value_pct_change_3'] = site_df['value'].pct_change(3)
            
            # Weather features 
            if 'precipitation' in df.columns and 'evaporation' in df.columns:
                for lag in [1, 2, 3]:
                    site_df[f'precipitation_lag_{lag}'] = site_df['precipitation'].shift(lag)
                    site_df[f'evaporation_lag_{lag}'] = site_df['evaporation'].shift(lag)
                
                for window in [3, 7]:
                    site_df[f'precip_rolling_sum_{window}'] = site_df['precipitation'].rolling(window).sum()
                    site_df[f'evap_rolling_mean_{window}'] = site_df['evaporation'].rolling(window).mean()
                
                # Precipitation/evaporation ratio
                site_df['precip_evap_ratio'] = site_df['precipitation'] / (site_df['evaporation'] + 0.01)
            
            # Time features
            site_df[tcol] = pd.to_datetime(site_df[tcol])
            site_df['day_of_year'] = site_df[tcol].dt.dayofyear
            site_df['month'] = site_df[tcol].dt.month
            site_df['day_of_week'] = site_df[tcol].dt.dayofweek
            
            features_list.append(site_df)
        
        return pd.concat(features_list, ignore_index=True)
        
    def create_targets(self, df):
        """Create target variables for next 3 timesteps"""
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        
        # Detect column names
        if 'site' in df.columns:
            col = 'site'
            tcol = 'date'
        else:
            col = 'item_id'
            tcol = 'timestamp'
            df = df.reset_index()
        
        df = df.copy().sort_values([col, tcol])
        
        targets_list = []
        
        for site in df[col].unique():
            site_df = df[df[col] == site].copy()
            site_df = site_df.sort_values(tcol).reset_index(drop=True)
            
            if 'peaks' not in site_df.columns:
                print(f"Warning: 'peaks' column not found for site {site}. Creating a default column of zeros.")
                site_df['peaks'] = 0
            
            # Create targets for next 3 timesteps
            for step in range(1, self.prediction_length + 1):
                site_df[f'peak_target_{step}'] = site_df['peaks'].shift(-step)
            
            targets_list.append(site_df)
        
        return pd.concat(targets_list, ignore_index=True)
    
    def prepare_training_data(self, df):
        """Prepare features and targets for training"""
        df_features = self.create_features(df)
        df_with_targets = self.create_targets(df_features)
        
        # Detect column names
        if 'site' in df_with_targets.columns:
            col = 'site'
            tcol = 'date'
        else:
            col = 'item_id'
            tcol = 'timestamp'
            df_with_targets = df_with_targets.reset_index()        
        
        # Encode site names
        df_with_targets['site_encoded'] = self.site_encoder.fit_transform(df_with_targets[col])
        
        # Select feature columns
        # IMPORTANT: Exclude peaks to prevent data leakage
        exclude_cols = [col, tcol, 'peaks', 'index', 'level_0'] + [f'peak_target_{i}' for i in range(1, self.prediction_length + 1)]
        feature_cols = [column for column in df_with_targets.columns if column not in exclude_cols]
        
        target_cols = [f'peak_target_{i}' for i in range(1, self.prediction_length + 1)]
        
        df_clean = df_with_targets.dropna()
        
        X = df_clean[feature_cols]
        y = df_clean[target_cols]
        
        self.feature_names = feature_cols
        
        return X, y, df_clean
    
    def train(self, df, test_size=0.2, random_state=42):
        """Train the multi-output XGBoost model"""
        print("Preparing training data...")
        X, y, df_clean = self.prepare_training_data(df)
        
        print(f"X columns: {X.columns}")
        print(f"y columns: {y.columns}")
        
        print(f"Training data shape: X={X.shape}, y={y.shape}")
        
        if X.empty or y.empty:
            raise ValueError("Training data is empty or insufficient for training.")
        # print(y)
        y_numpy = y.to_numpy()
        
        # For stratification, use the first target column that has positive samples
        stratify_col = None
        try:
            column_pos_samples = np.any(y_numpy == 1, axis=0)
            if np.any(column_pos_samples):
                stratify_col = y_numpy[:, np.argmax(column_pos_samples)]
        except Exception as e:
            print(f"Error in stratification: {e}")
            stratify_col = None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=stratify_col
        )
        
        # Calculate overall class weights for multi-output model
        # Use the average positive weight across all targets
        pos_weights = []
        for step in range(1, self.prediction_length + 1):
            target_col = f'peak_target_{step}'
            pos_samples = (y_train[target_col] == 1).sum()
            neg_samples = (y_train[target_col] == 0).sum()
            pos_weight = neg_samples / pos_samples if pos_samples > 0 else 1.0
            pos_weights.append(pos_weight)
        
        avg_pos_weight = np.mean(pos_weights)
        print(f"Average positive weight: {avg_pos_weight:.4f}")
        
        # base XGBoost classifier
        base_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            scale_pos_weight=avg_pos_weight,
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1
        )
        
        # Wrap in MultiOutputClassifier
        print("Training multi-output model...")
        self.model = MultiOutputClassifier(base_model, n_jobs=-1)
        
        # Convert targets to numpy array for multi-output training
        y_train_array = y_train.astype(int).values
        y_test_array = y_test.astype(int).values
        
        # Fit
        self.model.fit(X_train, y_train_array)
        
        # Evaluate each output separately
        y_pred_array = self.model.predict(X_test)
        y_pred_proba_array = self.model.predict_proba(X_test)
        
        for step in range(self.prediction_length):
            target_col = f'peak_target_{step + 1}'
            y_test_step = y_test_array[:, step]
            y_pred_step = y_pred_array[:, step]
            
            # Get probabilities for positive class
            y_pred_proba_step = y_pred_proba_array[step][:, 1]
            
            try:
                auc = roc_auc_score(y_test_step, y_pred_proba_step)
                print(f"Timestep {step + 1} - Test AUC: {auc:.4f}")
                print(f"Timestep {step + 1} - Classification Report:")
                print(classification_report(y_test_step, y_pred_step))
                print("-" * 50)
                
            except Exception as e:
                print(f"Could not calculate AUC for timestep {step + 1}: {e}")
        
        print("Multi-output training completed!")
        return self
    
    def predict(self, df, target_step=None):
        """
        Predict peak probabilities for specific timestep or all timesteps
        
        Args:
            df: Input dataframe
            target_step: If specified (1, 2, or 3), returns only that timestep's predictions
                        If None, returns predictions for all timesteps
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        df_features = self.create_features(df)
        
        # Detect column names
        if 'site' in df_features.columns:
            col = 'site'
            tcol = 'date'
        else:
            col = 'item_id'
            tcol = 'timestamp'
            if 'level_0' not in df_features.columns:
                df_features = df_features.reset_index()
        
        df_features['site_encoded'] = self.site_encoder.transform(df_features[col])
        X = df_features[self.feature_names].fillna(0)  
        
        # Get predictions for all timesteps
        predictions_proba = self.model.predict_proba(X)
        
        if target_step is not None:
            if target_step not in range(1, self.prediction_length + 1):
                raise ValueError(f"target_step must be between 1 and {self.prediction_length}")
            
            # Extract probabilities for specific timestep (positive class)
            step_predictions = predictions_proba[target_step - 1][:, 1]
            
            result_df = df[[col, tcol]].copy()
            result_df['peak_probability'] = step_predictions
            
            return result_df
        else:
            # Return all timesteps
            result_df = df[[col, tcol]].copy()
            for step in range(self.prediction_length):
                step_predictions = predictions_proba[step][:, 1]
                result_df[f'peak_probability_step_{step + 1}'] = step_predictions
            
            return result_df
    
    def add_peak_covariates_to_autogluon_data(self, df):
        """
        Add peak probability covariates to AutoGluon training data
        For training data, we add the 1-step-ahead peak probability as a feature
        """
        # Get peak predictions for 1 step ahead (next day)
        df = df.reset_index()
        df_with_peaks = self.predict(df, target_step=1)
        print(df_with_peaks.columns)
        if 'peaks' in df_with_peaks.columns:
            df_with_peaks = df_with_peaks.drop(columns='peaks')
        # Add the peak probability as a single column
        df_copy = df.copy()
        df_copy['peak_probability'] = df_with_peaks['peak_probability']
        
        return df_copy
    
    def get_future_covariates(self, df):
        """
        Predict the future probability covariates for prediction_length steps into the future
        
        Args:
            df: Multi-index timeseries dataframe with (item_id, timestamp) index
                containing historical data up to a certain date
            
        Returns:
            Multi-index timeseries dataframe with future dates and peak_probability column
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        df = df.copy()
        df_reset = df.reset_index()
        
        # Detect column names
        if 'site' in df_reset.columns:
            col = 'site'
            tcol = 'date'
        else:
            col = 'item_id'
            tcol = 'timestamp'
        
        df_reset[tcol] = pd.to_datetime(df_reset[tcol])
        
        future_predictions = []
        
        # Get all predictions at once using the multi-output model
        all_predictions = self.predict(df_reset)  # Returns all timesteps
        
        # Process each site
        for site in df_reset[col].unique():
            site_data = df_reset[df_reset[col] == site].copy()
            site_data = site_data.sort_values(tcol)
            last_date = site_data[tcol].max()
            
            # Get the predictions for this site (last row)
            site_mask = all_predictions[col] == site
            site_predictions = all_predictions[site_mask].iloc[-1]
            
            # For each future timestep
            for step in range(1, self.prediction_length + 1):
                future_date = last_date + pd.Timedelta(days=step)
                
                # Get the prediction for this timestep
                peak_prob = site_predictions[f'peak_probability_step_{step}']
                future_predictions.append({
                    col: site,
                    tcol: future_date,
                    'peak_probability': peak_prob
                })
        
        future_df = pd.DataFrame(future_predictions)
        future_df = future_df.set_index([col, tcol])
        
        return future_df
    
def retrieve_best_subsets(BO_files,preloaded):
    """Retrieve best subsets found by BO in previous experiments. Find the prediction file with the lowest error, and retrieve the subset on which the model was trained for each site.

    Args:
        BO_files (list): list of files with predictions from the BO experiments
        preloaded (dict): dictionary with preloaded timeseries

    Returns:
        dict: dictionary with target as key and best subset found by BO as list
    """

    best_subsets = {}
    
    METRICS_DIR = ".\\BOSP_preds\\"
    all_pred_files = [f for f in os.listdir(METRICS_DIR) if f.endswith(".csv")]

    # extract site ids from filenames
    site_pattern = re.compile(r'([a-zA-Z-_]+_\d+_[0-9a-zA-Z]+)')
    sites = set()
    for f in all_pred_files:
        m = site_pattern.search(f)
        if m:
            sites.add(m.group(1))

    def compute_mae_for_file(site, pred_path):
        """Compute MAE for one saved prediction CSV against ground truth for a single site."""
        df_pred = pd.read_csv(pred_path)
        if 'item_id' not in df_pred.columns:
            raise ValueError(f"'item_id' missing in {pred_path}; make sure index was saved to CSV.")
        df_pred = df_pred[df_pred['item_id'] == site].copy()
        if df_pred.empty:
            return None
        # normalize timestamp col name
        ts_col = 'timestamp' if 'timestamp' in df_pred.columns else 'date'
        df_pred[ts_col] = pd.to_datetime(df_pred[ts_col])
        df_pred = df_pred.sort_values(ts_col)
        # Ground truth slice
        gt = preloaded[site]
        mask = (gt['date'] >= df_pred[ts_col].min()) & (gt['date'] <= df_pred[ts_col].max())
        gt_slice = gt.loc[mask].copy().sort_values('date')
        if len(gt_slice) != len(df_pred):
            # skip misaligned files
            return None
        _, _, mae = calculate_metrics(gt_slice['value'].values, df_pred['mean'].values)
        return mae

    def get_strategy_from_file(pred_path):
        """Read one row to get strategy label saved by AG_DT.py (column 'strategy')."""
        df = pd.read_csv(pred_path, nrows=5)
        return str(df['strategy'].iloc[0]) if 'strategy' in df.columns else None

    def get_subset_from_file(pred_path, target_site):
        """The prediction CSV contains predictions for all item_ids in the trained set.
        That list is the (target + selected series). We return the selected series (exclude target)."""
        df = pd.read_csv(pred_path, usecols=['item_id']).drop_duplicates()
        items = df['item_id'].unique().tolist()
        selected = [x for x in items if x != target_site]
        return selected

    rows = []
    for site in sorted(sites):
        # all files for this site
        site_files = [f for f in all_pred_files if f.startswith(site)]
        if not site_files:
            continue
        
        # best baseline & best BOSP across seeds/timestamps
        best_bosp = (np.inf, None)

        for fname in site_files:
            p = os.path.join(METRICS_DIR, fname)
            mae = compute_mae_for_file(site, p)
            if mae is None:
                continue
            strat = get_strategy_from_file(p) or ""
            if strat.lower().startswith('baseline'):
                continue
                # if mae < best_baseline[0]:
                #     best_baseline = (mae, fname)
            elif 'bo' in strat.lower():  # catch 'bo_subset_*' etc.
                if mae < best_bosp[0]:
                    best_bosp = (mae, fname)

        # subset composition from the best BOSP file
        best_bosp_path = os.path.join(METRICS_DIR, best_bosp[1])
        selected_series = get_subset_from_file(best_bosp_path, target_site=site)
        best_subsets[site] = selected_series
    print(best_subsets)
    return best_subsets

def train_predict(best_subset, preloaded, prediction_length, num_val_windows, random_seed,target):
    concat_df = pd.DataFrame()
    concat_df_xgb = pd.DataFrame()
    for file_name in best_subset:
        temp_df = preloaded[file_name].copy()
        temp_df = temp_df.dropna(axis=0)
        temp_df = temp_df.sort_index()
        concat_df = pd.concat([concat_df, temp_df])
        xgb_df = temp_df[:-prediction_length*num_val_windows]
        xgb_df['peaks'] = create_peak_targets(xgb_df['value'].values)
        concat_df_xgb = pd.concat([concat_df_xgb, xgb_df])

    # Train the peak predictor on the XGBoost data
    peak_predictor = PeakPredictor(prediction_length=prediction_length)
    peak_predictor.train(concat_df_xgb)

    
    # Create time series dataframe and split
    ts_df = to_timeseriesdataframe(concat_df)
    splitter = ExpandingWindowSplitter(prediction_length=prediction_length, num_val_windows=num_val_windows)
    train_data, test_data = ts_df.train_test_split(prediction_length=prediction_length*num_val_windows)
    
    ts_df_enhanced = peak_predictor.add_peak_covariates_to_autogluon_data(train_data)

    ts_df_enhanced =TimeSeriesDataFrame.from_data_frame(
        df=ts_df_enhanced,
        id_column='item_id',
        timestamp_column='timestamp'
    )

    known_cov_names = [col for col in ts_df.columns if col not in ['site', 'date', 'value', 'delta_3d', 'is_rise']] + ['peak_probability']

    predictor_specialist = TimeSeriesPredictor(
    prediction_length=prediction_length,
    freq='D',
    cache_predictions=False,  
    eval_metric="MAE",
    target="value", 
    known_covariates_names=known_cov_names,
    ).fit(
        ts_df_enhanced, 
        random_seed=random_seed,
        presets='high_quality',
        excluded_model_types=["AutoETS", "AutoARIMA", "DynamicOptimizedTheta", "SeasonalNaive", "AutoETS"],
    )
    
    site_results = []
    all_pred = pd.DataFrame()

    for window_idx, (train_split, val_split) in enumerate(splitter.split(test_data)):
        # For realistic prediction, we need to:
        # 1. Get the historical data up to the prediction point
        # 2. Use the peak predictor to predict peak probabilities for the future dates
        # 3. Add these peak probabilities to the known covariates
        
        # Get future peak probabilities for the prediction window
        train_split = peak_predictor.add_peak_covariates_to_autogluon_data(train_split)
        future_peak_probs = peak_predictor.get_future_covariates(train_split)
        
        # Create a copy of val_split to add peak probabilities
        val_split_with_peaks = val_split.copy()
        
        # Add the peak probabilities to the validation split
        # This simulates what would happen in production where we predict peak probabilities
        for idx in future_peak_probs.index:
            if idx in val_split_with_peaks.index:
                val_split_with_peaks.loc[idx, 'peak_probability'] = future_peak_probs.loc[idx, 'peak_probability']
        val_split_with_peaks.update(train_split[['peak_probability']])

        # Select known covariates including peak_probability
        known_covariates_cols = [col for col in val_split_with_peaks.columns 
                            if col not in ['site', 'date', 'value', 'value_diff1', 'value_diff2', 'value_lag1', 'value_lag2']]
        
        train_split =TimeSeriesDataFrame.from_data_frame(
                df=train_split,
                id_column='item_id',
                timestamp_column='timestamp'
            )
        # Make predictions with the peak probability included
        y_pred = predictor_specialist.predict(
            train_split, 
            known_covariates=val_split_with_peaks[known_covariates_cols]
        )
        
        all_pred = pd.concat([all_pred, y_pred])
        
        y_true = val_split.groupby(level='item_id').tail(prediction_length)
        
        mask = y_pred.index.get_level_values('item_id') == target
        site_predictions = y_pred.loc[mask]
        site_true_unscaled = y_true.loc[mask]
        
        MAPE, MSE, MAE = calculate_metrics(site_true_unscaled['value'], site_predictions['mean'])
        
        site_results.append({
            'MAPE': MAPE,
            'MSE': MSE,
            'MAE': MAE
        })
    all_pred.to_csv(f'./peak_predictor_files/10windowpred_{target}_{random_seed}.csv')
    
def main():
    os.makedirs('peak_predictor_files', exist_ok=True)
    seed = 123

    BO_files = os.listdir('.\\BOSP_preds\\')
    base_path = find_base_path()
    files = os.listdir(base_path + '\\2_Hydraulic head data/Sensor data')
    prediction_length = 3
    num_val_windows = 10

    preloaded = preload_timeseries(files=files)
    
    subset_dict = retrieve_best_subsets(BO_files=BO_files, preloaded=preloaded)
    done = [d[14:-7] for d in os.listdir('./peak_predictor_files/')]
    to_do = [t for t in subset_dict.keys() if t not in done]
    print(f"Running 10 window preds with best subset for these files: {to_do}")
    for target in to_do:
        for seed in [123,124,125]:
            np.random.seed(seed)
            random.seed(seed)
            best_subset = subset_dict[target] + [target]
            print(f"Starting with {target}")
            train_predict(best_subset=best_subset, 
                        preloaded=preloaded, 
                        prediction_length=prediction_length, 
                        num_val_windows=num_val_windows, 
                        random_seed =seed,
                        target=target)
        
if __name__ == "__main__":
    main()
