import pandas as pd
from tslearn.metrics import dtw
from sklearn.feature_selection import mutual_info_regression
from tsfresh.feature_extraction import extract_features
from sklearn.cluster import KMeans
import numpy as np
import os
import random
from load_function import load_timeseries, find_base_path
from AG_TS import evaluate_autogluon
from GLobal_univariate.AGTS_proxy import evaluate_autogluon as autogluon_proxy
from tslearn.metrics import dtw
from datetime import datetime

def run_experiments(df, prediction__length, random_seed, target, strat, num_val_windows, n_runs, covariates, evaluate_global):    
    """Runs the experiments with the full autogluon model.

    Args:
        df (dataframe): df with all time series
        prediction__length (int): lenght of prediction window
        random_seed (int): random seed
        target (string): name of the target time series
        strat (int): strategy number, added to the results
        num_val_windows (int): number of validation windows
        n_runs (int): amount of runs per subset
        covariates (bool): consider covariates or not
        evaluate_global (bool): Runs a full global model without a target series

    Returns:
        dict: Dictionary with metrics for all validation windows
    """
    results = []
    for i in range(n_runs):
        run_seed = random_seed + i
        run_df = df.copy(deep=True)
        
        np.random.seed(run_seed)
        random.seed(run_seed)
        
        print(f"Running experiment with seed {run_seed}")
        
        run_results = evaluate_autogluon(
            run_df, 
            prediction_length=prediction__length,
            random_seed=run_seed,
            target=target,
            strat=strat,
            num_val_windows=num_val_windows,
            covariates=covariates,
            evaluate_global=evaluate_global
        )
        
        results.extend(run_results)

    return results

def run_proxy_experiments(df, prediction__length, random_seed, target, strat, num_val_windows, n_runs, covariates):    
    """Runs experiments for the random search proxy. Calls autogluon_proxy from AG_DT_p, which runs only the directtabular model from Autogluon.

    Args:
        df (dataframe): df with all time series
        prediction__length (int): lenght of prediction window
        random_seed (int): random seed
        target (string): name of the target time series
        strat (int): strategy number, added to the results
        num_val_windows (int): number of validation windows
        n_runs (int): amount of runs per subset
        covariates (bool): consider covariates or not

    Returns:
        dict: Dictionary with metrics for all validation windows
    """
    results = []
    for i in range(n_runs):
        run_seed = random_seed + i
        run_df = df.copy(deep=True)
        
        np.random.seed(run_seed)
        random.seed(run_seed)
        
        print(f"Running proxy experiment with seed {run_seed}")
        
        run_results = autogluon_proxy(
            run_df, 
            prediction_length=prediction__length,
            random_seed=run_seed,
            target=target,
            strat=strat,
            num_val_windows=num_val_windows,
            covariates=covariates
        )
        
        results.extend(run_results)

    return results

def preload_timeseries(files):
    """Preloads all time series specified in the files list.

    Args:
        files (list): All filenames to consider

    Returns:
        (dict): dictionary of preloaded time series in dataframe format
    """
    ts = {}
    for fname in files:
        df = load_timeseries(fname).copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        ts[fname] = df

    max_start = max(df['date'].min() for df in ts.values())
    return {f: df.loc[df['date'] >= max_start] for f, df in ts.items()}

def preload_features(files, preloaded):
    """Preloads all features for k-means

    Args:
        files (list): All filenames to consider
        preloaded (dict): dictionary of preloaded time series in dataframe format

    Returns:
        (dict): dictionary of preloaded features per time series
    """
    f = files.copy()
    dfs = []
    for file in f:
        df = preloaded[file].copy()
        df = df[['value']].reset_index().rename(columns={'index': 'time'}).diff()
        # df = df['value']
        df['id'] = file
        dfs.append(df)
    combined = pd.concat(dfs).dropna(axis=0)
    features = extract_features(combined, column_id='id', column_sort='time', n_jobs=0)
    features = features.dropna(axis=1, how='any')  # Drop features with NaNs
    return features

def select_multiple_timeseries(files, target, strat, preloaded, features):
    """Function with all selective pooling strategies. 

    Args:
        files (list): All filenames to consider
        target (string): name of the target time series
        strat (int): Strategy number, corresponds to a selective pooling strategy
        preloaded (dict): dictionary of preloaded time series in dataframe format
        features (dict): dictionary of preloaded features per time series

    Returns:
        list: list of files similar to and including the target series
    """

    f = files.copy()
    if not isinstance(strat, int):
        print(f"Expected int for strat, got {type(strat)}: {strat}")
    
    print(f"strat start: {strat}")
    if strat == 0:
        # print(target)
        return [target]
    
    # 1. Get all similarly placed timeseries
    elif strat == 1:  
        num = target.split('_')[1]
        # print(f"target: {target}, num: {num}")
        similar_position = [x for x in f if x.split('_')[1] == num]
        # print(similar_position)
        return similar_position
        
    # 2. Get timeseries with similar variance, std' or just pearson
        # Can also try filtering further with mean and std within some range of the target df. Commented out now.
    elif strat == 2:
        target_df = preloaded[target].copy()
        # target_df['value'] = target_df['value'].diff()
        target_df['value'] = target_df['value'].diff()
        target_df['value'] = target_df['value'].dropna()
        target_mean, target_std = target_df['value'].mean(), target_df['value'].std()
        corr_list = []
        
        for file in f:
            # keep the target in there, instead of adding it back later
            # if file == target:
            #     continue
            comp_df = preloaded[file].copy()
            # comp_df['value'] = comp_df['value'].diff()
            comp_df['value'] = comp_df['value'].diff()
            comp_df['value'] = comp_df['value'].dropna()
            comp_mean, comp_std = comp_df['value'].mean(), comp_df['value'].std()
            mean_pct_diff, std_pct_diff = ((abs(comp_mean- target_mean) / target_mean) * 100), ((abs(comp_std - target_std) / target_std) * 100)
            corr = abs(target_df['value'].corr(comp_df['value']))
            corr_list.append({
                'correlation': corr,
                'file':file,
                'mean_pct_diff': mean_pct_diff,
                'std_pct_diff': std_pct_diff
            })
        corr_df =pd.DataFrame(corr_list)
        corr_df = corr_df.sort_values(by=['correlation'], ascending=False)
        
        # values above 0.3 are moderate to high correlation, keep all above .3 > increase to .4 for more selective approach
        corr_df = corr_df[corr_df['correlation'] > 0.4]
        # print(f"similar files: \n {list(corr_df['file'].values)}")
        return list(corr_df['file'].values)
    
    # 3. Get timeseries from same location
    elif strat == 3:  
        name = target.split('_')[0]
        similar_position = [x for x in f if x.split('_')[0] == name]
        # print(similar_position)
        return similar_position
    
    # 4. # DTW Similarity
    elif strat == 4:  
        target_series = preloaded[target]['value'].diff().dropna().values
        dtw_scores = []

        for file in f:
            comp_series = preloaded[file]['value']  .diff().dropna().values
            try:
                dist = dtw(target_series, comp_series)
                dtw_scores.append({'file': file, 'dtw_distance': dist})
            except Exception as e:
                print(f"Skipping {file} due to DTW error: {e}")
        
        dtw_df = pd.DataFrame(dtw_scores).sort_values(by='dtw_distance')
        return list(dtw_df['file'].values[:5])
    
    # 5. Feature-based clustering
    elif strat == 5:  
        kmeans = KMeans(n_clusters=5, random_state=0).fit(features)
        cluster_labels = kmeans.labels_
        target_cluster = cluster_labels[list(features.index.get_level_values(0)).index(target)]
        selected = [i for i, lbl in zip(features.index.get_level_values(0), cluster_labels) if lbl == target_cluster]
        return selected
    
    # 6. Mutual Information
    elif strat == 6:  
        target_series = (
            preloaded[target]['value'].diff().dropna().reset_index(drop=True)
        )
        mi_scores = []

        for file in f:
            comp_series   = (
                preloaded[file]['value']  .diff().dropna().reset_index(drop=True)
            )
            min_len = min(len(target_series), len(comp_series))
            if min_len < 30:
                continue  
            x = comp_series[:min_len]
            y = target_series[:min_len]
            mi = mutual_info_regression(x.values.reshape(-1, 1), y.values)[0]
            mi_scores.append({'file': file, 'mutual_info': mi})
        
        mi_df = pd.DataFrame(mi_scores).sort_values(by='mutual_info', ascending=False)
        return list(mi_df['file'].values[:5]) 
    
    # Feedforward proxy correlation
    elif strat == 8:  
        
        print(f"strat here: {strat}")
        target_df = preloaded[target].copy()
        target_name = target_df['site'].unique()[0]

        def prepare(df):
            out = df.copy()
            out["value"] = out["value"]  # same transform everywhere, or apply diff if intended
            return out.dropna()

        # Baseline
        baseline_proxy_results = run_proxy_experiments(
            prepare(target_df), prediction__length=3, random_seed=123,
            target=target_name, strat="baseline",
            num_val_windows=10, n_runs=3, covariates=True
        )
        baseline_proxy_mean = np.mean([i["MAE_mean"] for i in baseline_proxy_results])
        
        candidate_files = [f for f in preloaded.keys() if f != target]
        print(f"Testing {len(candidate_files)} candidate files for pairing")
        
        EVAL_KW = dict(prediction__length=3, random_seed=123, target=target_name,
               num_val_windows=10, n_runs=3, covariates=True)

        # First, test each file individually with the target
        individual_results = []
        for candidate in candidate_files:
            second_df = preloaded[candidate].copy()
            combined_df = pd.concat([target_df.copy(), second_df], ignore_index=True)
            pair_results = run_proxy_experiments(prepare(combined_df), strat=f"pair_{os.path.basename(candidate)}", **EVAL_KW)
            pair_mean = np.mean([r["MAE_mean"] for r in pair_results])
            delta = baseline_proxy_mean - pair_mean
            if delta > 0:
                individual_results.append(dict(candidate=candidate, delta=delta,
                                            sites=combined_df["site"].unique().tolist(),
                                            combined_df=combined_df))
        if not individual_results:
            return None

        individual_results.sort(key=lambda x: x["delta"], reverse=True)
        best = individual_results[0]
        current_best_df = best["combined_df"].copy()
        current_sites = set(best["sites"])
        current_loss = baseline_proxy_mean - best["delta"]  # equals pair_mean

        
        #start with the best performing candidate according to proxy
        if len(individual_results) > 0:
            best_candidate = individual_results[0]
            current_best_df = best_candidate['combined_df'].copy()
            current_sites = set(best_candidate['sites'])
            
            remaining_candidates = individual_results[1:]
            
            for nxt in individual_results[1:]:
                # derive the non target site
                nxt_sites = set(nxt["sites"])
                new_sites = nxt_sites - current_sites
                if not new_sites:
                    continue
                test_df = pd.concat([current_best_df, nxt["combined_df"]], ignore_index=True)
                test_df = test_df.drop_duplicates(subset=["site", "date"])
                combo_results = run_proxy_experiments(prepare(test_df), strat=f"combo_{len(current_sites)+1}", **EVAL_KW)
                combo_mean = np.mean([r["MAE_mean"] for r in combo_results])
                if combo_mean < current_loss:
                    current_loss = combo_mean
                    current_best_df = test_df
                    current_sites = set(test_df["site"].unique())

        if len(current_sites) <= 1:
            return None

        site_to_file = {}
        for filename, df in preloaded.items():
            site_name = df["site"].unique()[0]
            site_to_file[site_name] = filename

        toreturn = []
        for site in current_sites:
            if site in site_to_file:
                toreturn.append(site_to_file[site])

        return toreturn if toreturn else None

    else:
        print(f"wrong strat number")

def main():
    # Set seeds
    seed = 123
    np.random.seed(seed)
    random.seed(seed)
    random_seed = seed
    
    # initial variables. 
    base_path = find_base_path()
    files = os.listdir(base_path + '\\2_Hydraulic head data\\Sensor data')
    prediction__length = 3
    evaluate_global = False # only when running global model

    # Preload time series and features
    preloaded = preload_timeseries(files=files)
    features = preload_features(files, preloaded)
    # features = []

    # Create necessary directories
    RESULT_DIR = 'Selective_pooling'
    DUMP_DIR = 'proxy_metric_dump'
    for dir in [RESULT_DIR,  DUMP_DIR]:
        if not os.path.exists(dir):
            os.makedirs(dir)
    
    # seperate part for evaluating the global model. This excludes all selective pooling and runs AGTS on all time series in preloaded.
    if evaluate_global:
        np.random.seed(seed)
        random.seed(seed)
        concat_df = pd.DataFrame()
        for file_name in files:
            temp_df = preloaded[file_name].copy()
            
            
            temp_df['value'] = temp_df['value'] #.diff()
            temp_df = temp_df.dropna(axis=0)
            
            concat_df = pd.concat([concat_df, temp_df])
        res = run_experiments(
            concat_df.copy(deep=True), 
            prediction__length=prediction__length, 
            random_seed=random_seed,
            target=None, 
            strat=None,
            num_val_windows=10, 
            n_runs=3, 
            covariates=True,
            evaluate_global = evaluate_global
        )
        
        # result_df = pd.DataFrame(res)
        # result_df.to_csv(f'./global_metric/{file}_strat_{strat}_seed_{random_seed}_metrics_{datetime.now().strftime("%m_%d_%Y_%H_%M_%S")}.csv')
    
    # For all selective pooling strategies
    else:
        for file in files:
            target_name = preloaded[file]['site'].unique()[0]
            
            # select strategies to be included
            for strat in [3]: # 0,3,4,5,8
                np.random.seed(seed)
                random.seed(seed)
                
                # Finds all similar files, returns the subset including the target
                similar_files = select_multiple_timeseries(files, target=file, strat=strat, preloaded=preloaded, features=features)
                if similar_files ==None:
                    print(f"No similar files found for {file} during strategy {strat}")
                    continue
                print(f"Strategy {strat} selected files for {file}:")
                print(similar_files)
                
                # Concatenate the time series for the subset into a single dataframe
                concat_df = pd.DataFrame()
                for file_name in similar_files:
                    temp_df = preloaded[file_name].copy()
                    temp_df = temp_df.dropna(axis=0)
                    
                    concat_df = pd.concat([concat_df, temp_df])
                
                # Run experiments. Runs the full AGTS model in AG_TS.py over multiple runs. 
                res = run_experiments(
                    concat_df.copy(deep=True), 
                    prediction__length=prediction__length, 
                    random_seed=random_seed,
                    target=target_name, 
                    strat=strat,
                    num_val_windows=10, 
                    n_runs=3, 
                    covariates=True,
                    evaluate_global = evaluate_global
                )
                
                result_df = pd.DataFrame(res)
                result_df.to_csv(f'./{RESULT_DIR}/{file}_strat_{strat}_seed_{random_seed}_metrics_{datetime.now().strftime("%m_%d_%Y_%H_%M_%S")}.csv')
            
if __name__ == "__main__":
    main()