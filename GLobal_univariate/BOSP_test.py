import pandas as pd
import numpy as np
import os
import random
from load_function import load_timeseries, find_base_path
from AG_TS import evaluate_autogluon
import multiprocessing as mp
import os

def run_experiments(df, prediction__length, random_seed, target, strat, num_val_windows, n_runs, covariates, evaluate_global):    
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

def preload_timeseries(files,hhnk=False):
    ts = {}
    for fname in files:
        df = load_timeseries(fname, hhnk=hhnk).copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        ts[fname] = df

    max_start = max(df['date'].min() for df in ts.values())
    return {f: df.loc[df['date'] >= max_start] for f, df in ts.items()}


def process_single_file(args):
    """Process a single file with BO - designed to be run in parallel"""
    file, seed_offset, base_path = args
    
    from BOSP_function import bayesian_optimization_selection
    import numpy as np
    import random
    
    process_seed = 123 + seed_offset
    np.random.seed(process_seed)
    random.seed(process_seed)
    
    # Load data 
    files = os.listdir(base_path)
    preloaded = preload_timeseries(files=files, hhnk=True)
    
    print(f"Process {os.getpid()}: Starting BO for {file}")
    
    try:
        # Run BO selection
        similar_files = []
        best_score = np.inf
        
        for i in range(3): 
            run_seed = process_seed + i
            best_subset, best_score_temp = bayesian_optimization_selection(
                files=files,
                target_file=file,
                preloaded=preloaded,
                n_initial=30,
                max_iterations=30,
                prediction_length=3,
                random_seed=run_seed,
                num_val_windows=10,
                verbose=True
            )
            if best_score_temp < best_score:
                best_score = best_score_temp
                similar_files = best_subset
        
        print(f"Process {os.getpid()}: Completed BO for {file}")
        print(f"Selected files: {[f.split('_')[0] + '_' + f.split('_')[1] for f in similar_files]}")
        
        return file, similar_files, best_score
        
    except Exception as e:
        print(f"Process {os.getpid()}: Error processing {file}: {e}")
        return file, None, None
    
def main_multiprocessing_simple():
    os.makedirs('BOSP',exist_ok=True)
    os.makedirs('plots',exist_ok=True)

    base_path = find_base_path()
    files = os.listdir(base_path + '2_Hydraulic head data\\Sensor data')    
    args_list = [(file, i, base_path) for i, file in enumerate(files)]
    
    with mp.Pool(processes=1) as pool:
        results = pool.map(process_single_file, args_list)
    
    for file, similar_files, best_score in results:
        if similar_files is not None:
            print(f"File: {file}, Score: {best_score:.4f}")
            print(f"Selected: {similar_files}")


if __name__ == "__main__":
    if __name__ == "__main__":
        main_multiprocessing_simple()