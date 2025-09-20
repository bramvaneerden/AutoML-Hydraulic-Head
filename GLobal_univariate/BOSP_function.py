import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm
from tqdm import tqdm
import matplotlib.pyplot as plt
from AGTS_BOSP import evaluate_autogluon as autogluon_proxy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tsfresh.feature_extraction import extract_features
from datetime import datetime

def preload_features(files, preloaded):
    """Preloads features using TSFresh. Final 30 datapoints are removed, as this is the test set in the rest of the experiments.

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
        df = df[:-30] # remove prediction window
        df = df[['value']].reset_index().rename(columns={'index': 'time'})
        df['id'] = file
        dfs.append(df)
    combined = pd.concat(dfs).dropna(axis=0)
    features = extract_features(combined, column_id='id', column_sort='time', n_jobs=0)
    features = features.dropna(axis=1, how='any')  # drop features with NaNs
    return features

class TimeSeriesPoolOptimizer:
    def __init__(self, all_timeseries, target_series, preloaded_data, 
                 n_initial=50, max_iterations=50, prediction_length=3, 
                 random_seed=123, num_val_windows=10, verbose=True):
        """
        Bayesian Optimization for Selective Pooling.
        
        Args:
            all_timeseries: List of all available time series file names
            target_series: The target time series file name
            preloaded_data: Dictionary of preloaded dataframes for each time series
            n_initial: Number of random initial points to evaluate
            max_iterations: Maximum number of optimization iterations
            prediction_length: Prediction horizon length
            random_seed: Random seed for reproducibility
            num_val_windows: Number of validation windows for evaluation
            verbose: Whether to print detailed information
        """
        self.all_timeseries = [ts for ts in all_timeseries if ts != target_series]
        self.target_series = target_series
        self.preloaded_data = preloaded_data
        self.n_initial = min(n_initial, 2**len(self.all_timeseries) - 1)
        self.max_iterations = max_iterations
        self.prediction_length = prediction_length
        self.base_random_seed = random_seed
        self.num_val_windows = num_val_windows
        self.verbose = verbose
        
        # Create feature df
        features = preload_features(all_timeseries, self.preloaded_data)
        scaler = StandardScaler()
        features_scaled_arr = scaler.fit_transform(features)
        features_scaled = pd.DataFrame(features_scaled_arr,index=features.index, columns=features.columns)
        
        pca = PCA(n_components=19)
        features_pca_arr = pca.fit_transform(features_scaled)
        self.pca_df = pd.DataFrame(features_pca_arr, index = features_scaled.index)
        
        # For storing results
        self.X = []  # binary vectors
        self.y = []  # performance scores per vector
        self.best_score = float('inf')
        self.best_subset = []
        
        # GP kernel, should try a couple more different ones
        self.kernel = ConstantKernel() * Matern(nu=2.5) + WhiteKernel(noise_level=1e-5)
        # self.kernel = Matern(nu=2.5, length_scale=[1.0] * 19, length_scale_bounds=(0.01, 100.0))
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=random_seed
        )
        
        self.target_name = self.preloaded_data[self.target_series]['site'].unique()[0]
        
        # Get a baseline score first to compare. This is a single eval ,local model
        self.baseline_score = self._evaluate_single_series(self.target_series)
        if self.verbose:
            print(f"Baseline score (target only): {self.baseline_score:.4f}")
    
    def _binary_vector_to_subset(self, binary_vector):
        """Convert binary vector to list of selected time series"""
        return [self.all_timeseries[i] for i, bit in enumerate(binary_vector) if bit == 1]
    
    def _evaluate_single_series(self, series_name):
        """Evaluate performance using only a single series"""
        df = self.preloaded_data[series_name].copy()
        df['value'] = df['value'] # .diff()
        df = df.dropna()
        
        results = autogluon_proxy(
            df.copy(deep=True),
            prediction_length=self.prediction_length,
            random_seed=self.base_random_seed,
            target=self.target_name,
            strat="baseline",
            num_val_windows=self.num_val_windows,
            covariates=True
        )
        
        return np.mean([r['MAE_mean'] for r in results])
    
    def _evaluate_subset(self, binary_vector):
        """Evaluate performance on a subset of time series"""
        if sum(binary_vector) == 0:
            return self.baseline_score
        
        subset = self._binary_vector_to_subset(binary_vector)
        
        # Concatenate all dataframes in the subset plus target
        concat_df = pd.DataFrame()
        all_series = [self.target_series] + subset
        
        for file_name in all_series:
            temp_df = self.preloaded_data[file_name].copy()
            temp_df['value'] = temp_df['value'] #.diff()
            temp_df = temp_df.dropna()
            concat_df = pd.concat([concat_df, temp_df])
        
        # Run proxy model evaluation
        results = autogluon_proxy(
            concat_df.copy(deep=True),
            prediction_length=self.prediction_length,
            random_seed=self.base_random_seed,
            target=self.target_name,
            strat=f"bo_subset_{sum(binary_vector)}",
            num_val_windows=self.num_val_windows,
            covariates=True
        )
        
        score = np.mean([r['MAE_mean'] for r in results])
        
        if self.verbose:
            series_names = [f"{ts.split('_')[0]}_{ts.split('_')[1]}" for ts in subset]
            print(f"Subset size: {len(subset)}, Score: {score:.4f}, Series: {series_names}")
        
        return score

    def _feature_representation(self, binary_vector):
        """feature representation for a subset"""
        if sum(binary_vector) == 0:
            return np.zeros(self.pca_df.shape[1] * 2)  # handle empty case
            
        # Get features
        selected_features = [self.pca_df.iloc[i].values for i, bit in enumerate(binary_vector) if bit == 1]
        selected_features = np.vstack(selected_features)
        
        # calculate mean and variance
        mean_features = np.mean(selected_features, axis=0)
        var_features = np.var(selected_features, axis=0)
        
        #include similarity to target series
        target_features = self.pca_df.loc[self.target_series].values
        sim_to_target = [np.linalg.norm(target_features - self.pca_df.iloc[i].values) 
                        for i, bit in enumerate(binary_vector) if bit == 1]
        min_dist = min(sim_to_target) if sim_to_target else 999.0
        
        #return concatenated features
        return np.concatenate([mean_features, var_features, [min_dist, sum(binary_vector)]])
    
    def _expected_improvement(self, X_new, xi=0.01):
        """Calculate expected improvement acquisition function"""
        X_new_pca = []
        # for candidate in X_new:
        #     X_new_pca.append(np.mean(np.stack([self.pca_df.iloc[i].values for i, bit in enumerate(candidate) if bit == 1]), axis=0)) 
        for candidate in X_new:
            X_new_pca.append(self._feature_representation(candidate)) 

        mu, sigma = self.gp.predict(X_new_pca, return_std=True)
        mu = np.exp(mu) # log transform back
        
        # zero case
        if sigma == 0:
            return 0
        
        # minimizing, so improvement is negative of predicted value minus best so far
        # invert to make it a maximization problem 
        z = (self.best_score - mu - xi) / sigma
        ei = (self.best_score - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)
        
        return ei
    
    def _hamming_distance(self, vec1, vec2):
        """Calculate Hamming distance between binary vectors"""
        return sum(b1 != b2 for b1, b2 in zip(vec1, vec2))
    
    def _generate_diverse_initial_points(self):
        """Generate diverse initial points with preference for smaller subsets"""
        points = []
        
        # Start with some pairs (2 series)
        n_pairs = min(10, len(self.all_timeseries) * (len(self.all_timeseries) - 1) // 2)
        pairs_added = 0
        while pairs_added < n_pairs and len(points) < self.n_initial:
            i, j = np.random.choice(len(self.all_timeseries), 2, replace=False)
            point = np.zeros(len(self.all_timeseries), dtype=int)
            point[i] = point[j] = 1
            
            # check if pair is already in points
            if not any(np.array_equal(point, p) for p in points):
                points.append(point)
                pairs_added += 1
        
        # Add a few triples (3 series)
        n_triples = min(5, self.n_initial - len(points))
        triples_added = 0
        while triples_added < n_triples and len(points) < self.n_initial:
            indices = np.random.choice(len(self.all_timeseries), 3, replace=False)
            point = np.zeros(len(self.all_timeseries), dtype=int)
            point[indices] = 1
            
            # check if triple is already in points
            if not any(np.array_equal(point, p) for p in points):
                points.append(point)
                triples_added += 1
        
        # Fill the rest
        #size distribution: 40% small, 40% medium (4-len/2), 20% larger
        while len(points) < self.n_initial:
            rand_val = np.random.random()
            if rand_val < 0.4:  # 40% small subsets (4-5 features)
                size = np.random.randint(4, min(6, len(self.all_timeseries)))
            elif rand_val < 0.8:  # 40% medium subsets (6-len/2 features)
                size = np.random.randint(6, max(7, len(self.all_timeseries) // 2))
            else:  # 20% larger subsets
                size = np.random.randint(len(self.all_timeseries) // 2, 
                                    max(len(self.all_timeseries) // 2 + 1, 
                                        len(self.all_timeseries) * 3 // 4))
            
            # Create random subset of the determined size
            indices = np.random.choice(len(self.all_timeseries), size, replace=False)
            candidate = np.zeros(len(self.all_timeseries), dtype=int)
            candidate[indices] = 1
            
            # Add if diverse enough
            min_distance = min(self._hamming_distance(candidate, p) for p in points) if points else 0
            if min_distance >= min(3, size // 2 + 1):
                points.append(candidate)
        
        return points[:self.n_initial]
    
    def _sample_next_candidates(self, n_candidates=1000):
        """Sample next candidates using EI acquisition function"""
        # generate random candidates (including neighbors of best solution)
        candidates = []
        
        # add neighbors of the best solution (one bitflip)
        best_vector = self.X[np.argmin(self.y)]
        for i in range(len(best_vector)):
            neighbor = best_vector.copy()
            neighbor[i] = 1 - neighbor[i]
            candidates.append(neighbor)
        
        # random candidates
        while len(candidates) < n_candidates:
            candidate = np.random.randint(0, 2, len(self.all_timeseries))
            
            if sum(candidate) == 0:
                continue
                
            if any(np.array_equal(candidate, x) for x in self.X):
                continue
                
            candidates.append(candidate)
        
        # EI
        candidates = np.array(candidates)
        ei_values = [self._expected_improvement(candidate.reshape(1, -1)) for candidate in candidates]
        
        #candidate with highest EI
        best_idx = np.argmax(ei_values)
        return candidates[best_idx]
    
    def optimize(self):
        """Run the Bayesian Optimization process"""
        if self.verbose:
            print(f"Starting optimization with {self.n_initial} initial points...")
        
        initial_points = self._generate_diverse_initial_points()
        
        for i, binary_vector in enumerate(initial_points):
            score = self._evaluate_subset(binary_vector)
            self.X.append(binary_vector)
            self.y.append(score)
            
            if score < self.best_score:
                self.best_score = score
                self.best_subset = self._binary_vector_to_subset(binary_vector)
                
        # main optimization loop
        for iteration in tqdm(range(self.max_iterations), desc="BO Iterations"):
            # fit GP model with current data
            X_pca = []
            # for candidate in self.X:
            #     X_pca.append(np.mean(np.stack([self.pca_df.iloc[i].values for i, bit in enumerate(candidate) if bit == 1]), axis=0)) 
            for candidate in self.X:
                X_pca.append(self._feature_representation(candidate)) 
            
            X_array = np.array(X_pca)
            y_array = np.array(self.y)
            # self.gp.fit(X_array, y_array)
            self.gp.fit(X_array, np.log(y_array)) # log transform test
            
            next_candidate = self._sample_next_candidates()
            score = self._evaluate_subset(next_candidate)
            
            #Update data
            self.X.append(next_candidate)
            self.y.append(score)
            
            #update best solution if improved
            if score < self.best_score:
                self.best_score = score
                self.best_subset = self._binary_vector_to_subset(next_candidate)
                if self.verbose:
                    print(f"New best score: {self.best_score:.4f}, Improvement over baseline: {(self.baseline_score - self.best_score)/self.baseline_score*100:.2f}%")
                    print(f"Best subset size: {len(self.best_subset)}")
                    series_names = [f"{ts.split('_')[0]}_{ts.split('_')[1]}" for ts in self.best_subset]
                    print(f"Best subset: {series_names}")
            
            # early stopping
            if iteration > 35 and min(self.y[-5:]) >= self.best_score:
                if self.verbose:
                    print("Early stopping: No improvement for 5 iterations")
                break
        
        return self.best_subset, self.best_score
    
    def plot_optimization_progress(self):
        """Plot the optimization progress"""
        iterations = np.arange(len(self.y))
        best_so_far = np.minimum.accumulate(self.y)
        
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, self.y, 'o-', alpha=0.5, label='Evaluated subsets')
        plt.plot(iterations, best_so_far, 'r-', linewidth=2, label='Best so far')
        plt.axhline(y=self.baseline_score, color='g', linestyle='--', label='Baseline - local model')
        
        plt.xlabel('Iteration')
        plt.ylabel('MAE')
        plt.title('Bayesian optimization progress')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'./plots/BO_fig_{self.target_series}_seed_{self.base_random_seed}_{datetime.now().strftime("%m_%d_%Y_%H_%M_%S")}.png')
        # plt.show()
        


def bayesian_optimization_selection(files, target_file, preloaded, **kwargs):
    """
    Use Bayesian Optimization to select the best subset of time series.
    
    Args:
        files: List of all time series files
        target_file: Target time series file
        preloaded: Dictionary of preloaded time series dataframes
        **kwargs: Additional arguments for TimeSeriesPoolOptimizer
    
    Returns:
        best_subset, best_score
    """
    optimizer = TimeSeriesPoolOptimizer(
        all_timeseries=files,
        target_series=target_file,
        preloaded_data=preloaded,
        **kwargs
    )
    
    best_subset, best_score = optimizer.optimize()
    
    # plot the progress
    optimizer.plot_optimization_progress()
    best_subset = [target_file] + best_subset #include the target in the returned list
    return best_subset, best_score
