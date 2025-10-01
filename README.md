# AutoML-Hydraulic-Head
Code for the master thesis "AutoML for Hydraulic Head Forecasting in Dikes: A Selective Pooling and Peak-Aware Approach"  

Each folder contains the files for one of the modelling scenarios tested in the thesis: local univariate, global univariate and multivariate.  
The local univariate experiments compare three AutoML framewors to a set of statistical models. The global univariate experiments focus on 'selective pooling' strategies to select the time series that are most informative to forecasting the target time series. The multivariate experiments are focused on recreating the 'AutoGluon' approach in Darts.  

The best model (Global univariate combined with BOSP) is then extended with a peak-aware approach, where we foreacast peak probabilities with a separate XGBoost model, which we pass as additional covariates to AutoGluon-Timeseries to provide an explicit signal for potential peaks.  