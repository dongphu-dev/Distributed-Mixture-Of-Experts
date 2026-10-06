# Optimal Transport Aggregation for Distributed Mixture-of-Experts
Matlab source codes for the "Optimal Transport Aggregation for Distributed Mixture-of-Experts".

## Environment
Matlab R2018b. In our code, we also used `parfor` loop which requires there are more than one core to be affect. To check the number of physical cores, in Command Window, run `evalc("feature('numcores')")`

## Usage
`fit = Distributed_MixtureOfExperts_Gaussian(X_train, Y_train, K, M, options);`

#### Input
- `X_train`: matrix of size `m-by-d`, where `m` is the number of individuals, `d` is the number of feature
- `Y_train`: vector of size `m-by-1`
- `K`: the number of experts, supposed to be greater than one
- `M`: the number of machines, supposed to be greater than one
- `options`: the struct contains options for the algorithm. Set `options = get_options('default')` to use the default ones.

#### Output
The model returns a `fit` object contains the following fields:
- `experts`: matrix of size `d-by-K`, corresponds to the K fitted experts of the model;
- `gates`: matrix of size `d-by-K`, the K coresponding gates, with the last column is zeros;
- `variances`: the K coresponding variances;
- and other statistics for inspecting the implementation such as `learning_time`, `reduction_time`, `local_times`, `large_mixture`, `reduced_mixture`, `transportation_plan`, `local_estimates`, etc.

## Evaluation
The evaluation metrics will look like:
```
Learning Time  : 232.054 (s)
Trans. Distance: 2.117 
Log-likelihood : -113390.103 
MSE with truth : 0.001 
Prediction evaluation
RelativePredErr: 0.120 
Correlation    : 0.938 
Rand Index (RI): 0.941 
Adjusted RI    : 0.825 
ClusteringErr  : 7.582 (%)
```
