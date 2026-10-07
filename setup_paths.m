%% SETUP_PATHS
% Initializes MATLAB search paths for the DME (Distributed Mixture-of-Experts) repository.
%
% Usage:
%   run setup_paths;

base_dir = fileparts(mfilename('fullpath'));

addpath(base_dir);
addpath(fullfile(base_dir, 'models'));
addpath(fullfile(base_dir, 'stattools'));
addpath(fullfile(base_dir, 'evaltools'));
addpath(fullfile(base_dir, 'datatools'));
addpath(fullfile(base_dir, 'data'));
addpath(fullfile(base_dir, 'experiments'));
addpath(fullfile(base_dir, 'toy_examples'));
addpath(fullfile(base_dir, 'real_data', 'beijing_air_quality'));
addpath(fullfile(base_dir, 'real_data', 'year_prediction_msd'));

fprintf('==> DME environment initialized successfully.\n');
fprintf('    All models, tools, and benchmark paths added.\n');
