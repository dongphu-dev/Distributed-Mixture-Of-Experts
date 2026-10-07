%% EXP_IID_BENCHMARK
% Master Experiment 1: Evaluates distributed MoE estimators under IID data partitioning
% across three expert proportion profiles:
%   1. Balanced:            pi_k = 0.20
%   2. Moderately Imbalanced: 0.10 <= pi_k <= 0.30
%   3. Highly Imbalanced:   pi_min = 0.02
%
% Evaluated Models (6):
%   - Global oracle (GLB)
%   - Proposed (DME)
%   - Greedy merging (GM)
%   - Aligned averaging (AAVR)
%   - Federated averaging (FED)
%   - Naive averaging (WAVR)
%
% Setup:
%   M = 16 machines, N = 100,000 points per dataset (80% train / 20% test)
%   DEV mode (macOS): 2 runs | PROD mode (Linux): 100 runs
%
% Output:
%   results/benchmark_iid/Result_IID_M16_N100k.mat
%   results/tables/table_exp1_iid_benchmark.tex

clear; clc;
fprintf('========================================================================\n');
fprintf('  MASTER EXPERIMENT 1: IID Data Partitioning Benchmark (M = 16, N = 100k)\n');
fprintf('========================================================================\n');

base_dir = fileparts(fileparts(mfilename('fullpath')));
addpath(base_dir);
addpath(fullfile(base_dir, 'models'));
addpath(fullfile(base_dir, 'datatools'));
addpath(fullfile(base_dir, 'evaltools'));
addpath(fullfile(base_dir, 'stattools'));
addpath(fullfile(base_dir, 'experiments'));
addpath(fullfile(base_dir, 'reporting'));

poolobj = gcp('nocreate');
if isempty(poolobj)
    parpool('Processes', 4);
elseif poolobj.NumWorkers ~= 4
    delete(poolobj);
    parpool('Processes', 4);
end

out_dir = fullfile(base_dir, 'results', 'benchmark_iid');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

K = 5;
d = 20;
M = 16;
N_total = 100000;
N_train = floor(0.8 * N_total);
N_test  = N_total - N_train;
N_m     = floor(N_train / M);

profiles = {'balanced', 'moderate', 'severe'};
models_to_run = {'DME', 'GM', 'AAVR', 'FED', 'WAVR'};
all_models    = [{'GLB'}, models_to_run];

% Optimized options for GEM and local EM convergence
options = get_options('default');
options.DME_verbose = 0;
options.verbose     = 0;
options.sample_size = 2000; % Supporting sample size S
options.IRLS_max_iter = 30;
options.IRLS_threshold = 1e-5;
options.max_iter       = 150; % Fast, guaranteed convergence with Random Hyperplane init
options.nb_EM_runs     = 2;   % 2 starts with random hyperplanes
options.FedAvg_rounds  = 5;
options.FedAvg_local_iters = 15;

All_results = struct();

for p = 1:length(profiles)
    prof_name = profiles{p};
    data_file = fullfile(base_dir, 'data', sprintf('dataset_N100k_K5_d20_%s.mat', prof_name));
    
    if ~exist(data_file, 'file')
        error('Dataset not found: %s. Run data_generator_profiles first!', data_file);
    end
    
    fprintf('\n------------------------------------------------------------------------\n');
    fprintf('  Processing Profile [%d/3]: %s\n', p, prof_name);
    fprintf('  Loading: %s\n', data_file);
    fprintf('------------------------------------------------------------------------\n');
    
    loaded = load(data_file);
    X_mat        = loaded.X_mat;
    Y_mat        = loaded.Y_mat;
    LABEL_mat    = loaded.LABEL_mat;
    true_mixture = loaded.true_mixture;
    num_runs     = size(X_mat, 3);
    
    fprintf('Found %d datasets in file. Evaluating M = %d clients...\n', num_runs, M);

    % Initialize storage for this profile
    All_results.(prof_name) = struct();
    for m = 1:length(all_models)
        All_results.(prof_name).(all_models{m}).metrics = zeros(num_runs, 9);
    end

    for r = 1:num_runs
        fprintf('  --> Run [%d/%d] (%s, IID)...\n', r, num_runs, prof_name);
        
        X_all     = X_mat(:, :, r);
        Y_all     = Y_mat(:, r);
        label_all = LABEL_mat(:, r);
        
        % 80% train / 20% test split
        X_train     = X_all(1:N_train, :);
        Y_train     = Y_all(1:N_train);
        label_train = label_all(1:N_train);
        
        X_test      = X_all(N_train+1:end, :);
        Y_test      = Y_all(N_train+1:end);
        label_test  = label_all(N_train+1:end);

        %% 1. Centralized Oracle (GLB) - Skipped as requested
        All_results.(prof_name).GLB.metrics(r, :) = NaN;

        %% 2. Generate Uniform IID Client Partitions
        rng(r * 1000 + p, 'twister');
        shuffled = randperm(N_train);
        client_indices = cell(1, M);
        for m = 1:M
            client_indices{m} = shuffled((m - 1) * N_m + 1 : m * N_m);
        end
        
        run_options = options;
        run_options.client_indices = client_indices;

        %% 3. Execute Distributed Estimators
        runner_res = core_runner(X_train, Y_train, X_test, Y_test, label_test, true_mixture, ...
                                 K, M, run_options, models_to_run, 0);

        for m = 1:length(models_to_run)
            mod_name = models_to_run{m};
            All_results.(prof_name).(mod_name).metrics(r, :) = runner_res.(mod_name).metrics;
        end
        
        fprintf('      Run %d finished. DME ARI = %.4f | AAVR ARI = %.4f | GM ARI = %.4f\n', ...
                r, All_results.(prof_name).DME.metrics(r, 8), ...
                All_results.(prof_name).AAVR.metrics(r, 8), ...
                All_results.(prof_name).GM.metrics(r, 8));
    end
end

%% Save Results
out_mat = fullfile(out_dir, 'Result_IID_M16_N100k.mat');
save(out_mat, 'All_results', 'profiles', 'all_models', 'M', 'N_total', 'K', 'd', 'options', '-v7.3');
fprintf('\n==> Successfully saved IID benchmark results to: %s\n', out_mat);

%% Export LaTeX Table 1
table_dir = fullfile(base_dir, 'results', 'tables');
export_latex_two_tables(out_mat, '', table_dir);
fprintf('==> LaTeX Table 1 exported successfully!\n');
