%% EXP_NON_IID_BENCHMARK
% Master Experiment 2: Evaluates distributed MoE estimators under Non-IID data partitioning
% (Dirichlet cluster skew alpha_dir = 0.5) across three expert proportion profiles:
%   1. Balanced:            pi_k = 0.20
%   2. Moderately Imbalanced: 0.10 <= pi_k <= 0.30
%   3. Highly Imbalanced:   pi_min = 0.02
%
% Evaluated Models (6):
%   - Global oracle (GLB) [Reused from IID or computed once]
%   - Proposed (DME)
%   - Greedy merging (GM)
%   - Aligned averaging (AAVR)
%   - Federated averaging (FED)
%   - Naive averaging (WAVR)
%
% Setup:
%   M = 16 machines, N = 100,000 points per dataset (80% train / 20% test)
%   Non-IID: Dirichlet allocation (alpha = 0.5) with minimum support floor (n_min = 15)
%   DEV mode (macOS): 2 runs | PROD mode (Linux): 100 runs
%
% Output:
%   results/benchmark_non_iid/Result_NonIID_M16_N100k.mat
%   results/tables/table_exp2_non_iid_benchmark.tex

clear; clc;
fprintf('========================================================================\n');
fprintf('  MASTER EXPERIMENT 2: Non-IID Data Partitioning Benchmark (M = 16, N = 100k)\n');
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

out_dir = fullfile(base_dir, 'results', 'benchmark_non_iid');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

K = 5;
d = 20;
M = 16;
N_total = 100000;
N_train = floor(0.8 * N_total);
N_test  = N_total - N_train;

shift_tau = 2.0; % Covariate shift domain bandwidth (controls spatial client specialization)

profiles = {'balanced', 'moderate', 'severe'};
models_to_run = {'DME', 'GM', 'AAVR', 'FED', 'WAVR'};
all_models    = [{'GLB'}, models_to_run];

% Check for precomputed GLB oracle in IID results
iid_mat_file = fullfile(base_dir, 'results', 'benchmark_iid', 'Result_IID_M16_N100k.mat');
has_cached_glb = exist(iid_mat_file, 'file');
if has_cached_glb
    fprintf('Found cached GLB oracle from IID benchmark: %s\n', iid_mat_file);
    iid_cached = load(iid_mat_file);
else
    fprintf('No cached GLB found. GLB oracle will be computed.\n');
end

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
    fprintf('  Processing Profile [%d/3]: %s (Non-IID Covariate Shift tau = %.1f)\n', p, prof_name, shift_tau);
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
        fprintf('  --> Run [%d/%d] (%s, Non-IID)...\n', r, num_runs, prof_name);
        
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

        %% 2. Generate Non-IID Covariate Shift Client Partitions
        client_indices = partition_covariate_shift(X_train, M, K, shift_tau, r * 5000 + p);
        
        run_options = options;
        run_options.client_indices = client_indices;

        %% 3. Execute Distributed Estimators
        runner_res = core_runner(X_train, Y_train, X_test, Y_test, label_test, true_mixture, ...
                                 K, M, run_options, models_to_run, 0);

        for m = 1:length(models_to_run)
            mod_name = models_to_run{m};
            All_results.(prof_name).(mod_name).metrics(r, :) = runner_res.(mod_name).metrics;
        end
        
        fprintf('      Run %d finished. DME ARI = %.4f | AAVR ARI = %.4f | FED ARI = %.4f\n', ...
                r, All_results.(prof_name).DME.metrics(r, 8), ...
                All_results.(prof_name).AAVR.metrics(r, 8), ...
                All_results.(prof_name).FED.metrics(r, 8));
    end
end

%% Save Results
out_mat = fullfile(out_dir, 'Result_NonIID_M16_N100k.mat');
save(out_mat, 'All_results', 'profiles', 'all_models', 'M', 'N_total', 'K', 'd', 'shift_tau', 'options', '-v7.3');
fprintf('\n==> Successfully saved Non-IID benchmark results to: %s\n', out_mat);

%% Export LaTeX Table 2 (and update Table 1 if present)
table_dir = fullfile(base_dir, 'results', 'tables');
export_latex_two_tables(iid_mat_file, out_mat, table_dir);
fprintf('==> LaTeX Table 2 exported successfully!\n');

%% Helper Function: Covariate Shift Partitioning via Spatial Domain Affinity
function client_indices = partition_covariate_shift(X_train, M, K, shift_tau, seed)
    if nargin >= 5, rng(seed, 'twister'); end
    [N_train, d] = size(X_train);

    % Identify K representative domain centers in feature space
    sample_idx = randperm(N_train, min(2500, N_train));
    [~, centers] = kmeans(X_train(sample_idx, :), K, 'Replicates', 3);

    % Distribute M clients across the K spatial domains with localized jitter
    client_centers = zeros(M, d);
    for m = 1:M
        k_dom = mod(m - 1, K) + 1;
        client_centers(m, :) = centers(k_dom, :) + randn(1, d) * 0.20;
    end

    % Compute squared distance of each point to each client domain center
    D2 = zeros(N_train, M);
    for m = 1:M
        diff_m = X_train - repmat(client_centers(m, :), N_train, 1);
        D2(:, m) = sum(diff_m.^2, 2);
    end

    % Softmax domain affinity with temperature shift_tau
    logits = -D2 / (2 * shift_tau^2);
    logits = logits - max(logits, [], 2);
    probs = exp(logits) ./ sum(exp(logits), 2);

    % Assign points probabilistically to clients
    client_indices = cell(1, M);
    for m = 1:M, client_indices{m} = []; end

    for i = 1:N_train
        m_chosen = find(mnrnd(1, probs(i, :)) == 1, 1);
        client_indices{m_chosen} = [client_indices{m_chosen}; i];
    end

    % Fallback guarantee: ensure every client has sufficient support
    for m = 1:M
        if length(client_indices{m}) < d + 5
            extra_idx = randperm(N_train, d + 5)';
            client_indices{m} = unique([client_indices{m}; extra_idx]);
        end
        % Shuffle indices within client
        client_indices{m} = client_indices{m}(randperm(length(client_indices{m})));
    end
end
