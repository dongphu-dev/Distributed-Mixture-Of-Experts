%% EXP_SCALING_SAMPLESIZE_N
% Evaluates candidate distributed MoE models against varying total dataset size N:
%   N in {100k, 300k, 1M} with fixed machine count M = 16 (K = 5, d = 20).
%
% Candidate Estimators:
%   1. DME : One-shot Optimal Transport aggregation
%   2. MED : Coordinate-wise Median aggregation
%   3. WAVR : Weighted Parameter Averaging
%   4. AAVR : Hungarian Matching on KL Divergence + Parameter Averaging
%   5. GM : Runnalls (2007) Greedy Merging from MK down to K + OT/IRLS gating
%   6. FED : Canonical Multi-Round Federated Averaging
%
% Dual-Mode Support:
%   - macOS (DEV mode): N in [100k, 300k], 3 runs per N
%   - Linux/VPS (PROD mode): N in [100k, 300k, 1M], 100 runs per N
%
% Results saved to:
%   results/scaling_samplesize_N/Result_Candidates_M16_N{N_str}.mat

clear; clc;
fprintf('========================================================================\n');
fprintf('  EXPERIMENT: Scaling Sample Size (N in {100k, 300k, 1M}, fixed M = 16)\n');
fprintf('========================================================================\n');

% Set up paths
base_dir = fileparts(fileparts(mfilename('fullpath')));
addpath(base_dir);
addpath(fullfile(base_dir, 'models'));
addpath(fullfile(base_dir, 'datatools'));
addpath(fullfile(base_dir, 'evaltools'));
addpath(fullfile(base_dir, 'stattools'));
addpath(fullfile(base_dir, 'validationtools'));
addpath(fullfile(base_dir, 'experiments'));

out_dir = fullfile(base_dir, 'results', 'scaling_samplesize_N');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

% Options
options = get_options('default');

% Environment configuration
fixed_M = 16;
if ismac
    N_list = [100000, 300000];     % DEV mode on Mac
    max_runs_override = 3;
    fprintf('Detected macOS: DEV mode active (N in [100k, 300k], max %d runs)\n', max_runs_override);
else
    N_list = [100000, 300000, 1000000]; % Full production on VPS
    max_runs_override = [];
    fprintf('Detected Linux/VPS: PROD mode active (N in [100k, 300k, 1M], 100 runs)\n');
end

models_to_run = {'DME', 'MED', 'WAVR', 'AAVR', 'GM', 'FED'};

%% Loop over Dataset Sizes N
for n_val = N_list
    if n_val == 100000,     N_str = '100k';
    elseif n_val == 300000, N_str = '300k';
    elseif n_val == 1000000, N_str = '1M';
    else, N_str = num2str(n_val);
    end
    
    data_file = fullfile(base_dir, 'data', sprintf('dataset_N%s_K5_d20.mat', N_str));
    if ~exist(data_file, 'file')
        warning('Dataset file not found: %s. Skipping N = %s.', data_file, N_str);
        continue;
    end
    
    fprintf('\n------------------------------------------------------------------------\n');
    fprintf('  Starting Benchmark for N = %s (M = %d machines)\n', N_str, fixed_M);
    fprintf('------------------------------------------------------------------------\n');
    
    loaded = load(data_file);
    X_mat     = loaded.X_mat;
    Y_mat     = loaded.Y_mat;
    LABEL_mat = loaded.LABEL_mat;
    true_mixture = loaded.true_mixture;
    K = loaded.K;
    d = loaded.d;
    total_datasets = size(X_mat, 3);
    
    if isempty(max_runs_override)
        num_runs = total_datasets;
    else
        num_runs = min(max_runs_override, total_datasets);
    end
    
    temp_run_results = cell(1, num_runs);
    
    parfor run = 1:num_runs
        fprintf('  --> [N=%s, M=%d] Processing Dataset Run %d/%d...\n', N_str, fixed_M, run, num_runs);
        
        X = X_mat(:, :, run);
        Y = Y_mat(:, run);
        true_labels = LABEL_mat(:, run);
        n_obs = size(X, 1);
        
        % 80/20 train/test split
        N_train   = floor(0.8 * n_obs);
        X_train   = X(1:N_train, :);
        Y_train   = Y(1:N_train);
        X_test    = X(N_train+1:end, :);
        Y_test    = Y(N_train+1:end);
        labels_test = true_labels(N_train+1:end);
        
        res = core_runner(X_train, Y_train, X_test, Y_test, labels_test, ...
                          true_mixture, K, fixed_M, options, models_to_run, 0);
        
        temp_run_results{run} = res;
    end
    
    % Unpack and aggregate results across runs
    STORED_METRICS = struct();
    STORED_PARAMS  = struct();
    for m_idx = 1:length(models_to_run)
        m_name = models_to_run{m_idx};
        STORED_METRICS.(m_name) = zeros(num_runs, 9);
        STORED_PARAMS.(m_name)  = cell(1, num_runs);
    end
    
    for run = 1:num_runs
        res = temp_run_results{run};
        for m_idx = 1:length(models_to_run)
            m_name = models_to_run{m_idx};
            STORED_METRICS.(m_name)(run, :) = res.(m_name).metrics;
            STORED_PARAMS.(m_name){run}     = res.(m_name).param;
        end
    end
    
    % Save checkpoint file
    save_file = fullfile(out_dir, sprintf('Result_Candidates_M%d_N%s.mat', fixed_M, N_str));
    save(save_file, ...
        'STORED_METRICS', 'STORED_PARAMS', ...
        'fixed_M', 'n_val', 'N_str', 'K', 'd', 'num_runs', 'options', '-v7.3');
    fprintf('==> Saved results for N = %s to: %s\n', N_str, save_file);
    
    % Print Summary Comparison Table
    fprintf('\n========================================================================================\n');
    fprintf('  BENCHMARK SUMMARY (N = %s, M = %d machines, %d runs)\n', N_str, fixed_M, num_runs);
    fprintf('========================================================================================\n');
    fprintf('%-8s | %-12s | %-12s | %-14s | %-12s | %-10s | %-10s | %-10s\n', ...
        'Model', 'Time(s)', 'Trandis', 'Loglik', 'MSE_param', 'RPE', 'ARI', 'ClustErr%');
    fprintf('----------------------------------------------------------------------------------------\n');
    
    for m_idx = 1:length(models_to_run)
        m_name = models_to_run{m_idx};
        mat = STORED_METRICS.(m_name);
        mu  = mean(mat, 1);
        sd  = std(mat, 0, 1);
        
        fprintf('%-8s | %5.2f±%-4.2f | %5.2f±%-4.2f | %7.1f±%-4.1f | %5.4f±%-4.4f | %5.3f±%-3.3f | %5.3f±%-3.3f | %5.2f±%-3.2f\n', ...
            m_name, ...
            mu(1), sd(1), mu(2), sd(2), mu(3), sd(3), mu(4), sd(4), mu(5), sd(5), mu(8), sd(8), mu(9), sd(9));
    end
    fprintf('========================================================================================\n\n');
end

fprintf('\nSample size scaling experiments completed successfully!\n');
