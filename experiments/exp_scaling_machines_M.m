%% EXP_SCALING_MACHINES_M
% Primary experiment for paper Figure 5 reproduction and extension:
% Evaluates 6 candidate distributed MoE models against varying machine counts M:
%   M in {4, 16, 64, 128} on fixed dataset N = 100k (K = 5, d = 20).
%
% Candidate Estimators:
%   1. DME : One-shot Optimal Transport aggregation (proposed)
%   2. MED : Coordinate-wise Median aggregation
%   3. WAVR : Weighted Parameter Averaging
%   4. AAVR : Hungarian Matching on KL Divergence + Parameter Averaging
%   5. GM : Runnalls (2007) Greedy Merging from MK down to K + OT/IRLS gating
%   6. FED : Canonical Multi-Round Federated Averaging
%
% Dual-Mode Support:
%   - macOS (DEV mode): M in [4, 16], 3 runs, rapid verification (< 2 min)
%   - Linux/VPS (PROD mode): M in [4, 16, 64, 128], 100 runs, full publication benchmark
%
% Results saved to:
%   results/scaling_machines_M/Result_Candidates_N100k_M{M}.mat

clear;
fprintf('========================================================================\n');
fprintf('  EXPERIMENT: Scaling Distributed Machines (M in {4, 16, 64, 128})\n');
fprintf('  Evaluation of 6 Candidate Estimators\n');
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

out_dir = fullfile(base_dir, 'results', 'scaling_machines_M');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

% Options
options = get_options('default');

% Environment configuration
if ismac
    M_list = [16];              % Fast dev run on Mac
    max_runs_override = 4;
    fprintf('Detected macOS: DEV mode active (M in [%s], max %d runs)\n', ...
        num2str(M_list), max_runs_override);
else
    M_list = [4, 16, 64, 128];     % Full production benchmark on VPS
    max_runs_override = [];
    fprintf('Detected Linux/VPS: PROD mode active (M in [%s], 100 runs)\n', ...
        num2str(M_list));
end

% Load Dataset
N_str = '100k';
data_file = fullfile(base_dir, 'data', sprintf('dataset_N%s_K5_d20.mat', N_str));
if ~exist(data_file, 'file')
    error('Dataset file not found: %s. Run data_generator_benchmark.m first!', data_file);
end

fprintf('Loading benchmark dataset: %s...\n', data_file);
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

models_to_run = {'DME', 'MED', 'WAVR', 'AAVR', 'GM', 'FED'};
metric_labels = {'Time', 'Trandis', 'Loglik', 'mse_param', 'RPE', 'Corr', 'RI', 'ARI', 'ClustErr'};

%% Loop over Machine Counts M
for M = M_list
    fprintf('\n------------------------------------------------------------------------\n');
    fprintf('  Starting Benchmark for M = %d machines (N = %s, %d runs)\n', M, N_str, num_runs);
    fprintf('------------------------------------------------------------------------\n');
    
    temp_run_results = cell(1, num_runs);
    
    parfor run = 1:num_runs
        fprintf('  --> [M=%d] Processing Dataset Run %d/%d...\n', M, run, num_runs);
        
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
        
        % Run all 6 candidate models via unified core_runner
        res = core_runner(X_train, Y_train, X_test, Y_test, labels_test, ...
                          true_mixture, K, M, options, models_to_run, 0);
        
        temp_run_results{run} = res;
    end
    
    % Unpack and aggregate results across runs
    STORED_METRICS = struct();
    STORED_PARAMS  = struct();
    
    % Initialize matrices
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
    
    % Legacy compatibility variables
    DME_STORED_EVALUATIONS_TEST = num2cell(STORED_METRICS.DME, 2)';
    MED_STORED_EVALUATIONS_TEST = num2cell(STORED_METRICS.MED, 2)';
    WAVR_STORED_EVALUATIONS_TEST = num2cell(STORED_METRICS.WAVR, 2)';
    AAVR_STORED_EVALUATIONS_TEST = num2cell(STORED_METRICS.AAVR, 2)';
    GM_STORED_EVALUATIONS_TEST = num2cell(STORED_METRICS.GM, 2)';
    FED_STORED_EVALUATIONS_TEST = num2cell(STORED_METRICS.FED, 2)';
    
    % Save checkpoint file for this M
    save_file = fullfile(out_dir, sprintf('Result_Candidates_N%s_M%d.mat', N_str, M));
    save(save_file, ...
        'STORED_METRICS', 'STORED_PARAMS', ...
        'DME_STORED_EVALUATIONS_TEST', 'MED_STORED_EVALUATIONS_TEST', ...
        'AVR_STORED_EVALUATIONS_TEST', 'AAVR_STORED_EVALUATIONS_TEST', ...
        'GM_STORED_EVALUATIONS_TEST', 'FED_STORED_EVALUATIONS_TEST', ...
        'M', 'K', 'd', 'N_str', 'num_runs', 'options', '-v7.3');
    fprintf('==> Saved results for M = %d to: %s\n', M, save_file);
    
    % Print Summary Comparison Table for this M
    fprintf('\n========================================================================================\n');
    fprintf('  BENCHMARK SUMMARY (M = %d machines, N = %s, %d runs)\n', M, N_str, num_runs);
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
            mu(1), sd(1), ...  % Time
            mu(2), sd(2), ...  % Trandis
            mu(3), sd(3), ...  % Loglik
            mu(4), sd(4), ...  % MSE_param
            mu(5), sd(5), ...  % RPE
            mu(8), sd(8), ...  % ARI
            mu(9), sd(9));     % ClustErr
    end
    fprintf('========================================================================================\n\n');
end

fprintf('\nAll machine scaling experiments completed successfully!\n');
