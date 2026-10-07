%% EXP_GLOBAL_CENTRALIZED_BASELINE
% Computes the centralized global MoE baseline (Global_MixtureOfExperts)
% on the full pooled dataset (no distribution).
%
% This serves as the theoretical upper bound / gold standard for:
%   - Trandis (transportation distance)
%   - Test Log-Likelihood
%   - Parameter MSE
%   - Relative Prediction Error (RPE)
%   - Adjusted Rand Index (ARI)
%   - Learning Time (benchmarked on single machine)
%
% Results saved to:
%   results/global_baseline/Result_Global_N{N_str}.mat

clear; clc;
fprintf('========================================================================\n');
fprintf('  EXPERIMENT: Centralized Global MoE Baseline (Gold Standard)\n');
fprintf('========================================================================\n');

% Set up paths
base_dir = fileparts(fileparts(mfilename('fullpath')));
addpath(base_dir);
addpath(fullfile(base_dir, 'models'));
addpath(fullfile(base_dir, 'datatools'));
addpath(fullfile(base_dir, 'evaltools'));
addpath(fullfile(base_dir, 'stattools'));
addpath(fullfile(base_dir, 'validationtools'));

out_dir = fullfile(base_dir, 'results', 'global_baseline');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

% Options
options = get_options('default');

% Datasets to evaluate
if ismac
    N_list = [100000]; % Fast dev run on Mac
else
    N_list = [100000, 300000, 1000000]; % Full production on VPS
end

for n_val = N_list
    if n_val == 100000,     N_str = '100k';
    elseif n_val == 300000, N_str = '300k';
    elseif n_val == 1000000, N_str = '1M';
    else, N_str = num2str(n_val);
    end
    
    data_file = fullfile(base_dir, 'data', sprintf('dataset_N%s_K5_d20.mat', N_str));
    if ~exist(data_file, 'file')
        warning('Dataset file %s not found. Skipping N = %s.', data_file, N_str);
        continue;
    end
    
    fprintf('\n--> Loading dataset: %s\n', data_file);
    loaded = load(data_file);
    X_mat     = loaded.X_mat;
    Y_mat     = loaded.Y_mat;
    LABEL_mat = loaded.LABEL_mat;
    true_mixture = loaded.true_mixture;
    K = loaded.K;
    d = loaded.d;
    total_datasets = size(X_mat, 3);
    
    if ismac
        num_runs = min(3, total_datasets);
    else
        num_runs = total_datasets;
    end
    
    fprintf('Running %d Monte Carlo evaluations for N = %s (Centralized Global MoE)...\n', num_runs, N_str);
    
    % Storage
    temp_GLB_STORED_estimated_Beta   = cell(1, num_runs);
    temp_GLB_STORED_estimated_beta0  = cell(1, num_runs);
    temp_GLB_STORED_estimated_Alpha  = cell(1, num_runs);
    temp_GLB_STORED_estimated_alpha0 = cell(1, num_runs);
    temp_GLB_STORED_estimated_sigma2 = cell(1, num_runs);
    temp_GLB_STORED_EVALUATIONS_TEST = cell(1, num_runs);
    temp_GLB_STORED_FITS             = cell(1, num_runs);
    metrics_mat                      = zeros(num_runs, 9);
    
    for run = 1:num_runs
        fprintf('  [Run %d/%d] Centralized Global MoE fitting...\n', run, num_runs);
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
        
        % Fit Global MoE
        tic;
        fit = Global_MixtureOfExperts(X_train, Y_train, K, options);
        fit.learning_time = toc;
        
        % Evaluate metrics on unseen test set
        [learning_time, trandis, loglik, mse_param, RPE_test, corr_test, RI_test, ARI_test, ClustErr_test] = ...
            compute_metrics(fit, true_mixture, X_test, Y_test, labels_test, 1, 'GLB');
        
        m_vec = [learning_time, trandis, loglik, mse_param, RPE_test, corr_test, RI_test, ARI_test, ClustErr_test];
        metrics_mat(run, :) = m_vec;
        
        temp_GLB_STORED_estimated_Beta{run}   = fit.param.Beta;
        temp_GLB_STORED_estimated_beta0{run}  = fit.param.beta0;
        temp_GLB_STORED_estimated_Alpha{run}  = fit.param.Alpha;
        temp_GLB_STORED_estimated_alpha0{run} = fit.param.alpha0;
        temp_GLB_STORED_estimated_sigma2{run} = fit.param.sigma2;
        temp_GLB_STORED_EVALUATIONS_TEST{run} = m_vec;
        
        % Compact fit
        if isfield(fit, 'X_val'), fit.X_val = []; end
        if isfield(fit, 'Y_val'), fit.Y_val = []; end
        temp_GLB_STORED_FITS{run}             = fit;
    end
    
    GLB_STORED_estimated_Beta   = temp_GLB_STORED_estimated_Beta;
    GLB_STORED_estimated_beta0  = temp_GLB_STORED_estimated_beta0;
    GLB_STORED_estimated_Alpha  = temp_GLB_STORED_estimated_Alpha;
    GLB_STORED_estimated_alpha0 = temp_GLB_STORED_estimated_alpha0;
    GLB_STORED_estimated_sigma2 = temp_GLB_STORED_estimated_sigma2;
    GLB_STORED_EVALUATIONS_TEST = temp_GLB_STORED_EVALUATIONS_TEST;
    GLB_STORED_FITS             = temp_GLB_STORED_FITS;
    
    % Save
    save_file = fullfile(out_dir, sprintf('Result_Global_N%s.mat', N_str));
    save(save_file, ...
        'GLB_STORED_EVALUATIONS_TEST', ...
        'metrics_mat', ...
        'GLB_STORED_estimated_Beta', ...
        'GLB_STORED_estimated_beta0', ...
        'GLB_STORED_estimated_Alpha', ...
        'GLB_STORED_estimated_alpha0', ...
        'GLB_STORED_estimated_sigma2', ...
        'GLB_STORED_FITS', ...
        'options', 'K', 'd', 'n_val', '-v7.3');
    fprintf('==> Saved baseline results to: %s\n', save_file);
    
    % Print Summary Table
    fprintf('\n--- [GLOBAL BASELINE SUMMARY: N = %s] ---\n', N_str);
    fprintf('%-12s %-12s %-12s %-12s %-12s %-12s %-12s\n', ...
        'Time(s)', 'Trandis', 'Loglik', 'MSE_param', 'RPE', 'ARI', 'ClustErr%');
    mean_m = mean(metrics_mat, 1);
    std_m  = std(metrics_mat, 0, 1);
    fprintf('%6.2f±%-4.2f %6.2f±%-4.2f %7.1f±%-4.1f %6.4f±%-4.4f %6.3f±%-4.3f %6.3f±%-4.3f %6.2f±%-4.2f\n', ...
        mean_m(1), std_m(1), mean_m(2), std_m(2), mean_m(3), std_m(3), ...
        mean_m(4), std_m(4), mean_m(5), std_m(5), mean_m(8), std_m(8), mean_m(9), std_m(9));
    fprintf('------------------------------------------------------------------------\n');
end
