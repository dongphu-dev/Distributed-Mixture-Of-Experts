function data_generator_benchmark(N_list, force_new_gt)
% DATA_GENERATOR_BENCHMARK
% Generates reproducible benchmark datasets for the Distributed-MoE evaluation.
%
% Outputs:
%   data/ground_truth_param_K5_d20.mat : Ground truth model parameters (fixed seed)
%   data/dataset_N100k_K5_d20.mat      : N = 100,000 (DEV: 3 runs, PROD: 100 runs)
%   data/dataset_N300k_K5_d20.mat      : N = 300,000
%   data/dataset_N1M_K5_d20.mat        : N = 1,000,000
%
% Configuration:
%   - K = 5 experts, d = 20 covariates
%   - beta0 = [0 0 0 0 0]: zero intercept forces expert separation purely by slope Beta,
%     creating realistic label switching to evaluate Hungarian alignment (AAVR) and OT (DME).

    current_dir = fileparts(mfilename('fullpath'));
    base_dir    = fullfile(current_dir, '..');
    addpath(base_dir);
    addpath(fullfile(base_dir, 'datatools'));

    % Auto-detect environment mode
    if ismac
        DEFAULT_NUM_DATASETS = 3;   % Fast dev/test on MacBook
        fprintf('Detected macOS: Running in DEV mode (%d datasets per N)\n', DEFAULT_NUM_DATASETS);
    else
        DEFAULT_NUM_DATASETS = 100; % Full production benchmark on Linux VPS
        fprintf('Detected Linux/VPS: Running in PROD mode (%d datasets per N)\n', DEFAULT_NUM_DATASETS);
    end

    d = 20;
    K = 5;
    num_datasets = DEFAULT_NUM_DATASETS;

    if nargin < 2 || isempty(force_new_gt)
        force_new_gt = false;
    end
    gt_seed = 28; % Seed 28 yields balanced clusters: [18.6%, 18.9%, 24.8%, 19.8%, 17.9%]

    %% 1. Load or Generate Ground Truth Parameters
    gt_file = fullfile(current_dir, 'ground_truth_param_K5_d20.mat');

    if exist(gt_file, 'file') && ~force_new_gt
        fprintf('Loading ground truth parameters from: %s\n', gt_file);
        load(gt_file, 'param', 'MU');
    else
        fprintf('Generating NEW balanced ground truth parameters (seed = %d)...\n', gt_seed);
        rng(gt_seed, 'twister');
        
        % Zero intercepts to induce label switching for Hungarian alignment & OT
        param.beta0  = zeros(1, K);
        
        % Distinct regression slope vectors
        param.Beta   = randi([-10, 10], d, K);
        for k = 2:K
            while norm(param.Beta(:, k) - param.Beta(:, k-1)) < 2.0
                param.Beta(:, k) = randi([-10, 10], d, 1);
            end
        end
        
        param.sigma2 = [3.0, 3.0, 3.0, 3.0, 3.0]; % Expert noise variances
        
        % Balanced gating parameters:
        % Scaled slopes prevent exponential softmax saturation, keeping groups balanced around 20%
        param.alpha0 = [zeros(1, K - 1), 0];
        param.Alpha  = [randn(d, K - 1) * 0.45, zeros(d, 1)];
        
        % Component-specific mean vectors for X
        for k = 1:K
            MU.(sprintf('MU%d', k)) = randi([-2, 2], 1, d);
        end
        
        save(gt_file, 'param', 'MU', 'd', 'K', 'gt_seed');
        fprintf('Saved ground truth parameters to %s\n', gt_file);
    end

    %% 2. Generate Datasets (DEV: 100k, 300k | PROD: 100k, 300k, 1M)
    if nargin < 1 || isempty(N_list)
        if ismac
            N_list = [100000, 300000];
        else
            N_list = [100000, 300000, 1000000];
        end
    end

    for n = N_list
        if n == 100000
            N_str = '100k';
        elseif n == 300000
            N_str = '300k';
        elseif n == 1000000
            N_str = '1M';
        else
            N_str = num2str(n);
        end
        
        out_file = fullfile(current_dir, sprintf('dataset_N%s_K%d_d%d.mat', N_str, K, d));
        fprintf('\n--> Generating %d datasets for N = %s (size: %d x %d)...\n', num_datasets, N_str, n, d);
        
        X_mat     = zeros(n, d, num_datasets);
        Y_mat     = zeros(n, num_datasets);
        LABEL_mat = zeros(n, num_datasets);
        
        rng(100, 'twister'); % Reproducible dataset sequence
        for i = 1:num_datasets
            [X, Y, true_mixture] = simulate_data(n, d, K, '', param, MU);
            X_mat(:, :, i)   = X;
            Y_mat(:, i)     = Y;
            LABEL_mat(:, i) = true_mixture.true_labels;
            if mod(i, 10) == 0 || i == num_datasets
                fprintf('  Dataset %d/%d generated\n', i, num_datasets);
            end
        end
        
        save(out_file, 'X_mat', 'Y_mat', 'LABEL_mat', 'true_mixture', 'param', 'MU', 'n', 'd', 'K', '-v7.3');
        fprintf('==> Successfully saved: %s (size: %.2f MB)\n', out_file, dir(out_file).bytes / 1e6);
    end

    fprintf('\nData generation complete!\n');
end
