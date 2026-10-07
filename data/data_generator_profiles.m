function data_generator_profiles(num_datasets, N_total, d, K)
%% DATA_GENERATOR_PROFILES
% Generates reproducible benchmark datasets for the Distributed-MoE evaluation
% across three expert proportion profiles:
%   1. Balanced:            target proportions [0.20, 0.20, 0.20, 0.20, 0.20]
%   2. Moderately Imbalanced: target proportions [0.10, 0.15, 0.20, 0.25, 0.30]
%   3. Highly Imbalanced:   target proportions [0.02, 0.08, 0.20, 0.30, 0.40]
%
% Controlled Scientific Principle:
%   All 3 profiles share the EXACT SAME regression slopes Beta and noise variances sigma2,
%   loaded from data/ground_truth_param_K5_d20.mat.
%
% Usage:
%   data_generator_profiles;                      % Auto-detect DEV (3 runs) vs PROD (100 runs), N = 100k
%   data_generator_profiles(num_datasets);         % Explicit number of runs, N = 100,000
%   data_generator_profiles(num_datasets, N_total); % Custom N_total (e.g., 300000, 1000000)
%   data_generator_profiles(num_datasets, N_total, d, K);

    current_dir = fileparts(mfilename('fullpath'));
    base_dir    = fullfile(current_dir, '..');
    addpath(base_dir);
    addpath(fullfile(base_dir, 'datatools'));

    if nargin < 4 || isempty(K)
        K = 5;
    end
    if nargin < 3 || isempty(d)
        d = 20;
    end
    if nargin < 2 || isempty(N_total)
        N_total = 100000; % Default benchmark sample size
    end

    if nargin < 1 || isempty(num_datasets)
        if ismac
            num_datasets = 5;   % Fast DEV mode on macOS (5 runs)
            fprintf('Detected macOS: Running in DEV mode (%d datasets per profile, N = %d)\n', num_datasets, N_total);
        else
            num_datasets = 50; % Full PROD benchmark on Linux VPS
            fprintf('Detected Linux/VPS: Running in PROD mode (%d datasets per profile, N = %d)\n', num_datasets, N_total);
        end
    end

    %% 1. Load Ground Truth Regression Slopes and Variances
    gt_file = fullfile(current_dir, 'ground_truth_param_K5_d20.mat');
    if ~exist(gt_file, 'file')
        fprintf('Ground truth file not found. Generating ground_truth_param_K5_d20.mat...\n');
        data_generator_benchmark(1, false);
    end
    loaded_gt = load(gt_file);
    param     = loaded_gt.param;
    
    % Shared regression parameters across all profiles
    shared_beta0  = param.beta0;  % Zero intercepts (1 x K)
    shared_Beta   = param.Beta;   % Fixed regression slopes (d x K)
    shared_sigma2 = param.sigma2; % Fixed noise variances [3, 3, 3, 3, 3]

    %% 2. 5-Mode Covariates Design with Controlled Overlap (Separation ~ 2.5 sigma)
    % Scale 0.28 provides moderate, realistic overlap: modes have distinct peaks,
    % but clustering X alone has error ~35-45% (necessitating MoE regression feedback).
    scale_mu = 0.28;
    MU_mat = zeros(K, d);
    if isfield(loaded_gt, 'MU') && isstruct(loaded_gt.MU)
        for k = 1:K
            MU_mat(k, :) = loaded_gt.MU.(sprintf('MU%d', k)) * scale_mu;
        end
    else
        rng(28, 'twister');
        MU_mat = randn(K, d);
        MU_mat = MU_mat ./ sqrt(sum(MU_mat.^2, 2)) * 2.5;
    end

    %% 3. Profile Definitions
    profiles = {
        'balanced', [0.20, 0.20, 0.20, 0.20, 0.20], 'Balanced (pi_k = 0.20)';
        'moderate', [0.10, 0.15, 0.20, 0.25, 0.30], 'Moderately Imbalanced (0.10 <= pi_k <= 0.30)';
        'severe',   [0.02, 0.08, 0.20, 0.30, 0.40], 'Highly Imbalanced (pi_min = 0.02)'
    };

    fprintf('\n========================================================================\n');
    fprintf('  GENERATING BENCHMARK DATASETS (5-Mode GMM Covariates, N = %d, K = %d, d = %d)\n', N_total, K, d);
    fprintf('========================================================================\n');

    for p = 1:size(profiles, 1)
        prof_name  = profiles{p, 1};
        prof_props = profiles{p, 2};
        prof_desc  = profiles{p, 3};

        N_str = format_N_str(N_total);
        out_file = fullfile(current_dir, sprintf('dataset_N%s_K%d_d%d_%s.mat', N_str, K, d, prof_name));
        fprintf('\n--> Profile [%s]: %s\n', prof_name, prof_desc);
        fprintf('    Target Proportions: %s\n', mat2str(prof_props, 3));
        fprintf('    Generating %d Monte Carlo datasets...\n', num_datasets);

        % Bayes Conjugate Gating (Linear Discriminant Analysis equivalence)
        % For X | Z=k ~ N(mu_k, I_d):
        %   Alpha(:, k) = mu_k - mu_K
        %   alpha0(k)   = -0.5 * ||mu_k||^2 + log(pi_k) - (-0.5 * ||mu_K||^2 + log(pi_K))
        prof_Alpha = MU_mat' - repmat(MU_mat(K, :)', 1, K);
        norm_mu2 = sum(MU_mat.^2, 2)';
        prof_alpha0 = -0.5 * norm_mu2 + log(prof_props) - (-0.5 * norm_mu2(K) + log(prof_props(K)));

        prof_param.beta0  = shared_beta0;
        prof_param.Beta   = shared_Beta;
        prof_param.sigma2 = shared_sigma2;
        prof_param.alpha0 = prof_alpha0;
        prof_param.Alpha  = prof_Alpha;

        true_mixture.experts     = [prof_param.beta0; prof_param.Beta];
        true_mixture.gates       = [prof_param.alpha0; prof_param.Alpha];
        true_mixture.variances   = prof_param.sigma2;
        true_mixture.weights     = prof_props;
        true_mixture.MU          = MU_mat;

        X_mat     = zeros(N_total, d, num_datasets);
        Y_mat     = zeros(N_total, num_datasets);
        LABEL_mat = zeros(N_total, num_datasets);

        rng(100 + p * 1000, 'twister'); % Reproducible seed per profile

        for i = 1:num_datasets
            % 1. Sample Latent Component Assignments Z ~ Categorical(prof_props)
            cum_p = cumsum(prof_props);
            r_vals = rand(N_total, 1);
            labels = sum(r_vals > cum_p, 2) + 1;
            labels = min(labels, K);

            % 2. Covariates X ~ N(mu_Z, I_d) (Multimodal Gaussian Mixture)
            X = zeros(N_total, d);
            Y = zeros(N_total, 1);
            for k = 1:K
                idx_k = (labels == k);
                n_k = sum(idx_k);
                if n_k > 0
                    X(idx_k, :) = repmat(MU_mat(k, :), n_k, 1) + randn(n_k, d);
                    
                    % 3. Responses Y ~ N(beta0_k + X * Beta_k, sigma2_k)
                    mu_y = shared_beta0(k) + X(idx_k, :) * shared_Beta(:, k);
                    Y(idx_k) = mu_y + sqrt(shared_sigma2(k)) * randn(n_k, 1);
                end
            end

            X_mat(:, :, i)   = X;
            Y_mat(:, i)     = Y;
            LABEL_mat(:, i) = labels;
        end

        % Empirical validation of cluster proportions on Dataset 1
        emp_counts = histcounts(LABEL_mat(:, 1), 0.5:1:K+0.5);
        emp_props  = emp_counts / N_total;
        fprintf('    Empirical proportions (Run 1): %s\n', mat2str(round(emp_props, 3)));

        % Check tolerance
        max_dev = max(abs(emp_props - prof_props));
        if max_dev > 0.025
            warning('Empirical proportions deviate by %.3f from target!', max_dev);
        else
            fprintf('    Validation Passed: Max deviation from target = %.4f (<= 0.025)\n', max_dev);
        end

        % Save dataset file
        save(out_file, 'X_mat', 'Y_mat', 'LABEL_mat', 'true_mixture', 'prof_param', ...
             'prof_name', 'prof_props', 'N_total', 'd', 'K', 'num_datasets', '-v7.3');
        fprintf('    Saved to: %s (Size: %.2f MB)\n', out_file, dir(out_file).bytes / 1e6);
    end

    fprintf('\nAll 3 profile datasets generated successfully!\n');
end
