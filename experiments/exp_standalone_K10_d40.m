function results = exp_standalone_K10_d40(imbalance_spec, K_input, d_input, N_input, M_input, num_runs_input)
%% EXP_STANDALONE_K10_D40 (Generalized Standalone MoE Benchmark)
% Fully self-contained, isolated benchmark experiment for Distributed Mixture-of-Experts
% supporting ARBITRARY K, ARBITRARY d, and AUTOMATIC IMBALANCED expert subpopulations.
%
% Key Features:
%   1. Automatic Imbalance for ANY K:
%      - 'severe' (or 'rare') : Minority subpopulation with pi_min = 2% (or min(0.02, 0.2/K)),
%                               followed by linear arithmetic ramp across all K components.
%      - 'moderate'           : Moderate imbalance with pi_max / pi_min = 3 (matches paper).
%      - 'balanced'           : Perfectly uniform proportions pi_k = 1/K.
%      - Custom ratio r       : Passing any numeric scalar r > 1 (e.g. 10) sets pi_max/pi_min = r.
%      - Custom vector props  : Passing any 1xK positive vector directly sets target proportions.
%   2. Bayes Conjugate Gating (LDA Equivalence):
%      Constructs closed-form ground-truth Softmax gating parameters (alpha0, Alpha)
%      from cluster centers and prior probabilities, guaranteeing that empirical
%      sample proportions strictly follow the specified imbalance profile.
%   3. 100% Standalone & Non-interfering:
%      Generates all synthetic data in memory; never modifies or overwrites any
%      existing project files or benchmark datasets.
%   4. Adaptive Parallel Computing Strategy:
%      - num_runs <= 4: INNER parallelization (parfor across M local machines, fast DEV test).
%      - num_runs > 4 : OUTER parallelization (parfor across Monte Carlo datasets, PROD batch).
%   5. Unique Timestamped Saving:
%      results/standalone_imbalance/Result_K{K}_d{d}_{profile}_N{N}_M{M}_{timestamp}.mat
%
% Usage:
%   exp_standalone_K10_d40;                           % Default: severe imbalance, K=6, d=30
%   exp_standalone_K10_d40('moderate');               % Moderate imbalance, K=6, d=30
%   exp_standalone_K10_d40('severe', 10, 40);         % Severe imbalance, K=10, d=40
%   exp_standalone_K10_d40(15, 8, 30);                % Custom ratio = 15, K=8, d=30
%   exp_standalone_K10_d40('severe', 6, 30, 50000, 8, 2); % Custom sample size & machines

    %% 1. Path Setup & Parameter Configuration
    current_dir = fileparts(mfilename('fullpath'));
    base_dir    = fileparts(current_dir);

    addpath(base_dir);
    addpath(fullfile(base_dir, 'models'));
    addpath(fullfile(base_dir, 'datatools'));
    addpath(fullfile(base_dir, 'evaltools'));
    addpath(fullfile(base_dir, 'stattools'));
    addpath(fullfile(base_dir, 'experiments'));

    % 1a. Imbalance profile (default: 'severe')
    if nargin >= 1 && ~isempty(imbalance_spec)
        imbalance_profile = imbalance_spec;
    else
        imbalance_profile = 'severe';
    end

    % 1b. Number of experts K (default: 6)
    if nargin >= 2 && ~isempty(K_input)
        K = K_input;
    else
        K = 6;
    end

    % 1c. Covariate dimension d (default: 30)
    if nargin >= 3 && ~isempty(d_input)
        d = d_input;
    else
        d = 30;
    end

    % 1d. Platform detection for N, M, and num_runs
    if ismac
        default_N = 20000;
        default_M = 8;
        default_runs = 1;
        env_mode = 'macOS (DEV mode - fast verification)';
    else
        default_N = 100000;
        default_M = 16;
        default_runs = 10;
        env_mode = 'Linux/VPS (PROD mode - full Monte Carlo)';
    end

    if nargin >= 4 && ~isempty(N_input)
        N = N_input;
    else
        N = default_N;
    end

    if nargin >= 5 && ~isempty(M_input)
        M = M_input;
    else
        M = default_M;
    end

    if nargin >= 6 && ~isempty(num_runs_input)
        num_runs = num_runs_input;
    else
        num_runs = default_runs;
    end

    if ischar(imbalance_profile) || isstring(imbalance_profile)
        prof_name = char(imbalance_profile);
    elseif isnumeric(imbalance_profile) && isscalar(imbalance_profile)
        prof_name = sprintf('ratio%.1f', imbalance_profile);
    else
        prof_name = 'custom_vector';
    end

    fprintf('========================================================================\n');
    fprintf('  STANDALONE MoE BENCHMARK: K = %d Experts, d = %d Covariates\n', K, d);
    fprintf('  Imbalance Profile: [%s]\n', prof_name);
    fprintf('  Environment: %s\n', env_mode);
    fprintf('  Settings: N = %d, M = %d machines, Runs = %d\n', N, M, num_runs);
    fprintf('========================================================================\n\n');

    % Create dedicated output folder
    out_dir = fullfile(base_dir, 'results', 'standalone_imbalance');
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    % Execution options
    options = get_options('default');
    options.DME_verbose   = 0;
    options.verbose       = 0;
    options.tol           = 1e-5;
    options.max_iter      = 100;
    options.nb_EM_runs    = 2;
    options.DME_tries     = 2;
    options.DME_maxiter   = 60;
    options.S             = min(2000, floor(0.8 * N));
    options.sample_size   = options.S;
    options.IRLS_max_iter = 100;
    options.FedAvg_rounds = 5;

    models_to_run = {'DME', 'AAVR', 'GM', 'MED', 'WAVR', 'FED'};
    n_models = length(models_to_run);

    % Metric containers: [Time, Trandis, Loglik, MSE_param, RPE, Corr, RI, ARI, ClustErr, Trandis_fW]
    num_metrics = 10;
    STORED_METRICS = struct();
    for m = 1:n_models
        STORED_METRICS.(models_to_run{m}) = zeros(num_runs, num_metrics);
    end

    % Ensure parallel pool is active upfront so the user experiences zero startup latency
    poolobj = gcp('nocreate');
    if isempty(poolobj)
        if ismac
            parpool('Processes', min(M, 8));
        else
            parpool;
        end
    end

    %% 2. Adaptive Monte Carlo Benchmark Execution
    % Only parallelize outer runs when num_runs > 4.
    % For num_runs <= 4 (e.g. 1 to 4 runs), keep outer loop sequential and
    % parallelize across M local machines (inner parfor) with real-time logging.
    if num_runs <= 4
        % -------------------------------------------------------------
        % INNER PARALLELIZATION: parfor across M local machines
        % Ideal for num_runs <= 4 (near M-fold speedup on local machines)
        % -------------------------------------------------------------
        options.parallel_machines = true;
        fprintf('Parallel Strategy: INNER parallelization enabled (parfor across M = %d local machines, num_runs = %d <= 4).\n\n', M, num_runs);

        for run = 1:num_runs
            fprintf('--> [Run %d/%d] Generating synthetic dataset (K=%d, d=%d, Profile=%s, N=%d)...\n', ...
                run, num_runs, K, d, prof_name, N);

            run_seed = 1000 + run * 79;
            [X, Y, true_mixture] = generate_synthetic_data_imbalance(N, d, K, imbalance_profile, run_seed);

            % 80/20 train/test split
            n_train = floor(0.8 * N);
            X_train = X(1:n_train, :);
            Y_train = Y(1:n_train);

            X_test  = X(n_train+1:end, :);
            Y_test  = Y(n_train+1:end);
            labels_test = true_mixture.true_labels(n_train+1:end);

            fprintf('    Dataset partitioned: %d train, %d test observations.\n', n_train, N - n_train);
            fprintf('    Executing candidate distributed models on M = %d machines (in parallel)...\n', M);

            % Run models via core_runner (verbose = 1 to display real-time model execution progress)
            res = core_runner(X_train, Y_train, X_test, Y_test, labels_test, ...
                              true_mixture, K, M, options, models_to_run, 1);

            for m = 1:n_models
                m_name = models_to_run{m};
                STORED_METRICS.(m_name)(run, :) = res.(m_name).metrics;
            end
            fprintf('    [Run %d/%d] Completed successfully.\n\n', run, num_runs);
        end
    else
        % -------------------------------------------------------------
        % OUTER PARALLELIZATION: parfor across Monte Carlo datasets
        % Active only when num_runs > 4 (high throughput batch on VPS)
        % -------------------------------------------------------------
        options.parallel_machines = false;
        fprintf('Parallel Strategy: OUTER parallelization enabled (parfor across %d Monte Carlo runs, num_runs > 4).\n\n', num_runs);

        temp_results = cell(1, num_runs);
        parfor run = 1:num_runs
            run_seed = 1000 + run * 79;
            [X, Y, true_mixture] = generate_synthetic_data_imbalance(N, d, K, imbalance_profile, run_seed);

            n_train = floor(0.8 * N);
            X_train = X(1:n_train, :);
            Y_train = Y(1:n_train);

            X_test  = X(n_train+1:end, :);
            Y_test  = Y(n_train+1:end);
            labels_test = true_mixture.true_labels(n_train+1:end);

            res = core_runner(X_train, Y_train, X_test, Y_test, labels_test, ...
                              true_mixture, K, M, options, models_to_run, 0);
            temp_results{run} = res;
        end

        for run = 1:num_runs
            res = temp_results{run};
            for m = 1:n_models
                m_name = models_to_run{m};
                STORED_METRICS.(m_name)(run, :) = res.(m_name).metrics;
            end
            fprintf('    [Run %d/%d] Aggregated.\n', run, num_runs);
        end
    end

    %% 3. Display Summary Comparison Table
    fprintf('\n===================================================================================================\n');
    fprintf('  BENCHMARK SUMMARY TABLE: K = %d, d = %d, Profile = [%s], N = %d, M = %d machines (%d runs)\n', ...
        K, d, prof_name, N, M, num_runs);
    fprintf('===================================================================================================\n');
    fprintf('%-8s | %-12s | %-12s | %-14s | %-12s | %-10s | %-10s | %-10s\n', ...
        'Model', 'Time (s)', 'Trandis', 'Test Loglik', 'MSE_param', 'Test RPE', 'Test ARI', 'ClustErr %');
    fprintf('---------------------------------------------------------------------------------------------------\n');

    for m = 1:n_models
        m_name = models_to_run{m};
        mat = STORED_METRICS.(m_name);
        mu  = mean(mat, 1);
        sd  = std(mat, 0, 1);

        % Metric indices: 1=Time, 2=Trandis, 3=Loglik, 4=MSE, 5=RPE, 8=ARI, 9=ClustErr
        fprintf('%-8s | %5.2f±%-5.2f | %5.2f±%-5.2f | %7.1f±%-5.1f | %6.4f±%-5.4f | %5.3f±%-4.3f | %5.3f±%-4.3f | %5.2f±%-4.2f\n', ...
            m_name, ...
            mu(1), sd(1), ...
            mu(2), sd(2), ...
            mu(3), sd(3), ...
            mu(4), sd(4), ...
            mu(5), sd(5), ...
            mu(8), sd(8), ...
            mu(9), sd(9));
    end
    fprintf('===================================================================================================\n\n');

    %% 4. Save Timestamped Results (Zero Collisions)
    timestamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
    save_filename = sprintf('Result_K%d_d%d_%s_N%d_M%d_%s.mat', K, d, prof_name, N, M, timestamp);
    save_filepath = fullfile(out_dir, save_filename);

    save(save_filepath, 'STORED_METRICS', 'K', 'd', 'N', 'M', 'num_runs', 'imbalance_profile', 'prof_name', 'options', '-v7.3');
    fprintf('==> Results saved independently to:\n    %s\n\n', save_filepath);

    results = STORED_METRICS;
end


%% ========================================================================
%  LOCAL HELPER: Generalized Synthetic Data Generator for Arbitrary K and Imbalance
%  ========================================================================
function [X, Y, true_mixture] = generate_synthetic_data_imbalance(n, d, K, profile, seed)
    if nargin >= 5 && ~isempty(seed)
        rng(seed, 'twister');
    end

    %% 1. Derive Exact Target Proportions for ANY Arbitrary K
    if ischar(profile) || isstring(profile)
        prof_str = lower(char(profile));
        switch prof_str
            case {'balanced', 'uniform'}
                target_props = ones(1, K) / K;

            case {'moderate', 'medium'}
                % Linear ramp with ratio max/min = 3 (for K=5: [0.10, 0.15, 0.20, 0.25, 0.30])
                w = linspace(1.0, 3.0, K);
                target_props = w / sum(w);

            case {'severe', 'rare', 'imbalance', 'imbalanced'}
                % Minority cluster has pi_min = 2% (or min(0.02, 0.2/K)), then linear ramp
                pi_min = min(0.02, 0.2 / K);
                delta  = 2.0 * (1.0 - K * pi_min) / (K * (K - 1));
                target_props = pi_min + delta * (0:K-1);
                target_props = target_props / sum(target_props);

            otherwise
                warning('Unknown profile string: %s. Falling back to severe imbalance.', prof_str);
                pi_min = min(0.02, 0.2 / K);
                delta  = 2.0 * (1.0 - K * pi_min) / (K * (K - 1));
                target_props = pi_min + delta * (0:K-1);
                target_props = target_props / sum(target_props);
        end
    elseif isnumeric(profile)
        if isscalar(profile) && profile > 1
            % User passed a ratio r = pi_max / pi_min
            r = double(profile);
            w = linspace(1.0, r, K);
            target_props = w / sum(w);
        elseif length(profile) == K
            % User passed an explicit 1xK proportion vector
            target_props = profile(:)' / sum(profile);
        else
            error('Numeric profile must be either a scalar ratio > 1 or a 1x%d vector.', K);
        end
    else
        target_props = ones(1, K) / K;
    end

    %% 2. K-Mode Covariates & Bayes Conjugate Gating (LDA Equivalence)
    % Cluster centers MU_mat (K x d) with moderate separation
    MU_mat = randn(K, d);
    MU_mat = MU_mat ./ sqrt(sum(MU_mat.^2, 2)) * 2.5;

    % Exact Bayes Conjugate Gating:
    % P(Z=k | x) = softmax(alpha0_k + x' * Alpha_k) where:
    %   Alpha(:, k) = mu_k - mu_K
    %   alpha0(k)   = -0.5 * ||mu_k||^2 + log(pi_k) - (-0.5 * ||mu_K||^2 + log(pi_K))
    prof_Alpha = (MU_mat - repmat(MU_mat(K, :), K, 1))'; % d x K
    norm_mu2   = sum(MU_mat.^2, 2)';                     % 1 x K
    prof_alpha0 = -0.5 * norm_mu2 + log(target_props) - (-0.5 * norm_mu2(K) + log(target_props(K)));

    %% 3. Sample Latent Indicators Z and Covariates X
    cum_p = cumsum(target_props);
    r_vals = rand(n, 1);
    labels = sum(r_vals > cum_p, 2) + 1;
    labels = min(labels, K);

    X = zeros(n, d);
    for k = 1:K
        idx_k = (labels == k);
        n_k = sum(idx_k);
        if n_k > 0
            % Covariates drawn around center mu_k with standard normal dispersion
            X(idx_k, :) = repmat(MU_mat(k, :), n_k, 1) + randn(n_k, d);
        end
    end

    % Verify empirical proportions
    emp_counts = histcounts(labels, 0.5:1:K+0.5);
    emp_props  = emp_counts / n;
    fprintf('    Target proportions:    %s\n', mat2str(round(target_props, 3)));
    fprintf('    Empirical proportions: %s\n', mat2str(round(emp_props, 3)));
    fprintf('    Empirical counts:      %s\n', mat2str(emp_counts));

    %% 4. Generate Ground Truth Expert Slopes & Variances
    % Zero intercepts to force model separation purely by regression slopes
    beta0 = zeros(1, K);

    % Regression slopes Beta (d x K) with guaranteed separation >= 5.0
    Beta = randi([-10, 10], d, K);
    for k = 2:K
        while true
            min_dist = inf;
            for prev_k = 1:k-1
                dist = norm(Beta(:, k) - Beta(:, prev_k));
                if dist < min_dist
                    min_dist = dist;
                end
            end
            if min_dist >= 5.0
                break;
            end
            Beta(:, k) = randi([-10, 10], d, 1);
        end
    end

    % Noise variances spread across components
    sigma2 = linspace(2.5, 3.5, K);

    %% 5. Generate Scalar Responses Y
    Y = zeros(n, 1);
    for k = 1:K
        idx_k = (labels == k);
        n_k = sum(idx_k);
        if n_k > 0
            mu_y = beta0(k) + X(idx_k, :) * Beta(:, k);
            Y(idx_k) = mu_y + sqrt(sigma2(k)) * randn(n_k, 1);
        end
    end

    %% 6. Pack true_mixture struct
    true_mixture = struct();
    true_mixture.experts     = [beta0; Beta];
    true_mixture.gates       = [prof_alpha0; prof_Alpha];
    true_mixture.variances   = sigma2;
    true_mixture.weights     = target_props;
    true_mixture.true_labels = labels;
    true_mixture.MU          = MU_mat;
    true_mixture.d           = d;
    true_mixture.K           = K;
end
