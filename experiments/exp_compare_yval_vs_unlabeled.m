%% EXP_COMPARE_YVAL_VS_UNLABELED
% Independent experiment comparing:
%   1. DME_Tau: Gating fitted via IRLS on posterior responsibilities Tau (requires Y_val)
%   2. DME_Unlabeled: Gating fitted via IRLS on marginal transport plan gatingProb (requires ONLY X_val, purely unlabeled)
% Evaluated on:
%   - Part 1: Simulated Non-IID Balanced Data (M=16, K=5, d=20, N=100k, S=2000)
%   - Part 2: Beijing Real-World Air Quality Data (M=12, K=4, d=19, N_train=315k, N_test=105k, S=2000)

function results = exp_compare_yval_vs_unlabeled()
    base_dir = fileparts(fileparts(mfilename('fullpath')));
    addpath(base_dir);
    addpath(fullfile(base_dir, 'models'));
    addpath(fullfile(base_dir, 'datatools'));
    addpath(fullfile(base_dir, 'evaltools'));
    addpath(fullfile(base_dir, 'stattools'));
    addpath(fullfile(base_dir, 'BEIJING'));

    results = struct();

    fprintf('========================================================================================\n');
    fprintf('  INDEPENDENT EXPERIMENT: DME GATING COMPARISON (Posterior Tau vs Pure Unlabeled OT)    \n');
    fprintf('========================================================================================\n\n');

    % =========================================================================
    % PART 1: SIMULATED DATA (Non-IID Balanced Profile)
    % =========================================================================
    fprintf('----------------------------------------------------------------------------------------\n');
    fprintf('>>> PART 1: Simulated Benchmark (Non-IID Balanced, M=16, K=5, d=20, S=2000)\n');
    fprintf('----------------------------------------------------------------------------------------\n');

    sim_file = fullfile(base_dir, 'data', 'dataset_N100k_K5_d20_balanced.mat');
    if ~exist(sim_file, 'file')
        error('Simulated dataset not found: %s', sim_file);
    end

    loaded = load(sim_file);
    X_mat        = loaded.X_mat;
    Y_mat        = loaded.Y_mat;
    LABEL_mat    = loaded.LABEL_mat;
    true_mixture = loaded.true_mixture;
    K = 5;
    d = 20;
    M = 16;
    S = 2000;
    N_total = size(X_mat, 1);
    N_train = floor(0.8 * N_total);
    N_test  = N_total - N_train;

    % Use replicate 1 for deterministic evaluation
    r = 1;
    X_all     = X_mat(:, :, r);
    Y_all     = Y_mat(:, r);
    label_all = LABEL_mat(:, r);

    X_train     = X_all(1:N_train, :);
    Y_train     = Y_all(1:N_train);
    label_train = label_all(1:N_train);

    X_test      = X_all(N_train+1:end, :);
    Y_test      = Y_all(N_train+1:end);
    label_test  = label_all(N_train+1:end);

    % Client partition (deterministic)
    rng(42, 'twister');
    shuffled = randperm(N_train);
    client_indices = cell(1, M);
    N_m = floor(N_train / M);
    for m = 1:M
        if m < M
            client_indices{m} = shuffled((m-1)*N_m + 1 : m*N_m)';
        else
            client_indices{m} = shuffled((m-1)*N_m + 1 : end)';
        end
    end

    % Base options
    options = get_options('default');
    options.DME_verbose = 0;
    options.verbose     = 0;
    options.S           = S;
    options.sample_size = S;
    options.IRLS_max_iter = 100;
    options.IRLS_threshold = 1e-5;
    options.max_iter       = 50;
    options.nb_EM_runs     = 5;

    % Subsample supporting dataset D_S
    rng(1234, 'twister');
    s_idx = randperm(N_train, S);
    X_val = [ones(S, 1), X_train(s_idx, :)];
    Y_val = Y_train(s_idx);

    % Fit local models once (shared across both variants)
    fprintf('  [1/3] Fitting local MoE models on M = %d machines (shared baseline)...\n', M);
    t_local_start = tic;
    local_fits = cell(1, M);
    local_times = zeros(1, M);
    for m = 1:M
        t_m = tic;
        idx_m = client_indices{m};
        opt_m = options;
        opt_m.verbose = 0;
        local_fits{m} = Global_MixtureOfExperts(X_train(idx_m, :), Y_train(idx_m), K, opt_m);
        local_times(m) = toc(t_m);
    end
    t_local_max = max(local_times);
    fprintf('        Local fitting complete. Max local time: %.2f s (Wall-clock: %.2f s)\n', ...
            t_local_max, toc(t_local_start));

    % 1A. Variant 1: DME with posterior Tau (uses Y_val)
    fprintf('  [2/3] Running Variant 1: DME_Tau (IRLS on posterior Tau using Y_val)...\n');
    opt_tau = options;
    opt_tau.local_estimates     = local_fits;
    opt_tau.local_times         = local_times;
    opt_tau.X_val               = X_val;
    opt_tau.Y_val               = Y_val;
    opt_tau.use_unlabeled_gates = false;

    t_tau = tic;
    fit_tau = Distributed_MixtureOfExperts_Gaussian(X_train, Y_train, K, M, opt_tau);
    time_tau = t_local_max + (toc(t_tau) - t_local_max); % aggregation time
    [t_l, td_tau, ll_tau, mse_tau, rpe_tau, cr_tau, ri_tau, ari_tau, ce_tau] = ...
        compute_metrics(fit_tau, true_mixture, X_test, Y_test, label_test, 0, 'DME');

    % 1B. Variant 2: DME with pure Unlabeled OT (IRLS on marginal gatingProb, NO Y_val)
    fprintf('  [3/3] Running Variant 2: DME_Unlabeled (IRLS on marginal gatingProb, ONLY X_val)...\n');
    opt_unlabeled = options;
    opt_unlabeled.local_estimates     = local_fits;
    opt_unlabeled.local_times         = local_times;
    opt_unlabeled.X_val               = X_val;
    % Crucial: NO Y_val provided or needed
    opt_unlabeled.use_unlabeled_gates = true;

    t_unlabeled = tic;
    fit_unlabeled = Distributed_MixtureOfExperts_Gaussian(X_train, Y_train, K, M, opt_unlabeled);
    time_unlabeled = t_local_max + (toc(t_unlabeled) - t_local_max);
    [t_l, td_un, ll_un, mse_un, rpe_un, cr_un, ri_un, ari_un, ce_un] = ...
        compute_metrics(fit_unlabeled, true_mixture, X_test, Y_test, label_test, 0, 'DME');

    % Summary Part 1
    fprintf('\n  >>> SIMULATED BENCHMARK RESULTS (Non-IID Balanced, N_test = %d):\n', N_test);
    fprintf('  ------------------------------------------------------------------------------------\n');
    fprintf('  %-20s | %-8s | %-10s | %-8s | %-8s | %-8s | %-8s\n', ...
            'Variant', 'RPE', 'Param MSE', 'Trandis', 'Test LL', 'ARI', 'Agg Time');
    fprintf('  ------------------------------------------------------------------------------------\n');
    fprintf('  %-20s | %8.4f | %10.4f | %8.4f | %8.2f | %8.4f | %7.2f s\n', ...
            'DME_Tau (with Y)', rpe_tau, mse_tau, td_tau, ll_tau, ari_tau, fit_tau.learning_time);
    fprintf('  %-20s | %8.4f | %10.4f | %8.4f | %8.2f | %8.4f | %7.2f s\n', ...
            'DME_Unlabeled (X only)', rpe_un, mse_un, td_un, ll_un, ari_un, fit_unlabeled.learning_time);
    fprintf('  ------------------------------------------------------------------------------------\n\n');

    results.simulated.tau       = struct('RPE', rpe_tau, 'MSE', mse_tau, 'Trandis', td_tau, 'LL', ll_tau, 'ARI', ari_tau, 'Time', fit_tau.learning_time);
    results.simulated.unlabeled = struct('RPE', rpe_un, 'MSE', mse_un, 'Trandis', td_un, 'LL', ll_un, 'ARI', ari_un, 'Time', fit_unlabeled.learning_time);

    % =========================================================================
    % PART 2: BEIJING REAL DATA
    % =========================================================================
    fprintf('----------------------------------------------------------------------------------------\n');
    fprintf('>>> PART 2: Beijing Air Quality Benchmark (M=12 Stations, K=4, d=19, S=2000)\n');
    fprintf('----------------------------------------------------------------------------------------\n');

    bj_file = fullfile(base_dir, 'BEIJING', 'beijing_air_quality_processed.mat');
    if ~exist(bj_file, 'file')
        error('Beijing dataset not found: %s', bj_file);
    end

    bj = load(bj_file);
    X_train_cells  = bj.X_train_cells;
    Y_train_cells  = bj.Y_train_cells;
    X_train_pooled = bj.X_train_pooled;
    Y_train_pooled = bj.Y_train_pooled;
    X_test_pooled  = bj.X_test_pooled;
    Y_test_pooled  = bj.Y_test_pooled;

    M_bj = length(X_train_cells);
    K_bj = 4;
    d_bj = size(X_train_pooled, 2);
    N_train_bj = size(X_train_pooled, 1);
    N_test_bj  = size(X_test_pooled, 1);

    % Supporting dataset D_S
    rng(42, 'twister');
    s_idx_bj = randperm(N_train_bj, S);
    X_val_bj = [ones(S, 1), X_train_pooled(s_idx_bj, :)];
    Y_val_bj = Y_train_pooled(s_idx_bj);

    opt_bj = get_options('default');
    opt_bj.verbose         = 0;
    opt_bj.DME_verbose     = 0;
    opt_bj.max_iter        = 20;
    opt_bj.nb_EM_runs      = 5;
    opt_bj.IRLS_max_iter   = 30;
    opt_bj.IRLS_threshold  = 1e-5;

    % Fit local station models once (shared)
    fprintf('  [1/3] Fitting local station models on M = %d stations (shared)...\n', M_bj);
    t_local_bj = tic;
    local_fits_bj = cell(1, M_bj);
    local_times_bj = zeros(1, M_bj);
    for m = 1:M_bj
        t_m = tic;
        opt_m = opt_bj;
        opt_m.verbose = 0;
        local_fits_bj{m} = Global_MixtureOfExperts(X_train_cells{m}, Y_train_cells{m}, K_bj, opt_m);
        local_times_bj(m) = toc(t_m);
    end
    t_local_max_bj = max(local_times_bj);
    fprintf('        Local fitting complete. Max station time: %.2f s (Wall-clock: %.2f s)\n', ...
            t_local_max_bj, toc(t_local_bj));

    % 2A. Variant 1: DME with posterior Tau (uses Y_val)
    fprintf('  [2/3] Running Variant 1: DME_Tau (with Y_val)...\n');
    opt_bj_tau = opt_bj;
    opt_bj_tau.local_estimates     = local_fits_bj;
    opt_bj_tau.local_times         = local_times_bj;
    opt_bj_tau.X_val               = X_val_bj;
    opt_bj_tau.Y_val               = Y_val_bj;
    opt_bj_tau.use_unlabeled_gates = false;

    fit_bj_tau = Distributed_MixtureOfExperts_Gaussian(X_train_pooled, Y_train_pooled, K_bj, M_bj, opt_bj_tau);
    [rmse_tau, mae_tau, rpe_bj_tau, corr_tau] = evaluate_bj(fit_bj_tau, X_test_pooled, Y_test_pooled, K_bj);

    % 2B. Variant 2: DME with pure Unlabeled OT (ONLY X_val)
    fprintf('  [3/3] Running Variant 2: DME_Unlabeled (ONLY X_val, NO Y_val)...\n');
    opt_bj_un = opt_bj;
    opt_bj_un.local_estimates     = local_fits_bj;
    opt_bj_un.local_times         = local_times_bj;
    opt_bj_un.X_val               = X_val_bj;
    opt_bj_un.use_unlabeled_gates = true;

    fit_bj_un = Distributed_MixtureOfExperts_Gaussian(X_train_pooled, Y_train_pooled, K_bj, M_bj, opt_bj_un);
    [rmse_un, mae_un, rpe_bj_un, corr_un] = evaluate_bj(fit_bj_un, X_test_pooled, Y_test_pooled, K_bj);

    % Summary Part 2
    fprintf('\n  >>> BEIJING AIR QUALITY RESULTS (Out-of-Time Test: N_test = %d):\n', N_test_bj);
    fprintf('  ------------------------------------------------------------------------------------\n');
    fprintf('  %-20s | %-12s | %-10s | %-8s | %-10s | %-8s\n', ...
            'Variant', 'RMSE (ug/m3)', 'MAE', 'RPE', 'Corr (r)', 'Agg Time');
    fprintf('  ------------------------------------------------------------------------------------\n');
    fprintf('  %-20s | %12.2f | %10.2f | %8.4f | %10.4f | %7.2f s\n', ...
            'DME_Tau (with Y)', rmse_tau, mae_tau, rpe_bj_tau, corr_tau, fit_bj_tau.learning_time);
    fprintf('  %-20s | %12.2f | %10.2f | %8.4f | %10.4f | %7.2f s\n', ...
            'DME_Unlabeled (X only)', rmse_un, mae_un, rpe_bj_un, corr_un, fit_bj_un.learning_time);
    fprintf('  ------------------------------------------------------------------------------------\n\n');

    results.beijing.tau       = struct('RMSE', rmse_tau, 'MAE', mae_tau, 'RPE', rpe_bj_tau, 'Corr', corr_tau, 'Time', fit_bj_tau.learning_time);
    results.beijing.unlabeled = struct('RMSE', rmse_un, 'MAE', mae_un, 'RPE', rpe_bj_un, 'Corr', corr_un, 'Time', fit_bj_un.learning_time);

    % Save results
    save_path = fullfile(base_dir, 'results', 'compare_yval_vs_unlabeled.mat');
    if ~exist(fileparts(save_path), 'dir'), mkdir(fileparts(save_path)); end
    save(save_path, 'results');
    fprintf('Results saved to: %s\n', save_path);
    fprintf('========================================================================================\n');
end

function [rmse, mae, rpe, corr_val] = evaluate_bj(fit, X, Y, K)
    N = size(X, 1);
    if isfield(fit.param, 'Alpha')
        if size(fit.param.Alpha, 2) == K - 1
            H = [fit.param.alpha0; fit.param.Alpha];
            gatingProb = multinomial_logistic(H, [ones(N, 1) X]);
        else
            H = fit.param.alpha0 + X * fit.param.Alpha;
            maxm = max(H, [], 2);
            H = H - maxm;
            gatingProb = exp(H) ./ sum(exp(H), 2);
        end
    else
        gatingProb = ones(N, K) / K;
    end
    gatingProb = max(gatingProb, 1e-12);
    Y_experts = fit.param.beta0 + X * fit.param.Beta;
    Y_pred    = sum(gatingProb .* Y_experts, 2);
    residuals = Y - Y_pred;
    rmse      = sqrt(mean(residuals.^2));
    mae       = mean(abs(residuals));
    var_total = sum((Y - mean(Y)).^2);
    rpe       = sum(residuals.^2) / var_total;
    c_mat     = corrcoef(Y, Y_pred);
    corr_val  = c_mat(1, 2);
end
