function results = exp_dme_vs_gm_stress()
% EXP_DME_VS_GM_STRESS
% Agglomerative Stress-Test Benchmark: DME vs GM vs Centralized Oracle
%
% Configuration:
%   - M = 16 machines, each observing K_m = 3 experts (L = 48 source components)
%   - Target: K = 5 global experts (requires 43 greedy merge steps in GM)
%   - Small local samples: N_m = 500 samples per machine (total N = 8,000)
%   - Severe subpopulation imbalance: Expert 1 is a rare subpopulation (5%),
%     observed only on Machines 1 and 2. Expert 2 is dominant (40%).
%   - Equal noise variances sigma2 = [3.0, 3.0, 3.0, 3.0, 3.0]
%
% Evaluated Models:
%   1. DME_Kmeans : Standalone DME with multi-start k-means initialization
%   2. DME_Warm   : DME initialized from GM (demonstrates MM error correction)
%   3. GM        : Runnalls (2007) greedy pairwise merging (43 merges)
%   4. GLB        : Centralized Oracle trained on pooled data
%   5. WAVR, MED, AAVR, FED : Marked INCOMPATIBLE due to heterogeneous K_m
%
% Output:
%   results/stress_Km/Result_Stress_M16_L48.mat

    fprintf('========================================================================================\n');
    fprintf('  STRESS-TEST BENCHMARK: DME vs GM (M = 16, L = 48 -> K = 5, Rare Expert 5%%)          \n');
    fprintf('========================================================================================\n');

    is_dev = ismac;
    if is_dev
        num_runs = 2;
        fprintf('Mode: DEV (MacBook detected, %d runs)\n', num_runs);
    else
        num_runs = 10;
        fprintf('Mode: PROD (Linux VPS detected, %d runs)\n', num_runs);
    end

    current_dir = fileparts(mfilename('fullpath'));
    base_dir    = fullfile(current_dir, '..');
    addpath(fullfile(base_dir, 'models'));
    addpath(fullfile(base_dir, 'stattools'));
    addpath(fullfile(base_dir, 'evaltools'));
    addpath(base_dir);

    % Ensure results directory exists
    res_dir = fullfile(base_dir, 'results', 'stress_Km');
    if ~exist(res_dir, 'dir'), mkdir(res_dir); end

    % Load stress dataset
    data_file = fullfile(base_dir, 'data', 'dataset_stress_M16_L48.mat');
    if ~exist(data_file, 'file')
        fprintf('Dataset not found. Generating stress dataset first...\n');
        addpath(fullfile(base_dir, 'data'));
        data_generator_stress_Km(123);
    end

    data = load(data_file);
    X_cells = data.X_cells;
    Y_cells = data.Y_cells;
    X_test  = data.X_test;
    Y_test  = data.Y_test;
    Z_test  = data.Z_test;
    param_gt = data.param;
    config  = data.config;

    M = config.M;
    K_vec = config.K_vec;
    K_target = config.K_target;
    d = config.d;

    fprintf('Configuration: M = %d, L = %d, K_target = %d, N_m = %d, N_test = %d\n', ...
            M, config.L, K_target, config.N_per_machine, config.N_test);

    % Ground truth mixture struct for standard metrics
    if length(param_gt.alpha0) == K_target - 1
        alpha0_full = [param_gt.alpha0(:)', 0];
        Alpha_full  = [param_gt.Alpha, zeros(d, 1)];
    else
        alpha0_full = param_gt.alpha0(:)';
        Alpha_full  = param_gt.Alpha;
    end
    true_mixture.experts   = [param_gt.beta0; param_gt.Beta];
    true_mixture.variances = param_gt.sigma2;
    true_mixture.gates     = [alpha0_full; Alpha_full];
    true_mixture.weights   = ones(1, K_target) / K_target;

    % Pooled data for centralized baseline
    X_pooled = [];
    Y_pooled = [];
    for m = 1:M
        X_pooled = [X_pooled; X_cells{m}]; %#ok<AGROW>
        Y_pooled = [Y_pooled; Y_cells{m}]; %#ok<AGROW>
    end

    % Base options
    options = get_options('default');
    options.nb_EM_runs = 2;
    options.max_iter   = 40;
    options.DME_maxiter = 100;
    options.DME_tol    = 1e-4;
    options.verbose    = 0;
    options.DME_verbose = 0;

    % Storage for metrics
    % Standard 9 metrics: Time, Trandis, Loglik, MSE_param, RPE, Corr, RI, ARI, ClustErr
    metrics_DME_Kmeans = zeros(num_runs, 9);
    metrics_DME_Warm   = zeros(num_runs, 9);
    metrics_GM        = zeros(num_runs, 9);
    metrics_GLB        = zeros(num_runs, 9);

    % Stress-specific diagnostic metrics:
    % [Rare_Beta_MSE, Rare_Sigma_MSE, Mean_Variance, Rc_Transport_Objective]
    diag_DME_Kmeans = zeros(num_runs, 4);
    diag_DME_Warm   = zeros(num_runs, 4);
    diag_GM        = zeros(num_runs, 4);
    diag_GLB        = zeros(num_runs, 4);

    last_fits = struct();

    % Benchmark Loop
    for run = 1:num_runs
        fprintf('\n--- Run %d/%d ---\n', run, num_runs);

        % 1. Local fits on each machine
        fprintf('  Fitting local EM models on %d machines (K_m = 3, N_m = %d)...\n', M, config.N_per_machine);
        local_estimates = cell(1, M);
        local_times     = zeros(1, M);
        for m = 1:M
            tic_m = tic;
            local_est = MixtureOfExperts(X_cells{m}, Y_cells{m}, K_vec(m), options);
            local_times(m) = toc(tic_m);
            local_estimates{m} = local_est;
        end

        % Validation support D_S (sample S = 1000 from pooled data)
        S = 1000;
        idx_val = randperm(size(X_pooled, 1), min(S, size(X_pooled, 1)));
        X_val   = [ones(length(idx_val), 1), X_pooled(idx_val, :)];
        Y_val   = Y_pooled(idx_val);

        dme_temp.local_estimates = local_estimates;
        dme_temp.local_times     = local_times;
        dme_temp.d               = d;
        dme_temp.X_val           = X_val;
        dme_temp.Y_val           = Y_val;
        dme_temp.n               = size(X_pooled, 1);

        % (a) GM: Runnalls (2007) Greedy Merging (43 sequential merges)
        fprintf('  Fitting GM (43 sequential greedy merges)...\n');
        tic_gm = tic;
        fit_gm = Greedy_MixtureOfExperts_Hetero(dme_temp, K_target, options);
        t_gm   = toc(tic_gm);
        if ~isfield(fit_gm, 'learning_time') || fit_gm.learning_time == 0
            fit_gm.learning_time = max(local_times) + t_gm;
        end
        [t_l, td, ll, mse, rpe, cr, ri, ari, ce] = compute_metrics(fit_gm, true_mixture, X_test, Y_test, Z_test, 0, 'GM');
        metrics_GM(run, :) = [t_l, td, ll, mse, rpe, cr, ri, ari, ce];
        
        [r_beta, r_sig, mean_var] = evaluate_rare_expert_and_variance(fit_gm, param_gt, X_test);
        rc_gm = fit_gm.transportdis;
        diag_GM(run, :) = [r_beta, r_sig, mean_var, rc_gm];
        fprintf('    GM: Loglik = %.1f, ARI = %.3f, Rare_Beta_MSE = %.4f, Mean_Var = %.2f (True=3.0), R_c = %.2f\n', ...
                ll, ari, r_beta, mean_var, rc_gm);

        % (b) DME_Kmeans: Standalone DME (Multi-start k-means, independent of GM)
        fprintf('  Fitting DME (Independent multi-start k-means)...\n');
        opt_dme_km = options;
        opt_dme_km.init_mode = 'kmeans';
        opt_dme_km.DME_tries = 3;
        tic_km = tic;
        fit_dme_km = Distributed_MixtureOfExperts_Hetero(dme_temp, K_target, opt_dme_km);
        t_km = toc(tic_km);
        if ~isfield(fit_dme_km, 'learning_time') || fit_dme_km.learning_time == 0
            fit_dme_km.learning_time = max(local_times) + t_km;
        end
        [t_l, td, ll, mse, rpe, cr, ri, ari, ce] = compute_metrics(fit_dme_km, true_mixture, X_test, Y_test, Z_test, 0, 'DME_Kmeans');
        metrics_DME_Kmeans(run, :) = [t_l, td, ll, mse, rpe, cr, ri, ari, ce];

        [r_beta, r_sig, mean_var] = evaluate_rare_expert_and_variance(fit_dme_km, param_gt, X_test);
        rc_dme_km = fit_dme_km.transportdis;
        diag_DME_Kmeans(run, :) = [r_beta, r_sig, mean_var, rc_dme_km];
        fprintf('    DME (Standalone): Loglik = %.1f, ARI = %.3f, Rare_Beta_MSE = %.4f, Mean_Var = %.2f, R_c = %.2f\n', ...
                ll, ari, r_beta, mean_var, rc_dme_km);

        % (c) DME_Warm: DME Warm-Started from GM (MM Error Correction)
        fprintf('  Fitting DME (Warm-started from GM -> MM descent)...\n');
        opt_dme_warm = options;
        opt_dme_warm.init_mode = 'grd';
        opt_dme_warm.DME_tries = 2;
        tic_warm = tic;
        fit_dme_warm = Distributed_MixtureOfExperts_Hetero(dme_temp, K_target, opt_dme_warm);
        t_warm = toc(tic_warm);
        if ~isfield(fit_dme_warm, 'learning_time') || fit_dme_warm.learning_time == 0
            fit_dme_warm.learning_time = max(local_times) + t_warm;
        end
        [t_l, td, ll, mse, rpe, cr, ri, ari, ce] = compute_metrics(fit_dme_warm, true_mixture, X_test, Y_test, Z_test, 0, 'DME_Warm');
        metrics_DME_Warm(run, :) = [t_l, td, ll, mse, rpe, cr, ri, ari, ce];

        [r_beta, r_sig, mean_var] = evaluate_rare_expert_and_variance(fit_dme_warm, param_gt, X_test);
        rc_dme_warm = fit_dme_warm.transportdis;
        diag_DME_Warm(run, :) = [r_beta, r_sig, mean_var, rc_dme_warm];
        fprintf('    DME (Warm-start): Loglik = %.1f, ARI = %.3f, Rare_Beta_MSE = %.4f, Mean_Var = %.2f, R_c = %.2f\n', ...
                ll, ari, r_beta, mean_var, rc_dme_warm);

        % (d) GLB: Centralized Oracle Baseline
        fprintf('  Fitting GLB (Centralized Oracle on pooled data)...\n');
        glb_opt = options;
        glb_opt.nb_EM_runs = 2;
        glb_opt.max_iter   = 30;
        tic_glb = tic;
        fit_glb = Global_MixtureOfExperts(X_pooled, Y_pooled, K_target, glb_opt);
        t_glb = toc(tic_glb);
        fit_glb.learning_time = t_glb;
        [t_l, td, ll, mse, rpe, cr, ri, ari, ce] = compute_metrics(fit_glb, true_mixture, X_test, Y_test, Z_test, 0, 'GLB');
        metrics_GLB(run, :) = [t_l, td, ll, mse, rpe, cr, ri, ari, ce];

        [r_beta, r_sig, mean_var] = evaluate_rare_expert_and_variance(fit_glb, param_gt, X_test);
        diag_GLB(run, :) = [r_beta, r_sig, mean_var, NaN];
        fprintf('    GLB (Oracle): Loglik = %.1f, ARI = %.3f, Rare_Beta_MSE = %.4f, Mean_Var = %.2f\n', ...
                ll, ari, r_beta, mean_var);

        if run == num_runs
            last_fits.GM        = fit_gm;
            last_fits.DME_Kmeans = fit_dme_km;
            last_fits.DME_Warm   = fit_dme_warm;
            last_fits.GLB        = fit_glb;
        end
    end

    % Assemble results struct
    models = {'GLB', 'DME_Kmeans', 'DME_Warm', 'GM', 'WAVR', 'MED', 'AAVR', 'FED'};
    status = {'ORACLE', 'COMPLETED', 'COMPLETED', 'COMPLETED', ...
              'INCOMPATIBLE', 'INCOMPATIBLE', 'INCOMPATIBLE', 'INCOMPATIBLE'};

    results.models   = models;
    results.status   = status;
    results.config   = config;
    results.num_runs = num_runs;

    results.metrics_raw.GLB        = metrics_GLB;
    results.metrics_raw.DME_Kmeans = metrics_DME_Kmeans;
    results.metrics_raw.DME_Warm   = metrics_DME_Warm;
    results.metrics_raw.GM        = metrics_GM;

    results.diag_raw.GLB        = diag_GLB;
    results.diag_raw.DME_Kmeans = diag_DME_Kmeans;
    results.diag_raw.DME_Warm   = diag_DME_Warm;
    results.diag_raw.GM        = diag_GM;

    results.metrics_mean.GLB        = mean(metrics_GLB, 1);
    results.metrics_std.GLB         = std(metrics_GLB, 0, 1);
    results.metrics_mean.DME_Kmeans = mean(metrics_DME_Kmeans, 1);
    results.metrics_std.DME_Kmeans  = std(metrics_DME_Kmeans, 0, 1);
    results.metrics_mean.DME_Warm   = mean(metrics_DME_Warm, 1);
    results.metrics_std.DME_Warm    = std(metrics_DME_Warm, 0, 1);
    results.metrics_mean.GM        = mean(metrics_GM, 1);
    results.metrics_std.GM         = std(metrics_GM, 0, 1);

    results.diag_mean.GLB        = mean(diag_GLB, 1);
    results.diag_std.GLB         = std(diag_GLB, 0, 1);
    results.diag_mean.DME_Kmeans = mean(diag_DME_Kmeans, 1);
    results.diag_std.DME_Kmeans  = std(diag_DME_Kmeans, 0, 1);
    results.diag_mean.DME_Warm   = mean(diag_DME_Warm, 1);
    results.diag_std.DME_Warm    = std(diag_DME_Warm, 0, 1);
    results.diag_mean.GM        = mean(diag_GM, 1);
    results.diag_std.GM         = std(diag_GM, 0, 1);

    results.last_fits = last_fits;

    % Save results
    out_res = fullfile(res_dir, 'Result_Stress_M16_L48.mat');
    save(out_res, 'results', '-v7.3');
    fprintf('\n==> Successfully saved benchmark results to: %s\n', out_res);

    % Display formatted comparison table
    print_stress_summary_table(results);

end


function [rare_beta_mse, rare_sig_mse, mean_variance] = evaluate_rare_expert_and_variance(fit, param_gt, X_test)
% Evaluates parameter recovery specifically for the rare subpopulation (Expert 1)
% using Hungarian matching between estimated experts and ground truth.

    K = length(param_gt.sigma2);
    N_test = size(X_test, 1);
    X_aug = [ones(N_test, 1), X_test];

    gt_experts = [param_gt.beta0; param_gt.Beta]; % (d+1) x K
    gt_sigma2  = param_gt.sigma2;                 % 1 x K

    est_experts = [fit.param.beta0; fit.param.Beta]; % (d+1) x K
    est_sigma2  = fit.param.sigma2;                  % 1 x K

    % Build K x K pairwise KL distance cost matrix
    C = zeros(K, K);
    for i = 1:K
        mu_true = X_aug * gt_experts(:, i);
        sig_true = gt_sigma2(i);
        for j = 1:K
            mu_est = X_aug * est_experts(:, j);
            sig_est = max(est_sigma2(j), 1e-4);
            diff_mu = sum((mu_true - mu_est).^2) / N_test;
            kl = 0.5 * (log(sig_est / sig_true) + (sig_true + diff_mu) / sig_est - 1);
            C(i, j) = max(kl, 0);
        end
    end

    % Solve linear assignment: best_p(i) is the estimated expert index matching true expert i
    best_p = solve_hungarian_exact(C, K);

    % True Expert 1 is the rare subpopulation
    est_match_for_rare = best_p(1);

    rare_beta_mse = mean((gt_experts(2:end, 1) - est_experts(2:end, est_match_for_rare)).^2);
    rare_sig_mse  = (gt_sigma2(1) - est_sigma2(est_match_for_rare))^2;
    mean_variance = mean(est_sigma2);

end


function best_p = solve_hungarian_exact(C, K)
% Exact brute force permutation matching for K <= 7
    all_perms = perms(1:K);
    n_perms = size(all_perms, 1);
    min_cost = inf;
    best_p = 1:K;
    for ip = 1:n_perms
        p = all_perms(ip, :);
        cost = 0;
        for k = 1:K
            cost = cost + C(k, p(k));
        end
        if cost < min_cost
            min_cost = cost;
            best_p = p;
        end
    end
end


function print_stress_summary_table(results)
    fprintf('\n');
    fprintf('=======================================================================================================================================\n');
    fprintf('  STRESS-TEST SUMMARY: DME vs GM vs ORACLE (M = 16, L = 48 -> K = 5, Rare Expert 5%%)                                                \n');
    fprintf('=======================================================================================================================================\n');
    fprintf('Model            | Loglik            | Param MSE         | Rare Beta MSE   | Trandis            | ARI             | Mean Var (True=3.0) \n');
    fprintf('---------------------------------------------------------------------------------------------------------------------------------------\n');

    eval_models = {'GLB', 'DME_Kmeans', 'DME_Warm', 'GM'};
    display_names = {'GLB (Oracle)     ', 'DME (Standalone) ', 'DME (Warm-GM)    ', 'GM (Runnalls)    '};

    for idx = 1:length(eval_models)
        m = eval_models{idx};
        dname = display_names{idx};
        m_mean = results.metrics_mean.(m);
        m_std  = results.metrics_std.(m);
        d_mean = results.diag_mean.(m);
        d_std  = results.diag_std.(m);

        % Metrics: 2:Trandis, 3:Loglik, 4:MSE_param, 8:ARI
        % Diag: 1:Rare_Beta_MSE, 3:Mean_Var, 4:Rc_Obj
        ll_str   = sprintf('%8.1f +/- %4.1f', m_mean(3), m_std(3));
        pmse_str = sprintf('%6.2f +/- %4.2f', m_mean(4), m_std(4));
        rb_str   = sprintf('%6.4f +/- %6.4f', d_mean(1), d_std(1));
        td_str   = sprintf('%9.1f +/- %5.1f', m_mean(2), m_std(2));
        ari_str  = sprintf('%5.3f +/- %5.3f', m_mean(8), m_std(8));
        var_str  = sprintf('%5.2f +/- %4.2f', d_mean(3), d_std(3));

        fprintf('%s | %s | %s | %s | %s | %s | %s\n', ...
                dname, ll_str, pmse_str, rb_str, td_str, ari_str, var_str);
    end

    fprintf('---------------------------------------------------------------------------------------------------------------------------------------\n');
    fprintf('Incompatible: WAVR, MED, AAVR, FED (Failed due to heterogeneous K_m across machines)\n');
    fprintf('=======================================================================================================================================\n\n');
end
