function results = exp_ablation_balanced_M16()
% EXP_ABLATION_BALANCED_M16
% Ablation Study: Isolating the M = 16 Compounding / Agglomerative Factor
% with PERFECTLY BALANCED subpopulations (NO rare expert).
%
% Configuration:
%   - M = 16 machines, K_m = 3 -> Total L = 48 components
%   - Target: K = 5 global experts (43 merges in GM)
%   - BALANCED: All 5 experts have equal prior proportion (~20% each).
%   - Small local samples: N_m = 500 samples per machine (total N = 8,000)
%   - Equal noise variances sigma2 = [3.0, 3.0, 3.0, 3.0, 3.0]
%
% Evaluated Models:
%   1. GLB        : Centralized Oracle on pooled data
%   2. DME_Kmeans : Standalone DME with independent k-means initialization
%   3. DME_Warm   : DME warm-started from GM (MM error correction)
%   4. GM        : Runnalls (2007) greedy pairwise merging (43 merges)
%   5. WAVR, MED, AAVR, FED : Incompatible (heterogeneous K_m)
%
% Output:
%   results/ablation_balanced_M16/Result_Ablation_Balanced_M16.mat

    fprintf('========================================================================================\n');
    fprintf('  ABLATION EXPERIMENT: BALANCED EXPERTS (NO RARE) WITH M = 16 (L = 48 -> K = 5)          \n');
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

    res_dir = fullfile(base_dir, 'results', 'ablation_balanced_M16');
    if ~exist(res_dir, 'dir'), mkdir(res_dir); end

    data_file = fullfile(base_dir, 'data', 'dataset_ablation_balanced_M16.mat');
    if ~exist(data_file, 'file')
        fprintf('Dataset not found. Generating dataset first...\n');
        addpath(fullfile(base_dir, 'data'));
        data_generator_ablation_balanced_M16(123);
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

    X_pooled = [];
    Y_pooled = [];
    for m = 1:M
        X_pooled = [X_pooled; X_cells{m}]; %#ok<AGROW>
        Y_pooled = [Y_pooled; Y_cells{m}]; %#ok<AGROW>
    end

    options = get_options('default');
    options.nb_EM_runs  = 5;
    options.max_iter    = 500;
    options.DME_maxiter = 500;
    options.DME_tol     = 1e-6;
    options.verbose     = 0;
    options.DME_verbose = 0;

    metrics_GLB        = zeros(num_runs, 9);
    metrics_DME_Kmeans = zeros(num_runs, 9);
    metrics_DME_Warm   = zeros(num_runs, 9);
    metrics_GM        = zeros(num_runs, 9);

    diag_GLB        = zeros(num_runs, 3);
    diag_DME_Kmeans = zeros(num_runs, 3);
    diag_DME_Warm   = zeros(num_runs, 3);
    diag_GM        = zeros(num_runs, 3);

    last_fits = struct();

    for run = 1:num_runs
        fprintf('\n--- Run %d/%d ---\n', run, num_runs);

        fprintf('  Fitting local EM models on %d machines (N_m = %d)...\n', M, config.N_per_machine);
        local_estimates = cell(1, M);
        local_times     = zeros(1, M);
        for m = 1:M
            tic_m = tic;
            local_est = MixtureOfExperts(X_cells{m}, Y_cells{m}, K_vec(m), options);
            local_times(m) = toc(tic_m);
            local_estimates{m} = local_est;
        end

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

        % (a) GM: Greedy Merging (43 merges)
        fprintf('  Fitting GM (43 merges)...\n');
        tic_gm = tic;
        fit_gm = Greedy_MixtureOfExperts_Hetero(dme_temp, K_target, options);
        t_gm   = toc(tic_gm);
        if ~isfield(fit_gm, 'learning_time') || fit_gm.learning_time == 0
            fit_gm.learning_time = max(local_times) + t_gm;
        end
        [t_l, td, ll, mse, rpe, cr, ri, ari, ce] = compute_metrics(fit_gm, true_mixture, X_test, Y_test, Z_test, 0, 'GM');
        metrics_GM(run, :) = [t_l, td, ll, mse, rpe, cr, ri, ari, ce];
        mean_var_gm = mean(fit_gm.param.sigma2);
        rc_gm = fit_gm.transportdis;
        diag_GM(run, :) = [mean_var_gm, rc_gm, 0];
        fprintf('    GM: Loglik = %.1f, Param_MSE = %.2f, Trandis = %.1f, ARI = %.3f, Mean_Var = %.2f\n', ...
                ll, mse, td, ari, mean_var_gm);

        % (b) DME_Kmeans: Standalone DME
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
        mean_var_km = mean(fit_dme_km.param.sigma2);
        rc_km = fit_dme_km.transportdis;
        diag_DME_Kmeans(run, :) = [mean_var_km, rc_km, 0];
        fprintf('    DME (Standalone): Loglik = %.1f, Param_MSE = %.2f, Trandis = %.1f, ARI = %.3f, Mean_Var = %.2f\n', ...
                ll, mse, td, ari, mean_var_km);

        % (c) DME_Warm: DME Warm-Started from GM
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
        mean_var_warm = mean(fit_dme_warm.param.sigma2);
        rc_warm = fit_dme_warm.transportdis;
        diag_DME_Warm(run, :) = [mean_var_warm, rc_warm, 0];
        fprintf('    DME (Warm-start): Loglik = %.1f, Param_MSE = %.2f, Trandis = %.1f, ARI = %.3f, Mean_Var = %.2f\n', ...
                ll, mse, td, ari, mean_var_warm);

        % (d) GLB: Oracle Centralized Baseline
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
        diag_GLB(run, :) = [mean(fit_glb.param.sigma2), NaN, 0];
        fprintf('    GLB (Oracle): Loglik = %.1f, Param_MSE = %.2f, Trandis = %.1f, ARI = %.3f\n', ...
                ll, mse, td, ari);

        if run == num_runs
            last_fits.GM        = fit_gm;
            last_fits.DME_Kmeans = fit_dme_km;
            last_fits.DME_Warm   = fit_dme_warm;
            last_fits.GLB        = fit_glb;
        end
    end

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

    out_res = fullfile(res_dir, 'Result_Ablation_Balanced_M16.mat');
    save(out_res, 'results', '-v7.3');
    fprintf('\n==> Successfully saved results to: %s\n', out_res);

    print_ablation_summary_table(results, 'ABLATION: BALANCED EXPERTS (NO RARE) WITH M = 16');
end


function print_ablation_summary_table(results, title_str)
    fprintf('\n');
    fprintf('=======================================================================================================================================\n');
    fprintf('  %s\n', title_str);
    fprintf('=======================================================================================================================================\n');
    fprintf('Model            | Param MSE         | Trandis (x10^3)    | ARI             | Mean Var (True=3.0) | Test Loglik       \n');
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

        pmse_str = sprintf('%6.2f +/- %4.2f', m_mean(4), m_std(4));
        td_str   = sprintf('%8.1f +/- %5.1f', m_mean(2) / 1e3, m_std(2) / 1e3);
        ari_str  = sprintf('%5.3f +/- %5.3f', m_mean(8), m_std(8));
        var_str  = sprintf('%5.2f +/- %4.2f', d_mean(1), d_std(1));
        ll_str   = sprintf('%8.1f +/- %4.1f', m_mean(3), m_std(3));

        fprintf('%s | %s | %s | %s | %s | %s\n', ...
                dname, pmse_str, td_str, ari_str, var_str, ll_str);
    end
    fprintf('---------------------------------------------------------------------------------------------------------------------------------------\n');
    fprintf('Incompatible: WAVR, MED, AAVR, FED (Failed due to heterogeneous K_m across machines)\n');
    fprintf('=======================================================================================================================================\n\n');
end
