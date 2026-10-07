function results = exp_heterogeneous_Km()
% EXP_HETEROGENEOUS_KM
% Benchmark driver for heterogeneous local experts where machine m has K_m experts.
% Evaluates candidate models on the non-IID subpopulation dataset:
%   - DME (Native Optimal Transport MM via Distributed_MixtureOfExperts_Hetero)
%   - GM (Greedy Merging Runnalls via Greedy_MixtureOfExperts_Hetero)
%   - GLB (Centralized Oracle Baseline on pooled data)
%   - WAVR, MED, AAVR, FED (Documented as INCOMPATIBLE due to dimension mismatch)
%
% Output:
%   Saved to results/heterogeneous_Km/Result_Hetero_Km_K5_M4.mat

    fprintf('========================================================================================\n');
    fprintf('  BENCHMARK: HETEROGENEOUS LOCAL EXPERTS (K_m Variable across M = 4 Machines)           \n');
    fprintf('========================================================================================\n');

    % 1. Determine execution mode (DEV vs PROD)
    is_dev = ismac;
    if is_dev
        num_runs = 2;
        fprintf('Mode: DEV (MacBook detected, %d runs)\n', num_runs);
    else
        num_runs = 10;
        fprintf('Mode: PROD (Linux VPS detected, %d runs)\n', num_runs);
    end

    % 2. Load dataset
    current_dir = fileparts(mfilename('fullpath'));
    base_dir    = fullfile(current_dir, '..');
    addpath(fullfile(base_dir, 'models'));
    addpath(fullfile(base_dir, 'stattools'));
    addpath(fullfile(base_dir, 'evaltools'));
    addpath(base_dir);

    data_file = fullfile(base_dir, 'data', 'dataset_hetero_Km_K5_d20.mat');
    if ~exist(data_file, 'file')
        fprintf('Dataset not found. Generating dataset first...\n');
        addpath(fullfile(base_dir, 'data'));
        data_generator_hetero_Km(42);
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

    fprintf('Configuration: M = %d, K_vec = [%s], Target K = %d, Total L = %d\n', ...
            M, num2str(K_vec), K_target, sum(K_vec));

    % Construct ground truth mixture struct for metric evaluation
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

    % Options
    options = get_options('default');
    options.nb_EM_runs = 2;
    options.max_iter   = 40;
    options.DME_tries  = 2;
    options.verbose    = 0;
    options.DME_verbose = 0;

    % Metrics storage: [Run x Metric]
    % Metrics: 1:Time, 2:Trandis, 3:Loglik, 4:MSE_param, 5:RPE, 6:Corr, 7:RI, 8:ARI, 9:ClustErr
    metrics_DME = zeros(num_runs, 9);
    metrics_GM = zeros(num_runs, 9);
    metrics_GLB = zeros(num_runs, 9);

    % Pooled data for centralized baseline
    X_pooled = [];
    Y_pooled = [];
    for m = 1:M
        X_pooled = [X_pooled; X_cells{m}]; %#ok<AGROW>
        Y_pooled = [Y_pooled; Y_cells{m}]; %#ok<AGROW>
    end

    % 3. Benchmark Execution Loop
    for run = 1:num_runs
        fprintf('\n--- Run %d/%d ---\n', run, num_runs);

        % (a) DME: Distributed MoE via Optimal Transport MM
        fprintf('  Fitting DME (Optimal Transport MM)...\n');
        tic_dme = tic;
        fit_dme = Distributed_MixtureOfExperts_Hetero(X_cells, Y_cells, K_vec, K_target, options);
        t_dme = toc(tic_dme);
        if ~isfield(fit_dme, 'learning_time') || fit_dme.learning_time == 0
            fit_dme.learning_time = t_dme;
        end
        [t_l, td, ll, mse, rpe, cr, ri, ari, ce] = compute_metrics(fit_dme, true_mixture, X_test, Y_test, Z_test, 0, 'DME');
        metrics_DME(run, :) = [t_l, td, ll, mse, rpe, cr, ri, ari, ce];
        fprintf('    DME: Time = %.2fs, Trandis = %.2f, Loglik = %.1f, MSE = %.4f, ARI = %.3f\n', ...
                t_l, td, ll, mse, ari);

        % (b) GM: Greedy Merging (Runnalls 2007)
        fprintf('  Fitting GM (Runnalls Greedy Merging)...\n');
        tic_gm = tic;
        % Reuse local estimates from DME for perfectly fair comparison and compute efficiency
        dme_temp.local_estimates = fit_dme.local_estimates;
        dme_temp.local_times     = fit_dme.local_times;
        dme_temp.d               = d;
        dme_temp.X_val           = fit_dme.X_val;
        dme_temp.n               = fit_dme.n;
        fit_gm = Greedy_MixtureOfExperts_Hetero(dme_temp, K_target, options);
        t_gm = toc(tic_gm);
        if ~isfield(fit_gm, 'learning_time') || fit_gm.learning_time == 0
            fit_gm.learning_time = max(fit_dme.local_times) + t_gm;
        end
        [t_l, td, ll, mse, rpe, cr, ri, ari, ce] = compute_metrics(fit_gm, true_mixture, X_test, Y_test, Z_test, 0, 'GM');
        metrics_GM(run, :) = [t_l, td, ll, mse, rpe, cr, ri, ari, ce];
        fprintf('    GM: Time = %.2fs, Trandis = %.2f, Loglik = %.1f, MSE = %.4f, ARI = %.3f\n', ...
                t_l, td, ll, mse, ari);

        % (c) GLB: Centralized Oracle Baseline
        fprintf('  Fitting GLB (Centralized Oracle on pooled data)...\n');
        glb_opt = options;
        glb_opt.nb_EM_runs = 2;
        glb_opt.max_iter   = 30;
        fit_glb = Global_MixtureOfExperts(X_pooled, Y_pooled, K_target, glb_opt);
        [t_l, td, ll, mse, rpe, cr, ri, ari, ce] = compute_metrics(fit_glb, true_mixture, X_test, Y_test, Z_test, 0, 'GLB');
        metrics_GLB(run, :) = [t_l, td, ll, mse, rpe, cr, ri, ari, ce];
        fprintf('    GLB: Time = %.2fs, Trandis = %.2f, Loglik = %.1f, MSE = %.4f, ARI = %.3f\n', ...
                t_l, td, ll, mse, ari);
    end

    % 4. Assemble Summary Statistics
    models = {'DME', 'GM', 'GLB', 'WAVR', 'MED', 'AAVR', 'FED'};
    status = {'COMPLETED', 'COMPLETED', 'ORACLE', ...
              'INCOMPATIBLE', 'INCOMPATIBLE', 'INCOMPATIBLE', 'INCOMPATIBLE'};

    results = struct();
    results.models   = models;
    results.status   = status;
    results.config   = config;
    results.num_runs = num_runs;

    results.metrics_raw.DME = metrics_DME;
    results.metrics_raw.GM = metrics_GM;
    results.metrics_raw.GLB = metrics_GLB;

    results.metrics_mean.DME = mean(metrics_DME, 1);
    results.metrics_std.DME  = std(metrics_DME, 0, 1);
    results.metrics_mean.GM = mean(metrics_GM, 1);
    results.metrics_std.GM  = std(metrics_GM, 0, 1);
    results.metrics_mean.GLB = mean(metrics_GLB, 1);
    results.metrics_std.GLB  = std(metrics_GLB, 0, 1);

    % Save last transport plan for heatmap visualization
    results.last_dme_plan = fit_dme.plan;

    % 5. Print Consolidated Benchmark Summary Table
    fprintf('\n========================================================================================================\n');
    fprintf('  HETEROGENEOUS BENCHMARK SUMMARY (M = 4, Km = [%s], L = %d -> K = %d, %d runs)\n', ...
            num2str(K_vec), sum(K_vec), K_target, num_runs);
    fprintf('========================================================================================================\n');
    fprintf('%-8s | %-12s | %-12s | %-12s | %-16s | %-14s | %-10s | %-10s\n', ...
            'Model', 'Status', 'Time(s)', 'Trandis', 'Loglik', 'MSE_param', 'ARI', 'ClustErr%');
    fprintf('--------------------------------------------------------------------------------------------------------\n');

    % Print DME
    m_d = results.metrics_mean.DME; s_d = results.metrics_std.DME;
    fprintf('%-8s | %-12s | %5.2f±%-5.2f | %5.2f±%-5.2f | %8.1f±%-6.1f | %6.4f±%-6.4f | %4.3f±%-4.3f | %5.2f±%-4.2f\n', ...
            'DME', 'COMPLETED', m_d(1), s_d(1), m_d(2), s_d(2), m_d(3), s_d(3), m_d(4), s_d(4), m_d(8), s_d(8), m_d(9), s_d(9));

    % Print GM
    m_g = results.metrics_mean.GM; s_g = results.metrics_std.GM;
    fprintf('%-8s | %-12s | %5.2f±%-5.2f | %5.2f±%-5.2f | %8.1f±%-6.1f | %6.4f±%-6.4f | %4.3f±%-4.3f | %5.2f±%-4.2f\n', ...
            'GM', 'COMPLETED', m_g(1), s_g(1), m_g(2), s_g(2), m_g(3), s_g(3), m_g(4), s_g(4), m_g(8), s_g(8), m_g(9), s_g(9));

    % Print GLB
    m_o = results.metrics_mean.GLB; s_o = results.metrics_std.GLB;
    fprintf('%-8s | %-12s | %5.2f±%-5.2f | %5.2f±%-5.2f | %8.1f±%-6.1f | %6.4f±%-6.4f | %4.3f±%-4.3f | %5.2f±%-4.2f\n', ...
            'GLB', 'ORACLE', m_o(1), s_o(1), m_o(2), s_o(2), m_o(3), s_o(3), m_o(4), s_o(4), m_o(8), s_o(8), m_o(9), s_o(9));

    % Print Incompatible Baselines
    incompatible = {'WAVR', 'MED', 'AAVR', 'FED'};
    for ib = 1:length(incompatible)
        fprintf('%-8s | %-12s | %-12s | %-12s | %-16s | %-14s | %-10s | %-10s\n', ...
                incompatible{ib}, 'INCOMPATIBLE', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A');
    end
    fprintf('========================================================================================================\n');

    % 6. Save results to disk
    res_dir = fullfile(base_dir, 'results', 'heterogeneous_Km');
    if ~exist(res_dir, 'dir'), mkdir(res_dir); end
    save_path = fullfile(res_dir, 'Result_Hetero_Km_K5_M4.mat');
    save(save_path, 'results');
    fprintf('Results saved to: %s\n', save_path);

end
