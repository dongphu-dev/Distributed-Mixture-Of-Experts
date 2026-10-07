function results = exp_official_heterogeneous_Km_benchmark(N_total, num_runs, S)
%% EXP_OFFICIAL_HETEROGENEOUS_KM_BENCHMARK
% Master Experiment 2 (Official): Evaluates distributed MoE aggregation under
% uncoordinated local expert capacities where each machine m observes K_m = 3 experts,
% producing L = 48 components across M = 16 machines to be aggregated into K_target = 5 global experts.
%
% Theoretical & Empirical Value:
%   - Parameter averaging baselines (AAVR, FED, WAVR, MED) are STRUCTURALLY INCOMPATIBLE
%     due to dimension mismatch (no 1-to-1 Hungarian matching possible).
%   - Directly proves DME's unique optimal-transport capability to aggregate arbitrary mixtures.
%
% Configuration:
%   M = 16 machines, K_m = 3, Total L = 48 components, Target K = 5, d = 20.
%   On macOS: 3 Monte Carlo runs for rapid verification (DEV mode).
%   On Linux: 10-100 Monte Carlo runs for full PROD benchmark.
%
% Usage:
%   results = exp_official_heterogeneous_Km_benchmark;              % Default: N = 100k, S = 2000
%   results = exp_official_heterogeneous_Km_benchmark(N_total);      % Custom N (e.g. 1000000 for 1M)
%   results = exp_official_heterogeneous_Km_benchmark(N_total, num_runs, S);

    if nargin < 3 || isempty(S),       S = 2000; end
    if nargin < 1 || isempty(N_total), N_total = 100000; end

    fprintf('========================================================================================\n');
    fprintf('  OFFICIAL EXPERIMENT 2: Heterogeneous Expert Capacity Benchmark (M=16, L=48 -> K=5)   \n');
    fprintf('========================================================================================\n');

    base_dir = fileparts(fileparts(mfilename('fullpath')));
    addpath(base_dir);
    addpath(fullfile(base_dir, 'models'));
    addpath(fullfile(base_dir, 'datatools'));
    addpath(fullfile(base_dir, 'evaltools'));
    addpath(fullfile(base_dir, 'stattools'));
    addpath(fullfile(base_dir, 'experiments'));
    addpath(fullfile(base_dir, 'reporting'));

    poolobj = gcp('nocreate');
    if isempty(poolobj)
        if ismac
            parpool('Processes', 4);
        else
            parpool;
        end
    end

    out_dir = fullfile(base_dir, 'results', 'official_heterogeneous');
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end

    N_str = format_N_str(N_total);
    data_dir = fullfile(base_dir, 'data');
    addpath(data_dir);
    data_file = fullfile(data_dir, sprintf('dataset_N%s_K5_d20_hetero_M16_Km3.mat', N_str));
    if ~exist(data_file, 'file')
        legacy_file = fullfile(data_dir, 'dataset_official_heterogeneous_M16_L48.mat');
        if exist(legacy_file, 'file')
            data_file = legacy_file;
        else
            error('Dataset not found: %s. Run data_generator_official_heterogeneous first!', data_file);
        end
    end

    loaded = load(data_file);
    all_runs     = loaded.all_runs;
    param_gt     = loaded.param;
    config       = loaded.config;
    true_mixture = loaded.true_mixture;

    if nargin < 2 || isempty(num_runs)
        num_runs = length(all_runs);
        fprintf('Executing all %d runs found in file: %s\n', num_runs, data_file);
    else
        num_runs = min(num_runs, length(all_runs));
        fprintf('Configured: executing %d / %d runs\n', num_runs, length(all_runs));
    end

    options = get_options('default');
    options.nb_EM_runs     = 5;
    options.IRLS_max_iter  = 100;
    options.max_iter       = 100;
    options.parallel_machines = false; % Guarantees outer parfor is over datasets, inner loop is serial over machines
    options.DME_maxiter = 100;
    options.DME_tol     = 1e-4;
    options.verbose     = 0;
    options.DME_verbose = 0;
    options.S           = S; % Tunable support sample size S = |D_S|
    options.sample_size = S;

    metrics_GLB        = NaN(num_runs, 10);
    metrics_DME_Kmeans = zeros(num_runs, 10);
    metrics_DME_Warm   = zeros(num_runs, 10);
    metrics_GM        = zeros(num_runs, 10);

    diag_GLB        = NaN(num_runs, 3);
    diag_DME_Kmeans = zeros(num_runs, 3);
    diag_DME_Warm   = zeros(num_runs, 3);
    diag_GM        = zeros(num_runs, 3);

    M        = config.M;
    K_vec    = config.K_vec;
    K_target = config.K_target;
    d        = config.d;

    parfor run = 1:num_runs
        t_run_start = tic;

        data_run = all_runs(run);
        X_cells = data_run.X_cells;
        Y_cells = data_run.Y_cells;
        X_test  = data_run.X_test;
        Y_test  = data_run.Y_test;
        Z_test  = data_run.Z_test;

        % 1. Local EM fitting across M machines
        local_estimates = cell(1, M);
        local_times     = zeros(1, M);
        for m = 1:M
            tic_m = tic;
            loc_opt = options;
            loc_opt.verbose = 0;
            local_est = MixtureOfExperts(X_cells{m}, Y_cells{m}, K_vec(m), loc_opt);
            local_times(m) = toc(tic_m);
            local_estimates{m} = local_est;
        end

        % Assemble dme_temp input struct
        S_supp = min(options.S, size(X_test, 1));
        dme_temp = struct();
        dme_temp.local_estimates = local_estimates;
        dme_temp.local_times     = local_times;
        dme_temp.d               = d;
        dme_temp.X_val           = [ones(S_supp, 1), X_test(1:S_supp, :)];
        dme_temp.n               = config.N_total_train;

        % 2. Runnalls Greedy Pairwise Merging (GM)
        tic_gm = tic;
        fit_gm = Greedy_MixtureOfExperts_Hetero(dme_temp, K_target, options);
        t_gm = toc(tic_gm);
        if ~isfield(fit_gm, 'learning_time') || fit_gm.learning_time == 0
            fit_gm.learning_time = max(local_times) + t_gm;
        end

        % 3. Proposed DME (Standalone Multi-start K-means)
        opt_dme_km = options;
        opt_dme_km.init_mode = 'kmeans';
        opt_dme_km.DME_tries = 3;
        tic_km = tic;
        fit_dme_km = Distributed_MixtureOfExperts_Hetero(dme_temp, K_target, opt_dme_km);
        t_km = toc(tic_km);
        if ~isfield(fit_dme_km, 'learning_time') || fit_dme_km.learning_time == 0
            fit_dme_km.learning_time = max(local_times) + t_km;
        end

        % 4. Proposed DME (Warm-started from GM -> MM Descent Guarantee)
        opt_dme_warm = options;
        opt_dme_warm.init_mode = 'grd';
        opt_dme_warm.DME_tries = 2;
        tic_warm = tic;
        fit_dme_warm = Distributed_MixtureOfExperts_Hetero(dme_temp, K_target, opt_dme_warm);
        t_warm = toc(tic_warm);
        if ~isfield(fit_dme_warm, 'learning_time') || fit_dme_warm.learning_time == 0
            fit_dme_warm.learning_time = max(local_times) + t_warm;
        end

        f_W = fit_dme_warm.large_mixture;

        [t_l, td, ll, mse, rpe, cr, ri, ari, ce, td_fW] = compute_metrics(fit_gm, true_mixture, X_test, Y_test, Z_test, 0, 'GM', f_W);
        res_gm = [t_l, td, ll, mse, rpe, cr, ri, ari, ce, td_fW];
        metrics_GM(run, :) = res_gm;
        diag_GM(run, :) = [mean(fit_gm.param.sigma2), 0, 0];

        [t_l, td, ll, mse, rpe, cr, ri, ari, ce, td_fW] = compute_metrics(fit_dme_km, true_mixture, X_test, Y_test, Z_test, 0, 'DME_Kmeans', f_W);
        res_dme_km = [t_l, td, ll, mse, rpe, cr, ri, ari, ce, td_fW];
        metrics_DME_Kmeans(run, :) = res_dme_km;
        diag_DME_Kmeans(run, :) = [mean(fit_dme_km.param.sigma2), fit_dme_km.transportdis, 0];

        [t_l, td, ll, mse, rpe, cr, ri, ari, ce, td_fW] = compute_metrics(fit_dme_warm, true_mixture, X_test, Y_test, Z_test, 0, 'DME_Warm', f_W);
        res_dme_warm = [t_l, td, ll, mse, rpe, cr, ri, ari, ce, td_fW];
        metrics_DME_Warm(run, :) = res_dme_warm;
        diag_DME_Warm(run, :) = [mean(fit_dme_warm.param.sigma2), fit_dme_warm.transportdis, 0];

        % 5. Oracle GLB (Centralized Oracle on pooled data)
        X_pooled = cat(1, X_cells{:});
        Y_pooled = cat(1, Y_cells{:});
        glb_opt = options;
        glb_opt.nb_EM_runs = options.nb_EM_runs;
        glb_opt.max_iter   = 40;
        glb_opt.verbose    = 0;
        tic_glb = tic;
        fit_glb = Global_MixtureOfExperts(X_pooled, Y_pooled, K_target, glb_opt);
        fit_glb.learning_time = toc(tic_glb);
        [t_l, td, ll, mse, rpe, cr, ri, ari, ce, td_fW] = compute_metrics(fit_glb, true_mixture, X_test, Y_test, Z_test, 0, 'GLB', f_W);
        res_glb = [t_l, td, ll, mse, rpe, cr, ri, ari, ce, td_fW];
        metrics_GLB(run, :) = res_glb;

        t_total = toc(t_run_start);
        t_stamp_done = datestr(now, 'HH:MM:SS');
        fprintf('[Exp 2 | Hetero Km=3] [Run %3d/%3d] [%s] Done (%5.1fs) | DME ARI=%0.4f | GM ARI=%0.4f | GLB ARI=%0.4f\n', ...
                run, num_runs, t_stamp_done, t_total, ...
                res_dme_warm(8), ...
                res_gm(8), ...
                res_glb(8));
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

    results.metrics_mean.GLB        = mean(metrics_GLB, 1, 'omitnan');
    results.metrics_std.GLB         = std(metrics_GLB, 0, 1, 'omitnan');
    results.metrics_mean.DME_Kmeans = mean(metrics_DME_Kmeans, 1, 'omitnan');
    results.metrics_std.DME_Kmeans  = std(metrics_DME_Kmeans, 0, 1, 'omitnan');
    results.metrics_mean.DME_Warm   = mean(metrics_DME_Warm, 1, 'omitnan');
    results.metrics_std.DME_Warm    = std(metrics_DME_Warm, 0, 1, 'omitnan');
    results.metrics_mean.GM        = mean(metrics_GM, 1, 'omitnan');
    results.metrics_std.GM         = std(metrics_GM, 0, 1, 'omitnan');

    out_res = fullfile(out_dir, sprintf('Result_Official_Heterogeneous_M16_N%s.mat', N_str));
    save(out_res, 'results', '-v7.3');
    % Also save to default unversioned name for standard reference
    save(fullfile(out_dir, 'Result_Official_Heterogeneous_M16.mat'), 'results', '-v7.3');
    fprintf('\n==> Successfully saved Official Heterogeneous results to: %s\n', out_res);

    % Export LaTeX Table (if exporter is available)
    if exist('export_latex_official_tables', 'file')
        table_dir = fullfile(base_dir, 'results', 'tables');
        if ~exist(table_dir, 'dir'), mkdir(table_dir); end
        tex_file = fullfile(table_dir, sprintf('table_official_exp2_heterogeneous_N%s.tex', N_str));
        export_latex_official_tables('', out_res, table_dir);
        copyfile(fullfile(table_dir, 'table_official_exp2_heterogeneous.tex'), tex_file);
        fprintf('==> Official LaTeX Table 2 exported successfully to: %s\n', tex_file);
    end
end
