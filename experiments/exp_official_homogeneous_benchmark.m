function All_results = exp_official_homogeneous_benchmark(N_total, num_runs, S)
%% EXP_OFFICIAL_HOMOGENEOUS_BENCHMARK
% Master Experiment 1 (Official): Evaluates distributed MoE estimators under
% standard homogeneous expert capacities (K_m = K = 5) across 3 subpopulation profiles:
%   1. Balanced:              pi_k = 0.20
%   2. Moderately Imbalanced: 0.10 <= pi_k <= 0.30
%   3. Highly Imbalanced:     pi_min = 0.02
%
% Configuration:
%   M = 16 machines, d = 20, K = 5.
%   On macOS: 3 datasets for fast validation (DEV mode).
%   On Linux: Full datasets for PROD Monte Carlo benchmark.
%
% Usage:
%   All_results = exp_official_homogeneous_benchmark;                 % Default: N = 100k, S = 2000
%   All_results = exp_official_homogeneous_benchmark(N_total);         % Custom N (e.g., 300000, 1000000)
%   All_results = exp_official_homogeneous_benchmark(N_total, num_runs, S);

    if nargin < 3 || isempty(S),       S = 2000; end
    if nargin < 1 || isempty(N_total), N_total = 100000; end

    N_str = format_N_str(N_total);
    fprintf('========================================================================\n');
    fprintf('  OFFICIAL EXPERIMENT 1: Homogeneous MoE Benchmark (M = 16, N = %s)      \n', N_str);
    fprintf('========================================================================\n');

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

    out_dir = fullfile(base_dir, 'results', 'official_homogeneous');
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end

    K = 5;
    d = 20;
    % Options
    options = get_options('default');
    options.DME_verbose = 0;
    options.verbose     = 0;
    options.S           = S; % Tunable support sample size S = |D_S|
    options.sample_size = S;
    options.IRLS_max_iter = 100;
    options.IRLS_threshold = 1e-5;
    options.max_iter       = 100;
    options.nb_EM_runs     = 5;
    options.parallel_machines = false; % Guarantees outer parfor is over datasets, inner loop is serial over machines
    options.FedAvg_rounds  = 5;
    options.FedAvg_local_iters = 10;

    if isfield(options, 'N_test') && ~isempty(options.N_test)
        N_test  = options.N_test;
        N_train = N_total - N_test;
    else
        N_train = floor(0.8 * N_total);
        N_test  = N_total - N_train;
    end

    M = 16;
    profiles = {'balanced', 'moderate', 'severe'};
    models_to_run = {'DME', 'GM', 'AAVR', 'FED_T1', 'FED_T5', 'FED_T10', 'FED_T20', 'WAVR', 'MED'};
    all_models    = [{'GLB'}, models_to_run];

    All_results = struct();

    for p = 1:length(profiles)
        prof_name = profiles{p};
        data_file = fullfile(base_dir, 'data', sprintf('dataset_N%s_K%d_d%d_%s.mat', N_str, K, d, prof_name));
    
    if ~exist(data_file, 'file')
        error('Dataset not found: %s. Run data_generator_profiles first!', data_file);
    end
    
    fprintf('\n------------------------------------------------------------------------\n');
    fprintf('  Processing Profile [%d/3]: %s\n', p, prof_name);
    fprintf('  Loading: %s\n', data_file);
    fprintf('------------------------------------------------------------------------\n');
    
    loaded = load(data_file);
    X_mat        = loaded.X_mat;
    Y_mat        = loaded.Y_mat;
    LABEL_mat    = loaded.LABEL_mat;
    true_mixture = loaded.true_mixture;
    
    if nargin < 2 || isempty(num_runs)
        runs_to_process = size(X_mat, 3);
        fprintf('Executing all %d datasets found in file: %s\n', runs_to_process, data_file);
    else
        runs_to_process = min(num_runs, size(X_mat, 3));
        fprintf('Configured: executing %d / %d datasets\n', runs_to_process, size(X_mat, 3));
    end

    % Initialize storage
    All_results.(prof_name) = struct();
    for m = 1:length(all_models)
        All_results.(prof_name).(all_models{m}).metrics = zeros(runs_to_process, 10);
    end

    % Temporary storage for parfor loop (cell array indexed by r is parfor-safe)
    run_metrics = cell(runs_to_process, 1);

    parfor r = 1:runs_to_process
        t_run = tic;
        
        X_all     = X_mat(:, :, r);
        Y_all     = Y_mat(:, r);
        label_all = LABEL_mat(:, r);
        
        X_train     = X_all(1:N_train, :);
        Y_train     = Y_all(1:N_train);
        label_train = label_all(1:N_train);
        
        X_test      = X_all(N_train+1:end, :);
        Y_test      = Y_all(N_train+1:end);
        label_test  = label_all(N_train+1:end);

        % 1. Uniform partition across M machines
        rng(r * 1000 + p, 'twister');
        shuffled = randperm(N_train);
        client_indices = cell(1, M);
        N_m = floor(N_train / M);
        for m = 1:M
            client_indices{m} = shuffled((m - 1) * N_m + 1 : m * N_m)';
        end
        
        run_options = options;
        run_options.client_indices = client_indices;

        % 2. Execute distributed estimators
        runner_res = core_runner(X_train, Y_train, X_test, Y_test, label_test, true_mixture, ...
                                 K, M, run_options, models_to_run, 0);

        r_struct = struct();
        for m = 1:length(models_to_run)
            mod_name = models_to_run{m};
            r_struct.(mod_name) = runner_res.(mod_name).metrics;
        end

        % 3. Centralized Oracle (GLB on pooled training data)
        t_glb = tic;
        glb_opt = options;
        glb_opt.nb_EM_runs = options.nb_EM_runs;
        glb_opt.max_iter   = 40;
        glb_opt.verbose    = 0;
        fit_glb = Global_MixtureOfExperts(X_train, Y_train, K, glb_opt);
        fit_glb.learning_time = toc(t_glb);
        [t_l, td, ll, mse, rpe, cr, ri, ari, ce, td_fW] = compute_metrics(fit_glb, true_mixture, X_test, Y_test, label_test, 0, 'GLB', runner_res.f_W);
        r_struct.GLB = [t_l, td, ll, mse, rpe, cr, ri, ari, ce, td_fW];
        
        run_metrics{r} = r_struct;

        t_run_total = toc(t_run);
        t_stamp_done = datestr(now, 'HH:MM:SS');
        fprintf('[Exp 1 | %-10s] [Run %3d/%3d] [%s] Done (%5.1fs) | DME ARI=%0.4f | GM ARI=%0.4f | AAVR ARI=%0.4f | GLB ARI=%0.4f\n', ...
                prof_name, r, runs_to_process, t_stamp_done, t_run_total, ...
                r_struct.DME(8), ...
                r_struct.GM(8), ...
                r_struct.AAVR(8), ...
                r_struct.GLB(8));
    end

    % Collect parallel results into All_results struct
    for r = 1:runs_to_process
        r_struct = run_metrics{r};
        for m = 1:length(all_models)
            mod_name = all_models{m};
            if isfield(r_struct, mod_name)
                All_results.(prof_name).(mod_name).metrics(r, :) = r_struct.(mod_name);
            end
        end
    end
end

% Save Results
out_mat = fullfile(out_dir, sprintf('Result_Official_Homogeneous_M16_N%s.mat', N_str));
save(out_mat, 'All_results', 'profiles', 'all_models', 'M', 'N_total', 'K', 'd', 'options', '-v7.3');
% Also save unversioned default name for standard reference
save(fullfile(out_dir, 'Result_Official_Homogeneous_M16_N100k.mat'), 'All_results', 'profiles', 'all_models', 'M', 'N_total', 'K', 'd', 'options', '-v7.3');
fprintf('\n==> Successfully saved Official Homogeneous results to: %s\n', out_mat);

% Export LaTeX Table (if exporter is available)
if exist('export_latex_official_tables', 'file')
    table_dir = fullfile(base_dir, 'results', 'tables');
    if ~exist(table_dir, 'dir'), mkdir(table_dir); end
    tex_file = fullfile(table_dir, sprintf('table_official_exp1_homogeneous_N%s.tex', N_str));
    export_latex_official_tables(out_mat, '', table_dir);
    copyfile(fullfile(table_dir, 'table_official_exp1_homogeneous.tex'), tex_file);
    fprintf('==> Official LaTeX Table 1 exported successfully to: %s\n', tex_file);
end
