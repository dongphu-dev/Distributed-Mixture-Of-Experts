function results = exp_sensitivity_support_S(num_runs, S_list, N_total, profile_name)
%% EXP_SENSITIVITY_SUPPORT_S
% Investigates the sensitivity of distributed MoE aggregators (DME, GM, MED)
% to the support sample size S = |D_S| across 5 theoretical regimes:
%   - Under-determined: S in {30, 50} (S < (d+1)(K-1) = 84)
%   - Sát ngưỡng lý thuyết: S = 100 (S >= 84)
%   - Ngưỡng trung bình: S in {250, 500}
%   - Ngưỡng an toàn: S in {1000, 2000}
%   - Ngưỡng tối đa / bão hòa: S = 5000 (S = N_m)
%
% Also evaluates 3 sources of D_S at S = 5000:
%   1. Pooled i.i.d.
%   2. One machine (all N_m points from Machine 1)
%   3. Simulated x (synthetic covariates from feature distribution)
%
% Reference Baselines (Independent of S):
%   - GLB (Centralized Oracle)
%   - AAVR (Aligned Parameter Averaging)
%
% Uses Decoupled Training:
%   Local models on M=16 machines are fitted ONCE per run.
%
% Usage:
%   results = exp_sensitivity_support_S;                                % Auto-detects DEV/PROD, moderate profile
%   results = exp_sensitivity_support_S(num_runs);                      % Explicit runs, moderate profile
%   results = exp_sensitivity_support_S(num_runs, S_list, N_total, profile_name);

    if nargin < 4 || isempty(profile_name), profile_name = 'moderate'; end
    if nargin < 3 || isempty(N_total), N_total = 100000; end
    if nargin < 2 || isempty(S_list)
        S_list = [30, 50, 100, 250, 500, 1000, 2000, 5000];
    end
    if nargin < 1 || isempty(num_runs)
        if ismac
            num_runs = 2;   % Fast DEV mode on Mac
        else
            num_runs = 50;  % Full PROD benchmark on VPS
        end
    end

    fprintf('========================================================================\n');
    fprintf('  EXPERIMENT: SENSITIVITY TO SUPPORT SAMPLE SIZE S AND DATA SOURCES     \n');
    fprintf('  Profile: %s | N = %d, M = 16, K = 5, d = 20, Runs = %d               \n', profile_name, N_total, num_runs);
    fprintf('  S list: [%s]\n', num2str(S_list));
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

    out_dir = fullfile(base_dir, 'results', 'sensitivity_support_S');
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end

    % Load benchmark data (Default: moderate profile)
    N_str = format_N_str(N_total);
    data_file = fullfile(base_dir, 'data', sprintf('dataset_N%s_K5_d20_%s.mat', N_str, profile_name));
    if ~exist(data_file, 'file')
        error('Data file not found: %s. Run data_generator_profiles first!', data_file);
    end

    fprintf('Loading dataset: %s...\n', data_file);
    loaded = load(data_file);
    X_mat        = loaded.X_mat;
    Y_mat        = loaded.Y_mat;
    LABEL_mat    = loaded.LABEL_mat;
    true_mixture = loaded.true_mixture;
    K = loaded.K;
    d = loaded.d;
    M = 16;

    total_available_runs = size(X_mat, 3);
    runs_to_process = min(num_runs, total_available_runs);
    fprintf('Processing %d runs...\n', runs_to_process);

    options = get_options('default');
    options.DME_verbose   = 0;
    options.verbose       = 0;
    options.IRLS_max_iter = 100;
    options.IRLS_threshold = 1e-5;
    options.max_iter       = 100;
    options.nb_EM_runs     = 5;
    options.parallel_machines = false; % Guarantees outer parfor is over datasets, inner loop is serial over machines

    N_train = floor(0.8 * N_total);
    N_test  = N_total - N_train;
    N_m     = floor(N_train / M); % 5000

    % Storage for parallel runs
    raw_runs = cell(1, runs_to_process);

    parfor r = 1:runs_to_process
        t_run_start = tic;
        fprintf('\n--> [Run %d/%d] Starting Decoupled Training...\n', r, runs_to_process);

        X_full = X_mat(:, :, r);
        Y_full = Y_mat(:, r);
        Z_full = LABEL_mat(:, r);

        X_train = X_full(1:N_train, :);
        Y_train = Y_full(1:N_train);
        X_test  = X_full(N_train+1:end, :);
        Y_test  = Y_full(N_train+1:end);
        Z_test  = Z_full(N_train+1:end);

        % Partition training data across M machines
        client_indices = cell(1, M);
        for m = 1:M
            client_indices{m} = (m-1)*N_m + 1 : m*N_m;
        end

        % 1. Local Training: Fit M local models ONCE
        local_estimates = cell(1, M);
        local_times     = zeros(1, M);
        for m = 1:M
            tic_m = tic;
            loc_opt = options;
            loc_opt.verbose = 0;
            idx_m = client_indices{m};
            loc_fit = MixtureOfExperts(X_train(idx_m, :), Y_train(idx_m), K, loc_opt);
            local_times(m) = toc(tic_m);
            local_estimates{m} = loc_fit;
        end
        max_local_time = max(local_times);

        % Pre-assemble base DME struct for one-shot reuse
        base_dme_opt = options;
        base_dme_opt.local_estimates = local_estimates;
        base_dme_opt.local_times     = local_times;
        base_dme_opt.client_indices  = client_indices;

        % 2. Baselines (Independent of S)
        % 2b. Aligned Averaging AAVR (Hungarian matching on local models)
        tic_aavr = tic;
        % Fit dummy DME with minimal sample just to construct AAVR input and extract f^W
        dummy_opt = base_dme_opt;
        dummy_opt.X_val = [ones(50, 1), X_train(1:50, :)];
        dummy_dme = Distributed_MixtureOfExperts_Gaussian(X_train, Y_train, K, M, dummy_opt);
        fit_aavr = Aligned_MixtureOfExperts(dummy_dme, K, M, options);
        t_aavr_agg = toc(tic_aavr);
        fit_aavr.learning_time = max_local_time + t_aavr_agg;
        f_W_pool = dummy_dme.large_mixture;

        [lt_aavr, td_aavr, ll_aavr, mse_aavr, rpe_aavr, cr_aavr, ri_aavr, ari_aavr, ce_aavr, td_fW_aavr] = ...
            compute_metrics(fit_aavr, true_mixture, X_test, Y_test, Z_test, 0, 'AAVR', f_W_pool);

        % 2a. Centralized Oracle GLB
        tic_glb = tic;
        glb_opt = options;
        glb_opt.nb_EM_runs = options.nb_EM_runs;
        glb_opt.max_iter   = 40;
        glb_opt.verbose    = 0;
        fit_glb = Global_MixtureOfExperts(X_train, Y_train, K, glb_opt);
        t_glb = toc(tic_glb);
        fit_glb.learning_time = t_glb;

        [~, td_glb, ll_glb, mse_glb, rpe_glb, cr_glb, ri_glb, ari_glb, ce_glb, td_fW_glb] = ...
            compute_metrics(fit_glb, true_mixture, X_test, Y_test, Z_test, 0, 'GLB', f_W_pool);

        run_data = struct();
        run_data.baselines.GLB.metrics  = [t_glb, td_glb, ll_glb, mse_glb, rpe_glb, cr_glb, ri_glb, ari_glb, ce_glb, td_fW_glb];
        run_data.baselines.GLB.agg_time = t_glb;
        run_data.baselines.AAVR.metrics  = [lt_aavr, td_aavr, ll_aavr, mse_aavr, rpe_aavr, cr_aavr, ri_aavr, ari_aavr, ce_aavr, td_fW_aavr];
        run_data.baselines.AAVR.agg_time = t_aavr_agg;

        % 3. Sweep over S scale (Pooled i.i.d.)
        for s_idx = 1:length(S_list)
            S_val = S_list(s_idx);
            
            % Draw S samples from pooled covariates
            rand_idx = randperm(N_train, S_val);
            X_s = X_train(rand_idx, :);

            % Fit DME, GM, MED with this X_val
            [m_dme, t_agg_dme, m_gm, t_agg_gm, m_med, t_agg_med] = ...
                evaluate_s_condition(X_train, Y_train, X_s, K, M, base_dme_opt, options, ...
                                     true_mixture, X_test, Y_test, Z_test, max_local_time);

            s_key = sprintf('S_%d', S_val);
            run_data.pooled.(s_key).DME.metrics  = m_dme;
            run_data.pooled.(s_key).DME.agg_time = t_agg_dme;
            run_data.pooled.(s_key).GM.metrics  = m_gm;
            run_data.pooled.(s_key).GM.agg_time = t_agg_gm;
            run_data.pooled.(s_key).MED.metrics  = m_med;
            run_data.pooled.(s_key).MED.agg_time = t_agg_med;
        end

        % 4. Source: One Machine (S = 5000, Machine 1)
        X_one = X_train(client_indices{1}, :);
        [m_dme_one, t_agg_dme_one, m_gm_one, t_agg_gm_one, m_med_one, t_agg_med_one] = ...
            evaluate_s_condition(X_train, Y_train, X_one, K, M, base_dme_opt, options, ...
                                 true_mixture, X_test, Y_test, Z_test, max_local_time);

        run_data.one_machine.DME.metrics  = m_dme_one;
        run_data.one_machine.DME.agg_time = t_agg_dme_one;
        run_data.one_machine.GM.metrics  = m_gm_one;
        run_data.one_machine.GM.agg_time = t_agg_gm_one;
        run_data.one_machine.MED.metrics  = m_med_one;
        run_data.one_machine.MED.agg_time = t_agg_med_one;

        % 5. Source: Simulated x (S = 5000, Gaussian match on marginals)
        mu_x  = mean(X_train, 1);
        std_x = std(X_train, 0, 1);
        X_sim = repmat(mu_x, N_m, 1) + randn(N_m, d) .* repmat(std_x, N_m, 1);

        [m_dme_sim, t_agg_dme_sim, m_gm_sim, t_agg_gm_sim, m_med_sim, t_agg_med_sim] = ...
            evaluate_s_condition(X_train, Y_train, X_sim, K, M, base_dme_opt, options, ...
                                 true_mixture, X_test, Y_test, Z_test, max_local_time);

        run_data.simulated_x.DME.metrics  = m_dme_sim;
        run_data.simulated_x.DME.agg_time = t_agg_dme_sim;
        run_data.simulated_x.GM.metrics  = m_gm_sim;
        run_data.simulated_x.GM.agg_time = t_agg_gm_sim;
        run_data.simulated_x.MED.metrics  = m_med_sim;
        run_data.simulated_x.MED.agg_time = t_agg_med_sim;

        t_run_end = toc(t_run_start);
        fprintf('--> [Run %d/%d] Completed in %.1fs\n', r, runs_to_process, t_run_end);
        raw_runs{r} = run_data;
    end

    % Aggregate Means and STDs
    results = aggregate_sensitivity_results(raw_runs, S_list, runs_to_process);

    % Save results
    if strcmp(profile_name, 'balanced')
        out_mat = fullfile(out_dir, sprintf('Result_Sensitivity_S_N%s_M%d.mat', N_str, M));
    else
        out_mat = fullfile(out_dir, sprintf('Result_Sensitivity_S_N%s_M%d_%s.mat', N_str, M, profile_name));
    end
    save(out_mat, 'results', 'raw_runs', 'S_list', 'N_total', 'N_m', 'M', 'K', 'd', 'runs_to_process', 'profile_name', '-v7.3');
    fprintf('\n==> Successfully saved Sensitivity results to: %s\n', out_mat);

    % Export LaTeX Table (if exporter is available)
    if exist('export_latex_sensitivity_table', 'file')
        table_dir = fullfile(base_dir, 'results', 'tables');
        export_latex_sensitivity_table(out_mat, table_dir);
        fprintf('==> Exported sensitivity LaTeX table successfully!\n');
    end

end

%% Helper: Evaluate DME, GM, MED for a given support set X_val
function [m_dme, t_agg_dme, m_gm, t_agg_gm, m_med, t_agg_med] = ...
    evaluate_s_condition(X_train, Y_train, X_s, K, M, base_dme_opt, options, ...
                         true_mixture, X_test, Y_test, Z_test, max_local_time)

    opt_s = base_dme_opt;
    opt_s.X_val = [ones(size(X_s, 1), 1), X_s];

    % DME
    tic_dme = tic;
    fit_dme = Distributed_MixtureOfExperts_Gaussian(X_train, Y_train, K, M, opt_s);
    t_agg_dme = toc(tic_dme);
    fit_dme.learning_time = max_local_time + t_agg_dme;
    f_W = fit_dme.large_mixture;

    [lt_dme, td_dme, ll_dme, mse_dme, rpe_dme, cr_dme, ri_dme, ari_dme, ce_dme, td_fW_dme] = ...
        compute_metrics(fit_dme, true_mixture, X_test, Y_test, Z_test, 0, 'DME', f_W);
    m_dme = [lt_dme, td_dme, ll_dme, mse_dme, rpe_dme, cr_dme, ri_dme, ari_dme, ce_dme, td_fW_dme];

    % GM
    tic_gm = tic;
    fit_gm = Greedy_MixtureOfExperts(fit_dme, K, M, options);
    t_agg_gm = toc(tic_gm);
    fit_gm.learning_time = max_local_time + t_agg_gm;

    [lt_gm, td_gm, ll_gm, mse_gm, rpe_gm, cr_gm, ri_gm, ari_gm, ce_gm, td_fW_gm] = ...
        compute_metrics(fit_gm, true_mixture, X_test, Y_test, Z_test, 0, 'GM', f_W);
    m_gm = [lt_gm, td_gm, ll_gm, mse_gm, rpe_gm, cr_gm, ri_gm, ari_gm, ce_gm, td_fW_gm];

    % MED
    tic_med = tic;
    fit_med = Median_MixtureOfExperts(fit_dme, K, M, options);
    t_agg_med = toc(tic_med);
    fit_med.learning_time = max_local_time + t_agg_med;

    [lt_med, td_med, ll_med, mse_med, rpe_med, cr_med, ri_med, ari_med, ce_med, td_fW_med] = ...
        compute_metrics(fit_med, true_mixture, X_test, Y_test, Z_test, 0, 'MED', f_W);
    m_med = [lt_med, td_med, ll_med, mse_med, rpe_med, cr_med, ri_med, ari_med, ce_med, td_fW_med];

end

%% Helper: Aggregate results across runs
function results = aggregate_sensitivity_results(raw_runs, S_list, num_runs)
    results = struct();
    models  = {'DME', 'GM', 'MED'};

    % 1. Pooled S_list
    for s_idx = 1:length(S_list)
        S_val = S_list(s_idx);
        s_key = sprintf('S_%d', S_val);
        for m_idx = 1:length(models)
            mod = models{m_idx};
            m_mat = zeros(num_runs, 10);
            t_mat = zeros(num_runs, 1);
            for r = 1:num_runs
                m_mat(r, :) = raw_runs{r}.pooled.(s_key).(mod).metrics;
                t_mat(r)    = raw_runs{r}.pooled.(s_key).(mod).agg_time;
            end
            results.pooled.(s_key).(mod).mean = mean(m_mat, 1);
            results.pooled.(s_key).(mod).std  = std(m_mat, 0, 1);
            results.pooled.(s_key).(mod).agg_time_mean = mean(t_mat);
            results.pooled.(s_key).(mod).agg_time_std  = std(t_mat);
        end
    end

    % 2. Sources
    sources = {'one_machine', 'simulated_x'};
    for sk = 1:length(sources)
        skey = sources{sk};
        for m_idx = 1:length(models)
            mod = models{m_idx};
            m_mat = zeros(num_runs, 10);
            t_mat = zeros(num_runs, 1);
            for r = 1:num_runs
                m_mat(r, :) = raw_runs{r}.(skey).(mod).metrics;
                t_mat(r)    = raw_runs{r}.(skey).(mod).agg_time;
            end
            results.(skey).(mod).mean = mean(m_mat, 1);
            results.(skey).(mod).std  = std(m_mat, 0, 1);
            results.(skey).(mod).agg_time_mean = mean(t_mat);
            results.(skey).(mod).agg_time_std  = std(t_mat);
        end
    end

    % 3. Baselines
    base_models = {'GLB', 'AAVR'};
    for bm = 1:length(base_models)
        mod = base_models{bm};
        m_mat = zeros(num_runs, 10);
        t_mat = zeros(num_runs, 1);
        for r = 1:num_runs
            m_mat(r, :) = raw_runs{r}.baselines.(mod).metrics;
            t_mat(r)    = raw_runs{r}.baselines.(mod).agg_time;
        end
        results.baselines.(mod).mean = mean(m_mat, 1);
        results.baselines.(mod).std  = std(m_mat, 0, 1);
        results.baselines.(mod).agg_time_mean = mean(t_mat);
        results.baselines.(mod).agg_time_std  = std(t_mat);
    end
end

function s = format_N_str(N)
    if N >= 1e6
        s = sprintf('%dM', round(N / 1e6));
    elseif N >= 1e3
        s = sprintf('%dk', round(N / 1e3));
    else
        s = sprintf('%d', N);
    end
end
