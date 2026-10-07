function results = run_year_msd_benchmark(use_sub10, K, S, max_iter, nb_EM_runs)
%% RUN_YEAR_MSD_BENCHMARK
% Evaluates Distributed Mixture-of-Experts on YearPredictionMSD (UCI ID: 203):
%   - M = 32 decentralized worker machines (parfor m = 1:M)
%   - Features (d = 90): 12 timbre average + 78 timbre covariance
%   - Target: Release year of song [1922, 2011] (Gaussian MoE regression)
%   - Train / Test split:
%       * use_sub10 = true  (default): 1/10 subset (Train N=46,372, Test N=5,163)
%       * use_sub10 = false: Full dataset (Train N=463,715, Test N=51,630)
%
% Usage:
%   cd DME_GitHub/real_data/year_prediction_msd
%   run_year_msd_benchmark;             % Default: use_sub10=true, K=4, S=2000
%   run_year_msd_benchmark(false);      % Full dataset production run

    if nargin < 5 || isempty(nb_EM_runs), nb_EM_runs = 2; end
    if nargin < 4 || isempty(max_iter),   max_iter = 30; end
    if nargin < 3 || isempty(S),          S = 2000; end
    if nargin < 2 || isempty(K),          K = 4; end
    if nargin < 1 || isempty(use_sub10),  use_sub10 = true; end

    current_dir = fileparts(mfilename('fullpath'));
    base_dir    = fullfile(current_dir, '..', '..');
    
    addpath(base_dir);
    addpath(fullfile(base_dir, 'models'));
    addpath(fullfile(base_dir, 'datatools'));
    addpath(fullfile(base_dir, 'stattools'));
    addpath(fullfile(base_dir, 'evaltools'));

    fprintf('========================================================================\n');
    fprintf('  YEARPREDICTIONMSD REAL-DATA BENCHMARK (M = 32 Worker Nodes)           \n');
    fprintf('  Target: Release Year [1922, 2011] | K = %d Experts | S = %d Support   \n', K, S);
    if use_sub10
        fprintf('  MODE: 1/10 Subsample Verification (Smoke Test)                        \n');
    else
        fprintf('  MODE: Full Production Dataset (N_train = 463,715, N_test = 51,630)    \n');
    end
    fprintf('========================================================================\n');

    if use_sub10
        mat_file = fullfile(current_dir, 'year_msd_sub10.mat');
    else
        mat_file = fullfile(current_dir, 'year_msd_processed.mat');
    end

    if ~exist(mat_file, 'file')
        error('Data file not found: %s. Run preprocess_year_msd.py first.', mat_file);
    end

    fprintf('Loading dataset from: %s ...\n', mat_file);
    data = load(mat_file);

    M              = double(data.M);
    d              = double(data.d);
    X_train_cells  = data.X_train_cells;
    Y_train_cells  = data.Y_train_cells;
    X_test_pooled  = data.X_test_pooled;
    Y_test_pooled  = data.Y_test_pooled;
    X_train_pooled = data.X_train_pooled;
    Y_train_pooled = data.Y_train_pooled;
    N_test         = size(X_test_pooled, 1);
    N_train        = size(X_train_pooled, 1);

    fprintf('Dataset Summary:\n');
    fprintf('  - Worker Nodes (M):     %d\n', M);
    fprintf('  - Predictors (d):       %d\n', d);
    fprintf('  - Total Train samples:  %d (%d samples / machine)\n', N_train, round(N_train / M));
    fprintf('  - Total Test samples:   %d (Strictly out-of-sample artist split)\n', N_test);

    % Algorithmic configuration
    options = get_options('default');
    options.nb_EM_runs = nb_EM_runs;
    options.max_iter   = max_iter;
    options.verbose    = 0;
    options.S          = S;
    options.threshold  = 1e-4;

    models_to_evaluate = {'GLB', 'DME', 'GM', 'AAVR', 'FED', 'WAVR', 'MED'};
    results = struct();

    % -------------------------------------------------------------------------
    % Step 1: Fit Local MoE Models on M=32 Machines in Parallel (parfor)
    % -------------------------------------------------------------------------
    fprintf('\n--> [1/3] Fitting local MoE models on %d decentralized workers (parfor m=1:%d)...\n', M, M);
    local_fits = cell(1, M);
    local_times = zeros(1, M);
    
    t_local_start = tic;
    parfor m = 1:M
        t_m = tic;
        X_m = X_train_cells{m};
        Y_m = Y_train_cells{m};
        opt_m = options;
        opt_m.verbose = 0;
        local_fits{m} = Global_MixtureOfExperts(X_m, Y_m, K, opt_m);
        local_times(m) = toc(t_m);
    end
    t_local_max = max(local_times);
    fprintf('    Local training complete. Max worker time: %.2f s (Wall-clock: %.2f s)\n', ...
            t_local_max, toc(t_local_start));

    % Subsample supporting dataset D_S uniformly from pooled train
    rng(42, 'twister');
    s_idx = randperm(N_train, min(S, N_train));
    options.X_val = [ones(min(S, N_train), 1), X_train_pooled(s_idx, :)];
    options.Y_val = Y_train_pooled(s_idx);

    % Pack into DMEfit structure for standard aggregator compatibility
    DMEfit = struct();
    DMEfit.local_estimates = local_fits;
    DMEfit.local_times     = local_times;
    DMEfit.M               = M;
    DMEfit.K               = K;
    DMEfit.d               = d;
    DMEfit.n               = N_train;
    DMEfit.X_val           = options.X_val;

    % Construct pooled mixture f^W for reduction divergence calculation D(f, f^W)
    experts   = [];
    gates     = [];
    variances = [];
    weights   = [];
    for m = 1:M
        loc_b0 = local_fits{m}.param.beta0;
        loc_B  = local_fits{m}.param.Beta;
        loc_a0 = local_fits{m}.param.alpha0;
        loc_A  = local_fits{m}.param.Alpha;
        if length(loc_a0) == K - 1
            loc_a0 = [loc_a0, 0];
            loc_A  = [loc_A, zeros(d, 1)];
        end
        experts   = [experts,   [loc_b0; loc_B]];
        gate_     = [loc_a0; loc_A];
        gates     = [gates,     repmat(gate_(:), 1, K)];
        variances = [variances, local_fits{m}.param.sigma2];
        weights   = [weights,   (1/M) * ones(1, K)];
    end
    f_W = struct('experts', experts, 'gates', gates, 'variances', variances, 'weights', weights);

    % -------------------------------------------------------------------------
    % Step 2: Fit and Evaluate Each Estimator
    % -------------------------------------------------------------------------
    fprintf('\n--> [2/3] Executing Aggregation & Oracle Benchmarks...\n');

    for i = 1:length(models_to_evaluate)
        mod = models_to_evaluate{i};
        fprintf('    Evaluating %-4s ... ', mod);
        t_eval = tic;

        switch mod
            case 'GLB'
                % Centralized Global Oracle on pooled training set
                opt_glb = options;
                opt_glb.verbose = 0;
                fit = Global_MixtureOfExperts(X_train_pooled, Y_train_pooled, K, opt_glb);
                learning_time = fit.learning_time;

            case 'DME'
                % One-shot Optimal Transport Aggregation
                opt_dme = options;
                opt_dme.local_estimates = local_fits;
                opt_dme.local_times     = local_times;
                fit = Distributed_MixtureOfExperts_Gaussian(X_train_pooled, Y_train_pooled, K, M, opt_dme);
                learning_time = fit.learning_time;

            case 'GM'
                fit = Greedy_MixtureOfExperts(DMEfit, K, M, options);
                learning_time = t_local_max + fit.learning_time;

            case 'AAVR'
                fit = Aligned_MixtureOfExperts(DMEfit, K, M, options);
                learning_time = t_local_max + fit.learning_time;

            case 'FED'
                % Simulated Federated Averaging (T=5 rounds)
                opt_fed = options;
                opt_fed.client_indices = cell(1, M);
                curr = 1;
                for m = 1:M
                    n_m = size(X_train_cells{m}, 1);
                    opt_fed.client_indices{m} = (curr : curr + n_m - 1)';
                    curr = curr + n_m;
                end
                fit = FedAvg_MixtureOfExperts(X_train_pooled, Y_train_pooled, K, M, opt_fed);
                learning_time = fit.learning_time;

            case 'WAVR'
                fit = Averaged_MixtureOfExperts(DMEfit, K, M, options);
                learning_time = t_local_max + fit.learning_time;

            case 'MED'
                fit = Median_MixtureOfExperts(DMEfit, K, M, options);
                learning_time = t_local_max + fit.learning_time;
        end

        % Compute Real-Data Prediction Metrics on Unseen Test Set
        [rmse, mae, rpe, corr_val, loglik, d_fW] = evaluate_realdata_prediction(fit, X_test_pooled, Y_test_pooled, K, f_W, options.X_val);

        results.(mod).RMSE          = rmse;
        results.(mod).MAE           = mae;
        results.(mod).RPE           = rpe;
        results.(mod).Correlation   = corr_val;
        results.(mod).Test_Loglik   = loglik;
        results.(mod).D_fW          = d_fW;
        results.(mod).Learning_Time = learning_time;
        results.(mod).fit           = fit;

        fprintf('Done (%5.1f s) | RMSE = %5.2f yrs | RPE = %0.4f | Corr = %0.4f | D(f^W) = %6.1f | Time = %5.1f s\n', ...
                toc(t_eval), rmse, rpe, corr_val, d_fW, learning_time);
    end

    % -------------------------------------------------------------------------
    % Step 3: Print Standardized Comparative Summary & Export LaTeX Table
    % -------------------------------------------------------------------------
    fprintf('\n====================================================================================================\n');
    fprintf('  YEARPREDICTIONMSD REAL-DATA BENCHMARK RESULTS (Test N=%d, d=90, M=32)                            \n', N_test);
    fprintf('====================================================================================================\n');
    fprintf('%-24s | %-10s | %-8s | %-8s | %-8s | %-10s | %-8s\n', ...
            'Estimator', 'RMSE (yrs)', 'MAE', 'RPE', 'Corr (r)', 'D(f^W)', 'Time (s)');
    fprintf('----------------------------------------------------------------------------------------------------\n');

    for i = 1:length(models_to_evaluate)
        mod = models_to_evaluate{i};
        r = results.(mod);
        fprintf('%-24s | %10.2f | %8.2f | %8.4f | %8.4f | %10.2f | %8.2f\n', ...
                model_display_name(mod), r.RMSE, r.MAE, r.RPE, r.Correlation, r.D_fW, r.Learning_Time);
    end
    fprintf('====================================================================================================\n');

    % Save results
    if use_sub10
        out_res_file = fullfile(current_dir, 'result_year_msd_sub10.mat');
        tex_file     = fullfile(current_dir, 'table_year_msd_sub10.tex');
    else
        out_res_file = fullfile(current_dir, 'result_year_msd.mat');
        tex_file     = fullfile(current_dir, 'table_year_msd.tex');
    end

    save(out_res_file, 'results', 'models_to_evaluate', 'M', 'K', 'd', 'options', 'use_sub10', '-v7.3');
    fprintf('Saved numerical results to: %s\n', out_res_file);

    % Export LaTeX table
    export_year_msd_latex(results, models_to_evaluate, tex_file, N_train, N_test, M, d, K);
    fprintf('Exported LaTeX table to:    %s\n', tex_file);
end

% -------------------------------------------------------------------------
% Helper: Compute real-world regression metrics + Reduction Divergence D(f, f^W)
% -------------------------------------------------------------------------
function [rmse, mae, rpe, corr_val, loglik, d_fW] = evaluate_realdata_prediction(fit, X, Y, K, f_W, X_val)
    N = size(X, 1);
    
    % Gating probability calculation
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

    % Expert predictions Y_pred = sum_k pi_k(x) * (beta0_k + x'*beta_k)
    Y_experts = fit.param.beta0 + X * fit.param.Beta;
    Y_pred    = sum(gatingProb .* Y_experts, 2);

    % Errors
    residuals = Y - Y_pred;
    rmse      = sqrt(mean(residuals.^2));
    mae       = mean(abs(residuals));
    var_total = sum((Y - mean(Y)).^2);
    if var_total > 0
        rpe   = sum(residuals.^2) / var_total;
    else
        rpe   = NaN;
    end
    c_mat     = corrcoef(Y, Y_pred);
    if numel(c_mat) >= 4
        corr_val  = c_mat(1, 2);
    else
        corr_val  = NaN;
    end

    % Out-of-sample log-likelihood
    log_phi = zeros(N, K);
    for k = 1:K
        s2 = max(fit.param.sigma2(k), 1e-4);
        res_k = Y - fit.param.beta0(k) - X * fit.param.Beta(:, k);
        log_phi(:, k) = -0.5 * log(2 * pi * s2) - 0.5 * (res_k.^2) / s2;
    end
    loglik = sum(logsumexp(log(gatingProb) + log_phi, 2));

    % Reduction divergence D(f, f^W) on supporting set X_val
    if ~isempty(f_W) && ~isempty(X_val)
        if isfield(fit, 'reduced_mixture')
            fit_eval = fit.reduced_mixture;
        else
            fit_eval = fit;
        end
        try
            [plan_fW, dist_fW] = argmin_transportation_plan(f_W, fit_eval, X_val);
            d_fW = sum(sum(sum(plan_fW .* dist_fW)));
        catch
            d_fW = NaN;
        end
    else
        d_fW = NaN;
    end
end

% -------------------------------------------------------------------------
% Helper: Display names
% -------------------------------------------------------------------------
function name = model_display_name(mod)
    switch mod
        case 'GLB', name = 'Global oracle (G)';
        case 'DME', name = 'Proposed (DME)';
        case 'GM', name = 'Greedy merging (GM)';
        case 'AAVR', name = 'Aligned averaging (AAVR)';
        case 'FED', name = 'Federated averaging (FED)';
        case 'WAVR', name = 'Naive averaging (WAVR)';
        case 'MED', name = 'Coordinate median (MED)';
        otherwise,  name = mod;
    end
end

% -------------------------------------------------------------------------
% Helper: Export LaTeX Table
% -------------------------------------------------------------------------
function export_year_msd_latex(results, models, out_path, N_train, N_test, M, d, K)
    fid = fopen(out_path, 'w');
    if fid == -1, return; end

    fprintf(fid, '%% Auto-generated LaTeX table for YearPredictionMSD Benchmark\n');
    fprintf(fid, '\\begin{table}[t]\n\\centering\n');
    fprintf(fid, '\\caption{YearPredictionMSD Benchmark: Evaluation of distributed MoE estimators on large-scale audio timbre regression ($M=%d$ worker nodes, $N_{\\text{train}}=%s$, out-of-sample test $N_{\\text{test}}=%s$, $d=%d$, $K=%d$). Best distributed performance is in \\textbf{bold}; Centralized Global Oracle ($G$) serves as upper-bound reference.}\n', ...
            M, num2str(N_train), num2str(N_test), d, K);
    fprintf(fid, '\\label{tab:year_msd_benchmark}\n\\vspace{1mm}\n');
    fprintf(fid, '\\setlength{\\tabcolsep}{7pt}\n');
    fprintf(fid, '\\begin{tabular}{lcccccc}\n');
    fprintf(fid, '\\hline\\hline\n');
    fprintf(fid, 'Estimator & RMSE (yrs) $\\downarrow$ & MAE $\\downarrow$ & RPE $\\downarrow$ & Corr ($r$) $\\uparrow$ & $\\mathcal{D}(f, f^W)$ $\\downarrow$ & Time (s) $\\downarrow$ \\\\\n');
    fprintf(fid, '\\hline\n');

    % Find best distributed values (excluding GLB)
    best_rmse = inf; best_mae = inf; best_rpe = inf; best_corr = -inf; best_dfW = inf;
    for i = 1:length(models)
        mod = models{i};
        if strcmp(mod, 'GLB'), continue; end
        r = results.(mod);
        if r.RMSE < best_rmse, best_rmse = r.RMSE; end
        if r.MAE < best_mae,   best_mae = r.MAE; end
        if r.RPE < best_rpe,   best_rpe = r.RPE; end
        if r.Correlation > best_corr, best_corr = r.Correlation; end
        if ~isnan(r.D_fW) && r.D_fW < best_dfW, best_dfW = r.D_fW; end
    end

    tol = 1e-4;
    for i = 1:length(models)
        mod = models{i};
        r = results.(mod);
        name = model_display_name(mod);
        is_glb = strcmp(mod, 'GLB');

        % Format strings
        str_rmse = sprintf('%.2f', r.RMSE);
        if ~is_glb && abs(r.RMSE - best_rmse) < tol, str_rmse = sprintf('\\textbf{%s}', str_rmse); end

        str_mae = sprintf('%.2f', r.MAE);
        if ~is_glb && abs(r.MAE - best_mae) < tol, str_mae = sprintf('\\textbf{%s}', str_mae); end

        str_rpe = sprintf('%.4f', r.RPE);
        if ~is_glb && abs(r.RPE - best_rpe) < tol, str_rpe = sprintf('\\textbf{%s}', str_rpe); end

        str_corr = sprintf('%.4f', r.Correlation);
        if ~is_glb && abs(r.Correlation - best_corr) < tol, str_corr = sprintf('\\textbf{%s}', str_corr); end

        if isnan(r.D_fW)
            str_dfW = '---';
        else
            str_dfW = sprintf('%.1f', r.D_fW);
            if ~is_glb && abs(r.D_fW - best_dfW) < tol, str_dfW = sprintf('\\textbf{%s}', str_dfW); end
        end

        str_time = sprintf('%.1f', r.Learning_Time);

        fprintf(fid, '%-24s & %s & %s & %s & %s & %s & %s \\\\\n', ...
                name, str_rmse, str_mae, str_rpe, str_corr, str_dfW, str_time);

        if is_glb
            fprintf(fid, '\\hline\n');
        end
    end

    fprintf(fid, '\\hline\\hline\n');
    fprintf(fid, '\\end{tabular}\n');
    fprintf(fid, '\\end{table}\n');
    fclose(fid);
end
