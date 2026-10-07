function results = run_beijing_benchmark(K, S, max_iter, nb_EM_runs)
%% RUN_BEIJING_BENCHMARK
% Evaluates Distributed Mixture-of-Experts on Beijing Multi-Site Air Quality Data:
%   - 12 real decentralized monitoring stations (M = 12 natural machines)
%   - Rigorous out-of-time test split:
%       Train: 2013-03-01 to 2016-02-29 (315,360 hourly samples)
%       Test:  2016-03-01 to 2017-02-28 (105,120 future samples, 1 full year)
%   - Features (d = 19): Autoregressive lags, meteorology, gas precursors, cyclical time
%   - Target: PM2.5 concentration (ug/m^3)
%   - Models: GLB, DME, GM (GM), AAVR (AAVR), FED (T=1, 5, 10, 20), WAVR (WAVR), MED
%   - Metrics: Trandis to f^W and GLB, RMSE, MAE, RPE, Correlation, Learning Time
%
% Usage:
%   run_beijing_benchmark;                % Default: K=4, S=2000, max_iter=20, nb_EM_runs=2
%   run_beijing_benchmark(K, S, max_iter, nb_EM_runs);

    if nargin < 4 || isempty(nb_EM_runs), nb_EM_runs = 2; end
    if nargin < 3 || isempty(max_iter),   max_iter = 20; end
    if nargin < 2 || isempty(S),          S = 2000; end
    if nargin < 1 || isempty(K),          K = 4; end

    current_dir = fileparts(mfilename('fullpath'));
    base_dir    = fullfile(current_dir, '..', '..');
    
    addpath(base_dir);
    addpath(fullfile(base_dir, 'models'));
    addpath(fullfile(base_dir, 'datatools'));
    addpath(fullfile(base_dir, 'stattools'));
    addpath(fullfile(base_dir, 'evaltools'));

    fprintf('========================================================================\n');
    fprintf('  BEIJING MULTI-SITE AIR QUALITY REAL-DATA BENCHMARK (M=12 Stations)   \n');
    fprintf('  Target: PM2.5 (ug/m^3) | K = %d Experts | S = %d Support Samples      \n', K, S);
    fprintf('========================================================================\n');

    mat_file = fullfile(current_dir, 'beijing_air_quality_processed.mat');
    if ~exist(mat_file, 'file')
        error('Data file not found: %s. Run preprocess_beijing.py first.', mat_file);
    end

    fprintf('Loading processed Beijing dataset from: %s\n', mat_file);
    data = load(mat_file);

    M              = double(data.M);
    d              = double(data.d);
    stations       = data.stations;
    X_train_cells  = data.X_train_cells;
    Y_train_cells  = data.Y_train_cells;
    X_test_pooled  = data.X_test_pooled;
    Y_test_pooled  = data.Y_test_pooled;
    X_train_pooled = data.X_train_pooled;
    Y_train_pooled = data.Y_train_pooled;
    N_test         = size(X_test_pooled, 1);
    N_train        = size(X_train_pooled, 1);

    fprintf('Dataset Summary:\n');
    fprintf('  - Stations (M):         %d\n', M);
    fprintf('  - Predictors (d):       %d\n', d);
    fprintf('  - Total Train samples:  %d (3 years: 2013-03 to 2016-02)\n', N_train);
    fprintf('  - Total Test samples:   %d (1 year:  2016-03 to 2017-02)\n', N_test);

    % Configure algorithmic options
    options = get_options('default');
    options.nb_EM_runs = nb_EM_runs;
    options.max_iter   = max_iter;
    options.verbose    = 0;
    options.S          = S;
    options.threshold  = 1e-4;

    models_to_evaluate = {'GLB', 'DME', 'GM', 'AAVR', 'FED_T1', 'FED_T5', 'FED_T10', 'FED_T20', 'WAVR', 'MED'};
    results = struct();

    % -------------------------------------------------------------------------
    % Step 1: Fit Local MoE Models on 12 Stations (Shared by One-Shot Aggregators)
    % -------------------------------------------------------------------------
    fprintf('\n--> [1/3] Fitting local MoE models on %d decentralized stations (parfor)...\n', M);
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
    fprintf('    Local training complete. Max station time: %.2f s (Total wall-clock: %.2f s)\n', ...
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

    % Build pooled local mixture f^W (48 components)
    f_W = struct();
    experts_W   = [];
    gates_W     = [];
    variances_W = [];
    weights_W   = [];
    for m = 1:M
        loc_p = local_fits{m}.param;
        experts_W = [experts_W, [loc_p.beta0(:)'; loc_p.Beta]];
        if length(loc_p.alpha0) == K - 1
            loc_a0 = [loc_p.alpha0(:)', 0];
            loc_A  = [loc_p.Alpha, zeros(d, 1)];
        else
            loc_a0 = loc_p.alpha0(:)';
            loc_A  = loc_p.Alpha;
        end
        gate_ = [loc_a0; loc_A];
        gates_W = [gates_W, repmat(gate_(:), 1, K)];
        variances_W = [variances_W, loc_p.sigma2(:)'];
        n_m = size(X_train_cells{m}, 1);
        weights_W = [weights_W, (n_m / N_train) * (1/K) * ones(1, K)];
    end
    f_W.experts   = experts_W;
    f_W.gates     = gates_W;
    f_W.variances = variances_W;
    f_W.weights   = weights_W;

    % Precompute PI_hat for f_W on options.X_val (S x L)
    PI_hat_W = [];
    for m = 1:M
        loc_p = local_fits{m}.param;
        if length(loc_p.alpha0) == K - 1
            loc_a0 = [loc_p.alpha0(:)', 0];
            loc_A  = [loc_p.Alpha, zeros(d, 1)];
        else
            loc_a0 = loc_p.alpha0(:)';
            loc_A  = loc_p.Alpha;
        end
        gate_m = [loc_a0; loc_A];
        logits_m = options.X_val * gate_m;
        logits_m = logits_m - max(logits_m, [], 2);
        prob_m = exp(logits_m) ./ sum(exp(logits_m), 2);
        n_m = size(X_train_cells{m}, 1);
        PI_hat_W = [PI_hat_W, (n_m / N_train) * prob_m];
    end
    f_W.PI_hat = PI_hat_W;

    % -------------------------------------------------------------------------
    % Step 1b: Pre-train FedAvg Trajectory (T=20 with snapshots)
    % -------------------------------------------------------------------------
    fprintf('\n--> Pre-training FedAvg trajectory (T=20, snapshots at T in {1, 5, 10, 20})...\n');
    opt_fed = options;
    opt_fed.FedAvg_rounds    = 20;
    opt_fed.FedAvg_snapshots = [1, 5, 10, 20];
    opt_fed.client_indices   = cell(1, M);
    curr = 1;
    for m = 1:M
        n_m = size(X_train_cells{m}, 1);
        opt_fed.client_indices{m} = (curr : curr + n_m - 1)';
        curr = curr + n_m;
    end
    fed_traj = FedAvg_MixtureOfExperts(X_train_pooled, Y_train_pooled, K, M, opt_fed);

    % -------------------------------------------------------------------------
    % Step 2: Fit and Evaluate Each Estimator
    % -------------------------------------------------------------------------
    fprintf('\n--> [2/3] Executing Aggregation & Oracle Benchmarks...\n');
    mix_glb = [];

    for i = 1:length(models_to_evaluate)
        mod = models_to_evaluate{i};
        fprintf('    Evaluating %-8s ... ', mod);
        t_eval = tic;

        switch mod
            case 'GLB'
                % Centralized Global Oracle on pooled training set
                opt_glb = options;
                opt_glb.verbose = 0;
                fit = Global_MixtureOfExperts(X_train_pooled, Y_train_pooled, K, opt_glb);
                learning_time = fit.learning_time;
                if isfield(fit, 'reduced_mixture')
                    mix_glb = fit.reduced_mixture;
                else
                    mix_glb = fit;
                end

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

            case {'FED_T1', 'FED_T5', 'FED_T10', 'FED_T20'}
                if isfield(fed_traj, 'snapshots') && isfield(fed_traj.snapshots, mod)
                    fit = fed_traj.snapshots.(mod);
                else
                    t_val = str2double(strrep(mod, 'FED_T', ''));
                    f_opt = opt_fed;
                    f_opt.FedAvg_rounds = t_val;
                    fit = FedAvg_MixtureOfExperts(X_train_pooled, Y_train_pooled, K, M, f_opt);
                end
                learning_time = fit.learning_time;

            case 'WAVR'
                fit = Averaged_MixtureOfExperts(DMEfit, K, M, options);
                learning_time = t_local_max + fit.learning_time;

            case 'MED'
                fit = Median_MixtureOfExperts(DMEfit, K, M, options);
                learning_time = t_local_max + fit.learning_time;
        end

        % Compute Real-Data Prediction Metrics on Unseen Test Set
        [rmse, mae, rpe, corr_val, loglik] = evaluate_realdata_prediction(fit, X_test_pooled, Y_test_pooled, K);

        % Extract mixture structure for OT transportation divergence
        if isfield(fit, 'reduced_mixture')
            mix_fit = fit.reduced_mixture;
        else
            mix_fit = fit;
        end

        % Compute Transportation Divergence to pooled mixture f^W
        try
            [plan_fW, dist_fW] = argmin_transportation_plan(f_W, mix_fit, options.X_val);
            trandis_fW = sum(sum(sum(plan_fW .* dist_fW)));
        catch ME
            fprintf('(trandis_fW err: %s) ', ME.message);
            trandis_fW = NaN;
        end

        % Compute Transportation Divergence to GLB (Centralized Oracle)
        if ~isempty(mix_glb)
            if strcmp(mod, 'GLB')
                trandis_glb = 0;
            else
                try
                    glb_src = struct();
                    glb_src.experts   = mix_glb.experts;
                    glb_src.variances = mix_glb.variances;
                    glb_src.weights   = ones(1, K) / K;
                    % Precompute PI_hat for GLB on options.X_val
                    if size(mix_glb.gates, 2) == K - 1
                        g_full = [mix_glb.gates, zeros(d+1, 1)];
                    else
                        g_full = mix_glb.gates;
                    end
                    logits_glb = options.X_val * g_full;
                    logits_glb = logits_glb - max(logits_glb, [], 2);
                    glb_src.PI_hat = exp(logits_glb) ./ sum(exp(logits_glb), 2);
                    glb_src.gates  = repmat(g_full(:), 1, K);

                    [plan_glb, dist_glb] = argmin_transportation_plan(glb_src, mix_fit, options.X_val);
                    trandis_glb = sum(sum(sum(plan_glb .* dist_glb)));
                catch ME
                    fprintf('(trandis_glb err: %s) ', ME.message);
                    trandis_glb = NaN;
                end
            end
        else
            trandis_glb = 0;
        end

        results.(mod).RMSE          = rmse;
        results.(mod).MAE           = mae;
        results.(mod).RPE           = rpe;
        results.(mod).Correlation   = corr_val;
        results.(mod).Test_Loglik   = loglik;
        results.(mod).Trandis_fW    = trandis_fW;
        results.(mod).Trandis_GLB   = trandis_glb;
        results.(mod).Learning_Time = learning_time;
        results.(mod).fit           = fit;

        fprintf('Done (%4.1fs) | Tc(fW)=%6.2f | RMSE=%5.2f | MAE=%5.2f | RPE=%0.4f | r=%0.4f | Time=%5.1fs\n', ...
                toc(t_eval), trandis_fW, rmse, mae, rpe, corr_val, learning_time);
    end

    % -------------------------------------------------------------------------
    % Step 3: Print Standardized Comparative Summary & Export LaTeX Table
    % -------------------------------------------------------------------------
    fprintf('\n========================================================================================================\n');
    fprintf('  BEIJING REAL-DATA BENCHMARK RESULTS (Out-of-Time Test: 2016-03 to 2017-02, N=105,120)                \n');
    fprintf('========================================================================================================\n');
    fprintf('%-18s | %-12s | %-12s | %-10s | %-10s | %-8s | %-8s | %-10s\n', ...
            'Estimator', 'Tc(.,GLB)', 'Tc(.,fW)', 'RMSE (ug/m3)', 'MAE', 'RPE', 'Corr (r)', 'Time (s)');
    fprintf('--------------------------------------------------------------------------------------------------------\n');

    for i = 1:length(models_to_evaluate)
        mod = models_to_evaluate{i};
        r = results.(mod);
        fprintf('%-18s | %12.2f | %12.2f | %14.2f | %10.2f | %8.4f | %8.4f | %10.2f\n', ...
                model_display_name(mod), r.Trandis_GLB, r.Trandis_fW, r.RMSE, r.MAE, r.RPE, r.Correlation, r.Learning_Time);
    end
    fprintf('========================================================================================================\n');

    % Save results mat
    out_res_file = fullfile(current_dir, 'result_beijing_air_quality.mat');
    save(out_res_file, 'results', 'models_to_evaluate', 'M', 'K', 'd', 'options', '-v7.3');
    fprintf('Saved numerical results to: %s\n', out_res_file);

    % Export LaTeX table
    tex_file = fullfile(current_dir, 'table_beijing_air_quality.tex');
    export_beijing_latex(results, models_to_evaluate, tex_file);
    fprintf('Exported LaTeX table to:    %s\n', tex_file);
end

% -------------------------------------------------------------------------
% Helper: Compute real-world regression metrics
% -------------------------------------------------------------------------
function [rmse, mae, rpe, corr_val, loglik] = evaluate_realdata_prediction(fit, X, Y, K)
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
    corr_val  = c_mat(1, 2);

    % Out-of-sample log-likelihood
    log_phi = zeros(N, K);
    for k = 1:K
        s2 = max(fit.param.sigma2(k), 1e-4);
        res_k = Y - fit.param.beta0(k) - X * fit.param.Beta(:, k);
        log_phi(:, k) = -0.5 * log(2 * pi * s2) - 0.5 * (res_k.^2) / s2;
    end
    loglik = sum(logsumexp(log(gatingProb) + log_phi, 2));
end

% -------------------------------------------------------------------------
% Helper: Display names
% -------------------------------------------------------------------------
function name = model_display_name(mod)
    switch mod
        case 'GLB',     name = 'GLB';
        case 'DME',     name = 'DME';
        case 'GM',     name = 'GM';
        case 'AAVR',     name = 'AAVR';
        case 'FED_T1',  name = 'FED ($T=1$)';
        case 'FED_T5',  name = 'FED ($T=5$)';
        case 'FED_T10', name = 'FED ($T=10$)';
        case 'FED_T20', name = 'FED ($T=20$)';
        case 'WAVR',     name = 'WAVR';
        case 'MED',     name = 'MED';
        otherwise,      name = mod;
    end
end

% -------------------------------------------------------------------------
% Helper: Export LaTeX Table
% -------------------------------------------------------------------------
function export_beijing_latex(results, models, out_path)
    fid = fopen(out_path, 'w');
    if fid == -1, return; end

    fprintf(fid, '%% Auto-generated LaTeX table for Beijing Multi-Site Air Quality Benchmark\n');
    fprintf(fid, '\\begin{table}[t]\n\\centering\n');
    fprintf(fid, '\\caption{Forecasting performance on the Beijing air quality sensor network ($M=12$ stations, $N_{\\text{train}}=315,360$, out-of-time test $N_{\\text{test}}=105,120$, $d=19$, $K=4$). RMSE and MAE are in $\\mu$g/m$^3$; Time is in seconds; $\\mathcal{T}_c$, RPE, and Pearson correlation ($r$) are dimensionless. Best distributed performance is in \\textbf{bold}; centralized oracle (GLB) serves as theoretical upper bound.}\n');
    fprintf(fid, '\\label{tab:beijing_air_quality}\n\\vspace{1mm}\n');
    fprintf(fid, '\\setlength{\\tabcolsep}{3.5pt}\n');
    fprintf(fid, '\\resizebox{\\columnwidth}{!}{\n');
    fprintf(fid, '\\begin{tabular}{lccccccc}\n\\toprule\n');
    fprintf(fid, '\\textbf{Estimator} & $\\mathcal{T}_c(\\cdot,\\widehat f_{\\mathrm{GLB}})$ & $\\mathcal{T}_c(\\cdot,\\bar f^W)$ & \\textbf{RMSE} & \\textbf{MAE} & \\textbf{RPE} & \\textbf{Corr. ($r$)} & \\textbf{Time (s)} \\\\\n');
    fprintf(fid, '\\midrule\n');

    % Global Oracle
    r = results.GLB;
    fprintf(fid, '%-16s & $0.00$ & $%.2f$ & $%.2f$ & $%.2f$ & $%.4f$ & $%.4f$ & $%.1f$ \\\\\n', ...
            'GLB', r.Trandis_fW, r.RMSE, r.MAE, r.RPE, r.Correlation, r.Learning_Time);
    fprintf(fid, '\\midrule\n');

    % Distributed estimators
    dist_mods = models(~strcmp(models, 'GLB'));

    % Rounded values for tie-breaking
    v_tc_glb = cellfun(@(m) sprintf('%.2f', results.(m).Trandis_GLB), dist_mods, 'UniformOutput', false);
    v_tc_fw  = cellfun(@(m) sprintf('%.2f', results.(m).Trandis_fW), dist_mods, 'UniformOutput', false);
    v_rmse   = cellfun(@(m) sprintf('%.2f', results.(m).RMSE), dist_mods, 'UniformOutput', false);
    v_mae    = cellfun(@(m) sprintf('%.2f', results.(m).MAE), dist_mods, 'UniformOutput', false);
    v_rpe    = cellfun(@(m) sprintf('%.4f', results.(m).RPE), dist_mods, 'UniformOutput', false);
    v_corr   = cellfun(@(m) sprintf('%.4f', results.(m).Correlation), dist_mods, 'UniformOutput', false);
    v_time   = cellfun(@(m) sprintf('%.1f', results.(m).Learning_Time), dist_mods, 'UniformOutput', false);

    best_tc_glb = sprintf('%.2f', min(cellfun(@(m) results.(m).Trandis_GLB, dist_mods)));
    best_tc_fw  = sprintf('%.2f', min(cellfun(@(m) results.(m).Trandis_fW, dist_mods)));
    best_rmse   = sprintf('%.2f', min(cellfun(@(m) results.(m).RMSE, dist_mods)));
    best_mae    = sprintf('%.2f', min(cellfun(@(m) results.(m).MAE, dist_mods)));
    best_rpe    = sprintf('%.4f', min(cellfun(@(m) results.(m).RPE, dist_mods)));
    best_corr   = sprintf('%.4f', max(cellfun(@(m) results.(m).Correlation, dist_mods)));
    best_time   = sprintf('%.1f', min(cellfun(@(m) results.(m).Learning_Time, dist_mods)));

    for i = 1:length(dist_mods)
        m = dist_mods{i};
        
        s_tc_glb = format_str(v_tc_glb{i}, strcmp(v_tc_glb{i}, best_tc_glb));
        s_tc_fw  = format_str(v_tc_fw{i}, strcmp(v_tc_fw{i}, best_tc_fw));
        s_rmse   = format_str(v_rmse{i}, strcmp(v_rmse{i}, best_rmse));
        s_mae    = format_str(v_mae{i}, strcmp(v_mae{i}, best_mae));
        s_rpe    = format_str(v_rpe{i}, strcmp(v_rpe{i}, best_rpe));
        s_corr   = format_str(v_corr{i}, strcmp(v_corr{i}, best_corr));
        s_time   = format_str(v_time{i}, strcmp(v_time{i}, best_time));

        fprintf(fid, '%-16s & %s & %s & %s & %s & %s & %s & %s \\\\\n', ...
                model_display_name(m), s_tc_glb, s_tc_fw, s_rmse, s_mae, s_rpe, s_corr, s_time);
    end

    fprintf(fid, '\\bottomrule\n\\end{tabular}\n}\n\\end{table}\n');
    fclose(fid);
end

function str = format_str(val, is_best)
    if is_best
        str = sprintf('$\\mathbf{%s}$', val);
    else
        str = sprintf('$%s$', val);
    end
end
