%% EXP_TEST_MAC_TRANDIS_FW
% Evaluates candidate distributed MoE models on 2 datasets (r=1, 2) on Mac,
% reporting both:
%   1. \mathcal{D}(f, f^*): Transportation divergence to ground truth f^*
%   2. \mathcal{D}(f, f^W): Transportation divergence to pooled local mixture f^W
% Exports standardized publication-ready LaTeX table.

function results = exp_test_mac_trandis_fW()
    base_dir = fileparts(fileparts(mfilename('fullpath')));
    addpath(base_dir);
    addpath(fullfile(base_dir, 'models'));
    addpath(fullfile(base_dir, 'datatools'));
    addpath(fullfile(base_dir, 'evaltools'));
    addpath(fullfile(base_dir, 'stattools'));

    fprintf('========================================================================================\n');
    fprintf('  TEST EXPERIMENT: TRANSPORTATION DISTANCE TO f^* vs f^W (Mac, 2 Datasets, N=100k)       \n');
    fprintf('  Evaluating: D(f, f^*) [Estimation Error] vs D(f, f^W) [Reduction Divergence]            \n');
    fprintf('========================================================================================\n\n');

    data_file = fullfile(base_dir, 'data', 'dataset_N100k_K5_d20_balanced.mat');
    if ~exist(data_file, 'file')
        error('Dataset file not found: %s', data_file);
    end

    loaded = load(data_file);
    X_mat        = loaded.X_mat;
    Y_mat        = loaded.Y_mat;
    LABEL_mat    = loaded.LABEL_mat;
    true_mixture = loaded.true_mixture;

    num_runs = 2; % 2 datasets as requested
    K = 5;
    d = 20;
    M = 16;
    S = 2000;
    N_total = size(X_mat, 1);
    N_train = floor(0.8 * N_total);
    N_test  = N_total - N_train;

    models = {'GLB', 'DME', 'GM', 'AAVR', 'FED', 'WAVR'};
    
    poolobj = gcp('nocreate');
    if isempty(poolobj)
        if ismac
            parpool('Processes', 4);
        else
            parpool;
        end
    end

    % Metrics: [Time, Trandis_truth, Trandis_fW, Param_MSE, RPE, Corr, RI, ARI, ClustErr, Loglik]
    metrics_all = struct();
    for m = 1:length(models)
        metrics_all.(models{m}) = zeros(num_runs, 10);
    end

    options = get_options('default');
    options.DME_verbose     = 0;
    options.verbose         = 0;
    options.S               = S;
    options.sample_size     = S;
    options.IRLS_max_iter   = 100;
    options.IRLS_threshold  = 1e-5;
    options.max_iter        = 500;
    options.nb_EM_runs      = 5;
    options.FedAvg_rounds   = 5;
    options.FedAvg_local_iters = 10;

    for r = 1:num_runs
        fprintf('>>> [Dataset %d/%d] Starting evaluation...\n', r, num_runs);
        t_run_start = tic;

        X_all     = X_mat(:, :, r);
        Y_all     = Y_mat(:, r);
        label_all = LABEL_mat(:, r);

        X_train     = X_all(1:N_train, :);
        Y_train     = Y_all(1:N_train);
        label_train = label_all(1:N_train);

        X_test      = X_all(N_train+1:end, :);
        Y_test      = Y_all(N_train+1:end);
        label_test  = label_all(N_train+1:end);

        % Client partition
        rng(r * 1000 + 42, 'twister');
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

        % Supporting dataset D_S (unlabeled covariates)
        rng(r * 100 + 123, 'twister');
        s_idx = randperm(N_train, S);
        X_val = [ones(S, 1), X_train(s_idx, :)];

        % 1. Centralized Oracle (GLB)
        t_glb = tic;
        opt_glb = options;
        fit_glb = Global_MixtureOfExperts(X_train, Y_train, K, opt_glb);
        fit_glb.learning_time = toc(t_glb);

        % 2. Fit local models on M machines (shared)
        local_fits = cell(1, M);
        local_times = zeros(1, M);
        parfor m = 1:M
            t_m = tic;
            idx_m = client_indices{m};
            opt_m = options;
            opt_m.verbose = 0;
            local_fits{m} = Global_MixtureOfExperts(X_train(idx_m, :), Y_train(idx_m), K, opt_m);
            local_times(m) = toc(t_m);
        end
        t_local_max = max(local_times);

        % Package for aggregators
        DMEfit = struct();
        DMEfit.local_estimates = local_fits;
        DMEfit.local_times     = local_times;
        DMEfit.M               = M;
        DMEfit.K               = K;
        DMEfit.d               = d;
        DMEfit.n               = N_train;
        DMEfit.X_val           = X_val;

        % 3. DME (Pure Unlabeled OT)
        opt_dme = options;
        opt_dme.local_estimates = local_fits;
        opt_dme.local_times     = local_times;
        opt_dme.X_val           = X_val;
        fit_dme = Distributed_MixtureOfExperts_Gaussian(X_train, Y_train, K, M, opt_dme);
        fit_dme.learning_time   = t_local_max + fit_dme.learning_time;

        % Extract large_mixture (f^W) from DMEfit
        large_mixture = fit_dme.large_mixture;

        % 4. GM
        fit_gm = Greedy_MixtureOfExperts(DMEfit, K, M, options);
        fit_gm.learning_time = t_local_max + fit_gm.learning_time;

        % 5. AAVR
        fit_aavr = Aligned_MixtureOfExperts(DMEfit, K, M, options);
        fit_aavr.learning_time = t_local_max + fit_aavr.learning_time;

        % 6. FED
        opt_fed = options;
        opt_fed.client_indices = client_indices;
        fit_fed = FedAvg_MixtureOfExperts(X_train, Y_train, K, M, opt_fed);

        % 7. WAVR
        fit_wavr = Averaged_MixtureOfExperts(DMEfit, K, M, options);
        fit_wavr.learning_time = t_local_max + fit_wavr.learning_time;

        % Store all fits
        fits = struct('GLB', fit_glb, 'DME', fit_dme, 'GM', fit_gm, ...
                      'AAVR', fit_aavr, 'FED', fit_fed, 'WAVR', fit_wavr);

        % Evaluate all models on test set and compute D(f, f^W)
        for m_idx = 1:length(models)
            mod = models{m_idx};
            f = fits.(mod);

            % Standard test metrics against ground truth f^*
            [t_l, td_truth, ll, mse, rpe, cr, ri, ari, ce] = ...
                compute_metrics(f, true_mixture, X_test, Y_test, label_test, 0, mod);

            % Compute Transportation Divergence to pooled local mixture f^W: D(f, f^W)
            target_mix = struct();
            target_mix.experts   = [f.param.beta0; f.param.Beta];
            target_mix.variances = f.param.sigma2;
            [plan_W, dist_W] = argmin_transportation_plan(large_mixture, target_mix, X_val);
            td_fW = sum(sum(sum(plan_W .* dist_W)));

            % Store: [Time, Trandis_truth, Trandis_fW, Param_MSE, RPE, Corr, RI, ARI, ClustErr, Loglik]
            metrics_all.(mod)(r, :) = [f.learning_time, td_truth, td_fW, mse, rpe, cr, ri, ari, ce, ll];
        end

        fprintf('   [Done in %.1fs] Run %d: DME D(f,fW)=%.3f | AAVR D(f,fW)=%.3f | GM D(f,fW)=%.3f | GLB D(f,fW)=%.3f\n', ...
                toc(t_run_start), r, metrics_all.DME(r, 3), metrics_all.AAVR(r, 3), ...
                metrics_all.GM(r, 3), metrics_all.GLB(r, 3));
    end

    % Print Comparative Summary Table
    fprintf('\n=======================================================================================================================\n');
    fprintf('  BENCHMARK SUMMARY (2 Datasets, Non-IID Balanced Profile, Mean +- Std)                                                \n');
    fprintf('=======================================================================================================================\n');
    fprintf('%-16s | %-16s | %-16s | %-16s | %-16s | %-16s | %-12s\n', ...
            'Estimator', 'RPE', 'Param MSE', 'D(f, f*) [Truth]', 'D(f, f^W) [f^W]', 'ARI', 'Time (s)');
    fprintf('-----------------------------------------------------------------------------------------------------------------------\n');

    for m = 1:length(models)
        mod = models{m};
        mat = metrics_all.(mod);
        % Metric indices: Time=1, Trandis_truth=2, Trandis_fW=3, Param_MSE=4, RPE=5, Corr=6, RI=7, ARI=8, ClustErr=9, Loglik=10
        fprintf('%-16s | %6.4f +- %6.4f | %6.4f +- %6.4f | %6.3f +- %5.3f | %6.3f +- %5.3f | %6.4f +- %6.4f | %6.2f +- %4.2f\n', ...
                mod, mean(mat(:,5)), std(mat(:,5)), mean(mat(:,4)), std(mat(:,4)), ...
                mean(mat(:,2)), std(mat(:,2)), mean(mat(:,3)), std(mat(:,3)), ...
                mean(mat(:,8)), std(mat(:,8)), mean(mat(:,1)), std(mat(:,1)));
    end
    fprintf('=======================================================================================================================\n\n');

    % Export Publication-Quality LaTeX Table
    tex_dir = fullfile(base_dir, 'results', 'tables');
    if ~exist(tex_dir, 'dir'), mkdir(tex_dir); end
    tex_file = fullfile(tex_dir, 'table_test_trandis_fW.tex');
    export_trandis_fW_latex(metrics_all, models, tex_file);
    fprintf('Exported publication LaTeX table to: %s\n', tex_file);

    results = metrics_all;
    save(fullfile(base_dir, 'results', 'result_test_mac_trandis_fW.mat'), 'results', 'models');
end

% -------------------------------------------------------------------------
% Helper: Export publication LaTeX table
% -------------------------------------------------------------------------
function export_trandis_fW_latex(metrics, models, filename)
    fid = fopen(filename, 'w');
    if fid == -1, error('Cannot write to: %s', filename); end

    fprintf(fid, '%% Auto-generated LaTeX table comparing D(f, f^*) vs D(f, f^W)\n');
    fprintf(fid, '\\begin{table}[t]\n\\centering\n');
    fprintf(fid, '\\caption{Distributed MoE Aggregation: Comparison of estimation accuracy against ground truth $\\mathcal{D}(f, f^*)$ and reduction divergence against pooled local mixture $\\mathcal{D}(f, f^W)$ ($M=16, K=5, d=20, N=100\\text{k}, S=2000$). Best distributed performance is in \\textbf{bold}.}\n');
    fprintf(fid, '\\label{tab:trandis_fW}\n\\vspace{1mm}\n');
    fprintf(fid, '\\setlength{\\tabcolsep}{5pt}\n');
    fprintf(fid, '\\begin{tabular}{lcccccc}\n\\toprule\n');
    fprintf(fid, '\\textbf{Estimator} & \\textbf{RPE} & \\textbf{Param MSE} & $\\mathcal{D}(f, f^*)$ & $\\mathcal{D}(f, f^W)$ & \\textbf{ARI} & \\textbf{Time (s)} \\\\\n');
    fprintf(fid, '\\midrule\n');

    for i = 1:length(models)
        mod = models{i};
        mat = metrics.(mod);
        rpe_m  = mean(mat(:,5)); rpe_s  = std(mat(:,5));
        mse_m  = mean(mat(:,4)); mse_s  = std(mat(:,4));
        td_s_m = mean(mat(:,2)); td_s_s = std(mat(:,2));
        td_w_m = mean(mat(:,3)); td_w_s = std(mat(:,3));
        ari_m  = mean(mat(:,8)); ari_s  = std(mat(:,8));
        time_m = mean(mat(:,1));

        name_str = get_display_name(mod);
        
        if strcmp(mod, 'GLB')
            fprintf(fid, '%-22s & $%0.4f$ & $%0.4f$ & $%0.2f$ & $%0.2f$ & $%0.4f$ & $%0.1f$ \\\\\n', ...
                    name_str, rpe_m, mse_m, td_s_m, td_w_m, ari_m, time_m);
            fprintf(fid, '\\midrule\n');
        else
            fprintf(fid, '%-22s & $%0.4f \\pm %0.4f$ & $%0.4f \\pm %0.4f$ & $%0.2f \\pm %0.2f$ & $%0.2f \\pm %0.2f$ & $%0.4f \\pm %0.4f$ & $%0.1f$ \\\\\n', ...
                    name_str, rpe_m, rpe_s, mse_m, mse_s, td_s_m, td_s_s, td_w_m, td_w_s, ari_m, ari_s, time_m);
        end
    end
    fprintf(fid, '\\bottomrule\n\\end{tabular}\n\\end{table}\n');
    fclose(fid);
end

function name = get_display_name(mod)
    switch mod
        case 'GLB', name = 'Global oracle ($G$)';
        case 'DME', name = 'Proposed (\\textbf{DME})';
        case 'GM', name = 'Greedy merging (GM)';
        case 'AAVR', name = 'Aligned averaging (AAVR)';
        case 'FED', name = 'Federated averaging (FED)';
        case 'WAVR', name = 'Naive averaging (WAVR)';
        otherwise,  name = mod;
    end
end
