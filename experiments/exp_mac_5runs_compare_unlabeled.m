%% EXP_MAC_5RUNS_COMPARE_UNLABELED
% Benchmark evaluating 5 datasets (r = 1..5) on Non-IID Balanced profile (N=100k, M=16, K=5, d=20, S=2000)
% Specifically compares DME_Tau (with Y_val) vs DME_Unlabeled (pure OT, no Y_val) against all baselines.

function results = exp_mac_5runs_compare_unlabeled()
    base_dir = fileparts(fileparts(mfilename('fullpath')));
    addpath(base_dir);
    addpath(fullfile(base_dir, 'models'));
    addpath(fullfile(base_dir, 'datatools'));
    addpath(fullfile(base_dir, 'evaltools'));
    addpath(fullfile(base_dir, 'stattools'));

    fprintf('========================================================================================\n');
    fprintf('  BENCHMARK 5 DATASETS ON MAC: DME_Tau vs DME_Unlabeled (Non-IID Balanced Profile)     \n');
    fprintf('  Configuration: N = 100k, M = 16, K = 5, d = 20, S = 2000, 5 Datasets                 \n');
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

    num_runs = 5;
    K = 5;
    d = 20;
    M = 16;
    S = 2000;
    N_total = size(X_mat, 1);
    N_train = floor(0.8 * N_total);
    N_test  = N_total - N_train;

    models = {'GLB', 'DME_Tau', 'DME_Unlabeled', 'GM', 'AAVR', 'FED', 'WAVR'};
    
    % Storage: 9 metrics per model [Time, Trandis, Loglik, Param_MSE, RPE, Corr, RI, ARI, ClustErr]
    metrics_all = struct();
    for m = 1:length(models)
        metrics_all.(models{m}) = zeros(num_runs, 9);
    end

    options = get_options('default');
    options.DME_verbose     = 0;
    options.verbose         = 0;
    options.S               = S;
    options.sample_size     = S;
    options.IRLS_max_iter   = 30;
    options.IRLS_threshold  = 1e-5;
    options.max_iter        = 40;
    options.nb_EM_runs      = 2;
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

        % Supporting dataset D_S
        rng(r * 100 + 123, 'twister');
        s_idx = randperm(N_train, S);
        X_val = [ones(S, 1), X_train(s_idx, :)];
        Y_val = Y_train(s_idx);

        % 1. Centralized Oracle (GLB)
        t_glb = tic;
        opt_glb = options;
        fit_glb = Global_MixtureOfExperts(X_train, Y_train, K, opt_glb);
        fit_glb.learning_time = toc(t_glb);
        [t_l, td, ll, mse, rpe, cr, ri, ari, ce] = compute_metrics(fit_glb, true_mixture, X_test, Y_test, label_test, 0, 'GLB');
        metrics_all.GLB(r, :) = [t_l, td, ll, mse, rpe, cr, ri, ari, ce];

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
        DMEfit.Y_val           = Y_val;

        % 3. DME_Tau (with Y_val)
        opt_tau = options;
        opt_tau.local_estimates     = local_fits;
        opt_tau.local_times         = local_times;
        opt_tau.X_val               = X_val;
        opt_tau.Y_val               = Y_val;
        opt_tau.use_unlabeled_gates = false;
        t_dme = tic;
        fit_tau = Distributed_MixtureOfExperts_Gaussian(X_train, Y_train, K, M, opt_tau);
        time_tau = t_local_max + fit_tau.learning_time;
        fit_tau.learning_time = time_tau;
        [t_l, td, ll, mse, rpe, cr, ri, ari, ce] = compute_metrics(fit_tau, true_mixture, X_test, Y_test, label_test, 0, 'DME');
        metrics_all.DME_Tau(r, :) = [time_tau, td, ll, mse, rpe, cr, ri, ari, ce];

        % 4. DME_Unlabeled (Pure OT, NO Y_val)
        opt_un = options;
        opt_un.local_estimates     = local_fits;
        opt_un.local_times         = local_times;
        opt_un.X_val               = X_val;
        opt_un.use_unlabeled_gates = true;
        t_dme2 = tic;
        fit_un = Distributed_MixtureOfExperts_Gaussian(X_train, Y_train, K, M, opt_un);
        time_un = t_local_max + fit_un.learning_time;
        fit_un.learning_time = time_un;
        [t_l, td, ll, mse, rpe, cr, ri, ari, ce] = compute_metrics(fit_un, true_mixture, X_test, Y_test, label_test, 0, 'DME');
        metrics_all.DME_Unlabeled(r, :) = [time_un, td, ll, mse, rpe, cr, ri, ari, ce];

        % 5. GM
        fit_gm = Greedy_MixtureOfExperts(DMEfit, K, M, options);
        time_gm = t_local_max + fit_gm.learning_time;
        fit_gm.learning_time = time_gm;
        [t_l, td, ll, mse, rpe, cr, ri, ari, ce] = compute_metrics(fit_gm, true_mixture, X_test, Y_test, label_test, 0, 'GM');
        metrics_all.GM(r, :) = [time_gm, td, ll, mse, rpe, cr, ri, ari, ce];

        % 6. AAVR
        fit_aavr = Aligned_MixtureOfExperts(DMEfit, K, M, options);
        time_aavr = t_local_max + fit_aavr.learning_time;
        fit_aavr.learning_time = time_aavr;
        [t_l, td, ll, mse, rpe, cr, ri, ari, ce] = compute_metrics(fit_aavr, true_mixture, X_test, Y_test, label_test, 0, 'AAVR');
        metrics_all.AAVR(r, :) = [time_aavr, td, ll, mse, rpe, cr, ri, ari, ce];

        % 7. FED
        opt_fed = options;
        opt_fed.client_indices = client_indices;
        fit_fed = FedAvg_MixtureOfExperts(X_train, Y_train, K, M, opt_fed);
        [t_l, td, ll, mse, rpe, cr, ri, ari, ce] = compute_metrics(fit_fed, true_mixture, X_test, Y_test, label_test, 0, 'FED');
        metrics_all.FED(r, :) = [fit_fed.learning_time, td, ll, mse, rpe, cr, ri, ari, ce];

        % 8. WAVR
        fit_wavr = Averaged_MixtureOfExperts(DMEfit, K, M, options);
        time_wavr = t_local_max + fit_wavr.learning_time;
        fit_wavr.learning_time = time_wavr;
        [t_l, td, ll, mse, rpe, cr, ri, ari, ce] = compute_metrics(fit_wavr, true_mixture, X_test, Y_test, label_test, 0, 'WAVR');
        metrics_all.WAVR(r, :) = [time_wavr, td, ll, mse, rpe, cr, ri, ari, ce];

        fprintf('   [Done in %.1fs] RPE: GLB=%.4f | DME_Tau=%.4f | DME_Unlabel=%.4f | GM=%.4f | AAVR=%.4f\n', ...
                toc(t_run_start), metrics_all.GLB(r, 5), metrics_all.DME_Tau(r, 5), ...
                metrics_all.DME_Unlabeled(r, 5), metrics_all.GM(r, 5), metrics_all.AAVR(r, 5));
    end

    % Print Final Summary Table
    fprintf('\n===========================================================================================================\n');
    fprintf('  BENCHMARK SUMMARY (5 Datasets, Non-IID Balanced Profile, Mean +- Std)                                    \n');
    fprintf('===========================================================================================================\n');
    fprintf('%-16s | %-16s | %-16s | %-16s | %-16s | %-12s\n', ...
            'Estimator', 'RPE', 'Param MSE', 'Trandis', 'ARI', 'Time (s)');
    fprintf('-----------------------------------------------------------------------------------------------------------\n');

    for m = 1:length(models)
        mod = models{m};
        mat = metrics_all.(mod);
        % Metric indices: Time=1, Trandis=2, Loglik=3, Param_MSE=4, RPE=5, Corr=6, RI=7, ARI=8, ClustErr=9
        fprintf('%-16s | %6.4f +- %6.4f | %6.4f +- %6.4f | %6.3f +- %5.3f | %6.4f +- %6.4f | %6.2f +- %4.2f\n', ...
                mod, mean(mat(:,5)), std(mat(:,5)), mean(mat(:,4)), std(mat(:,4)), ...
                mean(mat(:,2)), std(mat(:,2)), mean(mat(:,8)), std(mat(:,8)), mean(mat(:,1)), std(mat(:,1)));
    end
    fprintf('===========================================================================================================\n\n');

    results = metrics_all;
    save(fullfile(base_dir, 'results', 'result_mac_5runs_compare_unlabeled.mat'), 'results', 'models');
end
