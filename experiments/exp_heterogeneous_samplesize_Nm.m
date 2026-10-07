function results = exp_heterogeneous_samplesize_Nm(partition_type, profile_name, num_runs_input, N_total, M, S)
%% EXP_HETEROGENEOUS_SAMPLESIZE_NM
% Evaluates Distributed MoE aggregators when local workers hold HETEROGENEOUS
% sample sizes N_m (unbalanced local data volume across machines), reflecting
% realistic federated environments with varying edge/hub capacities.
%
% Key Theoretical Motivations:
%   1. DME naturally incorporates sample-weighting w_m = N_m / N in its pooled
%      mixture f^W = \sum w_m f_m and in optimal-transport marginals on D_S.
%   2. Naive parameter averaging (AAVR, WAVR) treats all machines equally (1/M),
%      causing small/noisy edge machines to disproportionately degrade the model.
%   3. Evaluates robustness of DME under extreme capacity skew.
%
% Supported Partition Profiles:
%   - 'skewed'  (Default): Power-law / Pareto decay (N_max / N_min ~ 12x, e.g. 21,000 to 1,700).
%   - 'bimodal'          : Hub-and-Spoke (4 large hubs with 12,500 + 12 edge nodes with 2,500).
%   - 'linear'           : Linear ramp from N_min = 1,000 to N_max = 9,000 (Ratio = 9.0x).
%   - Custom 1xM vector  : Directly provide target sample sizes [N_1, ..., N_M] summing to N_train.
%
% Options:
%   - nb_EM_runs = 5 (restarts per local EM fit)
%   - IRLS_max_iter = 100 (iterations for softmax gating)
%   - Profile: 'moderate' (default, matching benchmark)
%
% Usage:
%   exp_heterogeneous_samplesize_Nm;                     % Default: skewed, moderate, DEV/PROD auto
%   exp_heterogeneous_samplesize_Nm('bimodal');          % Bimodal hub-and-spoke
%   exp_heterogeneous_samplesize_Nm('linear', 'moderate', 3); % 3 runs smoke test

    %% 1. Path Setup & Parameters
    current_dir = fileparts(mfilename('fullpath'));
    base_dir    = fileparts(current_dir);

    addpath(base_dir);
    addpath(fullfile(base_dir, 'models'));
    addpath(fullfile(base_dir, 'datatools'));
    addpath(fullfile(base_dir, 'evaltools'));
    addpath(fullfile(base_dir, 'stattools'));
    addpath(fullfile(base_dir, 'experiments'));
    addpath(fullfile(base_dir, 'reporting'));

    if nargin < 1 || isempty(partition_type), partition_type = 'skewed'; end
    if nargin < 2 || isempty(profile_name),   profile_name   = 'moderate'; end
    if nargin < 4 || isempty(N_total),        N_total        = 100000; end
    if nargin < 5 || isempty(M),              M              = 16; end
    if nargin < 6 || isempty(S),              S              = 2000; end

    if nargin < 3 || isempty(num_runs_input)
        if ismac
            num_runs = 3;   % Fast DEV mode on macOS
        else
            num_runs = 50;  % Full PROD benchmark on VPS
        end
    else
        num_runs = num_runs_input;
    end

    K = 5;
    d = 20;
    N_str = format_N_str(N_total);

    % Configure train/test split
    N_train = floor(0.8 * N_total); % 80,000
    N_test  = N_total - N_train;    % 20,000

    % Compute heterogeneous local sample sizes N_m
    N_vec = compute_hetero_N_vector(N_train, M, partition_type);

    fprintf('========================================================================\n');
    fprintf('  EXPERIMENT: HETEROGENEOUS LOCAL SAMPLE SIZES N_m                      \n');
    fprintf('  Configuration: N = %s, M = %d, K = %d, d = %d, Profile = %s           \n', ...
            N_str, M, K, d, profile_name);
    fprintf('  Partition Type: %s (Ratio N_max/N_min = %.1fx)                        \n', ...
            char(string(partition_type)), max(N_vec) / min(N_vec));
    fprintf('  N_m Allocation: [%s]\n', num2str(N_vec));
    fprintf('  Executing %d Monte Carlo runs (nb_EM_runs = 5, IRLS_max_iter = 100)   \n', num_runs);
    fprintf('========================================================================\n');

    %% 2. Load Benchmark Dataset
    data_file = fullfile(base_dir, 'data', sprintf('dataset_N%s_K%d_d%d_%s.mat', N_str, K, d, profile_name));
    if ~exist(data_file, 'file')
        error('Dataset not found: %s. Run data_generator_profiles first!', data_file);
    end
    fprintf('Loading dataset: %s...\n', data_file);
    loaded = load(data_file);
    X_mat        = loaded.X_mat;
    Y_mat        = loaded.Y_mat;
    LABEL_mat    = loaded.LABEL_mat;
    true_mixture = loaded.true_mixture;

    total_avail = size(X_mat, 3);
    runs_to_process = min(num_runs, total_avail);
    fprintf('Processing %d / %d available datasets...\n', runs_to_process, total_avail);

    %% 3. Setup Parallel Pool & Execution Options
    poolobj = gcp('nocreate');
    if isempty(poolobj)
        if ismac
            parpool('Processes', min(4, runs_to_process));
        else
            parpool;
        end
    end

    options = get_options('default');
    options.DME_verbose       = 0;
    options.verbose           = 0;
    options.S                 = S;
    options.sample_size       = S;
    options.IRLS_max_iter     = 100;
    options.IRLS_threshold    = 1e-5;
    options.max_iter          = 100;
    options.nb_EM_runs        = 5;
    options.parallel_machines = false; % Guarantees outer parfor is over datasets
    options.FedAvg_rounds     = 5;
    options.FedAvg_local_iters= 10;

    models_to_run = {'DME', 'GM', 'AAVR', 'FED_T1', 'FED_T5', 'FED_T10', 'WAVR', 'MED'};
    all_models    = [{'GLB'}, models_to_run];

    % Temporary cell storage for parfor-safety
    run_metrics = cell(runs_to_process, 1);

    %% 4. Parallel Benchmark Execution
    t_suite_start = tic;
    parfor r = 1:runs_to_process
        t_run_start = tic;

        X_all     = X_mat(:, :, r);
        Y_all     = Y_mat(:, r);
        label_all = LABEL_mat(:, r);

        X_train     = X_all(1:N_train, :);
        Y_train     = Y_all(1:N_train);
        X_test      = X_all(N_train+1:end, :);
        Y_test      = Y_all(N_train+1:end);
        label_test  = label_all(N_train+1:end);

        % Construct heterogeneous client indices
        rng(r * 1000 + 77, 'twister');
        shuffled = randperm(N_train);
        client_indices = cell(1, M);
        idx_curr = 1;
        for m = 1:M
            client_indices{m} = shuffled(idx_curr : idx_curr + N_vec(m) - 1)';
            idx_curr = idx_curr + N_vec(m);
        end

        run_options = options;
        run_options.client_indices = client_indices;

        % Execute distributed models
        runner_res = core_runner(X_train, Y_train, X_test, Y_test, label_test, true_mixture, ...
                                 K, M, run_options, models_to_run, 0);

        r_struct = struct();
        for m = 1:length(models_to_run)
            mod_name = models_to_run{m};
            r_struct.(mod_name) = runner_res.(mod_name).metrics;
        end

        % Centralized Oracle (GLB on full pooled training data)
        t_glb = tic;
        glb_opt = options;
        glb_opt.nb_EM_runs = options.nb_EM_runs;
        glb_opt.max_iter   = 40;
        glb_opt.verbose    = 0;
        fit_glb = Global_MixtureOfExperts(X_train, Y_train, K, glb_opt);
        fit_glb.learning_time = toc(t_glb);
        [t_l, td, ll, mse, rpe, cr, ri, ari, ce, td_fW] = ...
            compute_metrics(fit_glb, true_mixture, X_test, Y_test, label_test, 0, 'GLB', runner_res.f_W);
        r_struct.GLB = [t_l, td, ll, mse, rpe, cr, ri, ari, ce, td_fW];

        run_metrics{r} = r_struct;

        t_run_dur = toc(t_run_start);
        fprintf('[Hetero Nm | %s] [Run %2d/%2d] Completed in %5.1fs | DME ARI = %.4f | AAVR ARI = %.4f | GLB ARI = %.4f\n', ...
                char(string(partition_type)), r, runs_to_process, t_run_dur, ...
                r_struct.DME(8), r_struct.AAVR(8), r_struct.GLB(8));
    end
    t_total_suite = toc(t_suite_start);

    %% 5. Aggregate Results & Statistics
    results = struct();
    results.partition_type = partition_type;
    results.profile_name   = profile_name;
    results.N_vec          = N_vec;
    results.M              = M;
    results.K              = K;
    results.d              = d;
    results.runs_processed = runs_to_process;
    results.all_models     = all_models;
    results.raw_metrics    = struct();
    results.summary        = struct();

    for m_i = 1:length(all_models)
        mod_name = all_models{m_i};
        mat = zeros(runs_to_process, 10);
        for r = 1:runs_to_process
            mat(r, :) = run_metrics{r}.(mod_name);
        end
        results.raw_metrics.(mod_name) = mat;
        
        stat = struct();
        stat.mean   = mean(mat, 1, 'omitnan');
        stat.std    = std(mat, 0, 1, 'omitnan');
        stat.median = median(mat, 1, 'omitnan');
        stat.iqr25  = prctile(mat, 25, 1);
        stat.iqr75  = prctile(mat, 75, 1);
        results.summary.(mod_name) = stat;
    end

    %% 6. Display Formatted Evaluation Table
    fprintf('\n========================================================================================================\n');
    fprintf('  BENCHMARK RESULTS: HETEROGENEOUS LOCAL SAMPLE SIZES (N_max/N_min = %.1fx, Runs = %d)\n', ...
            max(N_vec) / min(N_vec), runs_to_process);
    fprintf('========================================================================================================\n');
    fprintf('%-10s | %-17s | %-17s | %-16s | %-16s | %-12s\n', ...
            'Model', 'Tc(., f*)', '||theta - theta*||', 'RPE', 'ARI', 'Time (s)');
    fprintf('--------------------------------------------------------------------------------------------------------\n');

    metric_cols = [2, 4, 5, 8, 1]; % Tc, MSE, RPE, ARI, Time
    for m_i = 1:length(all_models)
        mod_name = all_models{m_i};
        st = results.summary.(mod_name);
        fprintf('%-10s | %8.1f +/- %6.1f | %8.4f +/- %6.4f | %6.4f +/- %5.4f | %6.4f +/- %5.4f | %5.2f +/- %4.2f\n', ...
                mod_name, ...
                st.mean(2), st.std(2), ...
                st.mean(4), st.std(4), ...
                st.mean(5), st.std(5), ...
                st.mean(8), st.std(8), ...
                st.mean(1), st.std(1));
    end
    fprintf('--------------------------------------------------------------------------------------------------------\n');
    fprintf('Robust Statistics (Median [IQR]):\n');
    for m_i = 1:length(all_models)
        mod_name = all_models{m_i};
        st = results.summary.(mod_name);
        fprintf('%-10s | Tc: %6.1f [%5.1f,%5.1f] | MSE: %6.4f [%5.4f,%5.4f] | RPE: %5.4f | ARI: %5.4f\n', ...
                mod_name, ...
                st.median(2), st.iqr25(2), st.iqr75(2), ...
                st.median(4), st.iqr25(4), st.iqr75(4), ...
                st.median(5), st.median(8));
    end
    fprintf('========================================================================================================\n');
    fprintf('Total execution time: %.1f seconds.\n', t_total_suite);

    %% 7. Unique Timestamped Saving
    out_dir = fullfile(base_dir, 'results', 'heterogeneous_Nm');
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    part_str = char(string(partition_type));
    out_mat = fullfile(out_dir, sprintf('Result_HeteroNm_%s_%s_N%s_M%d_%s.mat', ...
                                        part_str, profile_name, N_str, M, timestamp));
    save(out_mat, 'results', 'options', '-v7.3');
    fprintf('\n==> Successfully saved results to: %s\n', out_mat);

end

%% =========================================================================
%% Helper Function: Compute Heterogeneous Sample Size Vector N_m
%% =========================================================================
function N_vec = compute_hetero_N_vector(N_train, M, partition_type)
    if isnumeric(partition_type) && length(partition_type) == M
        N_vec = round(partition_type(:)' / sum(partition_type) * N_train);
        diff = N_train - sum(N_vec);
        N_vec(1) = N_vec(1) + diff;
        return;
    end

    switch lower(char(string(partition_type)))
        case 'skewed'
            % Power-law / Pareto decay: weights proportional to 1 / m^0.9
            m_idx = 1:M;
            w = 1.0 ./ (m_idx .^ 0.9);
            N_vec = round(N_train * (w / sum(w)));
            diff = N_train - sum(N_vec);
            N_vec(1) = N_vec(1) + diff;

        case 'bimodal'
            % Hub-and-Spoke: 4 large hubs (12,500) + 12 edge nodes (2,500)
            n_hubs = max(2, floor(M / 4));
            n_edge = M - n_hubs;
            hub_mass = 0.60 * N_train;
            edge_mass = N_train - hub_mass;
            N_hubs = round(repmat(hub_mass / n_hubs, 1, n_hubs));
            N_edge = round(repmat(edge_mass / n_edge, 1, n_edge));
            N_vec  = [N_hubs, N_edge];
            diff   = N_train - sum(N_vec);
            N_vec(1) = N_vec(1) + diff;

        case 'linear'
            % Linear ramp: N_min = 1000, N_max = 9000
            m_idx = 0:(M - 1);
            N_vec = round(1000 + m_idx * (8000 / (M - 1)));
            diff = N_train - sum(N_vec);
            N_vec(end) = N_vec(end) + diff;

        otherwise
            error('Unknown partition_type: %s. Use skewed, bimodal, or linear.', partition_type);
    end
end
