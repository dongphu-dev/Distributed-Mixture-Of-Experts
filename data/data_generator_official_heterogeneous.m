function data_generator_official_heterogeneous(num_datasets, N_total, d, K, M, K_m, N_test)
%% DATA_GENERATOR_OFFICIAL_HETEROGENEOUS
% Standardized multi-run pre-generator for Official Experiment 2 (Heterogeneous Km):
%   - Uncoordinated local capacities: M machines, each observing K_m experts
%   - Scalable to arbitrary sample sizes N_total in {100k, 300k, 1M, ...}
%   - Mathematically exact conditional sampling (no rejection sampling / pool bottlenecks)
%   - Generates reproducible Monte Carlo runs saved into a single .mat file
%
% Usage:
%   data_generator_official_heterogeneous;                               % Default: DEV (5 runs) or PROD (100 runs), N=100k
%   data_generator_official_heterogeneous(num_datasets);                  % Explicit runs, N = 100,000
%   data_generator_official_heterogeneous(num_datasets, N_total);          % Custom N (e.g., 1000000 for 1M)
%   data_generator_official_heterogeneous(num_datasets, N_total, d, K, M, K_m, N_test);
%
% Output:
%   data/dataset_N<N_str>_K<K>_d<d>_hetero_M<M>_Km<Km>.mat

    current_dir = fileparts(mfilename('fullpath'));
    base_dir    = fullfile(current_dir, '..');
    addpath(base_dir);
    addpath(fullfile(base_dir, 'datatools'));

    % Default parameters
    if nargin < 7 || isempty(N_test),  N_test = max(20000, floor(0.20 * N_total)); end
    if nargin < 6 || isempty(K_m), K_m = 3; end
    if nargin < 5 || isempty(M),   M = 16; end
    if nargin < 4 || isempty(K),   K = 5; end
    if nargin < 3 || isempty(d),   d = 20; end
    if nargin < 2 || isempty(N_total), N_total = 100000; end

    if nargin < 1 || isempty(num_datasets)
        if ismac
            num_datasets = 5;   % Fast DEV mode on macOS (5 runs)
            fprintf('Detected macOS: Running in DEV mode (%d datasets, N = %d)\n', num_datasets, N_total);
        else
            num_datasets = 50; % Full PROD benchmark on Linux VPS
            fprintf('Detected Linux/VPS: Running in PROD mode (%d datasets, N = %d)\n', num_datasets, N_total);
        end
    end

    N_str = format_N_str(N_total);
    out_file = fullfile(current_dir, sprintf('dataset_N%s_K%d_d%d_hetero_M%d_Km%d.mat', N_str, K, d, M, K_m));

    N_per_machine = floor(N_total / M);
    N_total_train = N_per_machine * M;
    K_vec  = K_m * ones(1, M);

    fprintf('\n========================================================================\n');
    fprintf('  PRE-GENERATING HETEROGENEOUS DATASETS (M=%d, Km=%d, L=%d -> K=%d, N=%s)\n', ...
            M, K_m, M * K_m, K, N_str);
    fprintf('  Local sample size per machine: N_m = %d (Total train = %d, Test = %d)\n', ...
            N_per_machine, N_total_train, N_test);
    fprintf('  Generating %d Monte Carlo runs...\n', num_datasets);
    fprintf('========================================================================\n');

    % 1. Construct Ground Truth Model (Fixed reference for fair evaluation)
    gt_file = fullfile(current_dir, 'ground_truth_param_K5_d20.mat');
    if ~exist(gt_file, 'file')
        fprintf('Ground truth file not found. Generating ground_truth_param_K5_d20.mat...\n');
        data_generator_benchmark(1, false);
    end
    if exist(gt_file, 'file')
        loaded_gt = load(gt_file);
        param.beta0  = loaded_gt.param.beta0;
        param.Beta   = loaded_gt.param.Beta;
        param.sigma2 = loaded_gt.param.sigma2;
        scale_mu = 0.28;
        MU_mat = zeros(K, d);
        for k = 1:K
            MU_mat(k, :) = loaded_gt.MU.(sprintf('MU%d', k)) * scale_mu;
        end
    else
        rng(42, 'twister');
        param.beta0 = zeros(1, K);
        param.Beta  = zeros(d, K);
        for k = 1:K
            param.Beta(:, k) = randi([-6, 6], d, 1);
            if k > 1
                while min(sqrt(sum((param.Beta(:, 1:k-1) - repmat(param.Beta(:, k), 1, k-1)).^2, 1))) < 3.0
                    param.Beta(:, k) = randi([-6, 6], d, 1);
                end
            end
        end
        param.sigma2 = 3.0 * ones(1, K);
        rng(28, 'twister');
        MU_mat = randn(K, d);
        MU_mat = MU_mat ./ sqrt(sum(MU_mat.^2, 2)) * 2.5;
    end

    % Bayes Conjugate Gating (LDA Equivalence)
    param.Alpha = MU_mat' - repmat(MU_mat(K, :)', 1, K);
    norm_mu2 = sum(MU_mat.^2, 2)';
    param.alpha0 = -0.5 * norm_mu2 - (-0.5 * norm_mu2(K)); % balanced log(1/K) cancels

    W_gate = [param.alpha0; param.Alpha];

    % True mixture struct
    true_mixture.experts   = [param.beta0; param.Beta];
    true_mixture.variances = param.sigma2;
    true_mixture.gates     = W_gate;
    true_mixture.weights   = ones(1, K) / K;
    true_mixture.MU        = MU_mat;

    % Subpopulation assignment per machine (balanced coverage across all K experts)
    subpopulations = cell(1, M);
    for m = 1:M
        % Cyclic permutation ensures all experts are equally covered across machines
        start_idx = mod(m - 1, K) + 1;
        subpopulations{m} = mod(start_idx - 1 + (0:K_m-1), K) + 1;
    end

    config.M             = M;
    config.K_vec         = K_vec;
    config.K_m           = K_m;
    config.K_target      = K;
    config.d             = d;
    config.N_total       = N_total;
    config.N_per_machine = N_per_machine;
    config.N_total_train = N_total_train;
    config.N_test        = N_test;
    config.subpopulations = subpopulations;

    all_runs = struct([]);

    for r = 1:num_datasets
        seed = r * 100 + 42;
        rng(seed, 'twister');
        if mod(r, 10) == 0 || r == 1 || r == num_datasets
            fprintf('  --> Generating Run [%d/%d] (seed = %d)...\n', r, num_datasets, seed);
        end

        % 2. 5-Mode Local Generation across M machines
        X_cells = cell(1, M);
        Y_cells = cell(1, M);
        Z_cells = cell(1, M);

        for m = 1:M
            active_k = subpopulations{m}; % Length K_m
            
            % Draw uniform local component assignments in active_k
            r_idx = randi(K_m, N_per_machine, 1);
            z_global = active_k(r_idx)';

            % Draw X ~ N(mu_Z, I_d)
            X_m = zeros(N_per_machine, d);
            Y_m = zeros(N_per_machine, 1);
            for idx = 1:K_m
                k_glob = active_k(idx);
                pts = (z_global == k_glob);
                n_pts = sum(pts);
                if n_pts > 0
                    X_m(pts, :) = repmat(MU_mat(k_glob, :), n_pts, 1) + randn(n_pts, d);
                    mu_y = param.beta0(k_glob) + X_m(pts, :) * param.Beta(:, k_glob);
                    Y_m(pts) = mu_y + sqrt(param.sigma2(k_glob)) * randn(n_pts, 1);
                end
            end

            X_cells{m} = X_m;
            Y_cells{m} = Y_m;
            Z_cells{m} = z_global;
        end

        % 3. Global Multimodal Test Set
        z_test = randi(K, N_test, 1);
        X_test = zeros(N_test, d);
        Y_test = zeros(N_test, 1);
        for k = 1:K
            pts = (z_test == k);
            n_pts = sum(pts);
            if n_pts > 0
                X_test(pts, :) = repmat(MU_mat(k, :), n_pts, 1) + randn(n_pts, d);
                mu_test = param.beta0(k) + X_test(pts, :) * param.Beta(:, k);
                Y_test(pts) = mu_test + sqrt(param.sigma2(k)) * randn(n_pts, 1);
            end
        end

        all_runs(r).X_cells = X_cells;
        all_runs(r).Y_cells = Y_cells;
        all_runs(r).Z_cells = Z_cells;
        all_runs(r).X_test  = X_test;
        all_runs(r).Y_test  = Y_test;
        all_runs(r).Z_test  = z_test;
    end

    % Save all pre-generated runs
    save(out_file, 'all_runs', 'param', 'config', 'true_mixture', 'num_datasets', ...
         'N_total', 'M', 'd', 'K', 'K_m', '-v7.3');
    fprintf('\n==> Successfully saved %d Heterogeneous datasets (N = %s) to:\n    %s (Size: %.2f MB)\n', ...
            num_datasets, N_str, out_file, dir(out_file).bytes / 1e6);
end
