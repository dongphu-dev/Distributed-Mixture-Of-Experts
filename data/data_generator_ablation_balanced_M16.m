function data_generator_ablation_balanced_M16(seed)
% DATA_GENERATOR_ABLATION_BALANCED_M16
% Generates an ablation benchmark dataset to isolate the Large-Scale / Compounding Merging factor:
%   - M = 16 machines, each observing K_m = 3 experts (Total L = 48 components)
%   - Target: K = 5 global experts (GM requires 43 merges)
%   - PERFECTLY BALANCED subpopulations: each expert is ~20% of the population.
%     NO rare expert exists!
%   - Small local samples: N_m = 500 samples per machine (total N_train = 8,000)
%   - Equal noise variances sigma2 = [3.0, 3.0, 3.0, 3.0, 3.0] and zero intercepts.
%
% Output:
%   data/dataset_ablation_balanced_M16.mat

    if nargin < 1 || isempty(seed), seed = 123; end
    rng(seed, 'twister');

    fprintf('========================================================================\n');
    fprintf('  Generating Ablation Dataset: Balanced Experts with M = 16 (L = 48 -> 5) \n');
    fprintf('========================================================================\n');

    current_dir = fileparts(mfilename('fullpath'));
    out_file    = fullfile(current_dir, 'dataset_ablation_balanced_M16.mat');

    d = 20;
    K = 5;

    % 1. Construct Ground Truth Model with Balanced Proportions
    param.beta0 = zeros(1, K);

    param.Beta = zeros(d, K);
    for k = 1:K
        param.Beta(:, k) = randi([-6, 6], d, 1);
        if k > 1
            while min(sqrt(sum((param.Beta(:, 1:k-1) - repmat(param.Beta(:, k), 1, k-1)).^2, 1))) < 3.0
                param.Beta(:, k) = randi([-6, 6], d, 1);
            end
        end
    end

    param.sigma2 = [3.0, 3.0, 3.0, 3.0, 3.0];

    % Balanced gating: all 5 experts around 20%
    target_proportions = [0.20, 0.20, 0.20, 0.20, 0.20];
    param.alpha0 = zeros(1, K);
    param.Alpha  = randn(d, K) * 0.25;
    param.Alpha  = param.Alpha - repmat(param.Alpha(:, K), 1, K);

    W_gate = [param.alpha0; param.Alpha];

    % 2. Sample Global Pool
    N_pool = 300000;
    X_pool = randn(N_pool, d);
    X_pool_aug = [ones(N_pool, 1), X_pool];

    logits = X_pool_aug * W_gate;
    logits = logits - max(logits, [], 2);
    prob_pool = exp(logits) ./ sum(exp(logits), 2);

    cum_prob = cumsum(prob_pool, 2);
    rand_vals = rand(N_pool, 1);
    Z_pool = sum(rand_vals > cum_prob, 2) + 1;
    Z_pool = min(Z_pool, K);

    emp_props = histcounts(Z_pool, 1:K+1) / N_pool;
    fprintf('Global pool proportions: [%s]\n', num2str(emp_props, '%.3f '));

    Y_pool = zeros(N_pool, 1);
    for k = 1:K
        idx_k = (Z_pool == k);
        mu_k  = param.beta0(k) + X_pool(idx_k, :) * param.Beta(:, k);
        sig_k = sqrt(param.sigma2(k));
        Y_pool(idx_k) = mu_k + sig_k * randn(sum(idx_k), 1);
    end

    % 3. Partition across M = 16 Machines (N_m = 500 per machine)
    % Uniformly distribute all 5 experts
    M = 16;
    N_per_machine = 500;
    K_vec = 3 * ones(1, M);

    subpopulations = { ...
        [1, 2, 3], ... % M1
        [2, 3, 4], ... % M2
        [3, 4, 5], ... % M3
        [4, 5, 1], ... % M4
        [5, 1, 2], ... % M5
        [1, 2, 3], ... % M6
        [2, 3, 4], ... % M7
        [3, 4, 5], ... % M8
        [4, 5, 1], ... % M9
        [5, 1, 2], ... % M10
        [1, 2, 3], ... % M11
        [2, 3, 4], ... % M12
        [3, 4, 5], ... % M13
        [4, 5, 1], ... % M14
        [5, 1, 2], ... % M15
        [1, 3, 5]  ... % M16
    };

    X_cells = cell(1, M);
    Y_cells = cell(1, M);
    Z_cells = cell(1, M);

    used_indices = false(N_pool, 1);

    fprintf('Partitioning %d machines (N_m = %d, K_m = 3)...\n', M, N_per_machine);
    for m = 1:M
        target_experts = subpopulations{m};
        eligible = find(ismember(Z_pool, target_experts) & ~used_indices);
        
        if length(eligible) < N_per_machine
            error('Machine %d: Not enough eligible samples (found %d, need %d).', ...
                  m, length(eligible), N_per_machine);
        end

        selected = eligible(randperm(length(eligible), N_per_machine));
        used_indices(selected) = true;

        X_cells{m} = X_pool(selected, :);
        Y_cells{m} = Y_pool(selected);
        Z_cells{m} = Z_pool(selected);

        local_counts = histcounts(Z_cells{m}, 1:K+1);
        fprintf('  Machine %2d (K_%d=%d): Active experts {%s}, Counts: [%s]\n', ...
                m, m, K_vec(m), num2str(target_experts), num2str(local_counts(target_experts)));
    end

    % 4. Generate Test Set (N_test = 20,000)
    N_test = 20000;
    X_test = randn(N_test, d);
    X_test_aug = [ones(N_test, 1), X_test];

    logits_test = X_test_aug * W_gate;
    logits_test = logits_test - max(logits_test, [], 2);
    prob_test   = exp(logits_test) ./ sum(exp(logits_test), 2);

    cum_prob_test = cumsum(prob_test, 2);
    rand_test = rand(N_test, 1);
    Z_test = sum(rand_test > cum_prob_test, 2) + 1;
    Z_test = min(Z_test, K);

    Y_test = zeros(N_test, 1);
    for k = 1:K
        idx_k = (Z_test == k);
        mu_k  = param.beta0(k) + X_test(idx_k, :) * param.Beta(:, k);
        sig_k = sqrt(param.sigma2(k));
        Y_test(idx_k) = mu_k + sig_k * randn(sum(idx_k), 1);
    end

    test_props = histcounts(Z_test, 1:K+1) / N_test;
    fprintf('Test set proportions: [%s]\n', num2str(test_props, '%.3f '));

    % 5. Save Output
    config.M = M;
    config.K_vec = K_vec;
    config.K_target = K;
    config.d = d;
    config.L = sum(K_vec);
    config.N_per_machine = N_per_machine;
    config.N_total_train = M * N_per_machine;
    config.N_test = N_test;
    config.subpopulations = subpopulations;
    config.target_proportions = target_proportions;

    save(out_file, 'X_cells', 'Y_cells', 'Z_cells', ...
                   'X_test', 'Y_test', 'Z_test', ...
                   'param', 'config', 'seed', '-v7.3');
    fprintf('==> Saved dataset successfully to: %s (%.2f MB)\n', ...
            out_file, dir(out_file).bytes / 1e6);
end
