function data_generator_ablation_rare_M4(seed)
% DATA_GENERATOR_ABLATION_RARE_M4
% Generates an ablation benchmark dataset to isolate the Rare Expert factor:
%   - M = 4 machines (small M, small L = 10 components)
%   - Target: K = 5 global experts
%   - Rare Expert 1: only 5% of global subpopulation (Expert 2 is 40%)
%   - Small local samples: N_m = 500 samples per machine (total N_train = 2,000)
%   - Partition:
%       Machine 1: Experts {1, 2}       -> K_1 = 2 (contains rare Expert 1)
%       Machine 2: Experts {2, 3, 4}    -> K_2 = 3
%       Machine 3: Experts {4, 5}       -> K_3 = 2
%       Machine 4: Experts {1, 3, 5}    -> K_4 = 3 (contains rare Expert 1)
%     Total L = 2 + 3 + 2 + 3 = 10 components (GM requires 5 merges).
%   - Equal noise variances sigma2 = [3.0, 3.0, 3.0, 3.0, 3.0] and zero intercepts.
%
% Output:
%   data/dataset_ablation_rare_M4.mat

    if nargin < 1 || isempty(seed), seed = 123; end
    rng(seed, 'twister');

    fprintf('========================================================================\n');
    fprintf('  Generating Ablation Dataset: Rare Expert 5%% with M = 4 (L = 10 -> 5)  \n');
    fprintf('========================================================================\n');

    current_dir = fileparts(mfilename('fullpath'));
    out_file    = fullfile(current_dir, 'dataset_ablation_rare_M4.mat');

    d = 20;
    K = 5;

    % 1. Construct Ground Truth Model with Severe Subpopulation Imbalance
    param.beta0 = zeros(1, K);

    % Well-separated regression slope vectors Beta (d x K)
    param.Beta = zeros(d, K);
    for k = 1:K
        param.Beta(:, k) = randi([-6, 6], d, 1);
        if k > 1
            while min(sqrt(sum((param.Beta(:, 1:k-1) - repmat(param.Beta(:, k), 1, k-1)).^2, 1))) < 3.0
                param.Beta(:, k) = randi([-6, 6], d, 1);
            end
        end
    end

    % Equal noise variances
    param.sigma2 = [3.0, 3.0, 3.0, 3.0, 3.0];

    % Imbalanced gating: Expert 1 is rare (5%), Expert 2 is dominant (40%)
    target_proportions = [0.05, 0.40, 0.20, 0.25, 0.10];
    param.alpha0 = log(target_proportions) - log(target_proportions(K));
    param.Alpha  = randn(d, K) * 0.25;
    param.Alpha  = param.Alpha - repmat(param.Alpha(:, K), 1, K);

    W_gate = [param.alpha0; param.Alpha]; % (d+1) x K

    % 2. Sample Global Pool for Local Machine Partitioning
    N_pool = 200000;
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

    % Generate continuous regression responses Y_pool
    Y_pool = zeros(N_pool, 1);
    for k = 1:K
        idx_k = (Z_pool == k);
        mu_k  = param.beta0(k) + X_pool(idx_k, :) * param.Beta(:, k);
        sig_k = sqrt(param.sigma2(k));
        Y_pool(idx_k) = mu_k + sig_k * randn(sum(idx_k), 1);
    end

    % 3. Partition across M = 4 Machines (N_m = 500 per machine)
    M = 4;
    N_per_machine = 500;
    subpopulations = {[1, 2], [2, 3, 4], [4, 5], [1, 3, 5]};
    K_vec = [2, 3, 2, 3];

    X_cells = cell(1, M);
    Y_cells = cell(1, M);
    Z_cells = cell(1, M);

    used_indices = false(N_pool, 1);

    fprintf('Partitioning %d machines (N_m = %d)...\n', M, N_per_machine);
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
        fprintf('  Machine %d (K_%d=%d): Active experts {%s}, Counts: [%s]\n', ...
                m, m, K_vec(m), num2str(target_experts), num2str(local_counts(target_experts)));
    end

    % 4. Generate Independent Unseen Test Set (N_test = 20,000)
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

    % 5. Save Structured Output
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
