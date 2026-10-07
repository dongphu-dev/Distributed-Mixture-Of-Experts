function data_generator_hetero_Km(seed)
% DATA_GENERATOR_HETERO_KM
% Generates non-IID partitioned datasets for M = 4 machines where each machine
% observes an honest subset of the K* = 5 global experts (Latent Subpopulation Partitioning).
%
% Partitioning Configuration:
%   Machine 1: Experts {1, 2}       -> K_1 = 2
%   Machine 2: Experts {2, 3, 4}    -> K_2 = 3
%   Machine 3: Experts {4, 5}       -> K_3 = 2
%   Machine 4: Experts {1, 3, 5}    -> K_4 = 3
% Total components: L = 2 + 3 + 2 + 3 = 10
% Target global experts: K_target = 5
%
% Output:
%   data/dataset_hetero_Km_K5_d20.mat

    if nargin < 1 || isempty(seed), seed = 42; end
    rng(seed);

    fprintf('========================================================\n');
    fprintf('  Generating Heterogeneous MoE Dataset (K_m Variable)  \n');
    fprintf('========================================================\n');

    % 1. Load or initialize ground truth parameters (K* = 5, d = 20)
    current_dir = fileparts(mfilename('fullpath'));
    gt_file = fullfile(current_dir, 'ground_truth_param_K5_d20.mat');

    if exist(gt_file, 'file') == 2
        gt_data = load(gt_file);
        param = gt_data.param;
        d = gt_data.d;
        K = gt_data.K;
        fprintf('Loaded existing ground truth: K = %d, d = %d\n', K, d);
    else
        d = 20;
        K = 5;
        param.beta0  = zeros(1, K);
        param.Beta   = randn(d, K) * 3;
        param.sigma2 = [3.0, 3.5, 4.0, 3.2, 3.8];
        param.alpha0 = zeros(1, K);
        param.Alpha  = randn(d, K) * 0.45;
        % Normalize reference component K = 0
        param.Alpha  = param.Alpha - repmat(param.Alpha(:, K), 1, K);
    end

    % Ensure full (d+1) x K gating parameter matrix W
    if length(param.alpha0) == K - 1
        alpha0_full = [param.alpha0(:)', 0];
        Alpha_full  = [param.Alpha, zeros(d, 1)];
    else
        alpha0_full = param.alpha0(:)';
        Alpha_full  = param.Alpha;
    end
    W_gate = [alpha0_full; Alpha_full]; % (d+1) x K

    % 2. Generate a large pool of global data points
    N_pool = 400000;
    X_pool = randn(N_pool, d);
    X_pool_aug = [ones(N_pool, 1), X_pool];

    % Gating probabilities for each point
    logits = X_pool_aug * W_gate;
    logits = logits - max(logits, [], 2);
    prob_pool = exp(logits) ./ sum(exp(logits), 2);

    % Sample latent expert assignment Z_i in {1, ..., K}
    cum_prob = cumsum(prob_pool, 2);
    rand_vals = rand(N_pool, 1);
    Z_pool = sum(rand_vals > cum_prob, 2) + 1;
    Z_pool = min(Z_pool, K);

    % Generate continuous regression response y_i
    Y_pool = zeros(N_pool, 1);
    for k = 1:K
        idx_k = (Z_pool == k);
        mu_k  = param.beta0(k) + X_pool(idx_k, :) * param.Beta(:, k);
        sig_k = sqrt(param.sigma2(k));
        Y_pool(idx_k) = mu_k + sig_k * randn(sum(idx_k), 1);
    end

    % 3. Partition samples into M = 4 machines according to subpopulation sets
    M = 4;
    subpopulations = {[1, 2], [2, 3, 4], [4, 5], [1, 3, 5]};
    K_vec = [2, 3, 2, 3];
    N_per_machine = 25000;

    X_cells = cell(1, M);
    Y_cells = cell(1, M);
    Z_cells = cell(1, M);

    for m = 1:M
        target_experts = subpopulations{m};
        eligible_indices = find(ismember(Z_pool, target_experts));
        
        if length(eligible_indices) < N_per_machine
            error('Machine %d: Not enough eligible samples (found %d, need %d). Increase N_pool.', ...
                  m, length(eligible_indices), N_per_machine);
        end

        selected = eligible_indices(randperm(length(eligible_indices), N_per_machine));
        X_cells{m} = X_pool(selected, :);
        Y_cells{m} = Y_pool(selected);
        Z_cells{m} = Z_pool(selected);

        % Remove selected from pool to ensure disjointness
        Z_pool(selected) = -1;

        fprintf('Machine %d: Km = %d, Subpopulations = [%s], Samples = %d\n', ...
                m, K_vec(m), num2str(target_experts), N_per_machine);
    end

    % 4. Generate independent global test set (N_test = 20,000)
    N_test = 20000;
    X_test = randn(N_test, d);
    X_test_aug = [ones(N_test, 1), X_test];
    logits_test = X_test_aug * W_gate;
    logits_test = logits_test - max(logits_test, [], 2);
    prob_test = exp(logits_test) ./ sum(exp(logits_test), 2);
    cum_prob_test = cumsum(prob_test, 2);
    Z_test = sum(rand(N_test, 1) > cum_prob_test, 2) + 1;
    Z_test = min(Z_test, K);

    Y_test = zeros(N_test, 1);
    for k = 1:K
        idx_k = (Z_test == k);
        mu_k  = param.beta0(k) + X_test(idx_k, :) * param.Beta(:, k);
        sig_k = sqrt(param.sigma2(k));
        Y_test(idx_k) = mu_k + sig_k * randn(sum(idx_k), 1);
    end

    % 5. Save dataset struct
    output_file = fullfile(current_dir, 'dataset_hetero_Km_K5_d20.mat');
    config.M = M;
    config.K_vec = K_vec;
    config.K_target = K;
    config.subpopulations = subpopulations;
    config.N_per_machine = N_per_machine;
    config.N_test = N_test;
    config.d = d;
    config.seed = seed;

    save(output_file, 'X_cells', 'Y_cells', 'Z_cells', 'X_test', 'Y_test', 'Z_test', ...
         'param', 'config', '-v7.3');

    fprintf('Dataset successfully saved to: %s\n', output_file);

end
