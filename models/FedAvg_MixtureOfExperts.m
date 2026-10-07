function solution = FedAvg_MixtureOfExperts(X, Y, K, M, options)
% FEDAVG_MIXTUREOFEXPERTS
% Fits a Mixture of Experts model using Federated Averaging (FedAvg) over T rounds.
% Uses the canonical MixtureOfExperts function for local client training.
%
% Reference: DME.md (line 256):
%   "FedAvg with T rounds: gradient-based training of the MoE log-likelihood
%    with local updates and T rounds of averaging, used as a reference with
%    T times more communication."
%
% Usage:
%   solution = FedAvg_MixtureOfExperts(X, Y, K, M, options);

    if nargin < 5, options = get_options('default'); end
    
    if isfield(options, 'FedAvg_rounds')
        T = options.FedAvg_rounds;
    else
        T = 5; % Default communication rounds
    end
    
    if isfield(options, 'FedAvg_local_iters')
        local_iters = options.FedAvg_local_iters;
    elseif isfield(options, 'max_iter')
        local_iters = max(10, round(options.max_iter / T));
    else
        local_iters = 25; % Local EM iterations per round
    end
    
    if isfield(options, 'verbose')
        verbose = options.verbose;
    else
        verbose = 0;
    end

    [n, d] = size(X);
    N = floor(n / M);
    
    % Partition data across M clients (custom client_indices if provided)
    client_X = cell(1, M);
    client_Y = cell(1, M);
    if isfield(options, 'client_indices') && ~isempty(options.client_indices)
        for m = 1:M
            idx_m = options.client_indices{m};
            client_X{m} = X(idx_m, :);
            client_Y{m} = Y(idx_m);
        end
    else
        indices_shuffled = randperm(n);
        for m = 1:M
            idx_m = indices_shuffled((m - 1) * N + 1 : m * N);
            client_X{m} = X(idx_m, :);
            client_Y{m} = Y(idx_m);
        end
    end

    % 1. Global Server Initialization (Round 0)
    % Initialize common global model via Client 1 local initialization (federated prior, no server data leakage)
    init_opt = options;
    init_opt.nb_EM_runs = 1;
    init_opt.max_iter   = 15;
    init_opt.verbose    = 0;
    tic_init = tic;
    init_est = MixtureOfExperts(client_X{1}, client_Y{1}, K, init_opt);
    init_time = toc(tic_init);
    global_param = init_est.param;
    
    total_learning_time = init_time;

    % 2. Federated Communication Rounds t = 1, ..., T
    for t = 1:T
        
        local_estimates = cell(1, M);
        local_times     = zeros(1, M);

        % Determine whether to parallelize across M local machines
        use_par_m = false;
        if isfield(options, 'parallel_machines')
            pm_val = options.parallel_machines;
            if islogical(pm_val) || isnumeric(pm_val)
                use_par_m = logical(pm_val);
            elseif ischar(pm_val) || isstring(pm_val)
                if strcmpi(pm_val, 'auto')
                    try
                        in_worker = ~isempty(getCurrentTask());
                    catch
                        in_worker = false;
                    end
                    pool_obj = gcp('nocreate');
                    use_par_m = ~in_worker && ~isempty(pool_obj);
                elseif strcmpi(pm_val, 'on') || strcmpi(pm_val, 'true')
                    use_par_m = true;
                end
            end
        end

        % Local training on each client starting from current global parameters
        if use_par_m
            parfor m = 1:M
                client_opt = options;
                client_opt.nb_EM_runs = 1;
                client_opt.max_iter   = local_iters;
                client_opt.init_param = global_param;
                client_opt.verbose    = 0;

                tic_loc = tic;
                loc_fit = MixtureOfExperts(client_X{m}, client_Y{m}, K, client_opt);
                local_times(m)     = toc(tic_loc);
                local_estimates{m} = loc_fit.param;
            end
        else
            for m = 1:M
                client_opt = options;
                client_opt.nb_EM_runs = 1;
                client_opt.max_iter   = local_iters;
                client_opt.init_param = global_param;
                client_opt.verbose    = 0;

                tic_loc = tic;
                loc_fit = MixtureOfExperts(client_X{m}, client_Y{m}, K, client_opt);
                local_times(m)     = toc(tic_loc);
                local_estimates{m} = loc_fit.param;
            end
        end

        % Server Aggregation (FedAvg with alignment to global_param)
        tic_agg = tic;
        beta0_mat  = zeros(M, K);
        Beta_mat   = zeros(d, K, M);
        sigma2_mat = zeros(M, K);
        alpha0_mat = zeros(M, K);
        Alpha_mat  = zeros(d, K, M);

        % Quick reference support for KL alignment across experts
        X_supp = [ones(min(200, N), 1), randn(min(200, N), d)];

        for m = 1:M
            p_m = local_estimates{m};
            
            % Safe variances
            loc_sigma2 = max(p_m.sigma2, 1e-4);
            glb_sigma2 = max(global_param.sigma2, 1e-4);

            % Match client m experts to current global_param via KL assignment
            % C(k_glb, j_loc): cost of assigning local expert j_loc to global expert k_glb
            C = zeros(K, K);
            for k_glb = 1:K
                e_glb.xBeta  = [global_param.beta0(k_glb); global_param.Beta(:, k_glb)];
                e_glb.sigma2 = glb_sigma2(k_glb);
                for j_loc = 1:K
                    e_loc.xBeta  = [p_m.beta0(j_loc); p_m.Beta(:, j_loc)];
                    e_loc.sigma2 = loc_sigma2(j_loc);
                    [~, kldiv]   = KL_distance(e_glb, e_loc, X_supp);
                    c_val = sum(kldiv);
                    if ~isfinite(c_val) || c_val < 0
                        c_val = 1e8;
                    end
                    C(k_glb, j_loc) = c_val;
                end
            end
            
            % Robust Hungarian assignment: best_p(k_glb) is the local expert for global expert k_glb
            best_p = solve_fedavg_assignment(C, K);
            
            % Ensure full K-dimensional gating parameters before reordering
            if length(p_m.alpha0) == K - 1
                p_alpha0 = [p_m.alpha0(:)', 0];
                p_Alpha  = [p_m.Alpha, zeros(d, 1)];
            else
                p_alpha0 = p_m.alpha0(:)';
                p_Alpha  = p_m.Alpha;
            end

            % Reorder client parameters to match global indices
            beta0_mat(m, :)    = p_m.beta0(best_p);
            Beta_mat(:, :, m)  = p_m.Beta(:, best_p);
            sigma2_mat(m, :)   = loc_sigma2(best_p);
            alpha0_mat(m, :)   = p_alpha0(best_p);
            Alpha_mat(:, :, m) = p_Alpha(:, best_p);
        end

        global_param.beta0  = mean(beta0_mat, 1);
        global_param.Beta   = mean(Beta_mat, 3);
        global_param.sigma2 = mean(sigma2_mat, 1);
        global_param.alpha0 = mean(alpha0_mat, 1);
        global_param.Alpha  = mean(Alpha_mat, 3);

        % Normalize softmax gating (reference component K = 0)
        global_param.alpha0 = global_param.alpha0 - global_param.alpha0(K);
        global_param.Alpha  = global_param.Alpha - repmat(global_param.Alpha(:, K), 1, K);
        agg_time = toc(tic_agg);

        % Distributed learning time convention: max(local_times) + aggregation_time
        round_time = max(local_times) + agg_time;
        total_learning_time = total_learning_time + round_time;
        
        if verbose
            fprintf('FedAvg Round %d/%d completed in %.3fs\n', t, T, round_time);
        end

        % Checkpoint snapshot along trajectory if requested
        if isfield(options, 'FedAvg_snapshots') && ismember(t, options.FedAvg_snapshots)
            snap = struct();
            snap.param     = global_param;
            snap.gates     = [global_param.alpha0; global_param.Alpha];
            snap.experts   = [global_param.beta0; global_param.Beta];
            snap.variances = global_param.sigma2;
            snap.weights   = ones(1, K);
            snap.reduced_mixture.gates     = snap.gates;
            snap.reduced_mixture.experts   = snap.experts;
            snap.reduced_mixture.variances = snap.variances;
            snap.reduced_mixture.weights   = snap.weights;
            snap.learning_time = total_learning_time;
            snap.rounds        = t;
            solution.snapshots.(sprintf('FED_T%d', t)) = snap;
        end
    end

    % Assemble solution struct
    solution.param     = global_param;
    solution.gates     = [global_param.alpha0; global_param.Alpha];
    solution.experts   = [global_param.beta0; global_param.Beta];
    solution.variances = global_param.sigma2;
    solution.weights   = ones(1, K);

    solution.reduced_mixture.gates     = solution.gates;
    solution.reduced_mixture.experts   = solution.experts;
    solution.reduced_mixture.variances = solution.variances;
    solution.reduced_mixture.weights   = solution.weights;
    
    solution.learning_time = total_learning_time;
    solution.rounds        = T;

end


function best_p = solve_fedavg_assignment(C, K)
% Solves min \sum_{k=1}^K C(k, p(k))
% Guaranteed to return a valid permutation of 1:K of length K.
    C(~isfinite(C)) = 1e8;
    C(C < 0) = 0;
    
    if K <= 7
        % Exact brute-force over all K! permutations (for K=5, 120 perms, <0.05ms)
        all_perms = perms(1:K);
        n_perms = size(all_perms, 1);
        min_cost = inf;
        best_p = 1:K;
        for ip = 1:n_perms
            p = all_perms(ip, :);
            total_cost = 0;
            for k = 1:K
                total_cost = total_cost + C(k, p(k));
            end
            if total_cost < min_cost
                min_cost = total_cost;
                best_p = p;
            end
        end
    else
        % Greedy matching with full coverage fallback
        C_copy = C;
        best_p = zeros(1, K);
        for k = 1:K
            [~, min_idx] = min(C_copy(k, :));
            best_p(k) = min_idx;
            C_copy(:, min_idx) = inf;
        end
    end
end
