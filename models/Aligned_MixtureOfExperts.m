function solution = Aligned_MixtureOfExperts(varargin)
% ALIGNED_MIXTUREOFEXPERTS
% Fits the Aligned Average estimator \bar{\theta}^A for distributed MoE.
%
% Idea:
%   "The experts of each local model are first matched to those of a
%    reference local model (Hungarian algorithm on the KL divergence between
%    experts), and the matched parameters are then averaged. This removes
%    the label switching problem of \bar{\theta}^W."
%
% Usage:
%   solution = Aligned_MixtureOfExperts(DME_solution, K, M, options);
%   or
%   solution = Aligned_MixtureOfExperts(X, Y, K, M, options);
%
% Outputs:
%   solution: struct containing fields (.param, .gates, .experts, .variances,
%             .weights, .reduced_mixture, .learning_time) compatible with
%             compute_metrics and all project evaluations.

    tic;

    % Parse inputs
    if isstruct(varargin{1}) && isfield(varargin{1}, 'local_estimates')
        DME_solution = varargin{1};
        K = varargin{2};
        M = varargin{3};
        if nargin >= 4, options = varargin{4}; else, options = get_options('default'); end
        
        local_estimates = DME_solution.local_estimates;
        local_times     = DME_solution.local_times;
        d               = DME_solution.d;
        if isfield(DME_solution, 'X_val') && ~isempty(DME_solution.X_val)
            X_val = DME_solution.X_val;
            if size(X_val, 2) == d
                X_val = [ones(size(X_val, 1), 1), X_val];
            end
        else
            X_val = [ones(100, 1), randn(100, d)];
        end
    else
        % Data passed directly: fit local models first
        X = varargin{1};
        Y = varargin{2};
        K = varargin{3};
        M = varargin{4};
        if nargin >= 5, options = varargin{5}; else, options = get_options('default'); end
        
        [n, d] = size(X);
        if isfield(options, 'S') && ~isempty(options.S)
            S_target = options.S;
        elseif isfield(options, 'sample_size') && ~isempty(options.sample_size)
            S_target = options.sample_size;
        else
            S_target = 2000;
        end
        S = min(N, S_target);
        idx_val = randperm(n, S);
        X_val = [ones(S, 1), X(idx_val, :)];
        
        idx_shuffled = randperm(n);
        local_times = zeros(1, M);
        local_estimates = cell(1, M);
        
        chi    = 0.1 * ones(1, K - 1);
        lambda = ones(1, K);
        
        for m = 1:M
            tic_loc = tic;
            idx_m = idx_shuffled((m - 1) * N + 1 : m * N);
            X_loc = X(idx_m, :);
            Y_loc = Y(idx_m);
            if isfield(options, 'LASSO') && options.LASSO
                local_est = MixtureOfExperts_LASSO(X_loc, Y_loc, K, chi, lambda, options);
            else
                local_est = MixtureOfExperts(X_loc, Y_loc, K, options);
            end
            local_estimates{m} = local_est;
            local_times(m) = toc(tic_loc);
        end
    end

    % Reference machine: Machine 1
    ref_param = local_estimates{1}.param;
    
    % Storage for aligned parameters across M machines
    aligned_beta0  = zeros(M, K);
    aligned_Beta   = zeros(d, K, M);
    aligned_sigma2 = zeros(M, K);
    aligned_alpha0 = zeros(M, K);
    aligned_Alpha  = zeros(d, K, M);
    
    % Ensure full K-dimensional gating parameters for reference machine
    ref_alpha0 = ref_param.alpha0;
    ref_Alpha  = ref_param.Alpha;
    if length(ref_alpha0) == K - 1
        ref_alpha0 = [ref_alpha0, 0];
        ref_Alpha  = [ref_Alpha, zeros(d, 1)];
    end
    
    % Machine 1 is the identity reference
    aligned_beta0(1, :)    = ref_param.beta0;
    aligned_Beta(:, :, 1)  = ref_param.Beta;
    aligned_sigma2(1, :)   = ref_param.sigma2;
    aligned_alpha0(1, :)   = ref_alpha0;
    aligned_Alpha(:, :, 1) = ref_Alpha;

    % Match each other machine m >= 2 to reference machine 1
    for m = 2:M
        cur_param = local_estimates{m}.param;
        
        % Build K x K cost matrix C:
        % C(k, j) = KL divergence between reference expert k and current expert j
        C = zeros(K, K);
        for k = 1:K
            expert_ref.xBeta  = [ref_param.beta0(k);  ref_param.Beta(:, k)];
            expert_ref.sigma2 = ref_param.sigma2(k);
            for j = 1:K
                expert_cur.xBeta  = [cur_param.beta0(j);  cur_param.Beta(:, j)];
                expert_cur.sigma2 = cur_param.sigma2(j);
                [cost, ~] = KL_distance(expert_ref, expert_cur, X_val);
                C(k, j) = cost;
            end
        end
        
        % Solve linear sum assignment (Hungarian matching)
        best_p = solve_assignment(C, K);
        
        % Ensure full K-dimensional gating parameters before reordering
        cur_alpha0 = cur_param.alpha0;
        cur_Alpha  = cur_param.Alpha;
        if length(cur_alpha0) == K - 1
            cur_alpha0 = [cur_alpha0, 0];
            cur_Alpha  = [cur_Alpha, zeros(d, 1)];
        end
        
        % Permute machine m parameters according to best_p:
        % Expert best_p(k) of machine m matches reference expert k
        aligned_beta0(m, :)    = cur_param.beta0(best_p);
        aligned_Beta(:, :, m)  = cur_param.Beta(:, best_p);
        aligned_sigma2(m, :)   = cur_param.sigma2(best_p);
        aligned_alpha0(m, :)   = cur_alpha0(best_p);
        aligned_Alpha(:, :, m) = cur_Alpha(:, best_p);
    end

    % Parameter Averaging across all M machines
    beta0  = mean(aligned_beta0, 1);
    Beta   = mean(aligned_Beta, 3);
    sigma2 = mean(aligned_sigma2, 1);
    alpha0 = mean(aligned_alpha0, 1);
    Alpha  = mean(aligned_Alpha, 3);

    % Normalize gating parameters so that component K is reference (0)
    alpha0 = alpha0 - alpha0(K);
    Alpha  = Alpha - repmat(Alpha(:, K), 1, K);

    % Assemble solution struct
    solution.param.alpha0 = alpha0;
    solution.param.Alpha  = Alpha;
    solution.param.beta0  = beta0;
    solution.param.Beta   = Beta;
    solution.param.sigma2 = sigma2;

    solution.gates     = [alpha0; Alpha];
    solution.experts   = [beta0; Beta];
    solution.variances = sigma2;
    solution.weights   = ones(1, K);

    solution.aligned_mixture.gates     = solution.gates;
    solution.aligned_mixture.experts   = solution.experts;
    solution.aligned_mixture.variances = solution.variances;
    solution.aligned_mixture.weights   = solution.weights;
    solution.reduced_mixture           = solution.aligned_mixture;

    align_time = toc;
    solution.learning_time = max(local_times) + align_time;

end


function best_p = solve_assignment(C, K)
% Solves min \sum_{k=1}^K C(k, p(k))
    % Sanitize cost matrix
    C(~isfinite(C)) = 1e8;
    C(C < 0) = 0;

    if K <= 7
        % Exact brute force via perms (K! <= 5040, takes < 0.1ms)
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
        % For larger K, use matchpairs if available, else greedy
        best_p = zeros(1, K);
        matched_row = false(1, K);
        matched_col = false(1, K);
        has_matchpairs = (exist('matchpairs', 'file') == 2) || (exist('matchpairs', 'builtin') == 5);

        if has_matchpairs
            try
                matches = matchpairs(C, 1e12);
                for i = 1:size(matches, 1)
                    r = matches(i, 1);
                    c = matches(i, 2);
                    if r >= 1 && r <= K && c >= 1 && c <= K
                        best_p(r) = c;
                        matched_row(r) = true;
                        matched_col(c) = true;
                    end
                end
            catch
                % Fall back to greedy below
            end
        end

        % Guarantee full 1-to-1 assignment: fill any unmatched rows
        unmatched_r = find(~matched_row);
        unmatched_c = find(~matched_col);
        for idx = 1:length(unmatched_r)
            r = unmatched_r(idx);
            if idx <= length(unmatched_c)
                best_p(r) = unmatched_c(idx);
            else
                best_p(r) = r;
            end
        end

        % Guarantee best_p is a strictly valid permutation of 1:K (no zeros, no duplicates)
        if any(best_p < 1) || any(best_p > K) || length(unique(best_p)) < K
            % Greedy matching
            C_copy = C;
            best_p = zeros(1, K);
            for k = 1:K
                [~, min_idx] = min(C_copy(k, :));
                best_p(k) = min_idx;
                C_copy(:, min_idx) = inf;
            end
            if any(best_p < 1) || any(best_p > K) || length(unique(best_p)) < K
                best_p = 1:K;
            end
        end
    end
end
