function solution = Greedy_MixtureOfExperts(varargin)
% GREEDY_MIXTUREOFEXPERTS
% Fits the Greedy Merging estimator \bar{\theta}^{GM} for distributed MoE.
%
% Reference: DME.md (line 254):
%   "The MK experts of f^W are merged two by two, starting with the pair
%    of smallest KL-based merging cost averaged over D_S (Runnalls, 2007),
%    until K experts remain; the gates are then fitted as for \bar{\theta}^R."
%
% Usage:
%   solution = Greedy_MixtureOfExperts(DME_solution, K, M, options);
%   or
%   solution = Greedy_MixtureOfExperts(X, Y, K, M, options);
%
% Outputs:
%   solution: struct containing fields (.param, .gates, .experts, .variances,
%             .weights, .reduced_mixture, .learning_time) compatible with
%             compute_metrics and all project evaluations.

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
        n = DME_solution.n;
    else
        % Data passed directly
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
        S = min(n, S_target);
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

    tic;
    S = size(X_val, 1);

    % Build initial large mixture of L = M*K experts
    experts   = [];
    gates     = [];
    variances = [];
    weights   = [];
    for m = 1:M
        experts   = [experts,   [local_estimates{m}.param.beta0;  local_estimates{m}.param.Beta]];
        gate_     = [local_estimates{m}.param.alpha0; local_estimates{m}.param.Alpha];
        gates     = [gates,     repmat(gate_(:), 1, K)];
        variances = [variances, local_estimates{m}.param.sigma2];
        if isfield(local_estimates{m}, 'stats') && isfield(local_estimates{m}.stats, 'n')
            weights = [weights, (local_estimates{m}.stats.n / n) * ones(1, K) / K];
        else
            weights = [weights, ones(1, K) / (M * K)];
        end
    end

    % Normalize weights
    weights = weights / sum(weights);

    large_mixture.experts   = experts;
    large_mixture.gates     = gates;
    large_mixture.variances = variances;
    large_mixture.weights   = ones(1, M*K) / M; % for argmin_transportation_plan compatibility

    % Components to be merged
    L = M * K;
    B = experts;       % (d+1) x L
    V = variances;     % 1 x L
    w = weights;       % 1 x L
    
    active = true(1, L);
    num_active = L;

    % Precompute pairwise merging costs for all active pairs
    C = inf(L, L);
    for i = 1:L
        for j = (i + 1):L
            C(i, j) = runnalls_merging_cost(B(:, i), V(i), w(i), B(:, j), V(j), w(j), X_val, S);
        end
    end

    % Greedy merging loop: merge pair with minimal Runnalls KL-cost until K remain
    while num_active > K
        % Find pair (i*, j*) with minimum merging cost
        [min_cost, idx] = min(C(:));
        if isinf(min_cost)
            break;
        end
        [i_star, j_star] = ind2sub([L, L], idx);
        
        % Compute merged component parameters
        w_m = w(i_star) + w(j_star);
        B_m = (w(i_star) * B(:, i_star) + w(j_star) * B(:, j_star)) / w_m;
        
        diff_mu = X_val * (B(:, i_star) - B(:, j_star));
        D2_mu = mean(diff_mu .^ 2);
        V_m = (w(i_star) * V(i_star) + w(j_star) * V(j_star)) / w_m + ...
              (w(i_star) * w(j_star) / (w_m ^ 2)) * D2_mu;
        
        % Replace i_star with merged component
        w(i_star) = w_m;
        B(:, i_star) = B_m;
        V(i_star) = V_m;
        
        % Deactivate j_star
        active(j_star) = false;
        num_active = num_active - 1;
        
        % Clear costs involving j_star
        C(j_star, :) = inf;
        C(:, j_star) = inf;
        
        % Update costs involving i_star
        for other = find(active)
            if other < i_star
                C(other, i_star) = runnalls_merging_cost(B(:, other), V(other), w(other), ...
                                                        B(:, i_star), V(i_star), w(i_star), X_val, S);
            elseif other > i_star
                C(i_star, other) = runnalls_merging_cost(B(:, i_star), V(i_star), w(i_star), ...
                                                        B(:, other), V(other), w(other), X_val, S);
            end
        end
    end

    % Collect remaining K merged experts
    active_indices = find(active);
    merged_mixture.experts   = B(:, active_indices);
    merged_mixture.variances = V(active_indices);
    merged_mixture.weights   = ones(1, K);
    merged_mixture.gates     = zeros(d + 1, K);

    % Fit gates as for \bar{\theta}^R:
    % 1. Compute optimal transportation plan from large_mixture to merged_mixture
    [plan, ~] = argmin_transportation_plan(large_mixture, merged_mixture, X_val);

    % 2. Target gating probabilities at supporting points
    gatingProb = squeeze(sum(plan, 1))';
    gatingProb = gatingProb ./ sum(gatingProb, 2);
    gatingProb = max(gatingProb, eps);
    gatingProb = gatingProb ./ sum(gatingProb, 2);

    % 3. Fit softmax gating network parameters via IRLS on X_val
    initial_gate = reshape(mean(large_mixture.gates, 2), d + 1, K);
    res = IRLS(X_val, gatingProb, initial_gate(:, 1:end-1), ones(S, 1), 10000, 1e-8, 0);
    merged_mixture.gates = [res.W, zeros(d + 1, 1)];

    % Assemble solution
    solution.param.alpha0 = merged_mixture.gates(1, :);
    solution.param.Alpha  = merged_mixture.gates(2:end, :);
    solution.param.beta0  = merged_mixture.experts(1, :);
    solution.param.Beta   = merged_mixture.experts(2:end, :);
    solution.param.sigma2 = merged_mixture.variances;

    solution.gates     = merged_mixture.gates;
    solution.experts   = merged_mixture.experts;
    solution.variances = merged_mixture.variances;
    solution.weights   = ones(1, K);

    solution.reduced_mixture = merged_mixture;
    solution.merged_mixture  = merged_mixture;

    greedy_time = toc;
    solution.reduction_time = greedy_time;
    solution.learning_time = max(local_times) + greedy_time;

end


function cost = runnalls_merging_cost(B1, V1, w1, B2, V2, w2, X_val, S)
% Computes Runnalls (2007) KL-based merging cost for conditional Gaussian experts
    w_m = w1 + w2;
    diff_mu = X_val * (B1 - B2);
    D2_mu = sum(diff_mu .^ 2) / S;
    
    V_m = (w1 * V1 + w2 * V2) / w_m + (w1 * w2 / (w_m ^ 2)) * D2_mu;
    
    % Runnalls discrimination / KL increment:
    cost = 0.5 * (w_m * log(V_m) - w1 * log(V1) - w2 * log(V2));
    if cost < 0 || isnan(cost)
        cost = 0;
    end
end
