function solution = Greedy_MixtureOfExperts_Hetero(varargin)
% GREEDY_MIXTUREOFEXPERTS_HETERO
% Runnalls (2007) greedy KL-based component merging for heterogeneous local
% MoE models where machine m has K_m experts (total L = sum(K_m) components),
% reducing them to K_target experts.
%
% Usage:
%   solution = Greedy_MixtureOfExperts_Hetero(DME_solution, K_target, options);
%   or
%   solution = Greedy_MixtureOfExperts_Hetero(X_cells, Y_cells, K_vec, K_target, options);

    % 1. Parse inputs
    if isstruct(varargin{1}) && isfield(varargin{1}, 'local_estimates')
        DME_solution    = varargin{1};
        K_target        = varargin{2};
        if nargin >= 3, options = varargin{3}; else, options = get_options('default'); end

        local_estimates = DME_solution.local_estimates;
        local_times     = DME_solution.local_times;
        d               = DME_solution.d;
        X_val           = DME_solution.X_val;
        M               = length(local_estimates);
        n               = DME_solution.n;
    else
        X_cells         = varargin{1};
        Y_cells         = varargin{2};
        K_vec           = varargin{3};
        K_target        = varargin{4};
        if nargin >= 5, options = varargin{5}; else, options = get_options('default'); end

        M = length(X_cells);
        d = size(X_cells{1}, 2);
        
        n_total = 0;
        for m = 1:M
            n_total = n_total + size(X_cells{m}, 1);
        end
        n = n_total;

        % Construct validation support X_val
        if isfield(options, 'S') && ~isempty(options.S)
            S_target = options.S;
        elseif isfield(options, 'sample_size') && ~isempty(options.sample_size)
            S_target = options.sample_size;
        else
            S_target = 2000;
        end
        S = min(S_target, n);
        X_all = [];
        for m = 1:M
            X_all = [X_all; X_cells{m}]; %#ok<AGROW>
            if size(X_all, 1) >= S, break; end
        end
        idx_val = randperm(size(X_all, 1), min(S, size(X_all, 1)));
        X_val = [ones(length(idx_val), 1), X_all(idx_val, :)];

        % Fit local models
        local_estimates = cell(1, M);
        local_times     = zeros(1, M);
        for m = 1:M
            tic_m = tic;
            local_est = MixtureOfExperts(X_cells{m}, Y_cells{m}, K_vec(m), options);
            local_times(m) = toc(tic_m);
            local_estimates{m} = local_est;
        end
    end

    tic;
    S = size(X_val, 1);

    % 2. Assemble L = sum(K_m) source components and source probabilities PI_hat
    experts   = [];
    variances = [];
    weights   = [];
    PI_hat    = [];

    for m = 1:M
        p_m = local_estimates{m}.param;
        Km  = length(p_m.beta0);

        % Local sample weight N_m / N
        if isfield(local_estimates{m}, 'stats') && isfield(local_estimates{m}.stats, 'n')
            w_m = local_estimates{m}.stats.n / n;
        else
            w_m = 1 / M;
        end

        % Extract full gating parameters (d+1) x Km
        if length(p_m.alpha0) == Km - 1
            alpha0_full = [p_m.alpha0(:)', 0];
            Alpha_full  = [p_m.Alpha, zeros(d, 1)];
        else
            alpha0_full = p_m.alpha0(:)';
            Alpha_full  = p_m.Alpha;
        end
        W_gate = [alpha0_full; Alpha_full];

        % Compute local softmax gating probabilities on support X_val
        logits = X_val * W_gate;
        logits = logits - max(logits, [], 2);
        prob_m = exp(logits) ./ sum(exp(logits), 2);

        % Weight source components by machine sample proportion
        PI_hat = [PI_hat, w_m * prob_m]; %#ok<AGROW>

        experts   = [experts,   [p_m.beta0; p_m.Beta]]; %#ok<AGROW>
        variances = [variances, max(p_m.sigma2, 1e-4)]; %#ok<AGROW>
        weights   = [weights,   (w_m / Km) * ones(1, Km)]; %#ok<AGROW>
    end

    % Normalize global source weights
    weights = weights / sum(weights);
    L = length(variances);

    % 3. Runnalls Greedy Pairwise Merging Loop (L -> K_target)
    B = experts;       % (d+1) x L
    V = variances;     % 1 x L
    w = weights;       % 1 x L
    
    active = true(1, L);
    num_active = L;

    % Precompute pairwise merging costs
    C = inf(L, L);
    for i = 1:L
        for j = (i + 1):L
            C(i, j) = runnalls_merging_cost(B(:, i), V(i), w(i), B(:, j), V(j), w(j), X_val, S);
        end
    end

    while num_active > K_target
        [min_cost, idx] = min(C(:));
        if isinf(min_cost)
            break;
        end
        [i_star, j_star] = ind2sub([L, L], idx);

        % Merged parameters via moment matching
        w_m = w(i_star) + w(j_star);
        B_m = (w(i_star) * B(:, i_star) + w(j_star) * B(:, j_star)) / w_m;

        diff_mu = X_val * (B(:, i_star) - B(:, j_star));
        D2_mu   = sum(diff_mu .^ 2) / S;
        V_m     = (w(i_star) * V(i_star) + w(j_star) * V(j_star)) / w_m + ...
                  (w(i_star) * w(j_star) / (w_m ^ 2)) * D2_mu;

        % Update i_star with merged component
        w(i_star)    = w_m;
        B(:, i_star) = B_m;
        V(i_star)    = V_m;

        % Deactivate j_star
        active(j_star) = false;
        num_active     = num_active - 1;

        % Clear costs for j_star
        C(j_star, :) = inf;
        C(:, j_star) = inf;

        % Update pairwise costs for i_star
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

    % Collect remaining K_target experts
    active_idx = find(active);
    merged_mixture.experts   = B(:, active_idx);
    merged_mixture.variances = V(active_idx);
    merged_mixture.weights   = ones(1, K_target);
    merged_mixture.gates     = zeros(d + 1, K_target);

    % 4. Fit Gating Network via OT on Support and IRLS
    large_mixture.experts   = experts;
    large_mixture.variances = variances;
    large_mixture.weights   = weights;
    large_mixture.PI_hat    = PI_hat; % Uses dual-path in argmin_transportation_plan

    [plan, distance_matrix] = argmin_transportation_plan(large_mixture, merged_mixture, X_val);
    transportdis = sum(sum(sum(plan .* distance_matrix)));

    % Target gating probabilities on support
    gatingProb = squeeze(sum(plan, 1))';
    gatingProb = gatingProb ./ sum(gatingProb, 2);
    gatingProb = max(gatingProb, eps);
    gatingProb = gatingProb ./ sum(gatingProb, 2);

    % Fit global softmax gating via IRLS
    initial_gate = zeros(d + 1, K_target - 1);
    res = IRLS(X_val, gatingProb, initial_gate, ones(S, 1), 10000, 1e-8, 0);
    merged_mixture.gates = [res.W, zeros(d + 1, 1)];

    % 5. Assemble standardized solution struct
    solution.param.alpha0 = merged_mixture.gates(1, :);
    solution.param.Alpha  = merged_mixture.gates(2:end, :);
    solution.param.beta0  = merged_mixture.experts(1, :);
    solution.param.Beta   = merged_mixture.experts(2:end, :);
    solution.param.sigma2 = merged_mixture.variances;

    solution.gates     = merged_mixture.gates;
    solution.experts   = merged_mixture.experts;
    solution.variances = merged_mixture.variances;
    solution.weights   = ones(1, K_target);

    solution.reduced_mixture = merged_mixture;
    solution.merged_mixture  = merged_mixture;
    solution.plan            = plan;
    solution.best_dis        = transportdis;
    solution.transportdis    = transportdis;
    solution.large_mixture   = large_mixture;

    greedy_time = toc;
    solution.learning_time = max(local_times) + greedy_time;

end


function cost = runnalls_merging_cost(B1, V1, w1, B2, V2, w2, X_val, S)
% Runnalls (2007) KL-based discrimination cost for Gaussian components
    w_m = w1 + w2;
    diff_mu = X_val * (B1 - B2);
    D2_mu   = sum(diff_mu .^ 2) / S;
    
    V_m = (w1 * V1 + w2 * V2) / w_m + (w1 * w2 / (w_m ^ 2)) * D2_mu;
    
    cost = 0.5 * (w_m * log(max(V_m, 1e-8)) - w1 * log(max(V1, 1e-8)) - w2 * log(max(V2, 1e-8)));
    if cost < 0 || ~isfinite(cost)
        cost = 0;
    end
end
