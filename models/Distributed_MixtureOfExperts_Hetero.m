function solution = Distributed_MixtureOfExperts_Hetero(varargin)
% DISTRIBUTED_MIXTUREOFEXPERTS_HETERO
% Optimal-transport aggregation of heterogeneous local MoE models where
% machine m has K_m experts (total L = sum(K_m) components), reducing them
% to a global target model with K_target experts.
%
% Usage:
%   solution = Distributed_MixtureOfExperts_Hetero(X, Y, K_vec, M, options);
%   or
%   solution = Distributed_MixtureOfExperts_Hetero(X_cells, Y_cells, K_vec, K_target, options);

    % 1. Input parsing
    if isstruct(varargin{1}) && isfield(varargin{1}, 'local_estimates')
        dme_input       = varargin{1};
        K_target        = varargin{2};
        if nargin >= 3, options = varargin{3}; else, options = get_options('default'); end

        local_estimates = dme_input.local_estimates;
        local_times     = dme_input.local_times;
        d               = dme_input.d;
        X_val           = dme_input.X_val;
        if isfield(dme_input, 'Y_val')
            Y_val = dme_input.Y_val;
        else
            Y_val = [];
        end
        M               = length(local_estimates);
        n_total         = dme_input.n;
        S               = size(X_val, 1);
        skip_local_em   = true;
    elseif iscell(varargin{1})
        X_cells  = varargin{1};
        Y_cells  = varargin{2};
        K_vec    = varargin{3};
        if nargin >= 4 && ~isempty(varargin{4})
            if isstruct(varargin{4})
                options  = varargin{4};
                K_target = 5;
            else
                K_target = varargin{4};
                if nargin >= 5, options = varargin{5}; else, options = get_options('default'); end
            end
        else
            K_target = 5;
            options  = get_options('default');
        end
        M = length(X_cells);
        d = size(X_cells{1}, 2);
    else
        X     = varargin{1};
        Y     = varargin{2};
        K_vec = varargin{3};
        M     = varargin{4};
        if nargin >= 5, options = varargin{5}; else, options = get_options('default'); end
        
        [n, d] = size(X);
        if isfield(options, 'K_target') && ~isempty(options.K_target)
            K_target = options.K_target;
        else
            K_target = 5;
        end

        % Partition X and Y into M blocks
        N = floor(n / M);
        idx_shuf = randperm(n);
        X_cells = cell(1, M);
        Y_cells = cell(1, M);
        for m = 1:M
            idx_m = idx_shuf((m - 1) * N + 1 : m * N);
            X_cells{m} = X(idx_m, :);
            Y_cells{m} = Y(idx_m);
        end
    end

    if isfield(options, 'K_target') && ~isempty(options.K_target)
        K_target = options.K_target;
    end

    if ~exist('skip_local_em', 'var') || ~skip_local_em
        % Total sample size across cells
        n_total = 0;
        for m = 1:M
            n_total = n_total + size(X_cells{m}, 1);
        end

        % Construct validation support X_val
        if isfield(options, 'S') && ~isempty(options.S)
            S = options.S;
        elseif isfield(options, 'sample_size') && ~isempty(options.sample_size)
            S = options.sample_size;
        else
            S = min(2000, n_total);
        end

        X_pooled = [];
        Y_pooled = [];
        for m = 1:M
            X_pooled = [X_pooled; X_cells{m}]; %#ok<AGROW>
            Y_pooled = [Y_pooled; Y_cells{m}]; %#ok<AGROW>
            if size(X_pooled, 1) >= S * 2, break; end
        end
        idx_val = randperm(size(X_pooled, 1), min(S, size(X_pooled, 1)));
        X_val = [ones(length(idx_val), 1), X_pooled(idx_val, :)];
        Y_val = Y_pooled(idx_val);
        S = size(X_val, 1);

        % 2. Local EM estimation on each machine (can use parfor)
        local_estimates = cell(1, M);
        local_times     = zeros(1, M);

        parfor m = 1:M
            tic_m = tic;
            local_est = MixtureOfExperts(X_cells{m}, Y_cells{m}, K_vec(m), options);
            local_times(m) = toc(tic_m);
            local_estimates{m} = local_est;
        end
    end

    % 3. Assemble large mixture of L = sum(K_m) components
    tic_agg = tic;
    experts   = [];
    variances = [];
    weights   = [];
    PI_hat    = [];

    for m = 1:M
        p_m = local_estimates{m}.param;
        Km  = length(p_m.beta0);
        if exist('X_cells', 'var') && length(X_cells) >= m
            N_m = size(X_cells{m}, 1);
        elseif isfield(local_estimates{m}, 'stats') && isfield(local_estimates{m}.stats, 'n')
            N_m = local_estimates{m}.stats.n;
        else
            N_m = n_total / M;
        end
        w_m = N_m / n_total;

        % Gating parameters (d+1) x Km
        if length(p_m.alpha0) == Km - 1
            alpha0_full = [p_m.alpha0(:)', 0];
            Alpha_full  = [p_m.Alpha, zeros(d, 1)];
        else
            alpha0_full = p_m.alpha0(:)';
            Alpha_full  = p_m.Alpha;
        end
        W_gate = [alpha0_full; Alpha_full];

        % Softmax probabilities on support X_val
        logits = X_val * W_gate;
        logits = logits - max(logits, [], 2);
        prob_m = exp(logits) ./ sum(exp(logits), 2);

        PI_hat    = [PI_hat, w_m * prob_m]; %#ok<AGROW>
        experts   = [experts,   [p_m.beta0; p_m.Beta]]; %#ok<AGROW>
        variances = [variances, max(p_m.sigma2, 1e-4)]; %#ok<AGROW>
        weights   = [weights,   (w_m / Km) * ones(1, Km)]; %#ok<AGROW>
    end

    weights = weights / sum(weights);
    L = length(variances);

    large_mixture.experts         = experts;
    large_mixture.variances       = variances;
    large_mixture.weights         = weights;
    large_mixture.PI_hat          = PI_hat;
    large_mixture.local_estimates = local_estimates;

    % 4. Multi-Start MM Algorithm for Heterogeneous Optimal Transport
    best_transportdis = inf;
    best_solution     = struct();
    best_history      = [];
    num_tries         = 2;
    if isfield(options, 'DME_tries'), num_tries = options.DME_tries; end

    use_gm_init = true;
    if isfield(options, 'init_mode') && strcmpi(options.init_mode, 'kmeans')
        use_gm_init = false;
    end

    dme_temp.local_estimates = local_estimates;
    dme_temp.local_times     = local_times;
    dme_temp.d               = d;
    dme_temp.X_val           = X_val;
    dme_temp.n               = n_total;

    for DME_try = 1:num_tries
        curr_try_history = [];
        
        % Initialization
        if DME_try == 1 && use_gm_init
            % Intelligent warm start via Greedy Merging:
            % Guarantees R_c(DME) <= R_c(GM) by MM descent property!
            try
                grd_init = Greedy_MixtureOfExperts_Hetero(dme_temp, K_target, options);
                reduced_mixture = grd_init.reduced_mixture;
            catch
                labels_ = kmeans(experts', K_target, 'Replicates', 3);
                reduced_mixture.experts   = zeros(d + 1, K_target);
                reduced_mixture.variances = zeros(1, K_target);
                for k = 1:K_target
                    reduced_mixture.experts(:, k) = mean(experts(:, labels_ == k), 2);
                    reduced_mixture.variances(k)  = mean(variances(labels_ == k));
                end
                reduced_mixture.weights = ones(1, K_target);
                reduced_mixture.gates   = zeros(d + 1, K_target);
            end
        else
            % Multi-replicate k-means on component parameters
            labels_ = kmeans(experts', K_target, 'Replicates', 5);
            reduced_mixture.experts   = zeros(d + 1, K_target);
            reduced_mixture.variances = zeros(1, K_target);
            for k = 1:K_target
                reduced_mixture.experts(:, k) = mean(experts(:, labels_ == k), 2);
                reduced_mixture.variances(k)  = mean(variances(labels_ == k));
            end
            reduced_mixture.weights = ones(1, K_target);
            reduced_mixture.gates   = zeros(d + 1, K_target);
        end

        % MM Optimization Loop
        converged = false;
        iteration = 0;
        prev_transportdis = -inf;
        max_iter = 100;
        tol = 1e-4;
        if isfield(options, 'DME_maxiter'), max_iter = options.DME_maxiter; end
        if isfield(options, 'DME_tol'), tol = options.DME_tol; end

        final_plan = [];

        while (~converged) && (iteration < max_iter)
            % Compute optimal transport plan using generalized dual-path
            [plan, distance_matrix] = argmin_transportation_plan(large_mixture, reduced_mixture, X_val);
            transportdis = sum(sum(sum(plan .* distance_matrix)));
            curr_try_history(end + 1) = transportdis; %#ok<AGROW>

            % Closed-form parameter update via weighted regression
            reduced_mixture = update_reduced_mixture(large_mixture, plan, X_val);

            converged = abs(transportdis - prev_transportdis) < tol;
            prev_transportdis = transportdis;
            iteration = iteration + 1;
            final_plan = plan;
        end

        % Check if best try
        if transportdis < best_transportdis
            best_transportdis = transportdis;
            best_mixture      = reduced_mixture;
            best_plan         = final_plan;
            best_history      = curr_try_history;
        end
    end

    % 5. Fit Global Gating Network via IRLS
    gatingProb = squeeze(sum(best_plan, 1))';
    gatingProb = gatingProb ./ sum(gatingProb, 2);
    gatingProb = max(gatingProb, eps);
    gatingProb = gatingProb ./ sum(gatingProb, 2);

    initial_gate = zeros(d + 1, K_target - 1);
    res = IRLS(X_val, gatingProb, initial_gate, ones(S, 1), 10000, 1e-8, 0);
    best_mixture.gates = [res.W, zeros(d + 1, 1)];

    % 6. Assemble output struct
    solution.param.alpha0 = best_mixture.gates(1, :);
    solution.param.Alpha  = best_mixture.gates(2:end, :);
    solution.param.beta0  = best_mixture.experts(1, :);
    solution.param.Beta   = best_mixture.experts(2:end, :);
    solution.param.sigma2 = best_mixture.variances;

    solution.gates     = best_mixture.gates;
    solution.experts   = best_mixture.experts;
    solution.variances = best_mixture.variances;
    solution.weights   = ones(1, K_target);

    solution.reduced_mixture = best_mixture;
    solution.best_dis        = best_transportdis;
    solution.transportdis    = best_transportdis;
    solution.R_c_history     = best_history;
    solution.large_mixture   = large_mixture;
    solution.plan            = best_plan;
    solution.local_estimates = local_estimates;
    solution.local_times     = local_times;
    solution.X_val           = X_val;
    solution.Y_val           = Y_val;
    solution.d               = d;
    solution.n               = n_total;

    reduction_time = toc(tic_agg);
    solution.learning_time = max(local_times) + reduction_time;

end


function reduced_mixture = update_reduced_mixture(large_mixture, plan, X_val)
% Closed-form MM parameter update for Gaussian experts with conditional KL ground cost
    [~, d_plus_1] = size(X_val);
    [L, K, S]     = size(plan);
    B = large_mixture.experts;       % (d+1) x L
    V = large_mixture.variances';    % L x 1

    reduced_mixture.experts   = zeros(d_plus_1, K);
    reduced_mixture.variances = zeros(1, K);
    reduced_mixture.gates     = zeros(d_plus_1, K);
    reduced_mixture.weights   = ones(1, K);

    % Precompute X_val * B once (S x L) outside the loop
    pred_B = X_val * B;

    for k = 1:K
        w  = squeeze(sum(plan(:, k, :), 1)); % (S x 1)
        Wk = squeeze(plan(:, k, :));         % (L x S)

        % Mathematically equivalent to X_val' * diag(w) * X_val (saves 32MB dense allocation)
        Xw   = X_val .* sqrt(max(w, 0));
        XtDX = Xw' * Xw + 1e-6 * eye(d_plus_1);

        % Numerically stable solve (mathematically identical to inv(...) * ...)
        rhs    = X_val' * sum(Wk' .* pred_B, 2);
        beta_k = XtDX \ rhs;

        pred_k    = X_val * beta_k;  % (S x 1)
        diff_pred = pred_k - pred_B; % (S x L)

        term1 = sum(V .* sum(Wk, 2));
        term2 = sum(sum(Wk' .* (diff_pred .^ 2)));

        tr_Dk   = max(sum(w), 1e-8);
        sigma_k = (term1 + term2) / tr_Dk;

        reduced_mixture.experts(:, k) = beta_k;
        reduced_mixture.variances(k)  = max(sigma_k, 1e-4);
    end
end
