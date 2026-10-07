function solution = Distributed_MixtureOfExperts_Gaussian(X, Y, K, M, options)

    % Delegation hook: If K is non-scalar, delegate to dedicated heterogeneous aggregator
    if ~isscalar(K)
        solution = Distributed_MixtureOfExperts_Hetero(X, Y, K, M, options);
        return;
    end

    chi    = .1*ones(1,K-1);
    lambda = ones(1,K);
    
    IRLS_max_iter = 10000;
    IRLS_threshold = 1e-8;
    IRLS_verbose = 0;
    
    [n,d] = size(X);

    N = floor(n/M); %number of observations in each subdataset
    
    % Configure supporting sample X_val (covariates only - purely unlabeled D_S)
    if isfield(options, 'X_val') && ~isempty(options.X_val)
        X_val = options.X_val;
        S = size(X_val, 1);
        if size(X_val, 2) == d
            X_val = [ones(S, 1), X_val];
        end
        if isfield(options, 'Y_val') && ~isempty(options.Y_val)
            Y_val = options.Y_val;
        end
    else
        if isfield(options, 'S') && ~isempty(options.S)
            S = min(options.S, n);
        elseif isfield(options, 'sample_size') && ~isempty(options.sample_size)
            S = min(options.sample_size, n);
        elseif isfield(options, 'S_ratio') && ~isempty(options.S_ratio)
            S = min(max(round(options.S_ratio * N), 10), n);
        else
            S = N;
        end
        indices_val = randperm(n, S);
        X_val = [ones(S, 1), X(indices_val, :)];
        if exist('Y', 'var') && ~isempty(Y)
            Y_val = Y(indices_val);
        end
    end

    % Check if precomputed local estimates are provided (avoids redundant local EM runs)
    if isfield(options, 'local_estimates') && ~isempty(options.local_estimates)
        local_estimates = options.local_estimates;
        if isfield(options, 'local_times') && ~isempty(options.local_times)
            local_times = options.local_times;
        else
            local_times = zeros(1, M);
        end
        if options.DME_verbose >= 2
            fprintf('Reusing %d precomputed local estimates.\n', M);
        end
    else
        has_custom_indices = isfield(options, 'client_indices') && ~isempty(options.client_indices);
        if has_custom_indices
            client_indices_cell = options.client_indices;
        else
            client_indices_cell = cell(1, M);
            indices_shuffled = randperm(n); %shuffle the indices
            for m = 1:M
                client_indices_cell{m} = indices_shuffled( (m-1)*N+1:m*N );
            end
        end
        local_times      = zeros(1, M);
        local_estimates  = cell(1, M);

        % Pre-slice client datasets for parallel/serial execution
        client_X = cell(1, M);
        client_Y = cell(1, M);
        for m = 1:M
            idx_m = client_indices_cell{m};
            client_X{m} = X(idx_m, :);
            client_Y{m} = Y(idx_m);
        end

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

        % Subdataset are disjoint and of size N
        % For each subdataset, estimate the mixture
        if use_par_m
            parfor m = 1:M
                t_m = tic;
                if options.LASSO
                    local_estimates{m} = MixtureOfExperts_LASSO(client_X{m}, client_Y{m}, K, chi, lambda, options);
                else
                    local_estimates{m} = MixtureOfExperts(client_X{m}, client_Y{m}, K, options);
                end
                local_times(m) = toc(t_m);
            end
        else
            for m = 1:M
                t_m = tic;
                if options.LASSO
                    local_est = MixtureOfExperts_LASSO(client_X{m}, client_Y{m}, K, chi, lambda, options);
                else
                    local_est = MixtureOfExperts(client_X{m}, client_Y{m}, K, options);
                end
                local_estimates{m} = local_est; %store local estimate
                local_times(m)     = toc(t_m);       %store local time

                if options.DME_verbose, fprintf('Machine: %i  localtime: %.5fs \n', m, local_times(m)); end
            end
        end
        if options.DME_verbose, fprintf('Maximum localtime: %.5fs \n', max(local_times)); end
    end
    
   
    %We organize all experts, gates, covariances into arrays
    %   - experts: size (d+1)  -by- KM
    %   - gates  : size (d+1)K -by- KM
    %   - variances: size    1 -by- KM
    %Why the length of each gate is (d+1)K ? Because each gate is not
    %independent, it depends on the others in its same local mixture.

    experts   = [];
    gates     = [];
    variances = [];
    weights   = [];
    for m = 1:M

        experts   = [experts   [local_estimates{m}.param.beta0;  local_estimates{m}.param.Beta]];
        loc_alpha0 = local_estimates{m}.param.alpha0;
        loc_Alpha  = local_estimates{m}.param.Alpha;
        if length(loc_alpha0) == K - 1
            loc_alpha0 = [loc_alpha0, 0];
            loc_Alpha  = [loc_Alpha, zeros(d, 1)];
        end
        gate_     = [loc_alpha0; loc_Alpha];
        gates     = [gates     repmat(gate_(:),1,K)];
        variances = [variances local_estimates{m}.param.sigma2];
        weights   = [weights local_estimates{m}.stats.n/n*ones(1,K)];

    end

    %Now, we have a "large mixture" contains KM components
    large_mixture.experts   = experts;
    large_mixture.gates     = gates;
    large_mixture.variances = variances;
    large_mixture.weights   = weights;


    %Begin reduction algorithm
    %----------------------------------------------------------------------
    tic
    best_transportdis = inf;
    
    for DME_try = 1:options.DME_tries

        if options.DME_verbose, fprintf('Try: %i\n', DME_try);end

        
        %Initialize the reduced_mixture
        %This reduced_mixture is our final goal. It contains K components.
        %We initialize it by clustering the experts into K groups, then
        %taking the averagred. The gates and the variances are accordingly.
        % Initialize the reduced_mixture
        % Try 1: Intelligent initialization using Hungarian matching (Aligned MoE)
        % By MM descent property, DME is guaranteed to achieve R_c <= R_c(AAVR)
        if DME_try == 1
            try
                dme_temp.local_estimates = local_estimates;
                dme_temp.local_times     = local_times;
                dme_temp.d               = d;
                dme_temp.X_val           = X_val;
                ali_init = Aligned_MixtureOfExperts(dme_temp, K, M, options);
                reduced_mixture = ali_init.reduced_mixture;
            catch ME
                if options.DME_verbose, fprintf('AAVR warm-start failed (%s), falling back to kmeans\n', ME.message); end
                labels_    = kmeans(experts', K, 'Replicates', 3);
                experts_   = zeros(d+1,K);
                gates_     = zeros((d+1)*K,K);
                variances_ = zeros(1,K);
                for k=1:K
                    experts_(:,k)  = mean(experts(:,labels_==k),2);
                    gates_(:,k)    = mean(gates(:,labels_==k),2);
                    variances_(k)  = mean(variances(labels_==k));
                end
                [~, order] = sort(variances_);
                reduced_mixture.experts   = experts_(:,order);
                reduced_mixture.gates     = gates_(:,order);
                reduced_mixture.variances = variances_(order);
            end
        else
            % Subsequent tries: kmeans with multiple replicates for global exploration
            labels_    = kmeans(experts', K, 'Replicates', 5);
            experts_   = zeros(d+1,K);
            gates_     = zeros((d+1)*K,K);
            variances_ = zeros(1,K);
            for k=1:K
                experts_(:,k)  = mean(experts(:,labels_==k),2);
                gates_(:,k)    = mean(gates(:,labels_==k),2);
                variances_(k)  = mean(variances(labels_==k));
            end
            if length(unique(round(experts_(1,:), 4))) == K
                [~, order] = sort(experts_(1,:));
            else
                [~, order] = sort(variances_);
            end
            reduced_mixture.experts   = experts_(:,order);
            reduced_mixture.gates     = gates_(:,order);
            reduced_mixture.variances = variances_(order);
        end


        %Some variables for storing
        %-----------------------------
        stored_distances  = [];
        stored_loglik     = [];
        loglik            = [];
        prev_transportdis = -inf;
        converged         = 0;
        iteration         = 0;
        %-----------------------------


        
        %We keep looping until the transportation_distance between the 
        %large_mixture and the reduced_mixture is stable.
        %-----------------------------
        while (~converged) && (iteration < options.DME_maxiter)

            %The plan of transporting large_mixture to reduced_mixture
            [plan, distance_matrix] = argmin_transportation_plan(large_mixture, reduced_mixture, X_val);

            %Given the plan, compute the transportation_distance
            transportdis = sum(sum(sum(plan .* distance_matrix)));
            
            if options.DME_verbose, fprintf('Iteration %i: Transportation distance (obj. func.): %.5f\n', iteration, transportdis); end
            stored_distances = [stored_distances transportdis];
            
            %Find the optimal reduced_mixture given the plan
            reduced_mixture = argmin_mixture(large_mixture, plan, X_val);

            %Check for convergence
            converged = abs(transportdis - prev_transportdis) < options.DME_tol;
            
            prev_transportdis  = transportdis;
            iteration = iteration + 1;
            
        end

        
        
        
        % Now, transportation_distance is converged. We estimate the gates
        % using IRLS algorithm directly on marginal OT plan gatingProb (Unlabeled OT)
        % ----------------------------------------------------------------
        gatingProb = squeeze(sum(plan, 1))';
        gatingProb = gatingProb ./ sum(gatingProb, 2);
        gatingProb = max(gatingProb, eps);
        gatingProb = gatingProb ./ sum(gatingProb, 2);
        
        % Legacy posterior refinement (only if explicitly requested via options.use_tau_gates)
        if isfield(options, 'use_tau_gates') && options.use_tau_gates && exist('Y_val', 'var') && ~isempty(Y_val)
            log_Phi_y  = -0.5*log(2*pi) - 0.5*log(reduced_mixture.variances) - 0.5*((Y_val - X_val*reduced_mixture.experts).^2)./reduced_mixture.variances;
            log_Phi_xy = log(gatingProb) + log_Phi_y;
            log_sum_PhixPhiy = logsumexp(log_Phi_xy, 2);
            loglik           = sum(log_sum_PhixPhiy);
            log_Tau = log_Phi_xy - log_sum_PhixPhiy*ones(1, K);
            Tau     = exp(log_Tau);
            target_gates = Tau ./ (sum(Tau, 2)*ones(1, K));
        else
            % Default (Paper formulation): Pure unlabeled OT marginal plan
            target_gates = gatingProb;
        end
        
        initial_gate = reshape(mean(large_mixture.gates, 2), d+1, K);
        res = IRLS(X_val, target_gates, initial_gate(:, 1:end-1), ones(S, 1), 10000, 1e-8, 0);
        reduced_mixture.gates = [res.W, zeros(d+1, 1)];


        
        %Assign solution corresponding to the best transportation distance
        %-----------------------------
        if transportdis < best_transportdis
            
            best_transportdis = transportdis;
            solution.best_dis = best_transportdis;
            solution.plan     = plan;
            
            solution.param.alpha0 = reduced_mixture.gates(1,:);
            solution.param.Alpha  = reduced_mixture.gates(2:end,:);
            solution.param.beta0  = reduced_mixture.experts(1,:);
            solution.param.Beta   = reduced_mixture.experts(2:end,:);
            solution.param.sigma2 = reduced_mixture.variances;
            
            solution.gates        = reduced_mixture.gates;
            solution.experts      = reduced_mixture.experts;
            solution.variances    = reduced_mixture.variances;

            solution.reduced_mixture        = reduced_mixture; %Solution
            solution.stats.stored_loglik    = stored_loglik;
            solution.stats.stored_distances = stored_distances;
            solution.stats.loglik_on_supporting_data = loglik;

        end
        %-----------------------------


    end %DME_try


    reduction_time = toc;
    solution.local_times     = local_times;
    solution.reduction_time  = reduction_time;
    solution.learning_time   = max(local_times) + reduction_time;
    solution.large_mixture   = large_mixture;
    solution.local_estimates = local_estimates;
    solution.X_val = X_val;
    if exist('Y_val', 'var')
        solution.Y_val = Y_val;
    else
        solution.Y_val = [];
    end
    solution.n = n;
    solution.d = d;
    

end %function












function reduced_mixture = argmin_mixture(large_mixture, plan, X_val)

    [~,d] = size(X_val);
    [L,K,S] = size(plan);
    B = large_mixture.experts;
    V = large_mixture.variances';
    reduced_mixture.experts   = zeros(d,K);
    reduced_mixture.gates     = zeros(d,K);
    reduced_mixture.variances = zeros(1,K);

    % Precompute X_val * B once (S x L) outside the loop
    pred_B = X_val * B;

    for k = 1:K
        w  = squeeze(sum(plan(:, k, :), 1)); % (S x 1)
        Wk = squeeze(plan(:, k, :));         % (L x S)

        % Mathematically equivalent to X_val' * diag(w) * X_val (saves 32MB dense allocation)
        Xw   = X_val .* sqrt(max(w, 0));
        XtDX = Xw' * Xw;

        % Numerically stable solve (mathematically identical to inv(...) * ...)
        rhs = X_val' * sum(Wk' .* pred_B, 2);

        if rcond(XtDX) < 1e-12
            beta_k = (XtDX + 1e-6 * eye(d)) \ rhs;
        else
            beta_k = XtDX \ rhs;
        end

        pred_k    = X_val * beta_k;  % (S x 1)
        diff_pred = pred_k - pred_B; % (S x L)

        term1 = sum(V .* sum(Wk, 2));
        term2 = sum(sum(Wk' .* (diff_pred .^ 2)));

        tr_Dk   = max(sum(w), 1e-12);
        sigma_k = (term1 + term2) / tr_Dk;

        reduced_mixture.experts(:, k)   = beta_k;
        reduced_mixture.variances(:, k) = max(sigma_k, 1e-4);
    end
    reduced_mixture.weights = ones(1, K);

end























