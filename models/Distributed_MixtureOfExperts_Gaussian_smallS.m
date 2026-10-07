function solution = Distributed_MixtureOfExperts_Gaussian_smallS(X, Y, K, M, options)

    chi    = .1*ones(1,K-1);
    lambda = ones(1,K);
    
    IRLS_max_iter = 10000;
    IRLS_threshold = 1e-8;
    IRLS_verbose = 0;
    
    [n,d] = size(X);

    N = floor(n/M); %number of observations in each subdataset
    
    %Take a sample of size N (randomly) as validation data
    S = floor(N/2);
    indices_val = randperm(n,S);
    X_val = [ones(S,1) X(indices_val,:)];
    Y_val = Y(indices_val);

    indices_shuffled = randperm(n); %shuffle the indices
    local_times      = zeros(1,M);
    local_estimates  = cell(1,M);

    
    %Subdataset are disjoint and of size N
    %For each subdataset, estimate the mixture
    parfor m =1:M
        
        tic
        indices_m = indices_shuffled( (m-1)*N+1:m*N );
        X_local   = X(indices_m,:);
        Y_local   = Y(indices_m);

        if options.LASSO
            local_est = MixtureOfExperts_LASSO(X_local, Y_local, K, chi, lambda, options);
        else
            local_est = MixtureOfExperts(X_local, Y_local, K, options);
        end

        local_estimates{m} = local_est; %store local estimate
        local_times(m)     = toc;       %store local time
        
        if options.DME_verbose, fprintf('Machine: %i  localtime: %.5fs \n', m, local_times(m)); end
    end
    if options.DME_verbose, fprintf('Maximum localtime: %.5fs \n', max(local_times)); end
    
   
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
        gate_     = [local_estimates{m}.param.alpha0; local_estimates{m}.param.Alpha];
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
%     keyboard;
    for DME_try = 1:options.DME_tries

        if options.DME_verbose, fprintf('Try: %i\n', DME_try);end

        
        %Initialize the reduced_mixture
        %This reduced_mixture is our final goal. It contains K components.
        %We initialize it by clustering the experts into K groups, then
        %taking the averagred. The gates and the variances are accordingly.
        %------------------------------
        labels_    = kmeans(experts', K);
        experts_   = zeros(d+1,K);
        gates_     = zeros((d+1)*K,K);
        variances_ = zeros(1,K);
        for k=1:K
            experts_(:,k)  = mean(experts(:,labels_==k),2);
            gates_(:,k)    = mean(gates(:,labels_==k),2);
            variances_(k)  = mean(variances(labels_==k));
        end

        [~, order] = sort(experts_(1,:));
        reduced_mixture.experts   = experts_(:,order);
        reduced_mixture.gates     = gates_(:,order);
        reduced_mixture.variances = variances_(order);
        %------------------------------


        %Some variables for storing
        %-----------------------------
        stored_distances  = [];
        stored_loglik     = [];
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

        
        
        
        %Now, transportation_distance is converged. We estimate the gates
        %using IRLS algorithm
        %----------------------------------------------------------------

        gatingProb = squeeze(sum(plan,1))';
        gatingProb = gatingProb./sum(gatingProb,2);
        log_Phi_y  = -0.5*log(2*pi) - 0.5*log(reduced_mixture.variances) -0.5*((Y_val - X_val*reduced_mixture.experts).^2)./reduced_mixture.variances;
        log_Phi_xy = log(gatingProb) + log_Phi_y;
        log_sum_PhixPhiy = logsumexp(log_Phi_xy,2);
        loglik           = sum(log_sum_PhixPhiy);
        
        
        %Compute posterior porobability
        log_Tau = log_Phi_xy - log_sum_PhixPhiy*ones(1,K);
        Tau     = exp(log_Tau);
        Tau     = Tau./(sum(Tau,2)*ones(1,K));
        
        %initial_gate = gate_; %i.e. gate of the last machine
        initial_gate = reshape(mean(large_mixture.gates,2), d+1, K);
        
        res = IRLS(X_val, Tau, initial_gate(:,1:end-1), ones(S,1), 5000, 1e-8, 0);
        reduced_mixture.gates = [res.W zeros(d+1,1)];


        
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
    solution.Y_val = Y_val;
    solution.n = n;
    solution.d = d;
    

end %function












function reduced_mixture = argmin_mixture(large_mixture, plan, X_val)

    [~,d] = size(X_val);
    [L,K,S] = size(plan);
    B = large_mixture.experts;
    V = large_mixture.variances';
    reduced_mixture.experts = zeros(d,K);
    reduced_mixture.gates   = zeros(d,K);
    reduced_mixture.variances = zeros(1,K);
    %idx = repmat([1:K],L/K);
    for k = 1:K
        Dk_temp = sparse(squeeze(sum(plan(:,k,:))));
        Dk = diag(Dk_temp);
        Wk = sparse(squeeze(plan(:,k,:)));
        WkB = Wk'*B';

        beta_k  = inv(X_val'*Dk*X_val)*X_val'* (X_val .* WkB)*ones(d,1);
        sigma_k = (V' * Wk * ones(S,1) + ones(1,L)*(Wk .* ((beta_k - B)' * X_val')) * X_val * (beta_k - B)*ones(L,1)) / trace(Dk);
        
        reduced_mixture.experts(:,k)   = beta_k;
        reduced_mixture.variances(:,k) = sigma_k;
        
    end
    reduced_mixture.weights = ones(1,K);

end























