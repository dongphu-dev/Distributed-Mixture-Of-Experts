function solution = Distributed_MixtureOfExperts(X, Y, K, M, options)

    chi    = .1*ones(1,K-1);
    lambda = ones(1,K);

    [n,d] = size(X);

    N = floor(n/M); %number of observations in each subdataset
    
    %Take a sample of size N (randomly) as validation data
    indices_val = randperm(n,N);
    X_val = [ones(N,1) X(indices_val,:)];
    Y_val = Y(indices_val);

    indices_shuffled = randperm(n); %shuffle the indices
    local_times      = zeros(1,M);
    local_estimates  = cell(1,M);

    
    %Subdataset are disjoint and of size N
    %For each subdataset, estimate the mixture
    for m =1:M
        
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

    for m = 1:M

        experts   = [experts   [local_estimates{m}.param.beta0;  local_estimates{m}.param.Beta]];
        gate_     = [local_estimates{m}.param.alpha0; local_estimates{m}.param.Alpha];
        gates     = [gates     repmat(gate_(:),1,K)];
        variances = [variances local_estimates{m}.param.sigma2];

    end

    %Now, we have a "large mixture" contains KM components
    large_mixture.experts   = experts;
    large_mixture.gates     = gates;
    large_mixture.variances = variances;


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
%         experts_   = experts(:,1:K);
%         gates_     = gates_(:,1:K);
%         variances_ = variances_(:,1:K);
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
        %large_mixture and the reduced_mixture is converged.
        %-----------------------------
        while (~converged) && (iteration < options.DME_maxiter)

            %The plan of transporting large_mixture to reduced_mixture
            plan = argmin_transportation_plan(large_mixture, reduced_mixture, X_val);

            %Given the plan, compute the transportation_distance
            transportdis = transportation_distance(large_mixture, reduced_mixture, plan, X_val);
            
            if options.DME_verbose, fprintf('Mixture distance: %.5f\n', transportdis); end
            stored_distances = [stored_distances transportdis];
            
            %Find the optimal reduced_mixture given the plan
            reduced_mixture = argmin_mixture(large_mixture, plan, X_val);

            %Check for convergence
            converged = abs(transportdis - prev_transportdis) < options.DME_tol;
            
            prev_transportdis  = transportdis;
            iteration = iteration + 1;
            
        end

        
        %Now, transportation_distance is converged. We aggregate the gates
        %belong to a same group.
        for k=1:K
            gate_k = [];
            for l=1:K*M
                if length(plan{l,k}) > 1
                    gate_k = [gate_k plan{l,k}];
                end
            end
            reduced_mixture.gates(:,k) = sum(gate_k,2)/M;

        end
        %-----------------------------


        %Assign solution corresponding to the best transportation distance
        %-----------------------------
        if transportdis < best_transportdis
            
            best_transportdis = transportdis;
            solution.best_dis = best_transportdis;

            solution.param.beta0  = reduced_mixture.experts(1,:);
            solution.param.Beta   = reduced_mixture.experts(2:end,:);
            Alpha_                = reshape(reduced_mixture.gates(:,1),d+1,K);
            solution.param.alpha0 = Alpha_(1,:);
            solution.param.Alpha  = Alpha_(2:end,:);
            solution.param.sigma2 = reduced_mixture.variances;
            
            solution.gates        = Alpha_;
            solution.experts      = reduced_mixture.experts;
            solution.variances    = reduced_mixture.variances;

            solution.reduced_mixture  = reduced_mixture;
            solution.stored_loglik    = stored_loglik;
            solution.stored_distances = stored_distances;

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
    

end %function












function reduced_mixture = argmin_mixture(large_mixture, plan, X_val)

    [n,d] = size(X_val);
    [L,K] = size(plan);
    experts   = large_mixture.experts;
    variances = large_mixture.variances;
    reduced_mixture.experts = zeros(d,K);
    reduced_mixture.gates   = zeros(d*K,K);
    reduced_mixture.variances = zeros(1,K);
    
    for k = 1:K
        
        weights = zeros(n,L);
        for l = 1:L
            if length(plan{l,k}) > 1
                expXA  = exp(X_val*reshape(plan{l,k}, d, K));
                temp   = expXA ./ sum(expXA,2);
                weights(:,l) = temp(:,k);
            end
                 
        end
        normalized_weights = weights./sum(weights,2);
        
        all_weighted_mean = [];
        for l = 1:L
            weighted_X_val  = X_val .* normalized_weights(:,l);
            weighted_mean   = weighted_X_val*experts(:,l);
            all_weighted_mean   = [all_weighted_mean weighted_mean];
        end
        
        %update expert
        Beta_k = inv(X_val' * X_val) * X_val' * all_weighted_mean;
        Beta_k = sum(Beta_k,2);
        
        
        
        %update variance
        weighted_sigma2                = normalized_weights .* variances;
        weighted_sigma2_across_experts = normalized_weights .* ( X_val*(experts-Beta_k) ).^2;
        variance_k = sum(sum(weighted_sigma2 + weighted_sigma2_across_experts))/n;
        
        
        
        reduced_mixture.experts(:,k) = Beta_k;
        reduced_mixture.variances(:,k) = variance_k;
    end

end

























