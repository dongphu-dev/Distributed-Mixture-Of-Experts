function solution = Median_MixtureOfExperts(DME_solution, K, M, options)

    tic
    X_val = DME_solution.X_val;
    local_estimates = DME_solution.local_estimates;
    local_times     = DME_solution.local_times;
    
    %organize all experts, gates, covariances into array of cells
    %each cell is one component named comp, contains: comp.expert, comp.gate, comp.variance

    
    prev_dis = inf;
    best_m = 1;
    for m=1:M
        median_mixture.experts   = [local_estimates{m}.param.beta0;  local_estimates{m}.param.Beta];
        gate_                    = [local_estimates{m}.param.alpha0;  local_estimates{m}.param.Alpha];
        median_mixture.gates     = repmat(gate_(:),1,K);
        median_mixture.variances = local_estimates{m}.param.sigma2;
        
        dis = 0;
        for i=1:M
            current_mixture.experts   = [local_estimates{i}.param.beta0;  local_estimates{i}.param.Beta];
            current_gate_             = [local_estimates{i}.param.alpha0;  local_estimates{i}.param.Alpha];
            current_mixture.gates     = repmat(current_gate_(:),1,K);
            current_mixture.variances = local_estimates{i}.param.sigma2;
            current_mixture.weights   = ones(1,K);
            [plan, distance_matrix]   = argmin_transportation_plan(current_mixture, median_mixture, X_val);
            dis = dis + sum(sum(sum(plan .* distance_matrix)));
        end
        
        
        if dis < prev_dis
            solution = median_mixture;
            prev_dis = dis;
            best_m = m;
        end
        
    end
    


    median_mixture.gates = [local_estimates{best_m}.param.alpha0;  local_estimates{best_m}.param.Alpha];

    %the solution to be returned
    %-----------------------------

    best_dis = dis;
    solution.best_dis = best_dis;
    
    
    [~, order] = sort(median_mixture.experts(1,:));
    median_mixture.experts = median_mixture.experts(:,order);
    median_mixture.variances = median_mixture.variances(order);
    
    %-----------------------------
    solution.param.alpha0 = median_mixture.gates(1,:);
    solution.param.Alpha  = median_mixture.gates(2:end,:);
    solution.param.beta0  = median_mixture.experts(1,:);
    solution.param.Beta   = median_mixture.experts(2:end,:);
    solution.param.sigma2 = median_mixture.variances;
    
    %-----------------------------
    solution.gates     = median_mixture.gates;
    solution.experts   = median_mixture.experts;
    solution.variances = median_mixture.variances;
    solution.weights   = ones(1,K);
    solution.median_mixture   = median_mixture;
    %-----------------------------



    find_median_time = toc;
    solution.learning_time = max(local_times) + find_median_time;


end %function















