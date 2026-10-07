function solution = Averaged_MixtureOfExperts(DME_solution, K, M, options)


    tic
    d = DME_solution.d;
    local_estimates = DME_solution.local_estimates;
    local_times     = DME_solution.local_times;
    
    %organize all experts, gates, covariances into array of cells
    %each cell is one component named comp, contains: comp.expert, comp.gate, comp.variance

    
    alpha0_ = [];
    Alpha_  = [];
    beta0_  = [];
    Beta_   = [];
    sigma2_ = [];
    for m=1:M
        alpha0_ = [alpha0_; local_estimates{m}.param.alpha0  ];
        Alpha_  = [Alpha_   local_estimates{m}.param.Alpha(:)];
        beta0_  = [beta0_;  local_estimates{m}.param.beta0   ];
        Beta_   = [Beta_    local_estimates{m}.param.Beta(:) ];
        sigma2_ = [sigma2_; local_estimates{m}.param.sigma2  ];
    end

    q_gate = size(local_estimates{1}.param.Alpha, 2);
    alpha0 = mean(alpha0_, 1);
    Alpha  = reshape(mean(Alpha_, 2), d, q_gate);
    beta0  = mean(beta0_, 1);
    Beta   = reshape(mean(Beta_, 2), d, K);
    sigma2 = mean(sigma2_, 1);
    
    if q_gate == K - 1
        alpha0_full = [alpha0, 0];
        Alpha_full  = [Alpha, zeros(d, 1)];
    else
        alpha0_full = alpha0;
        Alpha_full  = Alpha;
    end

    averaged_mixture.gates     = [alpha0_full; Alpha_full];
    averaged_mixture.experts   = [beta0; Beta];
    averaged_mixture.variances = sigma2;
    
    
    %-----------------------------
    solution.param.alpha0 = averaged_mixture.gates(1,:);
    solution.param.Alpha  = averaged_mixture.gates(2:end,:);
    solution.param.beta0  = averaged_mixture.experts(1,:);
    solution.param.Beta   = averaged_mixture.experts(2:end,:);
    solution.param.sigma2 = averaged_mixture.variances;
    
    %-----------------------------
    solution.gates     = averaged_mixture.gates;
    solution.experts   = averaged_mixture.experts;
    solution.variances = averaged_mixture.variances;
    solution.weights   = ones(1,K);
    solution.averaged_mixture = averaged_mixture;
    %-----------------------------


    find_averaged_time = toc;
    solution.learning_time = max(local_times) + find_averaged_time;

end %function


