function solution = Global_MixtureOfExperts(X, Y, K, options)

    tic
    if options.LASSO
        chi    = ones(1,K-1);
        lambda = ones(1,K);
        fit = MixtureOfExperts_LASSO(X, Y, K, chi, lambda, options);
    else
        fit = MixtureOfExperts(X, Y, K, options);
    end

    solution.param = fit.param;
    solution.stats = fit.stats;
    solution.experts = [fit.param.beta0; fit.param.Beta];
    solution.gates   = [fit.param.alpha0; fit.param.Alpha];
    solution.variances = fit.param.sigma2;
    solution.weights   = ones(1,K);
    solution.learning_time = toc;
    
end