function options = get_options(settings)

    if strcmp(settings, 'default')
        options.DME_tol     = 1e-6;
        options.DME_verbose = 1; % {0, 1, 2}
        options.DME_maxiter = 500;
        options.DME_tries   = 2;
        options.tol         = 1e-6;
        options.verbose     = 1; % {0, 1, 2}
        options.maxiter     = 4000;
        options.nb_EM_runs  = 5;
        options.IRLS_max_iter = 100;
        options.IRLS_threshold = 1e-8;
        options.initialize_strategy = 'LR';
        options.LASSO = false;
        options.S = 2000;           % Default support sample size S = |D_S|
        options.sample_size = 2000; % Backward compatibility alias for S
        options.parallel_machines = 'auto'; % {'auto', true, false}: parallelize local machines if no outer parfor

    else
        %
    end

end
