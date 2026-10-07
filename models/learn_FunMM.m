function solution = learn_FunMM(X, y, K, options)


warning off

%----------------------------------------
% Get values of options

if isfield(options, 'nb_EM_runs')
     nb_EM_runs = options.nb_EM_runs; 
else nb_EM_runs = 10; 
end

if isfield(options, 'max_iter')
     max_iter = options.max_iter; 
else max_iter = 1000; 
end

if isfield(options, 'tol')
     tol = options.tol; 
else tol = 1e-5; 
end

if isfield(options, 'verbose')
     verbose = options.verbose; 
else verbose = 0; 
end

if isfield(options, 'algo_logreg')
     algo_logreg = options.algo_logreg; 
else algo_logreg = 'NR'; 
end

if isfield(options, 'initialize_strategy')
     initialize_strategy = options.initialize_strategy; 
else initialize_strategy = 'zeros'; 
end

if isfield(options, 'IRLS_max_iter')
     IRLS_max_iter = options.IRLS_max_iter; 
else IRLS_max_iter = 1000; 
end

if isfield(options, 'IRLS_threshold')
     IRLS_threshold = options.IRLS_threshold; 
else IRLS_threshold = 1e-5; 
end

if isfield(options, 'IRLS_verbose')
     IRLS_verbose = options.IRLS_verbose; 
else IRLS_verbose = 0; 
end
%----------------------------------------


[n, p] = size(X);
if size(y,1)~=n, y=y'; end


best_loglik    = -inf;
stored_cputime = [];
averaged_iter  = [];
EM_try = 1;

if verbose>=1
    fprintf('Model : FunMM   |   K=%i  \n', K);
end

%Standardize the designed matrices
[X, mu_X, sigma_X] = zscore(X);

    
while (EM_try <= nb_EM_runs)
    if (nb_EM_runs>1 && verbose>=1), fprintf('EM try %2i : ', EM_try); end
    time = cputime;
    
    %% ------------------------ Initialisation ----------------------------
    [weights, beta0, Beta, sigma2] = initialize_FunMM(X, y, K);
    iter = 0;
    converge = 0;
    prev_loglik=-inf;
    stored_loglik=[];
    
    
    %% ----------------------------- EM -----------------------------------
    
    while ~converge && (iter< max_iter)
        % ------------------------ E-Step ---------------------------------
        % Gating network conditional distribution
        log_Phi_xy = zeros(n,K);
        for k = 1:K
            log_Phi_y =  -0.5*log(2*pi) - 0.5*log(sigma2(k)) -0.5*((y - beta0(k)*ones(n,1) - X*Beta(:,k)).^2)/sigma2(k);
            log_Phi_xy(:,k) = log(weights(k)) + log_Phi_y;
        end
        
        
        log_sum_PhixPhiy = logsumexp(log_Phi_xy,2);
        log_Tau = log_Phi_xy - log_sum_PhixPhiy*ones(1,K);
        Tau = exp(log_Tau);
        Tau = Tau./(sum(Tau,2)*ones(1,K));
        
        %------------------------------------------------------------------
        % FunME log-likelihood
        loglik = sum(log_sum_PhixPhiy);
        if verbose >=2 ,fprintf(1, 'EM for FRM: iter : %d  | log-lik : %f \n',  iter, loglik); end

        % ------------------------ M-Step ---------------------------------
        % Update weights
        for k=1:K
           weights(k) = sum(Tau(:,k))/n; 
        end
        
        
        %------------------------------------------------------------------
        % Update experts Network
        for k=1:K
            beta0(k) = (Tau(:,k)' * (y - X*Beta(:,k)))/sum(Tau(:,k)); % update the intercept beta0
            % update Beta
            Xk = sqrt(Tau(:,k))' .* X'; 
            yk = sqrt(Tau(:,k)) .* (y - beta0(k));
            Beta(:,k) = Xk*Xk'\Xk*yk;
            % update sigma2
            sigma2(k) = (Tau(:,k)' * (y - beta0(k) - X*Beta(:,k)).^2) / sum(Tau(:,k));
        end
        
        % Convergence test
        converge = abs(loglik-prev_loglik) <= tol; % || abs((loglik-prev_loglik)/prev_loglik) <= tol;
        prev_loglik = loglik;
        iter=iter+1;
        stored_loglik = [stored_loglik, loglik];
        
        if iter == max_iter && verbose >=1, fprintf('reached max_iter      | '); end
        if (converge && verbose >= 1), fprintf('converged | %3i EM iters | ', iter);end
        if converge || iter == max_iter, averaged_iter = [averaged_iter iter]; end        
    end
    
    %----------------------------------------------------------------------
    EM_try = EM_try +1;
    stored_cputime = [stored_cputime cputime-time];
    [beta0, Beta, weights, sigma2] = identify_network_MM(beta0, Beta, weights, sigma2);
   
    % Results
    param.weights = weights;
    param.beta0 = beta0;
    param.Beta = Beta;
    param.sigma2 = sigma2;
    solution.param = param;
    
    % Parameter vector of the estimated model
    Psi = [weights(:); beta0(:); Beta(:); sigma2(:)];
    solution.stats.Psi = Psi;
    
    solution.stats.Tau = Tau;
    solution.stats.log_Phi_x = weights;
    solution.stats.log_Phi_y = log_Phi_y;
    solution.stats.log_alpha_Phi_xy=log_Phi_xy;
    solution.stats.ml = loglik;
    solution.stats.stored_loglik = stored_loglik;
    
    %% classsification pour EM : MAP(piik) (cas particulier ici to ensure a convex segmentation of the curve(s).
    [klas, Zik] = MAP(Tau);
    solution.stats.klas = klas;
    
    %----------------------------------------------------------------------
    % BIC AIC and ICL
    nf = length(Psi);
    df = length(nonzeros(Psi));
    
    solution.stats.df = nf;
    solution.stats.BIC = solution.stats.ml - (nf*log(n)/2);
    solution.stats.mBIC = solution.stats.ml - (df*log(n));
    solution.stats.AIC = solution.stats.ml - nf;
    
    %----------------------------------------------------------------------
    % CL: complete-data loglikelihood
    zik_log_alpha_Phi_xy = Zik.*log_Phi_xy;
    sum_zik_log_fik = sum(zik_log_alpha_Phi_xy,2);
    comp_loglik = sum(sum_zik_log_fik);
    solution.stats.CL = comp_loglik;
    solution.stats.ICL = solution.stats.CL - (nf*log(n)/2);
    
    %----------------------------------------------------------------------
    if (nb_EM_runs>1 && verbose)
        fprintf(1,'ml = %f \n',solution.stats.ml);
    end
    if loglik > best_loglik
        best_solution = solution;
        best_loglik = loglik;
    end
end
solution = best_solution;

if verbose >=1, fprintf('Average iteration : %i\n', round(mean(averaged_iter))); end
if (nb_EM_runs>1 && verbose >=2),   fprintf(1,'Best loglik :  %f\n',solution.stats.ml); end

solution.stats.cputime = mean(stored_cputime);
solution.stats.stored_cputime = stored_cputime;
solution.param.weights = weights;


%De-standardize: reconstruct the true param using the standardize-info
solution.param.Beta    = solution.param.Beta ./ sigma_X';
solution.param.beta0   = solution.param.beta0 - mu_X * solution.param.Beta;




end