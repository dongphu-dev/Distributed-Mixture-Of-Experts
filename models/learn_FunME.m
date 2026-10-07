function solution = learn_FunME(Xa, Xb, y, K, options)

%--------------------------------------------------------------------------
% Fits a functional softmax-gated mixture-of-experts by the EM algorithm.
%
% by Faicel Chamroukhi, Thien N. Pham
%--------------------------------------------------------------------------

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

if isfield(options, 'linesearch')
     linesearch = options.linesearch; 
else linesearch = 1; 
end
%----------------------------------------

[~, p] = size(Xb);
[n, q] = size(Xa);
if size(y,1)~=n, y=y'; end

best_loglik    = -inf;
stored_cputime = [];
averaged_iter  = [];
EM_try = 1;

if verbose>=1
    fprintf('Model : FunME   |   K=%i  \n', K);
end

%Standardize the designed matrices
[Xa, mu_Xa, sigma_Xa] = zscore(Xa);
[Xb, mu_Xb, sigma_Xb] = zscore(Xb);


%==========================================================================

while (EM_try <= nb_EM_runs)
    if (nb_EM_runs>1 && verbose>=1), fprintf('EM try %2i : ', EM_try); end
    time = cputime;
    
    %% ------------------------ Initialisation ----------------------------
    [alpha0, Alpha, beta0, Beta, sigma2] = initialize_FunME(Xa, Xb, y, K, initialize_strategy);
    iter = 0;
    converge = 0;
    prev_loglik=-inf;
    stored_loglik=[];
    
    
    %% ----------------------------- EM -----------------------------------
    % Gating network conditional distribution
    gatingProb = multinomial_logistic([alpha0 0; Alpha zeros(q,1)], [ones(n,1) Xa]);
    
    while ~converge && (iter< max_iter)
        % ------------------------ E-Step ---------------------------------

        if (~linesearch) || (linesearch && iter == 0)
            % Only need for non-linesearch OR first step if linesearch is used
            %Compute log likelihood
            log_Phi_y  = -0.5*log(2*pi) - 0.5*log(sigma2) -0.5*((y - beta0 - Xb*Beta).^2)./sigma2;
            log_Phi_xy = log(gatingProb) + log_Phi_y;
            log_sum_PhixPhiy = logsumexp(log_Phi_xy,2);
            loglik = sum(log_sum_PhixPhiy);
        end
        
        %------------------------------------------------------------------
        %Compute posterior porobability
        log_Tau = log_Phi_xy - log_sum_PhixPhiy*ones(1,K);
        Tau = exp(log_Tau);
        Tau = Tau./(sum(Tau,2)*ones(1,K));
        
        %------------------------------------------------------------------
        if verbose >=2 ,fprintf(1, 'EM for FunME: iter : %d  | log-lik : %f \n',  iter, loglik); end
        if isnan(loglik)
            if verbose>=2, fprintf('loglik NaN. Re-initialize the networks \n');end
            [alpha0, Alpha, beta0, Beta, sigma2] = initialize_FunME(Xa, Xb, y, K, initialize_strategy);
            iter = 0;
            converge = 0;
            prev_loglik=-inf;
            stored_loglik=[];
            gatingProb = multinomial_logistic([alpha0 0; Alpha zeros(q,1)], [ones(n,1) Xa]);
            continue
        end
        % ------------------------ M-Step ---------------------------------
        % 1. UPDATE GATING NETWORK
        if strcmp(algo_logreg, 'CA')% Coordinate ascent
            [alpha0, Alpha, ~, gatingProb] = CoordAscent_multlogreg(Xa, Tau, alpha0, Alpha, 0); % replace 0 by to print the objective
        else
            % or use Newton-Raphson
            res = IRLS([ones(n,1) Xa], Tau, [alpha0; Alpha], ones(n,1), IRLS_max_iter, IRLS_threshold, IRLS_verbose);
            gatingProb = res.piik;
            alpha0 = res.W(1,:);
            Alpha = res.W(2:end,:);
        end
        %Prevent degeneration in any column
        sm = (gatingProb < eps);
        gatingProb = gatingProb + eps*sm;
        
        
        %------------------------------------------------------------------
        % 2. UPDATE EXPERT NETWORK
        
        %Store old values
        beta0_old  = beta0;
        Beta_old   = Beta;
        sigma2_old = sigma2;
        
        %Update
        for k=1:K
            % update the intercept beta0
            beta0(k) = (Tau(:,k)' * (y - Xb*Beta(:,k)))/sum(Tau(:,k)); 
            % update Beta
            Xk = sqrt(Tau(:,k))' .* Xb'; 
            yk = sqrt(Tau(:,k)) .* (y - beta0(k));
            Beta(:,k) = inv(Xk*Xk')*Xk*yk;
            % update sigma2
            sigma2(k) = (Tau(:,k)' * (y - beta0(k) - Xb*Beta(:,k)).^2) / sum(Tau(:,k));
        end
        
        %------------------------------------------------------------------
        if linesearch
            %The ascent dicrection to move forward
            beta0_direction  = beta0  - beta0_old;
            Beta_direction   = Beta   - Beta_old;
            sigma2_direction = sigma2 - sigma2_old;
            
            stepsize = 1;
            while 1
                
                %Move forward with current value of stepsize
                beta0  = beta0_old + stepsize * beta0_direction;
                Beta   = Beta_old  + stepsize * Beta_direction;
                sigma2 = sigma2_old + stepsize * sigma2_direction;
                stepsize = 0.5*stepsize;
                
                %Compute lasso-penalized log likelihood again
                log_Phi_y    = -0.5*log(2*pi) - 0.5*log(sigma2) -0.5*((y - beta0 - Xb*Beta).^2)./sigma2;
                log_Phi_xy = log(gatingProb) + log_Phi_y;
                log_sum_PhixPhiy = logsumexp(log_Phi_xy,2);
                loglik = sum(log_sum_PhixPhiy);
                
                %Stopping condition
                if loglik > (1+tol)*prev_loglik  ||  stepsize == 0
                    break;
                end
            end
        end
        
        
        %------------------------------------------------------------------
        % Convergence test
        converge    = abs(loglik-prev_loglik) <= tol || abs((loglik-prev_loglik)/prev_loglik) <= tol;
        prev_loglik = loglik;
        iter = iter+1;
        stored_loglik = [stored_loglik, loglik];
        
        if iter == max_iter && verbose >= 1, fprintf('reached max_iter      | '); end
        if converge && verbose >= 1, fprintf('converged | %3i EM iters | ', iter);end
        if converge || iter == max_iter, averaged_iter = [averaged_iter iter]; end

    end
    
    %----------------------------------------------------------------------
    EM_try = EM_try +1;
    stored_cputime = [stored_cputime cputime-time];
    
   
    % Results
    param.alpha0 = alpha0;
    param.Alpha  = Alpha; 
    param.beta0  = beta0;
    param.Beta   = Beta;
    param.sigma2 = sigma2;
    solution.param = param;
    
    % Parameter vector of the estimated model
    Psi = [alpha0(:); Alpha(:); beta0(:); Beta(:); sigma2(:)];
    solution.stats.Psi = Psi;
    
    %----------------------------------------------------------------------
    % Model stats
    solution.stats.Tau = Tau;
    solution.stats.ml = loglik;
    solution.stats.stored_loglik = stored_loglik;
    
    %% classsification pour EM : MAP(piik) (cas particulier ici to ensure a convex segmentation of the curve(s).
    [klas, Zik] = MAP(Tau);
    solution.stats.klas = klas;

    %----------------------------------------------------------------------
    % BIC AIC and ICL
    nf = length(Psi);
    df = length(nonzeros(Psi));
    solution.stats.nf = nf;
    solution.stats.df = df;
    solution.stats.BIC  = solution.stats.ml - nf*log(n)/2;
    solution.stats.mBIC = solution.stats.ml - df*log(n)/2; %modified BIC
    solution.stats.AIC  = solution.stats.ml - nf;
    
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

%De-standardize: reconstruct the true param using the standardize-info
solution.param.Alpha   = solution.param.Alpha ./ sigma_Xa';
solution.param.alpha0  = solution.param.alpha0 - mu_Xa * solution.param.Alpha;
solution.param.Beta    = solution.param.Beta ./ sigma_Xb';
solution.param.beta0   = solution.param.beta0 - mu_Xb * solution.param.Beta;
[solution.param.beta0, solution.param.Beta, solution.param.alpha0, solution.param.Alpha, solution.param.sigma2] = identify_network(solution.param.beta0, solution.param.Beta, solution.param.alpha0, solution.param.Alpha, solution.param.sigma2);
solution.param.alpha0 = [solution.param.alpha0 0];
solution.param.Alpha  = [solution.param.Alpha zeros(q,1)];

%==========================================================================


end