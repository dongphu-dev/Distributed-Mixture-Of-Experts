function solution = MixtureOfExperts_LASSO(X, y, K, chi, lambda, options)

%--------------------------------------------------------------------------
% Fits a high-dimensional functional softmax-gated mixture-of-experts by an EM-Lasso algorithm. The
% algorithm maximizes an l1-regularized log-likelihood where the regularized parameters are the
% regression coefficients for the Gaussian regressor expert network, and the softmax regression
% coefficients, for the gating network

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
else initialize_strategy = 'LR'; 
end

%max_iter for weighted_LASSO algorithms
if isfield(options, 'wL_max_iter')
     wL_max_iter = options.wL_max_iter; 
else wL_max_iter = 5000; 
end

%threshold for weighted_LASSO algorithms
if isfield(options, 'wL_threshold') 
     wL_threshold = options.wL_threshold; 
else wL_threshold = 1e-7; 
end

if isfield(options, 'wL_verbose') 
     wL_verbose = options.wL_verbose; 
else wL_verbose = 0; 
end

if isfield(options, 'linesearch')
     linesearch = options.linesearch; 
else linesearch = 1; 
end
%----------------------------------------


[n, p] = size(X);

if size(y,1)~=n, y=y'; end

best_loglik = -inf;
stored_cputime = [];
averaged_iter = [];
EM_try = 1;

if verbose>=1
    fprintf('Model : FunME_Lasso\n');
end

%Standardize the designed matrices
[X, mu_X, sigma_X] = zscore(X);

%==========================================================================

while (EM_try <= nb_EM_runs)

    if (nb_EM_runs>1 && verbose>=1), fprintf('EM-Lasso try %2i : ', EM_try); end
    time = cputime;
    
    %% ------------------------ Initialisation ----------------------------
    [alpha0, Alpha, beta0, Beta, sigma2] = initialize_MoE_LASSO(X, y, K, initialize_strategy);
    %
    iter = 0;
    converge = 0;
    prev_loglik=-inf;
    stored_loglik=[];
    
    %% ----------------------------- EM -----------------------------------
    % Gating network conditional distribution
    gatingProb = multinomial_logistic([alpha0 0; Alpha zeros(p,1)], [ones(n,1) X]);
    
    while ~converge && (iter< max_iter)
        % ------------------------ E-Step ---------------------------------

        if (~linesearch) || (linesearch && iter == 0)
            % Only need for non-linesearch OR first step if linesearch is used
            % Compute log likelihood
            log_Phi_y  = -0.5*log(2*pi) - 0.5*log(sigma2) -0.5*((y - beta0 - X*Beta).^2)./sigma2;
            log_Phi_xy = log(gatingProb) + log_Phi_y;
            log_sum_PhixPhiy = logsumexp(log_Phi_xy,2);
            %Compute lasso-penalized log likelihood
            Pen = sum(sum(chi.*abs(Alpha))) + sum(sum(lambda.*abs(Beta)));
            loglik = sum(log_sum_PhixPhiy) - Pen;
        end
        
        %------------------------------------------------------------------
        %Compute posterior porobability
        log_Tau = log_Phi_xy - log_sum_PhixPhiy*ones(1,K);
        Tau = exp(log_Tau);
        Tau = Tau./(sum(Tau,2)*ones(1,K));
        
        
        if (verbose>=2),fprintf(1, 'EM-Lasso FunME: iter : %3d  | lasso-pen log-lik : %f \n',  iter, loglik); end
        if (loglik<prev_loglik && verbose>=2),fprintf(1, '\nEM-Lasso  log-lik decreasing !'); end

        %------------------------------------------------------------------
        if isnan(loglik)
            if verbose >=2, fprintf('loglik NaN. Re-initialize the networks \n');end
            [alpha0, Alpha, beta0, Beta, sigma2] = initialize_MoE_LASSO(X, y, K, initialize_strategy);
            iter = 0;
            converge = 0;
            prev_loglik=-inf;
            stored_loglik=[];
            gatingProb = multinomial_logistic([alpha0 0; Alpha zeros(p,1)], [ones(n,1) X]);
            continue
        end
        
        % ------------------------ M-Step ---------------------------------
        % 1. UPDATE GATING NETWORK
        [alpha0, Alpha, ~, gatingProb] = LASSO_multlogreg(X, Tau, alpha0, Alpha, chi, wL_max_iter, wL_threshold, 0); %replace 0 by one to print the objective
        
        %Prevent degeneration in any column
        sm = (gatingProb == 0);
        gatingProb = gatingProb + eps*sm;
        
        
        %------------------------------------------------------------------
        % 2. UPDATE EXPERT NETWORK
        
        %Store old values
        beta0_old  = beta0;
        Beta_old   = Beta;
        sigma2_old = sigma2;
        
        for k = 1:K
            Tauk = Tau(:,k);     % weights
            [beta0k, betak, sigma2k] = weighted_LASSO_greg(X, y, lambda(k), Tauk, beta0(k), Beta(:,k), sigma2(k), wL_max_iter, wL_threshold, 0); %replace 0 by one to print the objective
            beta0(k) = beta0k;   % intercept update
            Beta(:,k) = betak;   % regression coefficients vector update
            sigma2(k) = sigma2k; % regressor variance update
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
                log_Phi_y    = -0.5*log(2*pi) - 0.5*log(sigma2) -0.5*((y - beta0 - X*Beta).^2)./sigma2;
                log_Phi_xy = log(gatingProb) + log_Phi_y;
                log_sum_PhixPhiy = logsumexp(log_Phi_xy,2);
                Pen = sum(sum(chi.*abs(Alpha))) + sum(sum(lambda.*abs(Beta)));
                loglik = sum(log_sum_PhixPhiy) - Pen;
                
                %Stopping condition
                if loglik > prev_loglik
                    break;
                elseif stepsize < eps
                    fprintf("Iter %i: Too small stepszie, break\n", iter);
                    break;
                end
            end
        end

        %------------------------------------------------------------------
        % convergence test
        converge = abs(loglik-prev_loglik) <= tol; % || abs((loglik-prev_loglik)/prev_loglik) <= tol;
        prev_loglik = loglik;
        iter=iter+1;
        stored_loglik = [stored_loglik, loglik];
        
        if iter == max_iter, fprintf('reached max_iter      | '); end
        if (converge && verbose >= 1), fprintf('converged | %3i EM iters | ', iter);end
        if converge || iter == max_iter, averaged_iter = [averaged_iter iter]; end
        
    end %while
    [beta0, Beta, alpha0, Alpha, sigma2] = identify_network(beta0, Beta, alpha0, Alpha, sigma2);
    %----------------------------------------------------------------------
    EM_try = EM_try +1;
    stored_cputime = [stored_cputime cputime-time];
        
    % results
    param.alpha0 = alpha0;
    param.Alpha = Alpha;
    
    param.beta0 = beta0;
    param.Beta = Beta;
    param.sigma2 = sigma2;
    solution.param = param;
    
    % parameter vector of the estimated model
    Psi = [alpha0(:); Alpha(:); beta0(:); Beta(:); sigma2(:)];
    solution.stats.Psi = Psi;
    solution.chi = chi;
    solution.lambda = lambda;
    
    %----------------------------------------------------------------------
    %model stats
%     solution.stats.Tau = Tau;
%     solution.stats.gatingProb = gatingProb;
%     solution.stats.log_Phi_y = log_Phi_y;
%     solution.stats.log_alpha_Phi_xy=log_Phi_xy;
    solution.stats.ml = loglik;
    solution.stats.stored_loglik = stored_loglik;

    %% classsification pour EM : MAP(piik) (cas particulier ici to ensure a convex segmentation of the curve(s).
%     [klas, Zik] = MAP(Tau);
%     solution.stats.klas = klas;

    %----------------------------------------------------------------------
    % BIC AIC and ICL
    nf = length(Psi);
    if all((lambda==0)) && all((chi == 0))
        df = length(Psi);
    elseif all((lambda==0))
        df = length([alpha0(:); Alpha(:); beta0(:); nonzeros(Beta); sigma2(:)]);
    elseif all((chi == 0))
        df = length([alpha0(:); nonzeros(Alpha); beta0(:); Beta(:); sigma2(:)]);
    else
        df = length([alpha0(:); nonzeros(Alpha); beta0(:); nonzeros(Beta); sigma2(:)]);
    end
    solution.stats.nf = nf;
    solution.stats.df = df;
    solution.stats.BIC  = solution.stats.ml - nf*log(n)/2;
    solution.stats.mBIC = solution.stats.ml - df*log(n)/2; %modified BIC
    solution.stats.AIC  = solution.stats.ml - nf;
    
    %----------------------------------------------------------------------
    % CL: complete-data loglikelihood
%     zik_log_alpha_Phi_xy = Zik.*log_Phi_xy;
%     sum_zik_log_fik = sum(zik_log_alpha_Phi_xy,2);
%     comp_loglik = sum(sum_zik_log_fik);
%     solution.stats.CL = comp_loglik;
%     solution.stats.ICL = solution.stats.CL - (nf*log(n)/2);
    %%
    if (nb_EM_runs>1 && verbose>=1)
        fprintf(1,'ml = %f \n',solution.stats.ml);
    end
    if loglik > best_loglik
        best_solution = solution;
        best_loglik = loglik;
    end
end
solution = best_solution;

if verbose >=1, fprintf('Average iteration : %i\n', round(mean(averaged_iter))); end
if (nb_EM_runs>1 && verbose >=2),   fprintf(1,'best penalized loglik :  %f\n',solution.stats.ml); end

%--------------------------------------------------------------------------
solution.stats.cputime = mean(stored_cputime);
solution.stats.stored_cputime = stored_cputime;

solution.stats.n = n;
solution.stats.K = K;

%De-standardize: reconstruct the true param using the standardize-info
solution.param.Alpha   = solution.param.Alpha ./ sigma_X';
solution.param.alpha0  = solution.param.alpha0 - mu_X * solution.param.Alpha;
solution.param.Beta    = solution.param.Beta ./ sigma_X';
solution.param.beta0   = solution.param.beta0 - mu_X * solution.param.Beta;

[solution.param.beta0, solution.param.Beta, solution.param.alpha0, solution.param.Alpha, solution.param.sigma2] = identify_network(solution.param.beta0, solution.param.Beta, solution.param.alpha0, solution.param.Alpha, solution.param.sigma2);
solution.param.alpha0 = [solution.param.alpha0 0];
solution.param.Alpha  = [solution.param.Alpha zeros(p,1)];

end