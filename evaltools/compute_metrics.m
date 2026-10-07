function [learning_time, trandis, loglik, mse_param, RPE_test, corr_test, RI_test, ARI_test, ClustErr_test, trandis_fW] = compute_metrics(fit, true_mixture, X, Y, true_labels, verbose, approach_name, f_W)

    if nargin < 6, verbose = 0; end
    if nargin < 7, approach_name = ''; end
    if nargin < 8, f_W = []; end

    N = size(X,1);
    K = length(true_mixture.variances);
    
    % Default outputs
    learning_time = NaN; trandis = NaN; loglik = NaN; mse_param = NaN;
    RPE_test = NaN; corr_test = NaN; RI_test = NaN; ARI_test = NaN; ClustErr_test = NaN;
    trandis_fW = NaN;
    
    %======================================================================
    % FOR GLOBAL, DISTRIBUTED, MEDIAN AND AVERAGED ME MODELS
    % 1. Learning times
    % 2. Transportation distance
    % 3. Log likelihood on X_test
    % 4. MSE between estimated and true parameters
    % 5. Relative Prediction Error, Correlation, RI, ARI, Clusstering Error
    % 6. Transportation distance to f^W (pooled local mixture)
    %======================================================================
    
    if isfield(fit, 'gates')
    

        %------------------------------------------------------------------
        %  COMPUTE TRANSPORTATION DISTANCE
        %------------------------------------------------------------------
        if isfield(fit, 'reduced_mixture')
            % For Global, Median and Average MoE
            [plan, distance_matrix] = argmin_transportation_plan(fit.reduced_mixture, true_mixture, [ones(N,1) X]);
            fit_eval = fit.reduced_mixture;
        else 
            % For Distributed MoE
            [plan, distance_matrix] = argmin_transportation_plan(fit, true_mixture, [ones(N,1) X]);
            fit_eval = fit;
        end
        trandis       = sum(sum(sum(plan .* distance_matrix)));
        
        % Distance to pooled local mixture f^W (reduction divergence)
        if ~isempty(f_W)
            [plan_fW, dist_fW] = argmin_transportation_plan(f_W, fit_eval, [ones(N,1) X]);
            trandis_fW = sum(sum(sum(plan_fW .* dist_fW)));
        end
        
        learning_time = fit.learning_time;
        mse_param     =  param_MSE(fit, true_mixture);
        

        %------------------------------------------------------------------
        %  PREDICTION
        %------------------------------------------------------------------

        estimated_Alpha  = fit.param.Alpha;
        estimated_alpha0 = fit.param.alpha0;
        estimated_Beta   = fit.param.Beta;
        estimated_beta0  = fit.param.beta0;
        estimated_sigma2 = fit.param.sigma2;


        H       = estimated_alpha0  +  X * estimated_Alpha;
        maxm    = max(H, [], 2);
        H       = H  -  maxm * ones(1,K);
        gatingProb = exp(H) ./ (sum(exp(H),2)*ones(1,K));

        % Predict the labels
        %[~, pred_labels] = max(gatingProb,[],2);

        % Predict the responses
        Y_K     = estimated_beta0  +  X * estimated_Beta;
        Y_pred = sum(Y_K .* gatingProb, 2);

        %------------------------------------------------------------------
        %  COMPUTE LOGLIK
        %------------------------------------------------------------------

        log_Phi_xy = zeros(N,K);
        for k = 1:K
            log_Phi_y =  -0.5*log(2*pi) - 0.5*log(estimated_sigma2(k)) -0.5*((Y - estimated_beta0(k)*ones(N,1) - X*estimated_Beta(:,k)).^2)/estimated_sigma2(k);
            log_Phi_xy(:,k) = log(gatingProb(:,k)) + log_Phi_y;
        end
        %------------------------------------------------------------------
        log_sum_PhixPhiy = logsumexp(log_Phi_xy,2);
        loglik           = sum(log_sum_PhixPhiy);
        log_Tau          = log_Phi_xy - log_sum_PhixPhiy*ones(1,K);
        Tau              = exp(log_Tau);
        Tau              = Tau./(sum(Tau,2)*ones(1,K));
        pred_labels      = MAP(Tau);




        %------------------------------------------------------------------
        %  COMPUTE CRITERIA
        %------------------------------------------------------------------
        % Relative Predictions Error (RPE)
        RPE_test   = RPE(Y, Y_pred);

        % Correlation
        corr_test  = corr(Y ,Y_pred);

        % ARI
        [ARI_test, RI_test]  = RandIndex(true_labels, pred_labels);

        % Clustering error
        ClustErr_test  = clusteringError(true_labels, pred_labels);

        % Print the setting and the evaluation indicators
        if verbose 
            if ~isempty(approach_name)
                fprintf('--- [%s] --------------------------------------------------------\n', approach_name);
            else
                fprintf('----------------------------------------------------------------\n');
            end
            fprintf('         Time      Trandis    Loglik    mse_param   RPE     Correlation  RI      ARI     ClustErr\n');
            fprintf('Test :   %4.2f    %5.2f    %6.2f   %4.5f    %1.3f   %1.3f        %1.3f   %1.3f   %1.3f \n\n', learning_time, trandis, loglik, mse_param, RPE_test,  corr_test,  RI_test,  ARI_test,  ClustErr_test)
        end


    
    %======================================================================
    % FOR MIXTURE OF LINEAR REGRESSION MODEL
    % No gating functions
    % Do not compute transporttation distance, MSE_param
    %======================================================================
    else
        
        trandis       = NaN;
        learning_time = fit.learning_time;
        mse_param     = NaN;
        
        
        estimated_Beta    = fit.param.Beta;
        estimated_beta0   = fit.param.beta0;
        estimated_sigma2  = fit.param.sigma2;
        estimated_weights = fit.param.weights;
        %

        % -----------------------------------------------------------------
        %  PREDICTING
        % -----------------------------------------------------------------
        log_Phi_xy = zeros(N,K);
        for k = 1:K
            log_Phi_y =  -0.5*log(2*pi) - 0.5*log(estimated_sigma2(k)) -0.5*((Y - estimated_beta0(k)*ones(N,1) - X*estimated_Beta(:,k)).^2)/estimated_sigma2(k);
            log_Phi_xy(:,k) = log(estimated_weights(:,k)) + log_Phi_y;
        end
        %------------------------------------------------------------------
        log_sum_PhixPhiy = logsumexp(log_Phi_xy,2);
        loglik           = sum(log_sum_PhixPhiy);
        log_Tau = log_Phi_xy - log_sum_PhixPhiy*ones(1,K);
        Tau = exp(log_Tau);
        Tau = Tau./(sum(Tau,2)*ones(1,K));
        pred_labels = MAP(Tau);

        Y_     = estimated_beta0  + X * estimated_Beta;
        Y_pred = sum(Y_ .* estimated_weights,2);

        %------------------------------------------------------------------
        %  COMPUTE CRITERIA
        %------------------------------------------------------------------
        
        % Relative Predictions Error (RPE)
        RPE_test   = RPE(Y, Y_pred);

        % Correlation
        corr_test  = corr(Y ,Y_pred);

        % ARI
        [ARI_test, RI_test]  = RandIndex(true_labels, pred_labels);

        % Clustering error
        ClustErr_test  = clusteringError(true_labels, pred_labels);

        
        % Print the setting and the evaluation indicators
        if verbose 
            if ~isempty(approach_name)
                fprintf('--- [%s] (MLR) --------------------------------------------------\n', approach_name);
            else
                fprintf('MIXTURE OF LINEAR REGRESSION --------------------------------------------------------------------\n');
            end
            fprintf('         Time      Trandis    Loglik    mse_param   RPE     Correlation  RI      ARI     ClustErr\n');
            fprintf('Test :   %4.2f   %5.2f        %6.2f  %4.5f      %1.3f   %1.3f        %1.3f   %1.3f   %1.3f \n\n', learning_time, trandis, loglik, mse_param, RPE_test,  corr_test,  RI_test,  ARI_test,  ClustErr_test)
        end
        
    end
    
    
    
    
 
    
end