function [beta0, beta, sigma2] = weighted_LASSO_greg(X, y, lambda, weights, beta0, beta, sigma2, max_iter, threshold, verbose)

    % fits a weighted Gaussian regression problem with lasso-regularized maximum likelihood by using a
    % coordinate ascent algorithm

    if nargin < 10, verbose=0;end
    if nargin < 9, threshold=1e-5;end
    if nargin < 8, max_iter=500;end

    iter = 0;
    converge = 0;


    [n, p] = size(X);
    Tau = weights;

    % Objective function for the initial model
    if sigma2 < 1e-100, sigma2 = 1e-100; end
    log_Phi_y   =  -0.5*log(2*pi) - 0.5*log(sigma2) -0.5*((y - beta0*ones(n,1) - X*beta).^2)/sigma2;
    Q_old       = (Tau' * log_Phi_y) - lambda*sum(abs(beta));

    if verbose, fprintf(1,'Coord. Ascent Experts Net: iter : %d Q_lambda: %f \n', iter, Q_old); end

    while ~converge && (iter<max_iter)
        for j=1:p

            Xj = X(:,j);
            % coordinate ascent for lasso to update the reg coefficients Beta's
            Rkj     = y - beta0*ones(n,1) -  X*beta + beta(j)*Xj;
            beta(j) = wthresh(Tau' * (Xj.*Rkj),'s',lambda*sigma2)/(Tau' * (Xj.^2)); % soft thresholding
        end
        % Objective function
        log_Phi_y =  -0.5*log(2*pi) - 0.5*log(sigma2) -0.5*((y - beta0*ones(n,1) - X*beta).^2)/sigma2;
        Q = Tau' * log_Phi_y - lambda*sum(abs(beta));

        % convergence test
        converge  = abs(Q-Q_old) <= threshold; % || abs((Q-Q_old)/Q_old) <= threshold;
        iter = iter+1;
        Q_old = Q;
        if verbose, fprintf(1,'Coord. Ascent Experts Net: iter : %d Q_lambda: %f \n', iter, Q); end
    end
    beta0 = (Tau' * (y - X*beta))/sum(Tau); % update the intercept
    sigma2 = (Tau' * ((y - beta0*ones(n,1) -X*beta).^2))/sum(Tau); % update the regressor variance
    if sigma2 < 1e-100, sigma2 = 1e-100; end
    
end