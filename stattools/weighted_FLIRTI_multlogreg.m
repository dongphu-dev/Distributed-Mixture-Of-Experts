function [w0, W, stored_loglik, Prob] = weighted_FLIRTI_multlogreg(X, Y, Tau, w0, W, lambda, FLIRTI_matrix, omega, max_iter, threshold, verbose)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This function updates the Multinomial Expert in Mixture of Experts models
% 
% Coordinate ascent based on the paper of Friedman et al. Regularization 
% Paths for Generalized Linear Models via Coordinate Descent
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin < 9, verbose=0;end
if nargin < 8, threshold=1e-4;end
if nargin < 6, max_iter=500;end

[n, p] = size(X);
[~, K] = size(Y);

prev_loglik = -inf;
stored_loglik = [];
linesearch = 1;
converge = 0;
iter=0;

[Prob, ~] = multinomial_logit([w0; W], [ones(n,1) X], Y);

while (~converge && (iter< max_iter))
    
    
    W_prev = W;
    w0_prev = w0;

    for k=1:K-1
        % Quadratic Taylor exapnsion of the the log-likelihood
        prob = Prob(:,k);

        %True response
        y = Y(:,k);
        
        %Working response
        z = w0(k) + X*W(:,k) + 4*(y - prob);
        
        %Weighted Dantzig selector for the approximated quadratic exapnsion
        fit = weighted_Dantzig_selector(X, z, Tau, lambda(k), FLIRTI_matrix, omega, 1);
        
        W(:,k) = fit.beta;
        w0(k) = (Tau' * (z - X*W(:,k)))/sum(Tau); 

    end
    
    W_direction  = W - W_prev;
    w0_direction = w0 - w0_prev;
    W  = W_prev  + W_direction;
    w0 = w0_prev + w0_direction;
    
    if linesearch
        
        %Compute the new log-likelihood
        [Prob, loglik] = multinomial_logit([w0; W], [ones(n,1) X], Y);
        loglik = loglik - sum(sum(lambda.*abs(W)));
        
        stepsize = 1;
        while loglik < prev_loglik
            stepsize = 0.5 * stepsize;
            W  = W_prev  + stepsize * W_direction;
            w0 = w0_prev + stepsize * w0_direction;
            [Prob, loglik] = multinomial_logit([w0; W], [ones(n,1) X], Y);
            loglik = loglik - sum(sum(lambda.*abs(W)));
        end
        
    end
    
    prev_loglik = loglik;

    if verbose
        fprintf('FLIRTI_multlogreg: iter: %d   Q_chi: %f \n',iter, loglik);
    end

    converge = abs(loglik-prev_loglik) < threshold;

    prev_loglik = loglik;   
    iter=iter+1;
    
end



