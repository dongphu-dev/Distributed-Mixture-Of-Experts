function [w0, W, stored_loglik, Prob] = FLIRTI_multlogreg(X, Tau, w0, W, chi, FLIRTI_matrix, omega, max_iter, threshold, verbose)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   IRLS-like algo for solving Multinomial Logistic Regression that imposed 
%   FLIRTI methodology.
%   Input:
%       X: n-by-q designed matrix
%       Tau: n-by-K partition (hard or smooth), K >= 2
%       w0: initial value of intercept
%       W : q-by-K matrix of initial values of parameters
%       chi: tuning parameter
%       FLIRTI_matrix: (q+1)x(q+1) FLIRTI matrix (see the paper).
%       omega: tuning parameter controls weigth of the second derivative
%       max_iter, threshold, verbose: as usual.
%   by Thien Pham
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


if nargin < 10, verbose  =0;    end
if nargin < 9,  threshold=1e-4; end
if nargin < 8,  max_iter =2000; end

[n, ~] = size(X);
[~, K] = size(Tau);

stored_loglik = [];

epsilon = 1e-5;
converge = 0;
iter=0;

[Prob, prev_loglik] = multinomial_logit([w0; W], [ones(n,1) X], Tau);
prev_loglik = prev_loglik - sum(sum(chi.*abs(W)));
linesearch = 1;


while (~converge && (iter< max_iter))
    
    W_prev = W;
    w0_prev = w0;

    for k=1:K-1
        % Quadratic Taylor exapnsion of the the log-likelihood
        prob = Prob(:,k);
        
        %Compute the weights and set extreme values to epsilon
        sm = prob < epsilon; % rows close to 0
        md = (epsilon <= prob) & (prob <= 1-epsilon); % % nomal rows
        hi = prob > 1-epsilon; % rows close to 1
        prob = 0*sm + prob.*md + 1*hi;
        weights = prob.*(1-prob) + epsilon*sm + epsilon*hi;
        
        %True response
        y = Tau(:,k);
        
        %Working response
        z = w0(k) + X*W(:,k) + (y - prob)./weights;
        
        
        %Weighted Dantzig selector for the approximated quadratic exapnsion

        fit = weighted_Dantzig_selector(X, z, weights, chi(k), FLIRTI_matrix, omega(k), 1);
        W(:,k) = fit.beta;
        w0(k) = (weights' * (z - X*W(:,k)))/sum(weights); 

    end
    
    
    W_direction  = W - W_prev;
    w0_direction = w0 - w0_prev;
    W  = W_prev  + W_direction;
    w0 = w0_prev + w0_direction;
    
    if linesearch
        
        %Compute the new log-likelihood
        [Prob, loglik] = multinomial_logit([w0; W], [ones(n,1) X], Tau);
        loglik = loglik - sum(sum(chi.*abs(W)));
        
        stepsize = 1;
        while loglik < prev_loglik
            stepsize = 0.5 * stepsize;
            W  = W_prev  + stepsize * W_direction;
            w0 = w0_prev + stepsize * w0_direction;
            [Prob, loglik] = multinomial_logit([w0; W], [ones(n,1) X], Tau);
            loglik = loglik - sum(sum(chi.*abs(W)));
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



