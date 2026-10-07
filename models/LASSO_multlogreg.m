function [w0, W, stored_loglik, Prob] = LASSO_multlogreg(X, Tau, w0, W, chi, max_iter, threshold, verbose)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Coordinate descent dased on the paper of Friedman et al. Regularization 
% Paths for Generalized Linear Models via Coordinate Descent
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin < 8, verbose=0;end
if nargin < 7, threshold=1e-6;end
if nargin < 6, max_iter=1000;end

[n, q] = size(X);
[~, K] = size(Tau);

stored_loglik = [];
epsilon = 1e-8;
converge = 0;
iter=0;
linesearch = 1;

[Prob, prev_loglik] = multinomial_logistic([w0; W], [ones(n,1) X], Tau);
prev_loglik = prev_loglik - sum(sum(chi.*abs(W)));

while (~converge && (iter< max_iter))

    W_prev = W;
    w0_prev = w0;
    
    for k=1:K-1
        %% Quadratic Taylor exapnsion of the the log-likelihood
        prob = Prob(:,k);
   
        sm = prob < epsilon; % rows close to 0
        md = (epsilon <= prob) & (prob <= 1-epsilon); % % nomal rows
        hi = prob > 1-epsilon; % rows close to 1
        prob = 0*sm + prob.*md + 1*hi;
        
        % Set extreme weights to epsilon
        weights = prob.*(1-prob) + epsilon*sm + epsilon*hi;
        
        % True response
        y = Tau(:,k);
        
        % Working response
        z = w0(k) + X*W(:,k) + (y - prob)./weights;

        %% Coordinate ascent;
        % Optimize element j of w
        for j=1:q
            % Exclude the contribution of the jth column of X get ith col of X
            xj = X(:,j);
            % Fitted value excluding jth col
            ytild_j = w0(k) + X*W(:,k) - xj*W(j,k); 
            W(j,k) = wthresh(weights' * (xj.*(z - ytild_j)),'s',chi(k))/(weights' * xj.^2); % soft thresholding
        end
        
        % Optimize w0
        w0(k) = (weights' * (z - X*W(:,k)))/sum(weights); 
%         [Prob, ~] = multinomial_logistic([w0; W], [ones(n,1) X], Tau);
%         Prob = normalize(Prob + 1e-6, 2);
    end
    
    
    W_direction  = W - W_prev;
    w0_direction = w0 - w0_prev;
    W  = W_prev  + W_direction;
    w0 = w0_prev + w0_direction;
    
    if linesearch
        
        %Compute the new log-likelihood
        [Prob, loglik] = multinomial_logistic([w0; W], [ones(n,1) X], Tau);
        loglik = loglik - sum(sum(chi.*abs(W)));
        
        stepsize = 1;
        while loglik < prev_loglik
            stepsize = 0.5 * stepsize;
            W  = W_prev  + stepsize * W_direction;
            w0 = w0_prev + stepsize * w0_direction;
            [Prob, loglik] = multinomial_logistic([w0; W], [ones(n,1) X], Tau);
            loglik = loglik - sum(sum(chi.*abs(W)));
        end
        
    else
        [Prob, loglik] = multinomial_logistic([w0; W], [ones(n,1) X], Tau);
        loglik = loglik - sum(sum(chi.*abs(W)));
    end

    if verbose
        fprintf(1,'Coord. Asc. Softmax: iter : %d Q_chi: %f \n',iter, loglik);
    end

    converge = abs((loglik-prev_loglik)/prev_loglik) < threshold; %|| ;

    prev_loglik = loglik;   
    iter=iter+1;
%     if iter == max_iter
%         fprintf(1,'LASSO_multlogreg reachs max iter \n');
%     end
end



