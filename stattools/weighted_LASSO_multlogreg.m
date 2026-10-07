function [w0, W, stored_loglik, Prob] = weighted_LASSO_multlogreg(X, Y, tau, w0, W, chi, max_iter, threshold, verbose)
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

converge = 0;
iter=0;

[Prob, ~] = multinomial_logit([w0; W], [ones(n,1) X], Y);
while (~converge && (iter< max_iter))

    for k=1:K-1
        
        %% Quadratic Taylor exapnsion of the the log-likelihood
        prob = Prob(:,k);
        
        % True response
        y = Y(:,k);
        
        % Working response
        z = w0(k) + X*W(:,k) + 4*(y - prob);
        %% Coordinate ascent;
        
        % Update element j of w
        for j=1:p
            % exclude the contribution of the jth column of X get ith col of X
            xj = X(:,j);
            % fitted value excluding jth col
            ytild_j = w0(k) + X*W(:,k) - xj*W(j,k);
            W(j,k) = wthresh(tau'/4 * (xj.*(z - ytild_j)),'s',chi(k))/(tau'/4 * xj.^2); % soft thresholding
        end
        
        % Update w0
        w0(k) = (tau' * (z - X*W(:,k)))/sum(tau); 

    end
    
    % Compute the log-likelihood
    [Prob, loglik] = multinomial_logit([w0; W], [ones(n,1) X], Y);

    loglik = loglik - sum(sum(chi.*abs(W))); 
    if verbose
        fprintf(1,'Coord. Asc. Softmax: iter : %d Q_chi: %f \n',iter, loglik);
    end

    converge = abs(loglik-prev_loglik) < threshold; %|| abs((loglik-prev_loglik)/prev_loglik) < threshold;
%     converge = all(max(abs(W - W_old)) <= threshold); %|| abs((loglik-prev_loglik)/prev_loglik) < threshold;
%     converge = norm(W(:) - W_old(:), inf) <= threshold;
    prev_loglik = loglik;   
    iter=iter+1;
end



