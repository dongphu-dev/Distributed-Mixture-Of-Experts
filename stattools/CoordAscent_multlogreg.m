function [w0, W, stored_loglik, Prob] = CoordAscent_multlogreg(X, Tau, w0, W, verbose)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Coordinate ascent for multiclass logistic regression
% Tau: soft or hard partition
%
% by Faicel Chamroukhi
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%if (size(y,2)~=1), y = y';end

[n, q] = size(X);
[n, K] = size(Tau);

tol = 1e-5;
converge = 0;
iter=0;
max_iter = 500;
prev_loglik = -inf;

stored_loglik = [];
% %  Initialization of the model parameters
% W = X'*X\ X'*Tau;
% w0 = mean(Tau); %w_0
%
% W_old = W;
% [Prob, loglik_old] = multinomial_logit(W, X, Tau);
[Prob, loglik_old] = multinomial_logit([w0; W], [ones(n,1) X], Tau);
while (~converge && iter <= max_iter)
    for k=1:K-1
        %% Quadratic Taylor exapnsion of the the log-likelihood
        weights = Prob(:,k).*(1-Prob(:,k));% weights : prob.*(1-prob);
        y = Tau(:,k); % soft labels
        z = w0(k) + X*W(:,k) + (y - Prob(:,k))./weights; % working response
        % coordinate ascent;
        for j=1:q
            % optimize element j of wk
            xj = X(:,j);% exclude the contribution of the jth column of X get ith col of X
            ytild_j = w0(k) + X*W(:,k) - xj*W(j,k); % fitted value excluding the contribution from the jth col
            W(j,k) = sum(weights.*xj.*(z - ytild_j))/sum(weights.*xj.^2);
        end
       % w0(k) = sum(weights.*(z - X*W(:,k)))/sum(weights);%intercept
       w0(k) = sum(weights.*(w0(k) + (y - Prob(:,k))./weights))/sum(weights);
        %Prob = exp(log_softmax(X, W) ([w0; W], [ones(n,1) X], Tau);
%                         
    end
    
    % compute the log-likelihood
    [Prob, loglik] = multinomial_logit([w0; W], [ones(n,1) X], Tau);
    %     Ww0=[w0;W];
    %     loglik = sum(sum((y.*([ones(n,1) X]*[Ww0 zeros(q+1,1)])) - (y.*log(sum(exp([ones(n,1) X]*[Ww0 zeros(q+1,1)]),2)*ones(1,K))),2))
    if verbose
        fprintf(1,'CoordAsc. softmax: iteration : %d loglik : %f \n',iter, loglik);
    end
    %if (max(abs(W - W_old)) <tol), converge = 1; end
    %W_old = W;
    if (abs((loglik-prev_loglik)/prev_loglik) < tol), converge = 1; end
    prev_loglik = loglik;
    iter=iter+1;
end



