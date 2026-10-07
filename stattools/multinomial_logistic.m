function [probs, loglik] = multinomial_logistic(W, X, Y)
%--------------------------------------------------------------------------
%
%   Compute probability matrix according to multinomial logistic model and 
%   log-likelihood
%   Input:
%           1. W: (p+1)-by-K parameter matrix. First row are the intercepts
%           2. X: n-by-(p+1) designed matrix. First column are ones
%           3. Y: n-by-K response matrix.
%   Output:
%           1. probs: probability matrix, i.e, 
%                       [p_11 ... p_1K]
%                       [p_21 ... p_2K]
%                       [...  ... ... ]
%                       [p_n1 ... p_nK]
%           2. loglik: log likelihood of parameter W
%
%   by Thien N. Pham
%--------------------------------------------------------------------------



if nargin > 2
    [~,K] = size(Y);
    
    %IF W doesnt contain the null vector associated with the last class
    if size(W,2)== (K-1)
        q = size(W,1);
        wK=zeros(q,1);
        W = [W wK];
    end
else
    [~,K] = size(W);
end

XW = X*W;
maxm = max(XW,[],2);
XW = XW - maxm*ones(1,K);
expXW = exp(XW);
probs = expXW./(sum(expXW,2)*ones(1,K));


if nargin>2 
    loglik = sum( sum( Y .* (XW  -  logsumexp(XW,2)) ) );
%     loglik = sum( Y .* (XW  -  logsumexp(XW,2)), 'all' );
    if isnan(loglik)
        % to avoid numerical overflow since exp(XW=-746)=0 and exp(XW=710)=inf)
        minm = -745.1;
        XW = max(XW, minm);
        maxm = 709.78;
        XW= min(XW,maxm);
        expXW = exp(XW);
        % log-likelihood
        loglik = sum(sum((Y.*XW) - (Y.*log(sum(expXW,2)*ones(1,K)+eps)),2));
    end
else
    loglik = [];
end
end