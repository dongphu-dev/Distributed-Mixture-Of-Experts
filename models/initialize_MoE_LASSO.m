function [alpha0, Alpha, beta0, Beta, sigma2] = initialize_MoE_LASSO(X, y, K, strategy)
% Initializes a mixture of high-dimensional softmax-gated mixture-of-experts and the EM-Lasso algorithm

[n, p] = size(X);

%% Intialise the softmax Gating Net parameters

if strcmp(strategy,'zeros')
    alpha0 = zeros(1,K-1);
    Alpha = zeros(p,K-1);
elseif strcmp(strategy,'random')
    alpha0 = rand(1,K-1);
    Alpha = rand(p,K-1);
else % if Logistic Regression (LR)
    alpha0 = rand(1,K-1);
    Alpha = rand(p,K-1);
    Z = zeros(n,K);
    [klas, ~] = kmeans(X, K);
    Z(klas*ones(1,K)==ones(n,1)*[1:K])=1;
    Tau = Z;
    max_iter = 1000; % need to fully optimize?
    verbose = -1; % forced silence, even without convergence
    threshold = 1e-5;
    res = IRLS([ones(n,1) X], Tau, [alpha0; Alpha], ones(n,1), max_iter, threshold, verbose);
    alpha0 = res.W(1,:);
    Alpha = res.W(2:end,:);
end

% Expert Net parameters
[klas, ~] = kmeans(X, K);
beta0 = zeros(1,K);
Beta = zeros(p, K);
sigma2 = zeros(1,K);
for k=1:K
    Xk = X(klas==k,:);
    yk = y(klas==k);
    nk= length(yk);
    %the regression coefficients
    if p<=n
        Beta(:,k) = Xk'*Xk\Xk'*yk;%ols
    end
    beta0(k) = sum(yk - Xk*Beta(:,k));
    %the variances sigma2k
    sigma2(k)= sum((yk - beta0(k)*ones(nk,1) - Xk*Beta(:,k)).^2)/nk;
end


if strcmp(strategy,'values')
    load initialPointForMMASH_K4_LASSO;
    fprintf("Initialized by LASSO result\n")
    Beta   = estimated_Beta;
    beta0  = estimated_beta0;
    Alpha  = estimated_Alpha(:,1:end-1);
    alpha0 = estimated_alpha0(1:end-1);
end


[beta0, Beta, alpha0, Alpha, sigma2] = identify_network(beta0, Beta, alpha0, Alpha, sigma2);
        
end