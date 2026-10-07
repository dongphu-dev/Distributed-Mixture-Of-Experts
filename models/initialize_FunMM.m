function [weights, beta0, Beta, sigma2] = initialize_FunMM(X, y, K)
% Initializes a mixture of functional mixture of Gaussian and the EM-Lasso algorithm
% Thien Pham

[n, p] = size(X);

if size(y,1)~=n, y=y'; end
G = size(y,2); 
if G > 1 
    expert_type = 'softmax';
else
    expert_type = 'Gaussian';
end

%% Intialise the weights parameters
weights = zeros(1,K);
res = myKmeans(X, K, 10, 1000, 0);
    
for k=1:K
    Xk = X(res.klas==k, :);
    weights(k) = length(Xk)/n;
end
%     
%% Intialise Expert Network's parameters
switch expert_type
    case 'Gaussian' %Gaussian experts for regression problems
        beta0 = zeros(1,K);
        Beta = zeros(p, K);
        sigma2 = zeros(1,K);
        [klas, ~] = kmeans(X, K);
        for k=1:K
            Xk = X(klas==k,:);
            yk = y(klas==k);
            nk= length(yk);
            %the regression coefficients
            if p<=n
                Beta(:,k) = Xk'*Xk\Xk'*yk;% OLS
            end
            beta0(k) = sum(yk - Xk*Beta(:,k));
            %the variances sigma2k
            sigma2(k)= sum((yk - beta0(k)*ones(nk,1) - Xk*Beta(:,k)).^2)/nk;
        end
        [beta0, Beta, weights, sigma2] = identify_network_MM(beta0, Beta, weights, sigma2);
        
    case 'softmax' %softmax expert for classification problem

        beta0  = rand(K,G-1);
        Beta   = rand(p,G-1,K);
        sigma2 = rand(K); %to be removed
        
end