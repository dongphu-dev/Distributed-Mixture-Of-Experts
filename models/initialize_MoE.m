function [alpha0, Alpha, beta0, Beta, sigma2] = initialize_MoE(Xa, y, K, strategy)
% Initializes a mixture of functional softmax-gated mixture-of-experts and the EM-Lasso algorithm FC

[n, p] = size(Xa);

if size(y,1)~=n, y=y'; end
G = size(y,2); 
if G > 1 
    expert_type = 'softmax';
else
    expert_type = 'Gaussian';
end

%% 1. Partition initialization using Random Subset Regression on (Xa, y)
if strcmp(expert_type, 'Gaussian')
    % RANSAC-style Random Hyperplane sampling: fits K initial candidate hyperplanes on random subsets
    sub_size = min(n, max(2*p + 2, 40));
    candidate_betas = zeros(p, K);
    candidate_beta0 = zeros(1, K);
    for k = 1:K
        sub_idx = randperm(n, sub_size);
        X_sub = [ones(sub_size, 1), Xa(sub_idx, :)];
        y_sub = y(sub_idx);
        cand_b = (X_sub' * X_sub + 1e-4 * eye(p+1)) \ (X_sub' * y_sub);
        candidate_beta0(k) = cand_b(1);
        candidate_betas(:, k) = cand_b(2:end);
    end
    resids = zeros(n, K);
    for k = 1:K
        resids(:, k) = (y - candidate_beta0(k) - Xa * candidate_betas(:, k)).^2;
    end
    [~, klas] = min(resids, [], 2);
    % Guarantee every cluster has sufficient support for well-conditioned OLS
    for k = 1:K
        if sum(klas == k) < p + 2
            klas(randperm(n, p + 2)) = k;
        end
    end
else
    [klas, ~] = kmeans(Xa, K);
end

%% 2. Initialise the softmax Gating Net parameters
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
    Z(klas*ones(1,K)==ones(n,1)*[1:K])=1;
    Tau = Z;
    max_iter = 5;
    verbose = -1;
    threshold = 1e-5;
    res = IRLS([ones(n,1) Xa], Tau, [alpha0; Alpha], ones(n,1), max_iter, threshold, verbose);
    alpha0 = res.W(1,:);
    Alpha = res.W(2:end,:);
end

%% 3. Initialise Expert Network's parameters
switch expert_type
    case 'Gaussian' % Gaussian experts for regression problems
        beta0 = zeros(1,K);
        Beta = zeros(p, K);
        sigma2 = zeros(1,K);
        for k=1:K
            Xk = Xa(klas==k,:);
            yk = y(klas==k);
            nk = length(yk);
            if p + 1 <= nk
                Xk_aug = [ones(nk, 1), Xk];
                b_all = (Xk_aug'*Xk_aug + 1e-6*eye(p+1)) \ (Xk_aug'*yk);
                beta0(k) = b_all(1);
                Beta(:,k) = b_all(2:end);
            else
                beta0(k) = mean(yk);
                Beta(:,k) = zeros(p, 1);
            end
            res_k = yk - beta0(k) - Xk*Beta(:,k);
            sigma2(k) = max(mean(res_k.^2), 1e-4);
        end
        [beta0, Beta, alpha0, Alpha, sigma2] = identify_network(beta0, Beta, alpha0, Alpha, sigma2);
        
    case 'softmax' %softmax expert for classification problem

        beta0  = rand(K,G);
        Beta   = rand(p,G,K);
        [beta0, Beta, alpha0, Alpha, ~] = identify_network(beta0, Beta, alpha0, Alpha);
        
end





if strcmp(strategy,'values')
    load initialPointForMMASH_K4_from_Distributed_M16;
    Beta   = estimated_Beta;
    beta0  = estimated_beta0;
    Alpha  = estimated_Alpha(:,1:end-1);
    alpha0 = estimated_alpha0(1:end-1);
end









end