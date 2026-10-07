function [X, Y, true_mixture] = simulate_data(n, d, K, filename, param, MU)

    if nargin < 3 || isempty(K)
        K = 4;
    end
    if nargin < 4
        filename = '';
    end

    %% Parameters
    if nargin < 5 || isempty(param)
        beta0  = linspace(-5, 5, K);
        Beta   = randi([-5, 5], d, K);
        sigma2 = randi([4, 5], 1, K);
        alpha0 = [randi([-5, 5], 1, K-1), 0];
        Alpha  = [randi([-5, 5], d, K-1), zeros(d, 1)];
        beta0  = sort(beta0);
    else
        beta0  = param.beta0;
        Beta   = param.Beta;
        sigma2 = param.sigma2;
        alpha0 = param.alpha0;
        Alpha  = param.Alpha;
    end

    %% Generate X
    % MU can be:
    % 1) Not provided -> generated automatically for all K components
    % 2) A K x d matrix (or 1 x d replicated)
    % 3) A struct with fields MU1, MU2, ..., MUk
    if nargin < 6 || isempty(MU)
        MU_mat = randi([-4, 4], K, d);
    elseif isstruct(MU)
        MU_mat = zeros(K, d);
        for k = 1:K
            field_name = sprintf('MU%d', k);
            if isfield(MU, field_name)
                MU_mat(k,:) = MU.(field_name);
            elseif isfield(MU, 'MU1')
                MU_mat(k,:) = MU.MU1;
            else
                MU_mat(k,:) = randi([-4, 4], 1, d);
            end
        end
    elseif isnumeric(MU)
        if size(MU, 1) == 1 && K > 1
            MU_mat = repmat(MU, K, 1);
        else
            MU_mat = MU;
        end
    else
        MU_mat = randi([-4, 4], K, d);
    end

    % Covariance matrix for X
    SIGMA = eye(d);
    for i = 1:d
        for j = 1:d
            SIGMA(i,j) = 1 / (4^(abs(i-j)));
        end
    end

    % Sample sizes per component summing exactly to n
    n_k = floor(n / K) * ones(1, K);
    n_k(end) = n - sum(n_k(1:K-1));

    X_cells = cell(K, 1);
    for k = 1:K
        X_cells{k} = mvnrnd(MU_mat(k,:), SIGMA, n_k(k));
    end
    X = cell2mat(X_cells);
    X = X(randperm(n), :);
    X = zscore(X);

    %% Generate scalar responses
    H = alpha0 + X * Alpha;
    PI = exp(H) ./ sum(exp(H), 2);
    Z = mnrnd(1, PI);
    [~, true_labels] = max(Z, [], 2);

    means_nxK  = beta0 + X * Beta;
    means_nx1  = sum(means_nxK .* Z, 2);
    sigma2_nx1 = sum(sigma2 .* Z, 2);

    reproducibility = rng;
    Y = normrnd(means_nx1, sqrt(sigma2_nx1), n, 1);

    %% Create true_mixture struct
    true_mixture.experts     = [beta0; Beta];
    true_mixture.gates       = [alpha0; Alpha];
    true_mixture.variances   = sigma2;
    true_mixture.true_labels = true_labels;

    if ~isempty(filename)
        save("./data/" + filename + ".mat", 'X', 'Y', 'true_mixture', 'reproducibility');
    end

    groups_stats = zeros(1, K);
    for k = 1:K
        groups_stats(k) = sum(true_labels == k);
    end
    fprintf('Group stat (true) [K=%d] : %s\n', K, num2str(groups_stats));

end
