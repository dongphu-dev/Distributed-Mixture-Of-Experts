function [plan, distance_matrix] = argmin_transportation_plan(large_mixture, reduced_mixture, X_val)

    L  = length(large_mixture.variances);
    K  = length(reduced_mixture.variances);
    [S,d] = size(X_val);
    % -------------------------------------------------------------------------
    % Compute source mixture probabilities PI_hat (L x S)
    % -------------------------------------------------------------------------
    if isfield(large_mixture, 'PI_hat') && ~isempty(large_mixture.PI_hat) && size(large_mixture.PI_hat, 1) == S
        % Fast path: Precomputed source probability matrix on matching support (S x L)
        PI_hat = large_mixture.PI_hat'; % Transpose to (L x S)
    elseif isfield(large_mixture, 'local_estimates') && ~isempty(large_mixture.local_estimates)
        % Heterogeneous path: Compute PI_hat directly on provided X_val (S x d)
        M_loc = length(large_mixture.local_estimates);
        d_feat = d - 1;
        PI_hat_S = [];
        for m = 1:M_loc
            p_m = large_mixture.local_estimates{m}.param;
            Km  = length(p_m.beta0);
            if length(p_m.alpha0) == Km - 1
                alpha0_full = [p_m.alpha0(:)', 0];
                Alpha_full  = [p_m.Alpha, zeros(d_feat, 1)];
            else
                alpha0_full = p_m.alpha0(:)';
                Alpha_full  = p_m.Alpha;
            end
            W_gate = [alpha0_full; Alpha_full];
            logits = X_val * W_gate;
            logits = logits - max(logits, [], 2);
            prob_m = exp(logits) ./ sum(exp(logits), 2);
            w_m = 1 / M_loc;
            PI_hat_S = [PI_hat_S, w_m * prob_m]; %#ok<AGROW>
        end
        PI_hat = PI_hat_S'; % (L x S)
    else
        % Legacy path: Preserved verbatim for homogeneous K experiments
        M = L / K;
        if size(large_mixture.gates, 1) == d
            large_mixture.gates = repmat(large_mixture.gates(:), 1, K);
        end

        PI_hat = [];
        for m = 1:M
            X_val_times_Gate = X_val * reshape(large_mixture.gates(:, m*K), d, K);
            maxx             = max(X_val_times_Gate, [], 2);
            X_val_times_Gate = X_val_times_Gate - maxx;
            exp_X_val_times_Gate = exp(X_val_times_Gate);
            PI_temp = exp_X_val_times_Gate ./ sum(exp_X_val_times_Gate, 2);
            PI_hat = [PI_hat PI_temp];
        end
        % Ensure sum(PI_hat, 2) == 1 across source components
        PI_hat = PI_hat .* large_mixture.weights;
        PI_hat = PI_hat';
    end
    
    % Compute the tensor of distances [L-K-S] (Vectorized prediction precomputation)
    distance_matrix = zeros(L,K,S);
    Mean_source = X_val * large_mixture.experts;   % (S x L) precomputed once
    Mean_target = X_val * reduced_mixture.experts; % (S x K) precomputed once

    for l = 1:L
        sig1 = large_mixture.variances(l);
        m1   = Mean_source(:, l);
        
        for k = 1:K
            sig2 = reduced_mixture.variances(k);
            m2   = Mean_target(:, k);

            mean_diff_sq = (m2 - m1).^2;
            KLdisvec = 0.5 * (log(sig2 / sig1) + sig1 / sig2 + mean_diff_sq / sig2 - 1);
            distance_matrix(l, k, :) = KLdisvec;
        end
    end
    
    %Assign plan for each x in X_S
    plan  = zeros(L,K,S);
    
    [~,mindis_index] = min(distance_matrix,[],2);
    mindis_index = squeeze(mindis_index);
    for l=1:L
        for k=1:K
            plan(l,k,:) = PI_hat(l,:).*(mindis_index(l,:)==k);
        end
    end
    
end



