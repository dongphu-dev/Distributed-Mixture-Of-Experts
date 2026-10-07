function red = param_MSE(estimated_mixture, true_mixture)
% PARAM_MSE Computes parameter Mean Squared Error modulo label switching (Hungarian matching).
% Finds the optimal permutation of estimated experts to ground truth experts to eliminate label switching.

    [d_plus_1, K] = size(true_mixture.experts);

    true_exp  = true_mixture.experts;      % (d+1) x K
    true_sig2 = true_mixture.variances(:); % K x 1
    true_gate = true_mixture.gates;        % (d+1) x K
    
    est_exp  = estimated_mixture.experts;
    est_sig2 = estimated_mixture.variances(:);
    est_gate = estimated_mixture.gates;

    denom = length(true_exp(:)) + length(true_sig2) + length(true_gate(:)) - d_plus_1;

    if K <= 7
        % Exact search over all K! permutations (for K=5, 120 perms take < 0.1 ms)
        all_perms = perms(1:K);
        n_perms = size(all_perms, 1);
        min_mse = inf;
        
        for ip = 1:n_perms
            p = all_perms(ip, :);
            
            e_p = est_exp(:, p);
            s_p = est_sig2(p);
            g_p = est_gate(:, p);
            % Re-reference gating so reference column K is 0 (matching true_gate canonical form)
            g_p = g_p - repmat(g_p(:, K), 1, K);
            
            p_true = [true_exp(:); true_sig2; true_gate(:)];
            p_est  = [e_p(:); s_p; g_p(:)];
            
            cur_mse = sum((p_true - p_est).^2) / denom;
            if cur_mse < min_mse
                min_mse = cur_mse;
            end
        end
        red = min_mse;
    else
        % For larger K, use Hungarian matching on pairwise expert cost
        cost = zeros(K, K);
        for i = 1:K
            for j = 1:K
                cost(i, j) = sum((est_exp(:, i) - true_exp(:, j)).^2) + (est_sig2(i) - true_sig2(j)).^2;
            end
        end

        % Sanitize cost matrix: replace NaN, Inf, and negative values
        cost(~isfinite(cost)) = 1e8;
        cost(cost < 0) = 0;

        best_p = zeros(1, K);
        matched_true = false(1, K);
        matched_est  = false(1, K);

        has_matchpairs = (exist('matchpairs', 'file') == 2) || (exist('matchpairs', 'builtin') == 5);
        if has_matchpairs
            try
                % Threshold 1e12 ensures valid matching across all finite costs
                matches = matchpairs(cost, 1e12);
                for m = 1:size(matches, 1)
                    r = matches(m, 1); % est_exp index
                    c = matches(m, 2); % true_exp index
                    if c >= 1 && c <= K && r >= 1 && r <= K
                        best_p(c) = r;
                        matched_true(c) = true;
                        matched_est(r)  = true;
                    end
                end
            catch
                % Fall back to greedy matching below
            end
        end

        % Guarantee full 1-to-1 assignment: fill any unmatched true expert
        unmatched_true = find(~matched_true);
        unmatched_est  = find(~matched_est);
        for idx = 1:length(unmatched_true)
            c = unmatched_true(idx);
            if idx <= length(unmatched_est)
                best_p(c) = unmatched_est(idx);
            else
                best_p(c) = c;
            end
        end

        % Guarantee best_p is a strictly valid permutation of 1:K (no zeros, no duplicates)
        if any(best_p < 1) || any(best_p > K) || length(unique(best_p)) < K
            % Greedy matching fallback
            cost_copy = cost;
            best_p = zeros(1, K);
            for j = 1:K
                [~, min_i] = min(cost_copy(:, j));
                best_p(j) = min_i;
                cost_copy(min_i, :) = inf;
            end
            % Final safety check
            if any(best_p < 1) || any(best_p > K) || length(unique(best_p)) < K
                best_p = 1:K;
            end
        end

        e_p = est_exp(:, best_p);
        s_p = est_sig2(best_p);
        g_p = est_gate(:, best_p);
        g_p = g_p - repmat(g_p(:, K), 1, K);
        p_true = [true_exp(:); true_sig2; true_gate(:)];
        p_est  = [e_p(:); s_p; g_p(:)];
        red = sum((p_true - p_est).^2) / denom;
    end

end