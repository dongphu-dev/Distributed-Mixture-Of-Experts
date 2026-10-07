function pourcent_misclassified = clusteringError(true_labels, estimated_labels)
% CLUSTERINGERROR
% Computes the percentage of misclassified points across all possible label permutations.
% Robust to degenerate/dropped clusters.

    K = max([max(true_labels), max(estimated_labels)]);
    if isempty(K) || K == 0, K = 1; end

    % Build full K x K contingency table
    crtb = zeros(K, K);
    for i = 1:length(true_labels)
        t = true_labels(i);
        e = estimated_labels(i);
        if t >= 1 && t <= K && e >= 1 && e <= K
            crtb(t, e) = crtb(t, e) + 1;
        end
    end

    % Optimal permutation match
    if K <= 8
        a = perms(1:K);
        correct = 0;
        for i = 1:size(a, 1)
            cross_sum = trace(crtb(:, a(i, :)));
            if cross_sum > correct
                correct = cross_sum;
            end
        end
    else
        % For large K, use Hungarian matching on cost = -crtb with greedy fallback
        has_matchpairs = (exist('matchpairs', 'file') == 2) || (exist('matchpairs', 'builtin') == 5);
        matched_ok = false;
        if has_matchpairs
            try
                match = matchpairs(-crtb, 1e9);
                correct = 0;
                for m = 1:size(match, 1)
                    correct = correct + crtb(match(m, 1), match(m, 2));
                end
                matched_ok = true;
            catch
                matched_ok = false;
            end
        end
        if ~matched_ok
            % Greedy matching fallback
            crtb_copy = crtb;
            correct = 0;
            for k = 1:K
                [max_val, max_idx] = max(crtb_copy(:));
                if max_val <= 0, break; end
                [r, c] = ind2sub([K, K], max_idx);
                correct = correct + max_val;
                crtb_copy(r, :) = -inf;
                crtb_copy(:, c) = -inf;
            end
        end
    end

    n = length(true_labels);
    misclassified = n - correct;
    pourcent_misclassified = (misclassified / n) * 100;

end