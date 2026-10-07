function [beta0_out, Beta_out, alpha0_out, Alpha_out, sigma2_out] = identify_network(beta0, Beta, alpha0, Alpha, sigma2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Make Expert and Gating network identifiable by canonical re-ordering
% according to:
%   1. beta0 (intercepts)
%   2. sigma2 (noise variances, if beta0 tied)
%   3. norm(Beta) (column norms of slopes, if both beta0 and sigma2 tied)
%
% Ref: W.Jiang, M.Tanner, On the Identifiability of Mixtures-of-Experts
% Author: T. Pham (Extended for robust tie-breaking)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    K = length(beta0);
    q = size(Alpha, 1);
    has_sigma2 = (nargin >= 5 && ~isempty(sigma2));

    % 1. Determine canonical ordering
    if length(unique(round(beta0, 6))) == K
        % Criterion 1: Distinct intercepts
        [beta0_out, order] = sort(beta0);
        if has_sigma2
            sigma2_out = sigma2(order);
        else
            sigma2_out = [];
        end
    elseif has_sigma2 && length(unique(round(sigma2, 6))) == K
        % Criterion 2: Distinct variances (fallback when beta0 tied)
        [sigma2_out, order] = sort(sigma2);
        beta0_out = beta0(order);
    else
        % Criterion 3: Column norms of Beta (fallback when both beta0 and sigma2 tied)
        beta_norms = sqrt(sum(Beta.^2, 1));
        if length(unique(round(beta_norms, 6))) == K
            [~, order] = sort(beta_norms);
        else
            order = 1:K; % Preserve input order if perfectly identical
        end
        beta0_out = beta0(order);
        if has_sigma2
            sigma2_out = sigma2(order);
        else
            sigma2_out = [];
        end
    end

    % 2. Re-order regression slopes Beta
    Beta_out = Beta(:, order);

    % 3. Re-order Gating Network
    % Check if alpha0 / Alpha are already K-dimensional or (K-1)-dimensional
    if length(alpha0) == K - 1
        alpha0_full = [alpha0, 0];
    else
        alpha0_full = alpha0;
    end

    if size(Alpha, 2) == K - 1
        Alpha_full = [Alpha, zeros(q, 1)];
    else
        Alpha_full = Alpha;
    end

    % Permute gates according to canonical expert order
    alpha0_perm = alpha0_full(order);
    alpha0_perm = alpha0_perm - alpha0_perm(end); % Reference component K = 0

    Alpha_perm  = Alpha_full(:, order);
    Alpha_perm  = Alpha_perm - repmat(Alpha_perm(:, end), 1, K); % Reference component K = 0

    % Return (K-1) free gate parameters
    alpha0_out = alpha0_perm(1:K-1);
    Alpha_out  = Alpha_perm(:, 1:K-1);

end