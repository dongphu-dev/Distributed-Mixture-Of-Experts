function [beta0_out, Beta_out, weights_out, sigma2_out] = identify_network_MM(beta0, Beta, weights, sigma2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Re-order the components of GMM according to:
%   1. beta0
%   2. sigma2 (fallback if beta0 tied)
%   3. norm(Beta) (fallback if both tied)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    K = length(beta0);

    if length(unique(round(beta0, 6))) == K
        [beta0_out, order] = sort(beta0);
        sigma2_out = sigma2(order);
    elseif length(unique(round(sigma2, 6))) == K
        [sigma2_out, order] = sort(sigma2);
        beta0_out = beta0(order);
    else
        beta_norms = sqrt(sum(Beta.^2, 1));
        if length(unique(round(beta_norms, 6))) == K
            [~, order] = sort(beta_norms);
        else
            order = 1:K;
        end
        beta0_out = beta0(order);
        sigma2_out = sigma2(order);
    end

    Beta_out    = Beta(:, order);
    weights_out = weights(order);

end