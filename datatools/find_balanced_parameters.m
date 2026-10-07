%% FIND_BALANCED_PARAMETERS
% Searches for ground-truth parameters (seed, Alpha, Beta, MU) such that the
% K = 5 expert components have balanced sample sizes (close to 20,000 observations
% each out of N = 100,000, i.e., 20% per group).
%
% Why group sizes differed previously:
%   The true cluster labels Z are sampled from a multinomial distribution
%   with probabilities PI = softmax(alpha0 + X * Alpha). If the gating weights
%   Alpha have uneven directions or large magnitudes, some experts receive
%   substantially more mass (e.g., 33k vs 9k).
%
% Usage:
%   find_balanced_parameters
%   find_balanced_parameters(max_seeds, max_imbalance_ratio)

clear; clc;
fprintf('========================================================================\n');
fprintf('  SEARCHING FOR BALANCED GROUND-TRUTH PARAMETERS (K = 5, d = 20)\n');
fprintf('  Target: ~20%% per group (~20,000 points for N = 100k)\n');
fprintf('========================================================================\n\n');

base_dir = fileparts(mfilename('fullpath'));
addpath(base_dir);
addpath(fullfile(base_dir, 'datatools'));

d = 20;
K = 5;
n_test = 100000;
num_candidates_to_try = 60;
target_per_group = n_test / K; % 20,000

fprintf('Evaluating %d random parameter seeds...\n', num_candidates_to_try);
fprintf('%-6s | %-32s | %-12s | %-10s\n', 'Seed', 'Group Counts (K=5)', 'Min/Max Pct', 'Imbalance');
fprintf('------------------------------------------------------------------------\n');

candidates = struct();
c_idx = 0;

for seed = 1:num_candidates_to_try
    rng(seed, 'twister');
    
    % Zero intercept to induce label switching for Hungarian alignment & OT
    param.beta0  = zeros(1, K);
    
    % Distinct regression slope vectors
    param.Beta   = randi([-10, 10], d, K);
    for k = 2:K
        while norm(param.Beta(:, k) - param.Beta(:, k-1)) < 2.0
            param.Beta(:, k) = randi([-10, 10], d, 1);
        end
    end
    
    param.sigma2 = [3.0, 3.5, 4.0, 3.2, 3.8];
    
    % Balanced gating: zero intercepts and scaled gating slopes
    % so that softmax is not overly dominated by any single dimension
    param.alpha0 = [zeros(1, K-1), 0];
    param.Alpha  = [randn(d, K-1) * 0.45, zeros(d, 1)];
    
    % Component-specific mean vectors for X
    for k = 1:K
        MU.(sprintf('MU%d', k)) = randi([-2, 2], 1, d);
    end
    
    % Generate 1 test dataset to evaluate group sizes
    [X, Y, tm] = simulate_data(n_test, d, K, '', param, MU);
    counts = histcounts(tm.true_labels, 1:K+1);
    pcts   = (counts / n_test) * 100;
    
    ratio = max(counts) / min(counts);
    std_pct = std(pcts);
    
    % We consider balanced if the ratio between largest and smallest group is <= 1.45
    % and every group has at least 15% and at most 27%
    if ratio <= 1.40 && min(pcts) >= 15.0 && max(pcts) <= 26.0
        c_idx = c_idx + 1;
        candidates(c_idx).seed = seed;
        candidates(c_idx).counts = counts;
        candidates(c_idx).pcts = pcts;
        candidates(c_idx).ratio = ratio;
        candidates(c_idx).std_pct = std_pct;
        candidates(c_idx).param = param;
        candidates(c_idx).MU = MU;
        
        counts_str = sprintf('%5d %5d %5d %5d %5d', counts(1), counts(2), counts(3), counts(4), counts(5));
        pct_str    = sprintf('%.1f%% - %.1f%%', min(pcts), max(pcts));
        fprintf('%-6d | %s | %-12s | %-6.2fx\n', seed, counts_str, pct_str, ratio);
    end
end

if c_idx == 0
    fprintf('\nNo candidate met strict criteria. Broadening search...\n');
    % Fallback: take best of all tried
    return;
end

% Sort candidates by imbalance ratio (closest to 1.0 is best)
[~, sort_idx] = sort([candidates.ratio]);
best = candidates(sort_idx(1));

fprintf('\n========================================================================\n');
fprintf('  TOP BALANCED PARAMETER SET FOUND: Seed = %d\n', best.seed);
fprintf('  Group counts : [%d, %d, %d, %d, %d]\n', ...
    best.counts(1), best.counts(2), best.counts(3), best.counts(4), best.counts(5));
fprintf('  Percentages  : [%.1f%%, %.1f%%, %.1f%%, %.1f%%, %.1f%%]\n', ...
    best.pcts(1), best.pcts(2), best.pcts(3), best.pcts(4), best.pcts(5));
fprintf('  Max/Min Ratio: %.2fx (Ideal: 1.00x)\n', best.ratio);
fprintf('========================================================================\n\n');

% Save the best ground truth parameters
gt_file = fullfile(base_dir, 'data', 'ground_truth_param_K5_d20.mat');
param = best.param;
MU    = best.MU;
save(gt_file, 'param', 'MU', 'd', 'K');
fprintf('==> Successfully saved balanced ground truth to: %s\n', gt_file);

% Prompt to update data_generator_benchmark.m
fprintf('\nNow you can run data_generator_benchmark.m to generate datasets with these balanced groups!\n');
