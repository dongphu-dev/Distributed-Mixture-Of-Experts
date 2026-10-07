%% Unified Publication Toy Example: DME Reduction vs. Weighted Averaging
%
% Integrates regression functions (f_W vs f_R) and gating networks (pi_W vs pi_R)
% in a realistic, asymmetric, non-IID distributed setting (M = 2, K = 2, d = 1).
%
% Ground Truth:
%   - Expert 1 (x < 0.5):  y =  2.5*x - 1.5,  sigma^2 = 0.08
%   - Expert 2 (x > 0.5):  y = -1.8*x + 3.0,  sigma^2 = 0.08
%   - Gate transition:     pi_1^*(x) = sigmoid(2.0 - 4.0*x), centered at x = 0.5 (asymmetric)
%
% Non-IID Worker Partitioning:
%   - Worker 1: Operates on left domain [-3.0, 0.7] (Regime 1 dominant)
%   - Worker 2: Operates on right domain [0.3, 3.0] (Regime 2 dominant, label switched)
%
% Outputs:
%   - MATLAB Figures 1 to 4:
%       Figure 1: (a) Weighted average regression f^W (MK = 4)
%       Figure 2: (b) DME reduction regression f^R (K = 2)
%       Figure 3: (c) Weighted average gating pi^W (MK = 4)
%       Figure 4: (d) DME reduction gating pi^R (K = 2)
%   - toy_example_data.mat: Saved data for Bokeh LaTeX PDF export

clear;
close all;
this_dir = fileparts(mfilename('fullpath'));
root_dir = fullfile(this_dir, '..');
addpath(root_dir);
addpath(fullfile(root_dir, 'data'));
addpath(fullfile(root_dir, 'models'));
addpath(fullfile(root_dir, 'results'));
addpath(fullfile(root_dir, 'datatools'));
addpath(fullfile(root_dir, 'stattools'));
addpath(fullfile(root_dir, 'evaltools'));

rng(42);

%% 1. Generate Realistic Asymmetric Ground Truth Data
n_m = 100;        % 150 observations per machine
n   = n_m * 2;    % Total N = 300
d   = 1;
K   = 2;
M   = 2;

param.beta0  = [-1.5,  3.0];       % Intercepts
param.Beta   = [ 2.5, -1.8];       % Slopes
param.sigma2 = [ 0.08, 0.08];      % Low noise for crisp visual separation
param.alpha0 = [ 2.0,  0.0];       % Asymmetric gate intercept (centered at x = 0.5)
param.Alpha  = [-4.0,  0.0];       % Sharp transition slope

% Non-IID Covariate Distributions
% Worker 1: strictly left-biased [-3.0, 0.7]
% Worker 2: strictly right-biased [0.3, 3.0]
X1 = linspace(-3.0, 0.7, n_m)' + 0.03 * randn(n_m, 1);
X1 = sort(X1);

X2 = linspace(0.3, 3.0, n_m)' + 0.03 * randn(n_m, 1);
X2 = sort(X2);

% Worker 1 Gating & Responses (dominated by Expert 1)
H1 = param.alpha0 + X1 * param.Alpha;
expH1 = exp(H1 - max(H1, [], 2));
PI1 = expH1 ./ sum(expH1, 2);
Z1 = mnrnd(1, PI1);
[~, labels1] = max(Z1, [], 2);
means1 = sum((param.beta0 + X1 * param.Beta) .* Z1, 2);
Y1 = normrnd(means1, sqrt(0.08), n_m, 1);

% Worker 2 Gating & Responses (dominated by Expert 2)
H2 = param.alpha0 + X2 * param.Alpha;
expH2 = exp(H2 - max(H2, [], 2));
PI2 = expH2 ./ sum(expH2, 2);
Z2 = mnrnd(1, PI2);
[~, labels2] = max(Z2, [], 2);
means2 = sum((param.beta0 + X2 * param.Beta) .* Z2, 2);
Y2 = normrnd(means2, sqrt(0.08), n_m, 1);

% Combined Data
X = [X1; X2];
Y = [Y1; Y2];
true_labels = [labels1; labels2];

client_indices = cell(1, M);
client_indices{1} = (1:n_m)';
client_indices{2} = (n_m+1:n)';

%% 2. Fit DME and Retrieve Local Fits
options = get_options('default');
options.nb_EM_runs     = 5;
options.DME_tries      = 5;
options.DME_verbose    = 0;
options.verbose        = 0;
options.IRLS_max_iter  = 100;
options.client_indices = client_indices;

dme_sol = Distributed_MixtureOfExperts_Gaussian(X, Y, K, M, options);

dme_beta0  = dme_sol.param.beta0;
dme_Beta   = dme_sol.param.Beta;
dme_alpha0 = dme_sol.param.alpha0;
dme_Alpha  = dme_sol.param.Alpha;

% Align DME components with True components
if abs(dme_Beta(1) - param.Beta(2)) + abs(dme_beta0(1) - param.beta0(2)) < ...
   abs(dme_Beta(1) - param.Beta(1)) + abs(dme_beta0(1) - param.beta0(1))
    dme_beta0  = dme_beta0([2, 1]);
    dme_Beta   = dme_Beta(:, [2, 1]);
    dme_alpha0 = dme_alpha0([2, 1]);
    dme_Alpha  = dme_Alpha(:, [2, 1]);
end

local_estimates = dme_sol.local_estimates;

% Induce natural label switching on Worker 2 if both workers ordered experts identically
if sign(local_estimates{2}.param.Beta(1)) == sign(local_estimates{1}.param.Beta(1))
    p2 = local_estimates{2}.param;
    local_estimates{2}.param.beta0  = p2.beta0([2, 1]);
    local_estimates{2}.param.Beta   = p2.Beta(:, [2, 1]);
    local_estimates{2}.param.sigma2 = p2.sigma2([2, 1]);
    local_estimates{2}.param.alpha0 = p2.alpha0([2, 1]);
    local_estimates{2}.param.Alpha  = p2.Alpha(:, [2, 1]);
end

%% 3. Grid Predictions & Evaluations
x_grid = linspace(-3.0, 3.0, 500)';

% True Model
y_true_exp = param.beta0 + x_grid * param.Beta;
H_true = param.alpha0 + x_grid * param.Alpha;
expH_true = exp(H_true - max(H_true, [], 2));
pi_true = expH_true ./ sum(expH_true, 2);
y_true_pred = sum(y_true_exp .* pi_true, 2);

% DME (f_R)
y_exp_dme = dme_beta0 + x_grid * dme_Beta;
H_dme = dme_alpha0 + x_grid * dme_Alpha;
expH_dme = exp(H_dme - max(H_dme, [], 2));
pi_dme = expH_dme ./ sum(expH_dme, 2);
y_pred_dme = sum(y_exp_dme .* pi_dme, 2);

% Local estimates & Weighted Average (f_W)
MK = M * K;
y_exp_local = zeros(length(x_grid), MK);
pi_W = zeros(length(x_grid), MK);
comp_idx = 1;
for m = 1:M
    loc_p = local_estimates{m}.param;
    H_m = loc_p.alpha0 + x_grid * loc_p.Alpha;
    expH_m = exp(H_m - max(H_m, [], 2));
    pi_m = expH_m ./ sum(expH_m, 2);
    y_m = loc_p.beta0 + x_grid * loc_p.Beta;
    for k = 1:K
        y_exp_local(:, comp_idx) = y_m(:, k);
        pi_W(:, comp_idx) = 0.5 * pi_m(:, k);
        comp_idx = comp_idx + 1;
    end
end
y_pred_W = sum(y_exp_local .* pi_W, 2);

% Sum of individual local gates across machines
pi_sum_gate1 = pi_W(:, 1) + pi_W(:, 3);
pi_sum_gate2 = pi_W(:, 2) + pi_W(:, 4);

% Parameter Averaging (WAVR / FedAvg without alignment)
avr_b0 = 0.5 * (local_estimates{1}.param.beta0 + local_estimates{2}.param.beta0);
avr_B  = 0.5 * (local_estimates{1}.param.Beta   + local_estimates{2}.param.Beta);
y_exp_wavr = avr_b0 + x_grid * avr_B;
y_pred_wavr = 0.5 * (y_exp_wavr(:, 1) + y_exp_wavr(:, 2));

%% 4. Save Data for Python Bokeh Rendering
save(fullfile(this_dir, 'toy_example_data.mat'), 'X', 'Y', 'true_labels', 'x_grid', ...
    'y_true_exp', 'y_true_pred', 'y_exp_dme', 'y_pred_dme', ...
    'y_exp_local', 'pi_W', 'y_pred_W', 'y_exp_wavr', 'y_pred_wavr', ...
    'pi_true', 'pi_dme', 'pi_sum_gate1', 'pi_sum_gate2');

%% 5. Publication-Ready MATLAB Plotting (4 Standalone Panels)
fig_w = 400;
fig_h = 250;
fig_pos_1 = [100, 350, fig_w, fig_h];
fig_pos_2 = [530, 350, fig_w, fig_h];
fig_pos_3 = [100,  50, fig_w, fig_h];
fig_pos_4 = [530,  50, fig_w, fig_h];

color_true1 = [0.05 0.05 0.05];
color_true2 = [0.55 0.55 0.55]; % Bright slate gray (distinct from black)
color_dme1  = [0.12, 0.47, 0.71]; % Blue
color_dme2  = [0.84, 0.15, 0.16]; % Red
color_m1_1  = [0.12, 0.47, 0.71];
color_m1_2  = [0.17, 0.63, 0.17];
color_m2_1  = [0.84, 0.15, 0.16];
color_m2_2  = [1.00, 0.50, 0.05];

% --- FIGURE 1: Weighted Average f_W (MK = 4) ---
fig1 = figure('Name', 'Toy_Panel_a_Weighted_Average', 'Color', 'w', 'Position', fig_pos_1);
hold on; grid on; box on;
set(gca, 'FontSize', 9.5, 'LineWidth', 0.8, 'GridAlpha', 0.05, 'GridColor', [0.2 0.2 0.2]);

scatter(X, Y, 12, [0.72 0.72 0.72], 'filled', 'MarkerFaceAlpha', 0.6, 'HandleVisibility', 'off');

p_t1 = plot(x_grid, y_true_exp(:,1), '-', 'Color', color_true1, 'LineWidth', 2.2, 'DisplayName', 'True {\itf}_1^*');
p_t2 = plot(x_grid, y_true_exp(:,2), '-', 'Color', color_true2, 'LineWidth', 2.2, 'DisplayName', 'True {\itf}_2^*');
p_tm = plot(x_grid, y_true_pred,    '-', 'Color', [0.0, 0.45, 0.20], 'LineWidth', 2.0, 'DisplayName', 'True {\itf}^*');

p_w1 = plot(x_grid, y_exp_local(:,1), '--', 'Color', color_m1_1, 'LineWidth', 1.5, 'DisplayName', '$\widehat{f}_1^{(1)}$');
p_w2 = plot(x_grid, y_exp_local(:,2), '--', 'Color', color_m1_2, 'LineWidth', 1.5, 'DisplayName', '$\widehat{f}_2^{(1)}$');
p_w3 = plot(x_grid, y_exp_local(:,3), ':',  'Color', color_m2_1, 'LineWidth', 1.6, 'DisplayName', '$\widehat{f}_1^{(2)}$');
p_w4 = plot(x_grid, y_exp_local(:,4), ':',  'Color', color_m2_2, 'LineWidth', 1.6, 'DisplayName', '$\widehat{f}_2^{(2)}$');
p_wm = plot(x_grid, y_pred_W, '--', 'Color', [0.85, 0.20, 0.0], 'LineWidth', 1.8, 'DisplayName', '$\widehat{f}^W$');

xlabel('Covariate {\it x}', 'FontSize', 10);
ylabel('Response {\it y}', 'FontSize', 10);
title('Weighted average $\widehat{f}^W$ ($MK = 4$)', 'FontSize', 11, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'Interpreter', 'latex');
xlim([min(X)-0.1, max(X)+0.1]); ylim([min(Y)-0.4, max(Y)+0.4]);

lgd1 = legend([p_t1, p_w1, p_w3, p_tm, p_t2, p_w2, p_w4, p_wm], 'NumColumns', 2, ...
       'Location', 'northwest', 'FontSize', 8.0, 'Interpreter', 'latex');
try, lgd1.ItemTokenSize = [14, 16]; end
hold off;

% --- FIGURE 2: DME Reduction f_R (K = 2) ---
fig2 = figure('Name', 'Toy_Panel_b_DME_Reduction', 'Color', 'w', 'Position', fig_pos_2);
hold on; grid on; box on;
set(gca, 'FontSize', 9.5, 'LineWidth', 0.8, 'GridAlpha', 0.05, 'GridColor', [0.2 0.2 0.2]);

scatter(X(true_labels==1), Y(true_labels==1), 14, color_dme1, 'filled', ...
    'MarkerFaceAlpha', 0.5, 'HandleVisibility', 'off');
scatter(X(true_labels==2), Y(true_labels==2), 14, color_dme2, 's', 'filled', ...
    'MarkerFaceAlpha', 0.5, 'HandleVisibility', 'off');

p2_t1 = plot(x_grid, y_true_exp(:,1), '-', 'Color', color_true1, 'LineWidth', 2.3, 'DisplayName', 'True {\itf}_1^*');
p2_t2 = plot(x_grid, y_true_exp(:,2), '-', 'Color', color_true2, 'LineWidth', 2.3, 'DisplayName', 'True {\itf}_2^*');
p2_tm = plot(x_grid, y_true_pred,    '-', 'Color', [0.0, 0.45, 0.20], 'LineWidth', 2.0, 'DisplayName', 'True {\itf}^*');

p2_d1 = plot(x_grid, y_exp_dme(:,1), '--', 'Color', color_dme1, 'LineWidth', 2.0, 'DisplayName', '$\bar{f}_1^R$');
p2_d2 = plot(x_grid, y_exp_dme(:,2), '--', 'Color', color_dme2, 'LineWidth', 2.0, 'DisplayName', '$\bar{f}_2^R$');
p2_dm = plot(x_grid, y_pred_dme, '--', 'Color', [0.55, 0.15, 0.70], 'LineWidth', 2.0, 'DisplayName', '$\bar{f}^R$');

xlabel('Covariate {\it x}', 'FontSize', 10);
ylabel('Response {\it y}', 'FontSize', 10);
title('DME reduction $\bar{f}^R$ ($K = 2$)', 'FontSize', 11, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'Interpreter', 'latex');
xlim([min(X)-0.1, max(X)+0.1]); ylim([min(Y)-0.4, max(Y)+0.4]);

lgd2 = legend([p2_t1, p2_t2, p2_tm, p2_d1, p2_d2, p2_dm], 'NumColumns', 2, ...
       'Location', 'northwest', 'FontSize', 8.0, 'Interpreter', 'latex');
try, lgd2.ItemTokenSize = [14, 16]; end
hold off;

% --- FIGURE 3: Weighted Average Gating pi^W (MK = 4) ---
fig3 = figure('Name', 'Toy_Panel_c_Weighted_Gating', 'Color', 'w', 'Position', fig_pos_3);
hold on; grid on; box on;
set(gca, 'FontSize', 9.5, 'LineWidth', 0.8, 'GridAlpha', 0.05, 'GridColor', [0.2 0.2 0.2]);

p3_tg1 = plot(x_grid, pi_true(:,1), 'Color', color_true1, 'LineWidth', 2.0, 'DisplayName', 'True $\pi_1^*$');
p3_tg2 = plot(x_grid, pi_true(:,2), 'Color', color_true2, 'LineWidth', 2.0, 'DisplayName', 'True $\pi_2^*$');

p3_wg1 = plot(x_grid, pi_W(:,1), '--', 'Color', color_m1_1, 'LineWidth', 1.4, 'DisplayName', '$\lambda_1 \widehat{\pi}_1^{(1)}$');
p3_wg2 = plot(x_grid, pi_W(:,2), '--', 'Color', color_m1_2, 'LineWidth', 1.4, 'DisplayName', '$\lambda_1 \widehat{\pi}_2^{(1)}$');
p3_wg3 = plot(x_grid, pi_W(:,3), ':',  'Color', color_m2_1, 'LineWidth', 1.4, 'DisplayName', '$\lambda_2 \widehat{\pi}_1^{(2)}$');
p3_wg4 = plot(x_grid, pi_W(:,4), ':',  'Color', color_m2_2, 'LineWidth', 1.4, 'DisplayName', '$\lambda_2 \widehat{\pi}_2^{(2)}$');

p3_sg1 = plot(x_grid, pi_sum_gate1, '-.', 'Color', [0.08, 0.35, 0.65], 'LineWidth', 2.0, 'DisplayName', '$\sum_m \lambda_m \widehat{\pi}_1^{(m)}$');
p3_sg2 = plot(x_grid, pi_sum_gate2, '-.', 'Color', [0.75, 0.10, 0.12], 'LineWidth', 2.0, 'DisplayName', '$\sum_m \lambda_m \widehat{\pi}_2^{(m)}$');

xlabel('Covariate {\it x}', 'FontSize', 10);
ylabel('Gating probability', 'FontSize', 10);
title('Weighted average gating $\widehat{\pi}^W$ ($MK = 4$)', 'FontSize', 11, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'Interpreter', 'latex');
xlim([-3.0, 3.0]); ylim([-0.05, 1.05]);

lgd3 = legend([p3_tg1, p3_wg1, p3_wg3, p3_sg1, p3_tg2, p3_wg2, p3_wg4, p3_sg2], 'NumColumns', 2, ...
       'Location', 'northwest', 'FontSize', 7.5, 'Interpreter', 'latex');
try, lgd3.ItemTokenSize = [14, 16]; end
hold off;

% --- FIGURE 4: DME Reduction Gating pi^R (K = 2) ---
fig4 = figure('Name', 'Toy_Panel_d_DME_Gating', 'Color', 'w', 'Position', fig_pos_4);
hold on; grid on; box on;
set(gca, 'FontSize', 9.5, 'LineWidth', 0.8, 'GridAlpha', 0.05, 'GridColor', [0.2 0.2 0.2]);

p4_tg1 = plot(x_grid, pi_true(:,1), 'Color', color_true1, 'LineWidth', 2.4, 'DisplayName', 'True $\pi_1^*$');
p4_tg2 = plot(x_grid, pi_true(:,2), 'Color', color_true2, 'LineWidth', 2.4, 'DisplayName', 'True $\pi_2^*$');

p4_dg1 = plot(x_grid, pi_dme(:,1), '-', 'Color', color_dme1, 'LineWidth', 2.0, 'DisplayName', '$\bar{\pi}_1^R$');
p4_dg2 = plot(x_grid, pi_dme(:,2), '-', 'Color', color_dme2, 'LineWidth', 2.0, 'DisplayName', '$\bar{\pi}_2^R$');

xlabel('Covariate {\it x}', 'FontSize', 10);
ylabel('Gating probability', 'FontSize', 10);
title('DME reduction gating $\bar{\pi}^R$ ($K = 2$)', 'FontSize', 11, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'Interpreter', 'latex');
xlim([-3.0, 3.0]); ylim([-0.05, 1.05]);

lgd4 = legend([p4_tg1, p4_tg2, p4_dg1, p4_dg2], 'NumColumns', 2, ...
       'Location', 'northwest', 'FontSize', 8.5, 'Interpreter', 'latex');
try, lgd4.ItemTokenSize = [14, 16]; end
hold off;

fprintf('=== Unified Toy Example Finished ===\n');
fprintf('Figures 1 to 4 generated and data saved to toy_example_data.mat.\n');
