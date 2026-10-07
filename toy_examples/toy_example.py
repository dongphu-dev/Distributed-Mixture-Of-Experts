#!/usr/bin/env python3
"""
Unified Publication Toy Example: DME Reduction vs. Weighted Averaging
Generates 4 publication-ready panels as 100% vector PDFs with native LaTeX typography.

Panels:
  - Panel 1: Weighted average regression f^W (MK = 4)
  - Panel 2: DME reduction regression f^R (K = 2)
  - Panel 3: Weighted average gating pi^W (MK = 4)
  - Panel 4: DME reduction gating pi^R (K = 2)
  - Combined: 2x2 publication grid
"""

import os
import sys
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from toy_utils import fit_moe_em, dme_reduce


def load_or_generate_unified_data(data_path):
    if not os.path.exists(data_path):
        alt_path = os.path.join(os.path.dirname(data_path), "..", "toy_example_data.mat")
        if os.path.exists(alt_path):
            data_path = alt_path

    if os.path.exists(data_path):
        print(f"Loading MATLAB dataset from: {data_path}")
        mat = sio.loadmat(data_path)
        X = mat['X']
        Y = mat['Y']
        true_labels = mat['true_labels'].ravel()
        x_grid = mat['x_grid'].ravel()
        y_true_exp = mat['y_true_exp']
        y_true_pred = mat['y_true_pred'].ravel()
        y_exp_dme = mat['y_exp_dme']
        y_pred_dme = mat['y_pred_dme'].ravel()
        y_exp_local = mat['y_exp_local']
        pi_W = mat['pi_W']
        y_pred_W = mat['y_pred_W'].ravel()
        y_exp_wavr = mat['y_exp_wavr']
        y_pred_wavr = mat['y_pred_wavr'].ravel()
        pi_true = mat['pi_true']
        pi_dme = mat['pi_dme']
        pi_sum_gate1 = mat['pi_sum_gate1'].ravel()
        pi_sum_gate2 = mat['pi_sum_gate2'].ravel()
        return (X, Y, true_labels, x_grid, y_true_exp, y_true_pred, y_exp_dme, y_pred_dme,
                y_exp_local, pi_W, y_pred_W, y_exp_wavr, y_pred_wavr,
                pi_true, pi_dme, pi_sum_gate1, pi_sum_gate2)

    print("MATLAB data file not found, simulating unified data in Python...")
    rng = np.random.RandomState(42)
    n_m = 150
    K = 2
    M = 2

    # Asymmetric Ground Truth
    param_beta0 = np.array([-1.5, 3.0])
    param_Beta = np.array([2.5, -1.8])
    param_alpha0 = np.array([2.0, 0.0])
    param_Alpha = np.array([-4.0, 0.0])

    # Worker 1 (left domain)
    X1 = np.sort(np.linspace(-3.0, 0.7, n_m) + 0.03 * rng.randn(n_m))[:, None]
    H1 = param_alpha0 + X1 * param_Alpha
    P1 = np.exp(H1 - np.max(H1, axis=1, keepdims=True))
    P1 = P1 / np.sum(P1, axis=1, keepdims=True)
    z1 = (rng.rand(n_m) < P1[:, 0]).astype(int)
    Y1 = np.where(z1[:, None] == 1, param_beta0[0] + param_Beta[0]*X1, param_beta0[1] + param_Beta[1]*X1) + np.sqrt(0.08)*rng.randn(n_m, 1)

    # Worker 2 (right domain)
    X2 = np.sort(np.linspace(0.3, 3.0, n_m) + 0.03 * rng.randn(n_m))[:, None]
    H2 = param_alpha0 + X2 * param_Alpha
    P2 = np.exp(H2 - np.max(H2, axis=1, keepdims=True))
    P2 = P2 / np.sum(P2, axis=1, keepdims=True)
    z2 = (rng.rand(n_m) < P2[:, 0]).astype(int)
    Y2 = np.where(z2[:, None] == 1, param_beta0[0] + param_Beta[0]*X2, param_beta0[1] + param_Beta[1]*X2) + np.sqrt(0.08)*rng.randn(n_m, 1)

    X = np.vstack([X1, X2])
    Y = np.vstack([Y1, Y2])
    true_labels = np.concatenate([np.where(z1 == 1, 1, 2), np.where(z2 == 1, 1, 2)])

    loc1 = fit_moe_em(X1, Y1, K=K, seed=1)
    loc2 = fit_moe_em(X2, Y2, K=K, seed=2)

    # Label switching on Worker 2
    if np.sign(loc2['Beta'][0]) == np.sign(loc1['Beta'][0]):
        loc2['beta0'] = loc2['beta0'][::-1]
        loc2['Beta'] = loc2['Beta'][::-1]
        loc2['sigma2'] = loc2['sigma2'][::-1]
        loc2['alpha0'] = loc2['alpha0'][::-1]
        loc2['Alpha'] = loc2['Alpha'][::-1]

    dme_sol = dme_reduce([loc1, loc2], X, K=K)
    if abs(dme_sol['Beta'][0] - param_Beta[1]) + abs(dme_sol['beta0'][0] - param_beta0[1]) < \
       abs(dme_sol['Beta'][0] - param_Beta[0]) + abs(dme_sol['beta0'][0] - param_beta0[0]):
        dme_sol['beta0'] = dme_sol['beta0'][::-1]
        dme_sol['Beta'] = dme_sol['Beta'][::-1]
        dme_sol['alpha0'] = dme_sol['alpha0'][::-1]
        dme_sol['Alpha'] = dme_sol['Alpha'][::-1]

    x_grid = np.linspace(-3.0, 3.0, 500)
    y_true_exp = np.column_stack([param_beta0[0] + param_Beta[0]*x_grid, param_beta0[1] + param_Beta[1]*x_grid])
    H_t = param_alpha0 + x_grid[:, None] * param_Alpha
    expH_t = np.exp(H_t - np.max(H_t, axis=1, keepdims=True))
    pi_true = expH_t / np.sum(expH_t, axis=1, keepdims=True)
    y_true_pred = np.sum(y_true_exp * pi_true, axis=1)

    y_exp_dme = np.column_stack([dme_sol['beta0'][0] + dme_sol['Beta'][0]*x_grid, dme_sol['beta0'][1] + dme_sol['Beta'][1]*x_grid])
    H_d = dme_sol['alpha0'] + x_grid[:, None] * dme_sol['Alpha']
    expH_d = np.exp(H_d - np.max(H_d, axis=1, keepdims=True))
    pi_dme = expH_d / np.sum(expH_d, axis=1, keepdims=True)
    y_pred_dme = np.sum(y_exp_dme * pi_dme, axis=1)

    MK = M * K
    y_exp_local = np.zeros((len(x_grid), MK))
    pi_W = np.zeros((len(x_grid), MK))
    comp_idx = 0
    local_estimates = [loc1, loc2]
    for m in range(M):
        loc_p = local_estimates[m]
        H_m = loc_p['alpha0'] + x_grid[:, None] * loc_p['Alpha']
        expH_m = np.exp(H_m - np.max(H_m, axis=1, keepdims=True))
        pi_m = expH_m / np.sum(expH_m, axis=1, keepdims=True)
        for k in range(K):
            y_exp_local[:, comp_idx] = (loc_p['beta0'][k] + loc_p['Beta'][k]*x_grid)
            pi_W[:, comp_idx] = 0.5 * pi_m[:, k]
            comp_idx += 1
    y_pred_W = np.sum(y_exp_local * pi_W, axis=1)

    pi_sum_gate1 = pi_W[:, 0] + pi_W[:, 2]
    pi_sum_gate2 = pi_W[:, 1] + pi_W[:, 3]

    avr_b0 = 0.5 * (loc1['beta0'] + loc2['beta0'])
    avr_B  = 0.5 * (loc1['Beta'] + loc2['Beta'])
    y_exp_wavr = np.column_stack([avr_b0[0] + avr_B[0]*x_grid, avr_b0[1] + avr_B[1]*x_grid])
    y_pred_wavr = 0.5 * (y_exp_wavr[:, 0] + y_exp_wavr[:, 1])

    return (X, Y, true_labels, x_grid, y_true_exp, y_true_pred, y_exp_dme, y_pred_dme,
            y_exp_local, pi_W, y_pred_W, y_exp_wavr, y_pred_wavr,
            pi_true, pi_dme, pi_sum_gate1, pi_sum_gate2)


def render_panel_a(ax, data):
    (X, Y, true_labels, x_grid, y_true_exp, y_true_pred, y_exp_dme, y_pred_dme,
     y_exp_local, pi_W, y_pred_W, y_exp_wavr, y_pred_wavr,
     pi_true, pi_dme, pi_sum_gate1, pi_sum_gate2) = data

    c_t1 = '#0d0d0d'
    c_t2 = '#7f8c8d'  # Bright slate gray (distinct from black)
    c_tm = '#007333'
    c_w1 = '#1f77b4'
    c_w2 = '#2ca02c'
    c_w3 = '#d62728'
    c_w4 = '#ff7f0e'
    c_wm = '#d93300'

    # Partition data points into 4 local expert components
    n_m = len(X) // 2
    mask_m1 = np.zeros(len(X), dtype=bool)
    mask_m1[:n_m] = True
    mask_m2 = ~mask_m1

    # Worker 1: expert 1 (label 1), expert 2 (label 2)
    # Worker 2: label switched: expert 1 (label 2), expert 2 (label 1)
    mask_w1 = mask_m1 & (true_labels == 1)
    mask_w2 = mask_m1 & (true_labels == 2)
    mask_w3 = mask_m2 & (true_labels == 2)
    mask_w4 = mask_m2 & (true_labels == 1)

    ax.scatter(X[mask_w1], Y[mask_w1], s=14, color=c_w1, alpha=0.5, edgecolors='none', label='_nolegend_')
    ax.scatter(X[mask_w2], Y[mask_w2], s=14, color=c_w2, marker='s', alpha=0.5, edgecolors='none', label='_nolegend_')
    ax.scatter(X[mask_w3], Y[mask_w3], s=14, color=c_w3, marker='^', alpha=0.5, edgecolors='none', label='_nolegend_')
    ax.scatter(X[mask_w4], Y[mask_w4], s=14, color=c_w4, marker='D', alpha=0.5, edgecolors='none', label='_nolegend_')

    p_t1, = ax.plot(x_grid, y_true_exp[:, 0], color=c_t1, lw=2.2)
    p_w1, = ax.plot(x_grid, y_exp_local[:, 0], color=c_w1, lw=1.5, ls='--')
    p_w3, = ax.plot(x_grid, y_exp_local[:, 2], color=c_w3, lw=1.6, ls=':')
    p_tm, = ax.plot(x_grid, y_true_pred, color=c_tm, lw=2.0)
    p_t2, = ax.plot(x_grid, y_true_exp[:, 1], color=c_t2, lw=2.2)
    p_w2, = ax.plot(x_grid, y_exp_local[:, 1], color=c_w2, lw=1.5, ls='--')
    p_w4, = ax.plot(x_grid, y_exp_local[:, 3], color=c_w4, lw=1.6, ls=':')
    p_wm, = ax.plot(x_grid, y_pred_W, color=c_wm, lw=1.8, ls='--')

    ax.set_xlabel(r'Covariate $\mathbf{x}$')
    ax.set_ylabel(r'Response $y$')
    ax.set_title(r'Weighted average $\widehat{f}^W$ ($MK = 4$)', fontweight='medium', pad=8, loc='center')
    ax.set_xlim([np.min(X)-0.1, np.max(X)+0.1])
    ax.set_ylim([np.min(Y)-0.4, np.max(Y)+0.4])
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.legend([p_t1, p_w1, p_w3, p_tm, p_t2, p_w2, p_w4, p_wm],
              [r'$f_1^*$', r'$\widehat{f}_1^{(1)}$', r'$\widehat{f}_1^{(2)}$', r'$f^*$',
               r'$f_2^*$', r'$\widehat{f}_2^{(1)}$', r'$\widehat{f}_2^{(2)}$', r'$\widehat{f}^W$'],
              loc='upper left', ncol=2, framealpha=0.9, fontsize=10, handlelength=1.5, columnspacing=0.8)


def render_panel_b(ax, data):
    (X, Y, true_labels, x_grid, y_true_exp, y_true_pred, y_exp_dme, y_pred_dme,
     y_exp_local, pi_W, y_pred_W, y_exp_wavr, y_pred_wavr,
     pi_true, pi_dme, pi_sum_gate1, pi_sum_gate2) = data

    c_t1 = '#0d0d0d'
    c_t2 = '#7f8c8d'  # Bright slate gray
    c_tm = '#007333'
    c_d1 = '#1f77b4'
    c_d2 = '#d62728'
    c_dm = '#8c26b3'

    mask1 = (true_labels == 1)
    mask2 = (true_labels == 2)
    ax.scatter(X[mask1], Y[mask1], s=14, color=c_d1, alpha=0.5, edgecolors='none', label='_nolegend_')
    ax.scatter(X[mask2], Y[mask2], s=14, color=c_d2, marker='s', alpha=0.5, edgecolors='none', label='_nolegend_')

    p2_t1, = ax.plot(x_grid, y_true_exp[:, 0], color=c_t1, lw=2.3)
    p2_t2, = ax.plot(x_grid, y_true_exp[:, 1], color=c_t2, lw=2.3)
    p2_tm, = ax.plot(x_grid, y_true_pred, color=c_tm, lw=2.0)
    p2_d1, = ax.plot(x_grid, y_exp_dme[:, 0], color=c_d1, lw=2.0, ls='--')
    p2_d2, = ax.plot(x_grid, y_exp_dme[:, 1], color=c_d2, lw=2.0, ls='--')
    p2_dm, = ax.plot(x_grid, y_pred_dme, color=c_dm, lw=2.0, ls='--')

    ax.set_xlabel(r'Covariate $\mathbf{x}$')
    ax.set_ylabel(r'Response $y$')
    ax.set_title(r'Reduction $\bar{f}^{\:R}$ ($K = 2$)', fontweight='medium', pad=8, loc='center')
    ax.set_xlim([np.min(X)-0.1, np.max(X)+0.1])
    ax.set_ylim([np.min(Y)-0.4, np.max(Y)+0.4])
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.legend([p2_t1, p2_d1, p2_tm, p2_t2, p2_d2, p2_dm],
              [r'$f_1^*$', r'$\bar{f}_1^{\:R}$', r'$f^*$',
               r'$f_2^*$', r'$\bar{f}_2^{\:R}$', r'$\bar{f}^{\:R}$'],
              loc='upper left', ncol=2, framealpha=0.9, fontsize=11, handlelength=1.5, columnspacing=0.8)


def render_panel_c(ax, data):
    (X, Y, true_labels, x_grid, y_true_exp, y_true_pred, y_exp_dme, y_pred_dme,
     y_exp_local, pi_W, y_pred_W, y_exp_wavr, y_pred_wavr,
     pi_true, pi_dme, pi_sum_gate1, pi_sum_gate2) = data

    c_t1 = '#0d0d0d'
    c_t2 = '#7f8c8d'
    c_w1 = '#1f77b4'
    c_w2 = '#2ca02c'
    c_w3 = '#d62728'
    c_w4 = '#ff7f0e'
    c_sg1 = '#1459a6'
    c_sg2 = '#bf1a1f'

    p3_tg1, = ax.plot(x_grid, pi_true[:, 0], color=c_t1, lw=2.0)
    p3_wg1, = ax.plot(x_grid, pi_W[:, 0], color=c_w1, lw=1.4, ls='--')
    p3_wg3, = ax.plot(x_grid, pi_W[:, 2], color=c_w3, lw=1.4, ls=':')
    p3_sg1, = ax.plot(x_grid, pi_sum_gate1, color=c_sg1, lw=2.0, dashes=[3, 1.2, 1, 1.2])
    p3_tg2, = ax.plot(x_grid, pi_true[:, 1], color=c_t2, lw=2.0)
    p3_wg2, = ax.plot(x_grid, pi_W[:, 1], color=c_w2, lw=1.4, ls='--')
    p3_wg4, = ax.plot(x_grid, pi_W[:, 3], color=c_w4, lw=1.4, ls=':')
    p3_sg2, = ax.plot(x_grid, pi_sum_gate2, color=c_sg2, lw=2.0, dashes=[3, 1.2, 1, 1.2])

    ax.set_xlabel(r'Covariate $\mathbf{x}$')
    ax.set_ylabel(r'Gating probability')
    ax.set_title(r'Weighted average gating $\widehat{\pi}^W$ ($MK = 4$)', fontweight='medium', pad=8, loc='center')
    ax.set_xlim([-3.0, 3.0])
    ax.set_ylim([-0.05, 1.05])
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.legend([p3_tg1, p3_wg1, p3_wg3, p3_sg1, p3_tg2, p3_wg2, p3_wg4, p3_sg2],
              [r'$\pi_1^*$', r'$\lambda_1 \widehat{\pi}_1^{(1)}$', r'$\lambda_2 \widehat{\pi}_1^{(2)}$', r'$\sum_m \lambda_m \widehat{\pi}_1^{(m)}$',
               r'$\pi_2^*$', r'$\lambda_1 \widehat{\pi}_2^{(1)}$', r'$\lambda_2 \widehat{\pi}_2^{(2)}$', r'$\sum_m \lambda_m \widehat{\pi}_2^{(m)}$'],
              bbox_to_anchor=(-0.01, 1.02), loc='upper left', ncol=2, framealpha=0.98, fontsize=8.5, handlelength=1.45, columnspacing=0.6)


def render_panel_d(ax, data):
    (X, Y, true_labels, x_grid, y_true_exp, y_true_pred, y_exp_dme, y_pred_dme,
     y_exp_local, pi_W, y_pred_W, y_exp_wavr, y_pred_wavr,
     pi_true, pi_dme, pi_sum_gate1, pi_sum_gate2) = data

    c_t1 = '#0d0d0d'
    c_t2 = '#7f8c8d'
    c_d1 = '#1f77b4'
    c_d2 = '#d62728'

    p4_tg1, = ax.plot(x_grid, pi_true[:, 0], color=c_t1, lw=2.4)
    p4_tg2, = ax.plot(x_grid, pi_true[:, 1], color=c_t2, lw=2.4)
    p4_dg1, = ax.plot(x_grid, pi_dme[:, 0], color=c_d1, lw=2.0)
    p4_dg2, = ax.plot(x_grid, pi_dme[:, 1], color=c_d2, lw=2.0)

    ax.set_xlabel(r'Covariate $\mathbf{x}$')
    ax.set_ylabel(r'Gating probability')
    ax.set_title(r'Reduction gating $\bar{\pi}^{\:R}$ ($K = 2$)', fontweight='medium', pad=8, loc='center')
    ax.set_xlim([-3.0, 3.0])
    ax.set_ylim([-0.05, 1.05])
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.legend([p4_tg1, p4_tg2, p4_dg1, p4_dg2],
              [r'$\pi_1^*$', r'$\pi_2^*$', r'$\bar{\pi}_1^{\:R}$', r'$\bar{\pi}_2^{\:R}$'],
              bbox_to_anchor=(0.0, 0.92), loc='upper left', ncol=2, framealpha=0.9, fontsize=11, handlelength=1.5, columnspacing=0.8)


def run_unified_toy_example():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(cur_dir, "toy_example_data.mat")
    data = load_or_generate_unified_data(data_path)

    # Configure publication matplotlib style
    plt.rcParams.update({
        'font.size': 10.5,
        'font.family': 'sans-serif',
        'mathtext.fontset': 'cm',
        'axes.labelsize': 14,
        'axes.titlesize': 15,
        'figure.dpi': 300,
        'pdf.fonttype': 42,
        'ps.fonttype': 42
    })

    fig_size = (4.8, 3.2)
    # Cấu hình lề ngoài (padding) khi xuất PDF:
    # 0.0: Sát mép 100% (không có lề thừa xung quanh)
    # Có thể tăng lên 0.01, 0.02,... (đơn vị inch) nếu muốn thêm khoảng cách nhỏ
    custom_pad_inches = 0.03

    # 1. Panel A: Weighted Average Regression
    fig, ax = plt.subplots(figsize=fig_size)
    render_panel_a(ax, data)
    plt.tight_layout()
    path_a = os.path.join(cur_dir, "toy_panel_a_weighted_average.pdf")
    fig.savefig(path_a, bbox_inches='tight', pad_inches=custom_pad_inches)
    plt.close(fig)
    print(f"Exported Vector PDF: {path_a}")

    # 2. Panel B: DME Reduction Regression
    fig, ax = plt.subplots(figsize=fig_size)
    render_panel_b(ax, data)
    plt.tight_layout()
    path_b = os.path.join(cur_dir, "toy_panel_b_dme_reduction.pdf")
    fig.savefig(path_b, bbox_inches='tight', pad_inches=custom_pad_inches)
    plt.close(fig)
    print(f"Exported Vector PDF: {path_b}")

    # 3. Panel C: Weighted Average Gating
    fig, ax = plt.subplots(figsize=fig_size)
    render_panel_c(ax, data)
    plt.tight_layout()
    path_c = os.path.join(cur_dir, "toy_panel_c_weighted_gating.pdf")
    fig.savefig(path_c, bbox_inches='tight', pad_inches=custom_pad_inches)
    plt.close(fig)
    print(f"Exported Vector PDF: {path_c}")

    # 4. Panel D: DME Reduction Gating
    fig, ax = plt.subplots(figsize=fig_size)
    render_panel_d(ax, data)
    plt.tight_layout()
    path_d = os.path.join(cur_dir, "toy_panel_d_dme_gating.pdf")
    fig.savefig(path_d, bbox_inches='tight', pad_inches=custom_pad_inches)
    plt.close(fig)
    print(f"Exported Vector PDF: {path_d}")

    # 5. 2x2 Combined Publication Grid
    fig, axs = plt.subplots(2, 2, figsize=(9.6, 6.4))
    render_panel_a(axs[0, 0], data)
    render_panel_b(axs[0, 1], data)
    render_panel_c(axs[1, 0], data)
    render_panel_d(axs[1, 1], data)
    plt.tight_layout()
    path_grid = os.path.join(cur_dir, "toy_example_unified_2x2.pdf")
    fig.savefig(path_grid, bbox_inches='tight', pad_inches=custom_pad_inches)
    plt.close(fig)
    print(f"Exported Vector PDF: {path_grid}")

    print("\nAll 4 Vector Panels and 2x2 Grid PDF generated successfully!")


if __name__ == "__main__":
    run_unified_toy_example()
