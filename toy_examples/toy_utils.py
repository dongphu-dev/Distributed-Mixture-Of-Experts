"""
Shared utilities for Toy Example simulations, MoE EM fitting, DME reduction,
and Bokeh Vector PDF export (SVG -> Vector PDF via CairoSVG/svglib).
"""

import os
import time
import numpy as np
from scipy.optimize import minimize

from bokeh.plotting import figure
from bokeh.io import export_svg
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import cairosvg


def fit_moe_em(X, Y, K=2, max_iter=200, tol=1e-6, seed=42):
    """
    Fits a K-expert Gaussian Mixture-of-Experts with softmax gating via EM.
    """
    rng = np.random.RandomState(seed)
    n = len(Y)
    X_mat = np.column_stack([np.ones(n), X])

    order = np.argsort(X.ravel())
    splits = np.array_split(order, K)

    beta0 = np.zeros(K)
    Beta = np.zeros(K)
    sigma2 = np.zeros(K)

    for k in range(K):
        idx = splits[k]
        p = np.polyfit(X[idx].ravel(), Y[idx].ravel(), 1)
        beta0[k] = p[1]
        Beta[k] = p[0]
        sigma2[k] = max(float(np.var(Y[idx])), 0.05)

    alpha0 = np.zeros(K)
    Alpha = np.zeros(K)
    Alpha[0] = 1.0

    prev_ll = -np.inf

    for it in range(max_iter):
        # E-step
        H = alpha0 + X * Alpha
        expH = np.exp(H - np.max(H, axis=1, keepdims=True))
        pi = expH / np.sum(expH, axis=1, keepdims=True)
        pi = np.clip(pi, 1e-12, 1.0)

        means = beta0 + X * Beta
        log_phi = -0.5 * np.log(2 * np.pi * sigma2) - 0.5 * ((Y - means) ** 2) / sigma2
        log_joint = np.log(pi) + log_phi
        log_sum = np.logaddexp(log_joint[:, 0], log_joint[:, 1])[:, None]
        tau = np.exp(log_joint - log_sum)
        tau = np.clip(tau, 1e-12, 1.0)
        tau = tau / np.sum(tau, axis=1, keepdims=True)

        # M-step: Experts
        for k in range(K):
            w = tau[:, k]
            W = np.diag(w)
            XtWX = X_mat.T @ W @ X_mat + 1e-7 * np.eye(2)
            XtWY = X_mat.T @ (w * Y.ravel())
            coef = np.linalg.solve(XtWX, XtWY)
            beta0[k] = coef[0]
            Beta[k] = coef[1]

            diff = Y.ravel() - (beta0[k] + Beta[k] * X.ravel())
            sigma2[k] = np.sum(w * (diff ** 2)) / np.sum(w)
            sigma2[k] = max(float(sigma2[k]), 1e-4)

        # M-step: Gating parameters via Logistic Regression
        def nll_gate(params):
            a0, a1 = params
            h = a0 + a1 * X.ravel()
            log1p = np.logaddexp(0, h)
            return -np.sum(tau[:, 0] * h - log1p)

        res = minimize(nll_gate, [alpha0[0], Alpha[0]], method='BFGS')
        alpha0[0], Alpha[0] = res.x
        alpha0[1], Alpha[1] = 0.0, 0.0

        ll = np.sum(log_sum)
        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

    return {
        'beta0': beta0,
        'Beta': Beta,
        'sigma2': sigma2,
        'alpha0': alpha0,
        'Alpha': Alpha
    }


def dme_reduce(local_estimates, X_val, K=2, max_iter=100, tol=1e-6):
    """
    DME reduction algorithm aggregating M local MoE models into K target experts.
    """
    M = len(local_estimates)
    components = []
    for m in range(M):
        est = local_estimates[m]
        for k in range(K):
            components.append({
                'm': m,
                'k': k,
                'beta0': est['beta0'][k],
                'Beta': est['Beta'][k],
                'sigma2': est['sigma2'][k],
                'alpha0': est['alpha0'],
                'Alpha': est['Alpha']
            })

    MK = len(components)
    S = len(X_val)
    X_mat = np.column_stack([np.ones(S), X_val])

    # Evaluate local gates on X_val
    pi_local = np.zeros((S, MK))
    for idx, comp in enumerate(components):
        m = comp['m']
        est = local_estimates[m]
        H = est['alpha0'] + X_val * est['Alpha']
        expH = np.exp(H - np.max(H, axis=1, keepdims=True))
        pi_m = expH / np.sum(expH, axis=1, keepdims=True)
        pi_local[:, idx] = (1.0 / M) * pi_m[:, comp['k']]

    # Initialize target components from local fits
    red_beta0 = np.array([components[0]['beta0'], components[1]['beta0']])
    red_Beta = np.array([components[0]['Beta'], components[1]['Beta']])
    red_sigma2 = np.array([components[0]['sigma2'], components[1]['sigma2']])

    prev_cost = 1e12
    plan = np.zeros((S, MK, K))

    for it in range(max_iter):
        plan.fill(0.0)
        total_cost = 0.0

        for s in range(S):
            x_s = X_val[s, 0]
            cost_mk_j = np.zeros((MK, K))
            for idx, comp in enumerate(components):
                mu_src = comp['beta0'] + comp['Beta'] * x_s
                var_src = comp['sigma2']
                for j in range(K):
                    mu_tgt = red_beta0[j] + red_Beta[j] * x_s
                    var_tgt = red_sigma2[j]
                    kl = 0.5 * (np.log(var_tgt / var_src) + (var_src + (mu_src - mu_tgt) ** 2) / var_tgt - 1.0)
                    cost_mk_j[idx, j] = kl

                best_j = np.argmin(cost_mk_j[idx])
                plan[s, idx, best_j] = pi_local[s, idx]
                total_cost += pi_local[s, idx] * cost_mk_j[idx, best_j]

        # Closed-form target updates
        for j in range(K):
            denom_mat = np.zeros((2, 2))
            numer_vec = np.zeros(2)
            total_w = 0.0

            for s in range(S):
                x_vec = X_mat[s]
                xxT = np.outer(x_vec, x_vec)
                for idx, comp in enumerate(components):
                    w_s = plan[s, idx, j]
                    if w_s > 0:
                        mu_src = comp['beta0'] + comp['Beta'] * X_val[s, 0]
                        denom_mat += w_s * xxT
                        numer_vec += w_s * x_vec * mu_src
                        total_w += w_s

            if total_w > 1e-8:
                coef = np.linalg.solve(denom_mat + 1e-7 * np.eye(2), numer_vec)
                red_beta0[j] = coef[0]
                red_Beta[j] = coef[1]

                var_num = 0.0
                for s in range(S):
                    for idx, comp in enumerate(components):
                        w_s = plan[s, idx, j]
                        if w_s > 0:
                            mu_src = comp['beta0'] + comp['Beta'] * X_val[s, 0]
                            var_src = comp['sigma2']
                            mu_tgt = red_beta0[j] + red_Beta[j] * X_val[s, 0]
                            var_num += w_s * (var_src + (mu_src - mu_tgt) ** 2)
                red_sigma2[j] = max(var_num / total_w, 1e-4)

        if abs(total_cost - prev_cost) < tol:
            break
        prev_cost = total_cost

    # Fit target gate parameters
    target_probs = np.sum(plan, axis=1)
    target_probs = target_probs / np.sum(target_probs, axis=1, keepdims=True)

    def nll_gate(params):
        a0, a1 = params
        h = a0 + a1 * X_val.ravel()
        log1p = np.logaddexp(0, h)
        return -np.sum(target_probs[:, 0] * h - log1p)

    res = minimize(nll_gate, [0.0, 1.0], method='BFGS')
    red_alpha0 = np.array([res.x[0], 0.0])
    red_Alpha = np.array([res.x[1], 0.0])

    return {
        'beta0': red_beta0,
        'Beta': red_Beta,
        'sigma2': red_sigma2,
        'alpha0': red_alpha0,
        'Alpha': red_Alpha
    }


def get_headless_driver():
    """Returns a headless Chrome webdriver for Bokeh SVG export."""
    driver_path = ChromeDriverManager().install()
    opts = Options()
    opts.add_argument('--headless=new')
    opts.add_argument('--no-sandbox')
    opts.add_argument('--disable-gpu')
    opts.binary_location = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
    svc = Service(executable_path=driver_path)
    return webdriver.Chrome(service=svc, options=opts)


def export_bokeh_to_pdf(plot_obj, output_pdf_path, driver=None):
    """
    Saves a Bokeh figure as SVG and converts it to a 100% vector, crystal-clear PDF
    using CairoSVG (StackOverflow standard).
    """
    if hasattr(plot_obj, 'output_backend'):
        plot_obj.output_backend = "svg"
    if hasattr(plot_obj, 'toolbar_location'):
        plot_obj.toolbar_location = None

    close_driver = False
    if driver is None:
        driver = get_headless_driver()
        close_driver = True

    svg_temp = output_pdf_path.replace('.pdf', '.svg')
    try:
        export_svg(plot_obj, filename=svg_temp, webdriver=driver)
        cairosvg.svg2pdf(url=svg_temp, write_to=output_pdf_path)
        print(f"Exported Vector PDF: {output_pdf_path} ({os.path.getsize(output_pdf_path)} bytes)")
    finally:
        if os.path.exists(svg_temp):
            os.remove(svg_temp)
        if close_driver:
            driver.quit()
