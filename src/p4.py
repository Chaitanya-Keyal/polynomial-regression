"""P4: Bayesian linear regression with fixed hyperparameters (degree 3, alpha = 0.05, beta = 100).

Posterior N(w | m_N, S_N) with S_N^-1 = alpha I + beta Phi^T Phi and m_N = beta S_N Phi^T t;
predictive N(m(x), s^2(x)) with m(x) = phi(x)^T m_N and s^2(x) = 1/beta + phi(x)^T S_N phi(x).
S_N^-1 is factored as L L^T (Cholesky), so S_N itself is never formed: m_N comes from two
triangular solves and phi^T S_N phi = ||L^-1 phi||^2.

python -m src.run p4
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.common.bases import monomial
from src.common.data import DATA_DIR, P4_SIZES, load_xy, p4_train
from src.common.linalg import ridge
from src.common.plots import BLUE, INK_2, ORANGE, RAMP, X_DENSE, end_label, points, truth

D = 3
ALPHA = 0.05  # prior precision
BETA = 100.0  # noise precision, 1 / sigma^2 with sigma^2 = 0.01
LAM = ALPHA / BETA  # the equivalent ridge penalty
Z = 1.96  # 95% interval
MAIN_N = 10  # the training set for the predictive plot and the equivalence check
FLOOR = 2 * Z / np.sqrt(BETA)  # interval width from observation noise alone


def posterior(Phi, t, alpha=ALPHA, beta=BETA):
    """m_N and the Cholesky factor L of the posterior precision S_N^-1 = L L^T."""
    L = np.linalg.cholesky(alpha * np.eye(Phi.shape[1]) + beta * Phi.T @ Phi)
    m = np.linalg.solve(L.T, np.linalg.solve(L, beta * Phi.T @ t))
    return m, L


def predictive(x, m, L, beta=BETA):
    """Predictive mean m(x) and the two parts of s^2(x): noise 1/beta, and the parameter
    uncertainty phi(x)^T S_N phi(x) = ||L^-1 phi(x)||^2."""
    Phi = monomial(x, D)
    param_var = (np.linalg.solve(L, Phi.T) ** 2).sum(axis=0)
    return Phi @ m, 1 / beta + param_var, param_var


def compute(data_dir=DATA_DIR):
    fits = {}
    for n in P4_SIZES:
        x, t = load_xy(Path(data_dir) / p4_train(n))
        m, L = posterior(monomial(x, D), t)
        mean, var, param_var = predictive(X_DENSE, m, L)
        fits[n] = {"x": x, "t": t, "m": m, "mean": mean, "var": var, "param_var": param_var}

    main = fits[MAIN_N]
    w_map = ridge(monomial(main["x"], D), main["t"], LAM)
    map_curve = monomial(X_DENSE, D) @ w_map

    numbers = {
        "alpha": ALPHA,
        "beta": BETA,
        "lam": LAM,
        "degree": D,
        "main_N": MAIN_N,
        "m_N": main["m"].tolist(),
        "w_map": w_map.tolist(),
        "w_map_minus_m_N": float(np.linalg.norm(w_map - main["m"])),
        "max_curve_gap": float(np.abs(map_curve - main["mean"]).max()),
        "noise_floor_width": float(FLOOR),
        "width": {
            n: {
                "N": len(f["x"]),
                "mean": float(2 * Z * np.sqrt(f["var"]).mean()),
                "min": float(2 * Z * np.sqrt(f["var"]).min()),
                "max": float(2 * Z * np.sqrt(f["var"]).max()),
                # share of the predictive variance that is parameter uncertainty, on average
                "param_share": float((f["param_var"] / f["var"]).mean()),
            }
            for n, f in fits.items()
        },
    }
    return {"numbers": numbers, "fits": fits, "map_curve": map_curve}


def predictive_figure(res):
    f = res["fits"][MAIN_N]
    sd = np.sqrt(f["var"])
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.fill_between(
        X_DENSE,
        f["mean"] - Z * sd,
        f["mean"] + Z * sd,
        color=BLUE,
        alpha=0.15,
        linewidth=0,
        label="95% predictive interval",
    )
    truth(ax)
    ax.plot(X_DENSE, f["mean"], color=BLUE, label="predictive mean m(x)")
    ax.plot(
        X_DENSE,
        res["map_curve"],
        color=ORANGE,
        linestyle=(0, (4, 3)),
        label=r"MAP / ridge fit, $\lambda = \alpha/\beta$",
    )
    points(ax, f["x"], f["t"])
    ax.set(
        xlabel="x",
        ylabel="t",
        title=rf"Degree {D}, N = {MAIN_N}, $\alpha$ = {ALPHA}, $\beta$ = {BETA:g}",
    )
    ax.set_ylim(-1.6, 1.6)
    ax.legend(loc="upper right", fontsize=7.5)
    return fig


def width_figure(res):
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for (n, f), color in zip(res["fits"].items(), RAMP, strict=True):
        width = 2 * Z * np.sqrt(f["var"])
        ax.plot(X_DENSE, width, color=color, label=f"N = {n}")
        end_label(ax, X_DENSE, width, f"N = {n}")
    ax.axhline(
        FLOOR,
        color=INK_2,
        linestyle="--",
        linewidth=1,
        label=r"noise floor $2 \times 1.96\, \beta^{-1/2}$",
    )
    ax.set(
        xlabel="x",
        ylabel=r"interval width $2 \times 1.96\, s(x)$",
        title="95% predictive interval width vs. N",
    )
    # a line chart, so no zero baseline: start just under the floor to show the gaps
    ax.set_ylim(FLOOR - 0.03, None)
    ax.set_xlim(0, 1.1)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=5, fontsize=7.5)
    return fig


def figures(res):
    return {"p4_predictive": predictive_figure(res), "p4_width": width_figure(res)}
