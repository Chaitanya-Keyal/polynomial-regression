"""P4: Bayesian linear regression with fixed hyperparameters (degree 3, alpha = 0.05, beta = 100).

Posterior N(w | m_N, S_N) with A = S_N^-1 = alpha I + beta Phi^T Phi and m_N = beta S_N Phi^T t;
predictive N(m(x), s^2(x)) with m(x) = phi(x)^T m_N and s^2(x) = 1/beta + phi(x)^T S_N phi(x).
S_N is never formed: every product with it is a linear solve with A.
"""

import matplotlib.pyplot as plt
import numpy as np

from src.common.bases import monomial
from src.common.data import P4_SIZES, load
from src.common.linalg import ridge
from src.common.plots import BLUE, INK_2, ORANGE, RAMP, X_DENSE, points, truth

D = 3
ALPHA = 0.05  # prior precision
BETA = 100.0  # noise precision, 1 / sigma^2 with sigma^2 = 0.01
LAM = ALPHA / BETA  # the equivalent ridge penalty
Z = 1.96  # 95% interval
MAIN_N = 10  # the training set for the predictive plot and the equivalence check
FLOOR = 2 * Z / np.sqrt(BETA)  # interval width from observation noise alone


def posterior(Phi, t):
    """m_N and the posterior precision A = S_N^-1."""
    A = ALPHA * np.eye(Phi.shape[1]) + BETA * Phi.T @ Phi
    return np.linalg.solve(A, BETA * Phi.T @ t), A


def predictive(x, m, A):
    """m(x) and s^2(x) = 1/beta + phi(x)^T S_N phi(x)."""
    G = monomial(x, D)
    return G @ m, 1 / BETA + (G.T * np.linalg.solve(A, G.T)).sum(axis=0)


def compute():
    fits = {}
    for n in P4_SIZES:
        x, t = load(f"p4_train_N{n}.csv")
        m, A = posterior(monomial(x, D), t)
        mean, var = predictive(X_DENSE, m, A)
        fits[n] = {"x": x, "t": t, "m": m, "mean": mean, "var": var}

    main = fits[MAIN_N]
    w_map = ridge(monomial(main["x"], D), main["t"], LAM)
    numbers = {
        "m_N": main["m"].tolist(),
        "w_map": w_map.tolist(),
        "w_map_minus_m_N": float(np.linalg.norm(w_map - main["m"])),
        "width": {
            n: {
                "mean": float(2 * Z * np.sqrt(f["var"]).mean()),
                "max": float(2 * Z * np.sqrt(f["var"]).max()),
                # the share of s^2(x) that is parameter uncertainty, on average over x
                "param_share": float((1 - 1 / (BETA * f["var"])).mean()),
            }
            for n, f in fits.items()
        },
    }
    return {"numbers": numbers, "fits": fits, "map_curve": monomial(X_DENSE, D) @ w_map}


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
        ax.plot(X_DENSE, 2 * Z * np.sqrt(f["var"]), color=color, label=f"N = {n}")
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
        title="95% predictive interval width vs. x",
    )
    # a line chart, so no zero baseline: start just under the floor to show the gaps
    ax.set_ylim(FLOOR - 0.03, None)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=5, fontsize=7.5)
    return fig


def figures(res):
    return {"p4_predictive": predictive_figure(res), "p4_width": width_figure(res)}
