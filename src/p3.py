"""P3: L1-regularised regression (Lasso) by subgradient descent in the shifted Legendre basis.

E(w) = 1/2 ||Phi w - t||^2 + lam ||w||_1 with Phi[n, j] = P_j(2 x_n - 1), j = 0..9. Every
reported quantity comes from the final iterate; the best-so-far E is only plotted
(assignment, P3 implementation note).
"""

import matplotlib.pyplot as plt
import numpy as np

from src.common import fmt
from src.common.bases import legendre
from src.common.data import TEST, TRAIN, load
from src.common.linalg import ridge
from src.common.metrics import lasso_objective, rmse
from src.common.plots import BLUE, INK_2, ORANGE, X_DENSE, points, rmse_vs_lambda, truth

D = 9
ETA = 0.05
ITERS = 20_000
LAMS = np.logspace(-4, 0, 21)  # the grid for every sweep
CHECK_LAM = 0.1  # deliverable 1
# The exact Lasso solution at lam = 0.1 (coordinate descent, rounded to 3 decimals; assignment)
E_REF = 0.2625
W_REF = np.array([0, -0.928, 0.045, 1.117, 0.032, -0.201, 0, -0.026, 0, 0])
SMALL = 0.01  # the |w_j| < 0.01 count in the Lasso vs. Ridge table


def near_zero(w, lam):
    """The "≈ 0" rule: with a constant step, coefficients chatter around zero with amplitude
    about ETA * lam, so |w_j| < 2 ETA lam counts as zero."""
    return np.abs(w) < 2 * ETA * lam


def subgradient_descent(Phi, t, lam):
    """ITERS steps of w <- w - ETA (Phi^T (Phi w - t) + lam sign(w)) from w = 0, with
    sign(0) = 0. Returns the final w and E after every iteration (index 0 is the start)."""
    w = np.zeros(Phi.shape[1])
    objs = [lasso_objective(Phi, w, t, lam)]
    for _ in range(ITERS):
        w = w - ETA * (Phi.T @ (Phi @ w - t) + lam * np.sign(w))
        objs.append(lasso_objective(Phi, w, t, lam))
    return w, np.array(objs)


def optimality(Phi, t, w, lam):
    """The optimality conditions with r = t - Phi w: Phi_j^T r = lam sign(w_j) where w_j != 0,
    |Phi_j^T r| <= lam where w_j = 0 (here: where w_j is ≈ 0)."""
    c = Phi.T @ (t - Phi @ w)
    zero = near_zero(w, lam)
    return {
        "max_active_violation": float(np.max(np.abs(c - lam * np.sign(w))[~zero], initial=0)),
        "max_zero_ratio": float(np.max(np.abs(c[zero]) / lam, initial=0)),
        "n_active": int((~zero).sum()),
        "n_zero": int(zero.sum()),
    }


def compute():
    x, t = load(TRAIN)
    x_test, t_test = load(TEST)
    Phi, Phi_test = legendre(x, D), legendre(x_test, D)

    w_check, objs_check = subgradient_descent(Phi, t, CHECK_LAM)
    sweep = []
    for lam in LAMS:
        w, _ = subgradient_descent(Phi, t, lam)
        sweep.append(
            {
                "lam": float(lam),
                "w": w.tolist(),
                "train_rmse": rmse(Phi @ w, t),
                "test_rmse": rmse(Phi_test @ w, t_test),
                "zero_idx": np.flatnonzero(near_zero(w, lam)).tolist(),
            }
        )
    best = min(sweep, key=lambda s: s["test_rmse"])

    ridge_ws = [ridge(Phi, t, lam) for lam in LAMS]
    i_ridge = int(np.argmin([rmse(Phi_test @ w, t_test) for w in ridge_ws]))
    table = {
        name: {
            "lam": float(lam),
            "test_rmse": rmse(Phi_test @ w, t_test),
            "w_norm": float(np.linalg.norm(w)),
            "n_small": int((np.abs(w) < SMALL).sum()),
            "w": w.tolist(),
        }
        for name, lam, w in (
            ("lasso", best["lam"], np.array(best["w"])),
            ("ridge", LAMS[i_ridge], ridge_ws[i_ridge]),
        )
    }
    numbers = {
        "cond_PtP": float(np.linalg.cond(Phi.T @ Phi)),
        "check": {
            "E_final": float(objs_check[-1]),
            "w": w_check.tolist(),
            "w_minus_w_ref": float(np.linalg.norm(w_check - W_REF)),
            "optimality": optimality(Phi, t, w_check, CHECK_LAM),
        },
        "sweep": sweep,
        "zero_idx_star": best["zero_idx"],
        "table": table,
    }
    return {"numbers": numbers, "x": x, "t": t, "objs_check": objs_check}


def convergence_figure(res):
    """Left, the deliverable: E and the best E so far. Right, both minus the reference E*,
    where the rises of a non-descent method become visible."""
    objs = res["objs_check"]
    best = np.minimum.accumulate(objs)
    k = np.arange(1, ITERS + 1)  # log x, or the drop in the first 100 iterations is invisible
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for ax, offset in ((ax1, 0.0), (ax2, E_REF)):
        ax.loglog(k, best[1:] - offset, color=ORANGE, linewidth=2.2, label="best E so far")
        ax.loglog(k, objs[1:] - offset, color=BLUE, linewidth=0.8, label="E, current iterate")
        ax.set_xlabel("iteration (log scale)")
    ax1.axhline(E_REF, color=INK_2, linestyle="--", linewidth=1, label=f"reference E* = {E_REF}")
    ax1.set(ylabel="E (log scale)", title="Lasso objective")
    ax1.legend()
    ax2.set(ylabel=r"$E - E^\star$ (log scale)", title=f"Gap to the reference E* = {E_REF}")
    fig.suptitle(f"Subgradient descent, λ = {CHECK_LAM}, η = {ETA}", y=1.02)
    fig.tight_layout()
    return fig


def rmse_figure(res):
    n = res["numbers"]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    rmse_vs_lambda(
        ax,
        [s["lam"] for s in n["sweep"]],
        [s["train_rmse"] for s in n["sweep"]],
        [s["test_rmse"] for s in n["sweep"]],
        n["table"]["lasso"]["lam"],
    )
    ax.set_title("Lasso (Legendre basis): RMSE vs. λ")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.9))
    return fig


def path_figure(res):
    """One small panel per coefficient (ten series are too many to tell apart on one axis),
    on one shared scale. Hollow markers are ≈ 0 by the 2 eta lam rule."""
    sweep = res["numbers"]["sweep"]
    lams = np.array([s["lam"] for s in sweep])
    W = np.array([s["w"] for s in sweep])
    zero = np.array([near_zero(w, lam) for w, lam in zip(W, lams, strict=True)])
    fig, axes = plt.subplots(2, 5, figsize=(7.4, 3.8), sharex=True, sharey=True)
    for j, ax in enumerate(axes.flat):
        ax.axhline(0, color=INK_2, linewidth=0.8)
        ax.semilogx(lams, W[:, j], color=BLUE, linewidth=1.2)
        ax.scatter(lams[~zero[:, j]], W[~zero[:, j], j], s=10, color=BLUE, zorder=3)
        ax.scatter(
            lams[zero[:, j]],
            W[zero[:, j], j],
            s=12,
            facecolor="white",
            edgecolor=BLUE,
            linewidth=0.9,
            zorder=3,
        )
        ax.set_title(f"$w_{j}$", fontsize=9)
        ax.tick_params(labelsize=7)
    lim = np.abs(W).max() * 1.1
    axes[0, 0].set_ylim(-lim, lim)
    for ax in axes[-1]:
        ax.set_xlabel("λ (log scale)", fontsize=8)
    fig.suptitle(
        r"Lasso coefficient path (hollow markers: $\approx 0$, $|w_j| < 2\eta\lambda$)", y=1.0
    )
    fig.tight_layout()
    return fig


def fits_figure(res):
    n = res["numbers"]["table"]
    Phi_dense = legendre(X_DENSE, D)
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    truth(ax)
    ax.plot(
        X_DENSE,
        Phi_dense @ n["lasso"]["w"],
        color=BLUE,
        label=rf"Lasso, $\lambda^\star = {fmt.lam(n['lasso']['lam'])}$",
    )
    ax.plot(
        X_DENSE,
        Phi_dense @ n["ridge"]["w"],
        color=ORANGE,
        linestyle=(0, (4, 3)),
        label=rf"Ridge, best $\lambda = {fmt.lam(n['ridge']['lam'])}$",
    )
    points(ax, res["x"], res["t"])
    ax.set(xlabel="x", ylabel="t", title="Lasso and Ridge at their best λ (Legendre basis, d = 9)")
    ax.set_ylim(-1.6, 1.6)
    ax.legend(loc="upper right")
    return fig


def figures(res):
    return {
        "p3_convergence": convergence_figure(res),
        "p3_rmse": rmse_figure(res),
        "p3_path": path_figure(res),
        "p3_fits": fits_figure(res),
    }
