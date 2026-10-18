"""P3: L1-regularised regression (Lasso) by subgradient descent in a Legendre basis.

E(w) = 1/2 ||Phi w - t||^2 + lam ||w||_1 with Phi[n, j] = P_j(2 x_n - 1), j = 0..9. Every
reported quantity comes from the final iterate; the best-so-far E is only for the
convergence plot (assignment, P3 implementation note).

python -m src.run p3
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.common.bases import legendre
from src.common.data import DATA_DIR, TEST, TRAIN, load_xy
from src.common.linalg import ridge
from src.common.metrics import lasso_objective, rmse
from src.common.plots import BLUE, INK, INK_2, MUTED, ORANGE, X_DENSE, points, truth

D = 9
ETA = 0.05
ITERS = 20_000
LAMS = np.logspace(-4, 0, 21)  # the grid for every sweep
CHECK_LAM = 0.1  # deliverable 1
# The exact Lasso solution at lam = 0.1 (coordinate descent, rounded to 3 decimals; assignment)
E_REF = 0.2625
W_REF = np.array([0, -0.928, 0.045, 1.117, 0.032, -0.201, 0, -0.026, 0, 0])
SMALL = 0.01  # the |w_j| < 0.01 count in the Lasso vs. Ridge table
NOISE_SD = 0.1


def near_zero(w, lam, eta=ETA):
    """The "≈ 0" rule: with a constant step, coefficients chatter around zero with amplitude
    about eta * lam, so |w_j| < 2 eta lam counts as zero."""
    return np.abs(w) < 2 * eta * lam


def subgradient_descent(Phi, t, lam, eta=ETA, iters=ITERS):
    """w <- w - eta (Phi^T (Phi w - t) + lam sign(w)) from w = 0, with sign(0) = 0.

    Returns the final w and E after every iteration (index 0 is the start).
    """
    w = np.zeros(Phi.shape[1])
    objs = np.empty(iters + 1)
    objs[0] = lasso_objective(Phi, w, t, lam)
    for k in range(1, iters + 1):
        w = w - eta * (Phi.T @ (Phi @ w - t) + lam * np.sign(w))
        objs[k] = lasso_objective(Phi, w, t, lam)
    return w, objs


def lasso_cd(Phi, t, lam, tol=1e-13, max_sweeps=1_000_000):
    """Exact Lasso by cyclic coordinate descent with soft-thresholding: a cross-check only,
    the method the assignment's reference solution was computed with."""
    w = np.zeros(Phi.shape[1])
    col_sq = (Phi**2).sum(axis=0)
    r = t.astype(float).copy()  # t - Phi w
    for _ in range(max_sweeps):
        biggest = 0.0
        for j in range(Phi.shape[1]):
            rho = Phi[:, j] @ r + col_sq[j] * w[j]
            new = np.sign(rho) * max(abs(rho) - lam, 0.0) / col_sq[j]
            r += Phi[:, j] * (w[j] - new)
            biggest = max(biggest, abs(new - w[j]))
            w[j] = new
        if biggest < tol:
            return w
    raise RuntimeError("coordinate descent did not converge")


def optimality(Phi, t, w, lam):
    """The subgradient conditions with r = t - Phi w: Phi_j^T r = lam sign(w_j) where w_j != 0,
    |Phi_j^T r| <= lam where w_j = 0 (here: where w_j is ≈ 0)."""
    c = Phi.T @ (t - Phi @ w)
    zero = near_zero(w, lam)
    return {
        "max_active_violation": float(np.max(np.abs(c - lam * np.sign(w))[~zero], initial=0)),
        "max_zero_ratio": float(np.max(np.abs(c[zero]) / lam, initial=0)),
        "n_active": int((~zero).sum()),
        "n_zero": int(zero.sum()),
    }


def compute(data_dir=DATA_DIR):
    x, t = load_xy(Path(data_dir) / TRAIN)
    x_test, t_test = load_xy(Path(data_dir) / TEST)
    Phi, Phi_test = legendre(x, D), legendre(x_test, D)

    sweep = []
    check = None
    for lam in LAMS:
        w, objs = subgradient_descent(Phi, t, lam)
        zero = near_zero(w, lam)
        sweep.append(
            {
                "lam": float(lam),
                "w": w.tolist(),
                "w_norm": float(np.linalg.norm(w)),
                "train_rmse": rmse(Phi @ w, t),
                "test_rmse": rmse(Phi_test @ w, t_test),
                "n_zero": int(zero.sum()),
                "zero_idx": np.flatnonzero(zero).tolist(),
            }
        )
        if np.isclose(lam, CHECK_LAM):
            check = (w, objs)
    if check is None:  # 0.1 = 10^-1 is on the grid, but do not rely on that
        check = subgradient_descent(Phi, t, CHECK_LAM)
    w_check, objs_check = check
    w_cd = lasso_cd(Phi, t, CHECK_LAM)

    best = int(np.argmin([s["test_rmse"] for s in sweep]))
    lasso_best = sweep[best]

    ridge_ws = [ridge(Phi, t, lam) for lam in LAMS]
    ridge_test = [rmse(Phi_test @ w, t_test) for w in ridge_ws]
    ridge_best = int(np.argmin(ridge_test))
    w_ridge = ridge_ws[ridge_best]
    w_lasso = np.array(lasso_best["w"])

    numbers = {
        "eta": ETA,
        "iters": ITERS,
        "cond_PtP": float(np.linalg.cond(Phi.T @ Phi)),
        "stability_limit": float(2 / np.linalg.eigvalsh(Phi.T @ Phi)[-1]),
        "check": {
            "lam": CHECK_LAM,
            "E_final": lasso_objective(Phi, w_check, t, CHECK_LAM),
            "E_best": float(objs_check.min()),
            "E_ref": E_REF,
            "w": w_check.tolist(),
            "w_ref": W_REF.tolist(),
            "w_minus_w_ref": float(np.linalg.norm(w_check - W_REF)),
            "optimality": optimality(Phi, t, w_check, CHECK_LAM),
            "w_cd": w_cd.tolist(),
            "E_cd": lasso_objective(Phi, w_cd, t, CHECK_LAM),
            "w_cd_minus_w_ref": float(np.linalg.norm(w_cd - W_REF)),
            "w_minus_w_cd": float(np.linalg.norm(w_check - w_cd)),
        },
        "sweep": sweep,
        "lam_star": lasso_best["lam"],
        "test_rmse_star": lasso_best["test_rmse"],
        "zero_idx_star": lasso_best["zero_idx"],
        "n_zero_star": lasso_best["n_zero"],
        "ridge": {
            "test_rmse": ridge_test,
            "lam_best": float(LAMS[ridge_best]),
        },
        "table": {
            name: {
                "lam": lam,
                "test_rmse": rmse(Phi_test @ w, t_test),
                "w_norm": float(np.linalg.norm(w)),
                "n_small": int((np.abs(w) < SMALL).sum()),
                "w": w.tolist(),
            }
            for name, lam, w in (
                ("lasso", lasso_best["lam"], w_lasso),
                ("ridge", float(LAMS[ridge_best]), w_ridge),
            )
        },
    }
    return {
        "numbers": numbers,
        "x": x,
        "t": t,
        "objs_check": objs_check,
        "w_lasso": w_lasso,
        "w_ridge": w_ridge,
    }


def convergence_figure(res):
    """Left, the deliverable: E and the best E so far. Right, the gap to the exact optimum
    (coordinate descent), where the rises of a non-descent method become visible."""
    objs = res["objs_check"]
    best = np.minimum.accumulate(objs)
    e_opt = res["numbers"]["check"]["E_cd"]
    k = np.arange(1, objs.size)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for ax, offset in ((ax1, 0.0), (ax2, e_opt)):
        ax.loglog(k, best[1:] - offset, color=ORANGE, linewidth=2.2, label="best E so far")
        ax.loglog(k, objs[1:] - offset, color=BLUE, linewidth=0.8, label="E, current iterate")
        ax.set_xlabel("iteration (log scale)")
    ax1.axhline(E_REF, color=INK_2, linestyle="--", linewidth=1, label=f"reference E* = {E_REF}")
    ax1.set(ylabel="E (log scale)", title="Lasso objective")
    ax1.legend()
    ax2.set(
        ylabel=r"$E - E_{opt}$ (log scale)",
        title=rf"Gap to the exact optimum $E_{{opt}}$ = {e_opt:.4f}",
    )
    fig.suptitle(f"Subgradient descent, λ = {CHECK_LAM}, η = {ETA}", y=1.02)
    fig.tight_layout()
    return fig


def rmse_figure(res):
    n = res["numbers"]
    lams = [s["lam"] for s in n["sweep"]]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.semilogx(
        lams,
        [s["train_rmse"] for s in n["sweep"]],
        color=BLUE,
        marker="o",
        markersize=3,
        label="train RMSE",
    )
    ax.semilogx(
        lams,
        [s["test_rmse"] for s in n["sweep"]],
        color=ORANGE,
        marker="o",
        markersize=3,
        label="test RMSE",
    )
    ax.axhline(NOISE_SD, color=INK_2, linestyle="--", linewidth=1, label="noise level 0.1")
    ax.axvline(n["lam_star"], color=MUTED, linewidth=1)
    ax.annotate(
        f"λ* = {n['lam_star']:.2g}",
        (n["lam_star"], 1),
        xycoords=("data", "axes fraction"),
        xytext=(4, -10),
        textcoords="offset points",
        fontsize=8,
    )
    ax.set(xlabel="λ (log scale)", ylabel="RMSE", title="Lasso (Legendre basis): RMSE vs. λ")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.9))
    return fig


def path_figure(res):
    """One small panel per coefficient (ten series are too many to tell apart on one axis), on
    one shared scale. The band is |w| < 2 eta lam; hollow markers fall inside it (≈ 0)."""
    sweep = res["numbers"]["sweep"]
    lams = np.array([s["lam"] for s in sweep])
    W = np.array([s["w"] for s in sweep])
    zero = np.array([near_zero(np.array(s["w"]), s["lam"]) for s in sweep])
    fig, axes = plt.subplots(2, 5, figsize=(7.4, 3.8), sharex=True, sharey=True)
    for j, ax in enumerate(axes.flat):
        ax.fill_between(lams, -2 * ETA * lams, 2 * ETA * lams, color=MUTED, alpha=0.25, lw=0)
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
        r"Lasso coefficient path (band and hollow markers: $\approx 0$, $|w_j| < 2\eta\lambda$)",
        y=1.0,
        color=INK,
    )
    fig.tight_layout()
    return fig


def fits_figure(res):
    n = res["numbers"]["table"]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    truth(ax)
    Phi_dense = legendre(X_DENSE, D)
    ax.plot(
        X_DENSE,
        Phi_dense @ res["w_lasso"],
        color=BLUE,
        label=f"Lasso, λ* = {n['lasso']['lam']:.2g}",
    )
    ax.plot(
        X_DENSE,
        Phi_dense @ res["w_ridge"],
        color=ORANGE,
        linestyle=(0, (4, 3)),
        label=f"Ridge, best λ = {n['ridge']['lam']:.2g}",
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
