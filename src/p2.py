"""P2: ridge regression as the MAP estimate, by closed form and by GD in the monomial basis.

E~(w) = 1/2 ||Phi w - t||^2 + lam/2 ||w||^2, with Phi the d = 9 monomial basis of P1.

python -m src.run p2
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.common.bases import monomial
from src.common.data import DATA_DIR, TEST, TRAIN, load_xy
from src.common.linalg import ridge
from src.common.metrics import ridge_objective, rmse
from src.common.plots import BLUE, INK_2, MUTED, ORANGE, X_DENSE, points, truth

D = 9
LAMS = np.logspace(-10, 2, 49)  # the grid for every sweep
GD_LAM = 1e-2
GD_ETA = 0.04
GD_TOL = 1e-6  # stop once ||w_GD - w_closed|| / ||w_closed|| is below this
GD_MAX_ITERS = 1_000_000
NOISE_SD = 0.1  # sqrt(0.01), the reference line on the RMSE plot
WITHIN = 1.05  # the flat region: test RMSE <= 1.05 x its minimum
OVERFIT_LAM, UNDERFIT_LAM = 1e-10, 1e-2


def ridge_gd(Phi, t, lam, eta, w_ref, tol=GD_TOL, max_iters=GD_MAX_ITERS):
    """Batch GD on E~ from w = 0: w <- w - eta (Phi^T (Phi w - t) + lam w), until w is within
    tol of w_ref in relative 2-norm. Returns w, E~ at every iteration (index 0 is the start)
    and the relative error at every iteration."""
    w = np.zeros(Phi.shape[1])
    ref_norm = np.linalg.norm(w_ref)
    objs, errs = [ridge_objective(Phi, w, t, lam)], [1.0]
    for _ in range(max_iters):
        w = w - eta * (Phi.T @ (Phi @ w - t) + lam * w)
        objs.append(ridge_objective(Phi, w, t, lam))
        errs.append(np.linalg.norm(w - w_ref) / ref_norm)
        if errs[-1] < tol:
            return w, np.array(objs), np.array(errs)
    raise RuntimeError(f"ridge GD did not reach {tol} in {max_iters} iterations")


def flat_interval(values, within=WITHIN):
    """The index of the minimum, and the contiguous run of indices around it whose values are
    at most `within` x the minimum. Also whether that run holds every such index."""
    best = int(np.argmin(values))
    ok = values <= within * values[best]
    lo = hi = best
    while lo > 0 and ok[lo - 1]:
        lo -= 1
    while hi < len(values) - 1 and ok[hi + 1]:
        hi += 1
    return best, lo, hi, bool(ok.sum() == hi - lo + 1)


def compute(data_dir=DATA_DIR):
    x, t = load_xy(Path(data_dir) / TRAIN)
    x_test, t_test = load_xy(Path(data_dir) / TEST)
    Phi, Phi_test = monomial(x, D), monomial(x_test, D)

    ws = np.array([ridge(Phi, t, lam) for lam in LAMS])
    train = np.array([rmse(Phi @ w, t) for w in ws])
    test = np.array([rmse(Phi_test @ w, t_test) for w in ws])
    best, lo, hi, contiguous = flat_interval(test)
    lam_star = float(LAMS[best])

    w_closed = ridge(Phi, t, GD_LAM)
    _, objs, errs = ridge_gd(Phi, t, GD_LAM, GD_ETA, w_closed)
    PtP = Phi.T @ Phi

    panels = {"overfit": OVERFIT_LAM, "best": lam_star, "underfit": UNDERFIT_LAM}
    panel_ws = {name: ridge(Phi, t, lam) for name, lam in panels.items()}

    numbers = {
        "lams": LAMS.tolist(),
        "train_rmse": train.tolist(),
        "test_rmse": test.tolist(),
        "w_norm": np.linalg.norm(ws, axis=1).tolist(),
        "gd": {
            "lam": GD_LAM,
            "eta": GD_ETA,
            "tol": GD_TOL,
            "iterations": len(objs) - 1,
            "rel_err": float(errs[-1]),
            "objective": float(objs[-1]),
            "objective_closed": ridge_objective(Phi, w_closed, t, GD_LAM),
            "cond_PtP": float(np.linalg.cond(PtP)),
            "cond_PtP_lam": float(np.linalg.cond(PtP + GD_LAM * np.eye(D + 1))),
            "stability_limit": float(2 / np.linalg.eigvalsh(PtP + GD_LAM * np.eye(D + 1))[-1]),
        },
        "lam_star": lam_star,
        "test_rmse_star": float(test[best]),
        "train_rmse_star": float(train[best]),
        "flat": {
            "within": WITHIN,
            "lam_lo": float(LAMS[lo]),
            "lam_hi": float(LAMS[hi]),
            "grid_points": hi - lo + 1,
            "contiguous": contiguous,
        },
        "panels": {
            name: {
                "lam": lam,
                "w_norm": float(np.linalg.norm(panel_ws[name])),
                "train_rmse": rmse(Phi @ panel_ws[name], t),
                "test_rmse": rmse(Phi_test @ panel_ws[name], t_test),
            }
            for name, lam in panels.items()
        },
    }
    return {
        "numbers": numbers,
        "x": x,
        "t": t,
        "objs": objs,
        "errs": errs,
        "panel_ws": panel_ws,
    }


def gd_figure(res):
    n = res["numbers"]["gd"]
    k = np.arange(len(res["objs"]))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2))
    ax1.loglog(k[1:], res["objs"][1:], color=BLUE, label="GD")
    ax1.axhline(
        n["objective_closed"], color=INK_2, linestyle="--", linewidth=1, label="closed form"
    )
    ax1.set(
        xlabel="iteration", ylabel=r"$\tilde{E}(\mathbf{w})$ (log scale)", title="Ridge objective"
    )
    ax1.legend()
    ax2.loglog(k[1:], res["errs"][1:], color=BLUE)
    ax2.axhline(n["tol"], color=INK_2, linestyle="--", linewidth=1)
    ax2.annotate(
        f"stop at {n['iterations']:,} iterations",
        (n["iterations"], n["tol"]),
        xytext=(-6, 10),
        textcoords="offset points",
        ha="right",
        fontsize=8,
    )
    ax2.set(
        xlabel="iteration",
        ylabel=r"$\|\mathbf{w} - \mathbf{w}_{closed}\| \,/\, \|\mathbf{w}_{closed}\|$",
        title="Distance to the closed form",
    )
    fig.suptitle(f"Batch GD on the ridge objective, λ = {n['lam']:g}, η = {n['eta']}", y=1.02)
    fig.tight_layout()
    return fig


def rmse_figure(res):
    n = res["numbers"]
    lams = np.array(n["lams"])
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.axvspan(
        n["flat"]["lam_lo"],
        n["flat"]["lam_hi"],
        color=ORANGE,
        alpha=0.08,
        linewidth=0,
        label=rf"test RMSE $\leq$ {WITHIN} $\times$ min",
    )
    ax.semilogx(lams, n["train_rmse"], color=BLUE, marker="o", markersize=3, label="train RMSE")
    ax.semilogx(lams, n["test_rmse"], color=ORANGE, marker="o", markersize=3, label="test RMSE")
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
    ax.set(xlabel="λ (log scale)", ylabel="RMSE", title=f"Ridge, d = {D}: RMSE vs. λ")
    ax.set_ylim(bottom=0)
    # opaque, so the λ* line does not run through the text
    ax.legend(loc="center left", bbox_to_anchor=(0, 0.6), frameon=True, edgecolor="none")
    return fig


def fits_figure(res):
    n = res["numbers"]["panels"]
    titles = {"overfit": "overfit", "best": "λ*", "underfit": "underfit"}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), sharey=True)
    for ax, (name, w) in zip(axes, res["panel_ws"].items(), strict=True):
        truth(ax)
        ax.plot(X_DENSE, monomial(X_DENSE, D) @ w, color=BLUE, label="ridge fit")
        points(ax, res["x"], res["t"])
        ax.set_title(f"{titles[name]}: λ = {n[name]['lam']:.2g}")
        ax.text(
            0.97,
            0.04,
            f"‖w‖ = {n[name]['w_norm']:.3g}",
            transform=ax.transAxes,
            ha="right",
            fontsize=7.5,
            color=INK_2,
        )
        ax.set_xlabel("x")
    axes[0].set_ylabel("t")
    axes[0].set_ylim(-1.8, 1.8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def figures(res):
    return {"p2_gd": gd_figure(res), "p2_rmse": rmse_figure(res), "p2_fits": fits_figure(res)}
