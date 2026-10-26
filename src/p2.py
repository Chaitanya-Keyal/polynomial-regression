"""P2: ridge regression as the MAP estimate, by closed form and by GD in the monomial basis.

E~(w) = 1/2 ||Phi w - t||^2 + lam/2 ||w||^2, with Phi the d = 9 monomial basis of P1.
"""

import matplotlib.pyplot as plt
import numpy as np

from src.common import fmt
from src.common.bases import monomial
from src.common.data import TEST, TRAIN, load
from src.common.linalg import ridge
from src.common.metrics import ridge_objective, rmse
from src.common.plots import BLUE, INK_2, X_DENSE, note, points, rmse_vs_lambda, truth

D = 9
LAMS = np.logspace(-10, 2, 49)  # the grid for every sweep
GD_LAM, GD_ETA, GD_TOL = 1e-2, 0.04, 1e-6  # stop once ||w - w_closed|| / ||w_closed|| < GD_TOL
WITHIN = 1.05  # the flat region: test RMSE <= 1.05 x its minimum
PANELS = {"overfit": 1e-10, "underfit": 1e-2}  # with lambda* between them


def ridge_gd(Phi, t, lam, eta, w_ref):
    """Batch GD on E~ from w = 0: w <- w - eta (Phi^T (Phi w - t) + lam w), until w is within
    GD_TOL of w_ref (relative 2-norm). Returns E~ and the relative error per iteration (index 0
    is the start)."""
    w = np.zeros(Phi.shape[1])
    objs, errs = [ridge_objective(Phi, w, t, lam)], [1.0]
    while errs[-1] >= GD_TOL:
        w = w - eta * (Phi.T @ (Phi @ w - t) + lam * w)
        objs.append(ridge_objective(Phi, w, t, lam))
        errs.append(np.linalg.norm(w - w_ref) / np.linalg.norm(w_ref))
    return np.array(objs), np.array(errs)


def within_band(test):
    """Grid indices with test RMSE <= WITHIN x the minimum: the index of the minimum, the
    contiguous run around it (lo..hi), and any qualifying indices outside that run."""
    best = int(np.argmin(test))
    ok = test <= WITHIN * test[best]
    lo = hi = best
    while lo > 0 and ok[lo - 1]:
        lo -= 1
    while hi < len(test) - 1 and ok[hi + 1]:
        hi += 1
    return best, lo, hi, [int(i) for i in np.flatnonzero(ok) if not lo <= i <= hi]


def compute():
    x, t = load(TRAIN)
    x_test, t_test = load(TEST)
    Phi, Phi_test = monomial(x, D), monomial(x_test, D)

    ws = [ridge(Phi, t, lam) for lam in LAMS]
    train = np.array([rmse(Phi @ w, t) for w in ws])
    test = np.array([rmse(Phi_test @ w, t_test) for w in ws])
    best, lo, hi, others = within_band(test)
    # the highest point between the run and the outside points, if there are any
    span = range(min([lo, *others]), max([hi, *others]) + 1)
    peak = max(span, key=lambda i: test[i])

    w_closed = ridge(Phi, t, GD_LAM)
    objs, errs = ridge_gd(Phi, t, GD_LAM, GD_ETA, w_closed)
    PtP = Phi.T @ Phi

    panel_lams = {
        "overfit": PANELS["overfit"],
        "best": float(LAMS[best]),
        "underfit": PANELS["underfit"],
    }
    panel_ws = {name: ridge(Phi, t, lam) for name, lam in panel_lams.items()}
    numbers = {
        "lams": LAMS.tolist(),
        "train_rmse": train.tolist(),
        "test_rmse": test.tolist(),
        "gd": {
            "iterations": len(objs) - 1,
            "objective": float(objs[-1]),
            "objective_closed": ridge_objective(Phi, w_closed, t, GD_LAM),
            "cond_PtP": float(np.linalg.cond(PtP)),
            "cond_PtP_lam": float(np.linalg.cond(PtP + GD_LAM * np.eye(D + 1))),
        },
        "flat": {
            "lam_lo": float(LAMS[lo]),
            "lam_hi": float(LAMS[hi]),
            "n_run": hi - lo + 1,
            "others": [float(LAMS[i]) for i in others],
            "peak_lam": float(LAMS[peak]),
            "peak_ratio": float(test[peak] / test[best]),
        },
        "panels": {
            name: {
                "lam": panel_lams[name],
                "w_norm": float(np.linalg.norm(w)),
                "train_rmse": rmse(Phi @ w, t),
                "test_rmse": rmse(Phi_test @ w, t_test),
            }
            for name, w in panel_ws.items()
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
    k = np.arange(1, len(res["objs"]))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2))
    ax1.semilogx(k, res["objs"][1:], color=BLUE, label="GD")
    ax1.axhline(
        n["objective_closed"], color=INK_2, linestyle="--", linewidth=1, label="closed form"
    )
    ax1.set(
        xlabel="iteration (log scale)", ylabel=r"$\tilde{E}(\mathbf{w})$", title="Ridge objective"
    )
    ax1.legend()
    ax2.loglog(k, res["errs"][1:], color=BLUE)
    ax2.axhline(GD_TOL, color=INK_2, linestyle="--", linewidth=1)
    ax2.annotate(
        f"stop at {n['iterations']:,} iterations",
        (n["iterations"], GD_TOL),
        xytext=(-6, 10),
        textcoords="offset points",
        ha="right",
        fontsize=8,
    )
    ax2.set(
        xlabel="iteration (log scale)",
        ylabel=r"$\|\mathbf{w} - \mathbf{w}_{closed}\| \,/\, \|\mathbf{w}_{closed}\|$",
        title="Distance to the closed form",
    )
    fig.suptitle(f"Batch GD on the ridge objective, λ = {GD_LAM:g}, η = {GD_ETA}", y=1.02)
    fig.tight_layout()
    return fig


def rmse_figure(res):
    n = res["numbers"]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    rmse_vs_lambda(ax, n["lams"], n["train_rmse"], n["test_rmse"], n["panels"]["best"]["lam"])
    ax.set_title(f"Ridge, d = {D}: RMSE vs. λ")
    ax.legend(loc="lower right", frameon=True, edgecolor="none")
    return fig


def fits_figure(res):
    n = res["numbers"]["panels"]
    titles = {"overfit": "overfit", "best": r"$\lambda^\star$", "underfit": "underfit"}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), sharey=True)
    for ax, (name, w) in zip(axes, res["panel_ws"].items(), strict=True):
        truth(ax)
        ax.plot(X_DENSE, monomial(X_DENSE, D) @ w, color=BLUE, label="ridge fit")
        points(ax, res["x"], res["t"])
        ax.set_title(rf"{titles[name]}: $\lambda = {fmt.lam(n[name]['lam'])}$")
        note(ax, rf"$\|\mathbf{{w}}\|_2 = {fmt.num(n[name]['w_norm'])}$")
        ax.set_xlabel("x")
    axes[0].set_ylabel("t")
    axes[0].set_ylim(-1.8, 1.8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def figures(res):
    return {"p2_gd": gd_figure(res), "p2_rmse": rmse_figure(res), "p2_fits": fits_figure(res)}
