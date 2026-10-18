"""P1: polynomial regression by least squares and by Batch, Stochastic and Mini-batch GD.

GD runs on the QR-reparametrised problem E(alpha) = 1/2 ||Q alpha - t||^2 with Phi = QR and
alpha = R w, from alpha = 0; w = R^-1 alpha is recovered by a triangular solve at the end.

python -m src.run p1
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.common.bases import monomial
from src.common.data import DATA_DIR, TRAIN, load_xy
from src.common.linalg import least_squares
from src.common.metrics import rmse, sse
from src.common.plots import (
    AQUA,
    BLUE,
    INK_2,
    ORANGE,
    SERIES,
    X_DENSE,
    YELLOW,
    points,
    truth,
)

SEED = 42  # one fresh np.random.default_rng(SEED) per SGD or Mini-batch run
DEGREES = range(1, 10)
D = 9
BATCH = 3  # Mini-batch size; with N = 10 each epoch is batches of 3, 3, 3, 1
ETA = 0.1  # the shared learning rate for the loss curves and the verification
EPOCHS = 2000  # every variant is flat well before this (verification, deliverable 5)
LOSS_PLOT_EPOCHS = 1500  # the range the loss-curve figure shows
LR_CASES = (("too small", 0.01), ("well chosen", 0.5), ("too large", 2.1))
LR_EDGE = 2.0  # the stability limit 2 / lambda_max(Q^T Q), shown dashed
LR_EPOCHS = 100
VARIANTS = ("batch", "sgd", "minibatch")
NAMES = {"batch": "Batch GD", "sgd": "Stochastic GD", "minibatch": f"Mini-batch GD (size {BATCH})"}


def batches(order, size):
    """Consecutive slices of a shuffled index order; the last one holds what is left."""
    return [order[i : i + size] for i in range(0, len(order), size)]


def gd(Q, t, eta, epochs, variant, rng=None):
    """GD on E(alpha) = 1/2 ||Q alpha - t||^2 from alpha = 0.

    batch:     one update per epoch with the full gradient Q^T (Q alpha - t), not divided by N.
    sgd:       shuffle, then one update per point with that point's gradient.
    minibatch: shuffle, then one update per batch with the batch's averaged gradient.

    Returns the final alpha and E after each epoch (index 0 is the starting point).
    """
    alpha = np.zeros(Q.shape[1])
    losses = np.empty(epochs + 1)
    losses[0] = sse(Q, alpha, t)
    size = 1 if variant == "sgd" else BATCH
    for k in range(1, epochs + 1):
        if variant == "batch":
            alpha = alpha - eta * Q.T @ (Q @ alpha - t)
        else:
            for idx in batches(rng.permutation(len(t)), size):
                Qb = Q[idx]
                alpha = alpha - eta * Qb.T @ (Qb @ alpha - t[idx]) / len(idx)
        losses[k] = sse(Q, alpha, t)
    return alpha, losses


def run_variant(Q, t, eta, epochs, variant):
    return gd(Q, t, eta, epochs, variant, np.random.default_rng(SEED))


def compute(data_dir=DATA_DIR):
    x, t = load_xy(Path(data_dir) / TRAIN)

    closed = {d: least_squares(monomial(x, d), t) for d in DEGREES}
    Phi = monomial(x, D)
    Q, R = np.linalg.qr(Phi)  # thin QR: Q is N x (D + 1) with orthonormal columns
    w_closed = closed[D]

    runs = {v: run_variant(Q, t, ETA, EPOCHS, v) for v in VARIANTS}
    verification = {}
    for v, (alpha, losses) in runs.items():
        w = np.linalg.solve(R, alpha)
        err = float(np.linalg.norm(w - w_closed))
        verification[v] = {
            "abs_err": err,
            "rel_err": err / float(np.linalg.norm(w_closed)),
            "final_E": float(losses[-1]),
            # (E_100 / E_0)^(1/100): the average factor E shrinks by per epoch
            "per_epoch_factor": float((losses[100] / losses[0]) ** (1 / 100)),
            "epochs_to_1e-12": int(np.argmax(losses < 1e-12)) if losses.min() < 1e-12 else None,
        }
    # SGD and Batch agree to rounding: each residual carries ~1e-16 of noise from Q alpha - t,
    # so the relative gap only grows once E nears the floating-point floor
    batch_E, sgd_E = runs["batch"][1], runs["sgd"][1]
    gap = np.abs(sgd_E - batch_E)
    well_above_floor = batch_E > 1e-10
    sgd_vs_batch = {
        "max_abs_diff": float(gap.max()),
        "max_rel_diff_while_E_above_1e-10": float(
            np.max(gap[well_above_floor] / batch_E[well_above_floor])
        ),
    }

    lr = {}
    for label, eta in (*LR_CASES, ("edge", LR_EDGE)):
        alpha, losses = gd(Q, t, eta, LR_EPOCHS, "batch")
        lr[label] = {"eta": eta, "losses": losses, "w": np.linalg.solve(R, alpha)}

    numbers = {
        "seed": SEED,
        "eta": ETA,
        "epochs": EPOCHS,
        "batch_size": BATCH,
        "N": len(t),
        "cond_phi": float(np.linalg.cond(Phi)),
        "cond_q": float(np.linalg.cond(Q)),
        "closed": {
            d: {
                "w": w.tolist(),
                "train_E": sse(monomial(x, d), w, t),
                "train_rmse": rmse(monomial(x, d) @ w, t),
            }
            for d, w in closed.items()
        },
        "w_closed_norm": float(np.linalg.norm(w_closed)),
        "verification": verification,
        "sgd_vs_batch": sgd_vs_batch,
        "lr": {
            label: {
                "eta": r["eta"],
                "E_0": float(r["losses"][0]),
                f"E_{LR_EPOCHS}": float(r["losses"][-1]),
                f"w_{LR_EPOCHS}": r["w"].tolist(),
            }
            for label, r in lr.items()
        },
        # at eta = 2.1, alpha_k = (1 - (-1.1)^k) alpha*, so w_100 is this multiple of w_closed
        "divergent_w_over_w_closed": float(lr["too large"]["w"] @ w_closed / (w_closed @ w_closed)),
    }
    return {
        "numbers": numbers,
        "x": x,
        "t": t,
        "closed": closed,
        "losses": {v: losses for v, (_, losses) in runs.items()},
        "lr": lr,
    }


def fits_figure(res):
    x, t = res["x"], res["t"]
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 6.4), sharex=True, sharey=True)
    for ax, d in zip(axes.flat, DEGREES, strict=True):
        w = res["closed"][d]
        truth(ax)
        ax.plot(X_DENSE, monomial(X_DENSE, d) @ w, color=BLUE, label="least-squares fit")
        points(ax, x, t)
        train_rmse = res["numbers"]["closed"][d]["train_rmse"]
        ax.set_title(f"d = {d}")
        ax.text(
            0.97,
            0.04,
            f"train RMSE {train_rmse:.3f}",
            transform=ax.transAxes,
            ha="right",
            fontsize=7.5,
            color=INK_2,
        )
    axes[0, 0].set_ylim(-1.8, 1.8)
    for ax in axes[-1]:
        ax.set_xlabel("x")
    for ax in axes[:, 0]:
        ax.set_ylabel("t")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def loss_figure(res):
    losses = res["losses"]
    k = np.arange(LOSS_PLOT_EPOCHS + 1)
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    styles = {"batch": (BLUE, "-"), "sgd": (ORANGE, (0, (4, 3))), "minibatch": (AQUA, "-")}
    for v in VARIANTS:
        color, ls = styles[v]
        ax.semilogy(k, losses[v][: k.size], color=color, linestyle=ls, label=NAMES[v])
    ax.set(xlabel="epoch", ylabel="E (log scale)", title=f"d = {D}, η = {ETA}, seed {SEED}")
    ax.set_xlim(0, LOSS_PLOT_EPOCHS)
    ax.legend(loc="upper right")
    return fig


def lr_figure(res):
    lr = res["lr"]
    k = np.arange(LR_EPOCHS + 1)
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for (label, _), color in zip(LR_CASES, SERIES, strict=False):
        r = lr[label]
        ax.semilogy(k, r["losses"], color=color, label=f"η = {r['eta']} ({label})")
    edge = lr["edge"]
    ax.semilogy(
        k, edge["losses"], color=YELLOW, linestyle=(0, (4, 3)), label=f"η = {edge['eta']} (edge)"
    )
    ax.set(xlabel="epoch", ylabel="E (log scale)", title=f"Batch GD, d = {D}")
    ax.set_xlim(0, LR_EPOCHS)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    return fig


def figures(res):
    return {"p1_fits": fits_figure(res), "p1_loss": loss_figure(res), "p1_lr": lr_figure(res)}
