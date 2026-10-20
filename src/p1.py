"""P1: polynomial regression by least squares and by Batch, Stochastic and Mini-batch GD.

GD runs on the QR-reparametrised problem E(alpha) = 1/2 ||Q alpha - t||^2, with Phi = QR and
alpha = R w, from alpha = 0; w = R^-1 alpha is recovered with a solve at the end.
"""

import matplotlib.pyplot as plt
import numpy as np

from src.common.bases import monomial
from src.common.data import TRAIN, load
from src.common.linalg import least_squares
from src.common.metrics import rmse, sse
from src.common.plots import AQUA, BLUE, ORANGE, X_DENSE, note, points, truth

SEED = 42  # a fresh np.random.default_rng(SEED) for every run
D = 9
BATCH = 3  # with N = 10, each epoch is batches of 3, 3, 3, 1
ETA = 0.1
EPOCHS = 2000  # every variant has flattened well before this (loss curves and verification)
LR_CASES = {"too small": 0.01, "well chosen": 0.5, "too large": 2.1}
LR_EPOCHS = 100
NAMES = {"batch": "Batch GD", "sgd": "Stochastic GD", "minibatch": f"Mini-batch GD (size {BATCH})"}


def batches(order, size):
    """Consecutive slices of a shuffled index order; the last one holds what is left."""
    return [order[i : i + size] for i in range(0, len(order), size)]


def gd(Q, t, eta, epochs, variant):
    """GD on E(alpha) = 1/2 ||Q alpha - t||^2 from alpha = 0.

    batch:     one update per epoch with the full gradient Q^T (Q alpha - t), not divided by N.
    sgd:       shuffle, then one update per point with that point's gradient.
    minibatch: shuffle, then one update per batch with the batch's averaged gradient.

    Returns the final alpha and E after each epoch (index 0 is the start).
    """
    rng = np.random.default_rng(SEED)
    alpha = np.zeros(Q.shape[1])
    losses = [sse(Q, alpha, t)]
    for _ in range(epochs):
        if variant == "batch":
            alpha = alpha - eta * Q.T @ (Q @ alpha - t)
        else:
            for idx in batches(rng.permutation(len(t)), 1 if variant == "sgd" else BATCH):
                Qb = Q[idx]
                alpha = alpha - eta * Qb.T @ (Qb @ alpha - t[idx]) / len(idx)
        losses.append(sse(Q, alpha, t))
    return alpha, np.array(losses)


def compute():
    x, t = load(TRAIN)
    closed = {d: least_squares(monomial(x, d), t) for d in range(1, D + 1)}
    w_closed = closed[D]
    Phi = monomial(x, D)
    Q, R = np.linalg.qr(Phi)  # thin QR; with N = D + 1 = 10, Q is square
    runs = {v: gd(Q, t, ETA, EPOCHS, v) for v in NAMES}
    lr = {label: gd(Q, t, eta, LR_EPOCHS, "batch") for label, eta in LR_CASES.items()}

    def verification(alpha, losses):
        err = float(np.linalg.norm(np.linalg.solve(R, alpha) - w_closed))
        return {
            "abs_err": err,
            "rel_err": err / float(np.linalg.norm(w_closed)),
            "final_E": float(losses[-1]),
            # the average factor E shrinks by per epoch, over the first 100 epochs
            "per_epoch_factor": float((losses[100] / losses[0]) ** (1 / 100)),
            "epochs_to_1e-12": int(np.argmax(losses < 1e-12)),
        }

    batch, sgd = runs["batch"][1], runs["sgd"][1]
    above_floor = batch > 1e-10  # below that, rounding differs between the two
    w_large = np.linalg.solve(R, lr["too large"][0])
    numbers = {
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
        "verification": {v: verification(*run) for v, run in runs.items()},
        "sgd_batch_rel_gap": float(np.max(np.abs(sgd - batch)[above_floor] / batch[above_floor])),
        "E_0": float(batch[0]),
        "E_100": {label: float(losses[-1]) for label, (_, losses) in lr.items()},
        "w_100_too_large": w_large.tolist(),
        # alpha_k = (1 - (-1.1)^k) alpha* at eta = 2.1, so w_100 is this multiple of w_closed
        "w_100_over_w_closed": float(w_large @ w_closed / (w_closed @ w_closed)),
    }
    return {
        "numbers": numbers,
        "x": x,
        "t": t,
        "losses": {v: losses for v, (_, losses) in runs.items()},
        "lr": {label: losses for label, (_, losses) in lr.items()},
    }


def fits_figure(res):
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 6.4), sharex=True, sharey=True)
    for ax, (d, c) in zip(axes.flat, res["numbers"]["closed"].items(), strict=True):
        truth(ax)
        ax.plot(X_DENSE, monomial(X_DENSE, d) @ c["w"], color=BLUE, label="least-squares fit")
        points(ax, res["x"], res["t"])
        ax.set_title(f"d = {d}")
        note(ax, f"train RMSE {c['train_rmse']:.3f}")
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
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    styles = ((BLUE, "-"), (ORANGE, (0, (4, 3))), (AQUA, "-"))
    for (v, losses), (color, ls) in zip(res["losses"].items(), styles, strict=True):
        ax.semilogy(losses, color=color, linestyle=ls, label=NAMES[v])
    ax.set(xlabel="epoch", ylabel="E (log scale)", title=f"d = {D}, η = {ETA}, seed {SEED}")
    ax.set_xlim(0, EPOCHS)
    ax.legend(loc="upper right")
    return fig


def lr_figure(res):
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for (label, losses), color in zip(res["lr"].items(), (BLUE, ORANGE, AQUA), strict=True):
        ax.semilogy(losses, color=color, label=f"η = {LR_CASES[label]} ({label})")
    ax.set(xlabel="epoch", ylabel="E (log scale)", title=f"Batch GD, d = {D}")
    ax.set_xlim(0, LR_EPOCHS)
    ax.legend(loc="center right")
    return fig


def figures(res):
    return {"p1_fits": fits_figure(res), "p1_loss": loss_figure(res), "p1_lr": lr_figure(res)}
