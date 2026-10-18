"""Objectives and errors, with lambda relative to the unnormalised SSE (assignment, P2 and P3)."""

import numpy as np


def sse(Phi, w, t):
    """E(w) = 1/2 ||Phi w - t||^2."""
    r = Phi @ w - t
    return 0.5 * float(r @ r)


def ridge_objective(Phi, w, t, lam):
    """E~(w) = 1/2 ||Phi w - t||^2 + lam/2 ||w||^2."""
    return sse(Phi, w, t) + 0.5 * lam * float(w @ w)


def lasso_objective(Phi, w, t, lam):
    """E(w) = 1/2 ||Phi w - t||^2 + lam ||w||_1."""
    return sse(Phi, w, t) + lam * float(np.abs(w).sum())


def rmse(y, t):
    """sqrt(mean squared error)."""
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(t)) ** 2)))
