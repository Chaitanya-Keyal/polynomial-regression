"""Closed-form solvers. Neither forms an explicit inverse (assignment, P1 to P4)."""

import numpy as np


def least_squares(Phi, t):
    """Minimiser of ||Phi w - t||^2 by np.linalg.lstsq, which works through the SVD of Phi.

    The normal equations would square cond(Phi), about 1e7 for d = 9, to about 1e14.
    """
    return np.linalg.lstsq(Phi, t, rcond=None)[0]


def ridge(Phi, t, lam):
    """Solution of (Phi^T Phi + lam I) w = Phi^T t, by a linear solve."""
    A = Phi.T @ Phi + lam * np.eye(Phi.shape[1])
    return np.linalg.solve(A, Phi.T @ t)
