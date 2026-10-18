"""Design matrices: row n is phi(x_n)^T, column j is the j-th basis function."""

import numpy as np


def monomial(x, d):
    """Phi[n, j] = x_n^j for j = 0..d (the Vandermonde matrix of P1, P2 and P4)."""
    return np.vander(np.asarray(x, dtype=float), d + 1, increasing=True)


def legendre(x, d):
    """Phi[n, j] = P_j(2 x_n - 1) for j = 0..d (P3), by the recurrence
    (j + 1) P_{j+1}(u) = (2j + 1) u P_j(u) - j P_{j-1}(u), with P_0 = 1 and P_1 = u.

    Equal to np.polynomial.legendre.legvander(2 * x - 1, d); tests/test_common.py checks it.
    """
    u = 2 * np.asarray(x, dtype=float) - 1
    Phi = np.empty((u.size, d + 1))
    Phi[:, 0] = 1
    if d >= 1:
        Phi[:, 1] = u
    for j in range(1, d):
        Phi[:, j + 1] = ((2 * j + 1) * u * Phi[:, j] - j * Phi[:, j - 1]) / (j + 1)
    return Phi
