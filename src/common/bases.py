"""Design matrices: row n is phi(x_n)^T."""

import numpy as np


def monomial(x, d):
    """[1, x, ..., x^d], the basis of P1, P2 and P4."""
    return np.vander(x, d + 1, increasing=True)


def legendre(x, d):
    """[P_0(2x - 1), ..., P_d(2x - 1)], the shifted Legendre basis of P3."""
    return np.polynomial.legendre.legvander(2 * x - 1, d)
