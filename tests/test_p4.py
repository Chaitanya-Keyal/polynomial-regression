"""P4's posterior and predictive distribution against the formulas, on an arbitrary target."""

import numpy as np

from src import p4
from src.common.bases import monomial
from src.common.linalg import ridge

PHI = monomial(np.linspace(0, 1, 10), p4.D)
T = np.array([0.3, -1.2, 0.8, 0.1, -0.5, 1.1, -0.9, 0.4, 0.0, -0.2])


def test_posterior_and_predictive():
    m, A = p4.posterior(PHI, T)
    np.testing.assert_allclose(A @ m, p4.BETA * PHI.T @ T)  # m_N = beta S_N Phi^T t
    x = np.linspace(0, 1, 7)
    mean, var = p4.predictive(x, m, A)
    G = monomial(x, p4.D)
    S_phi = np.linalg.solve(A, G.T)  # S_N phi(x), one column per x
    np.testing.assert_allclose(mean, G @ m)
    np.testing.assert_allclose(var, 1 / p4.BETA + np.einsum("ij,ji->i", G, S_phi))
    assert np.all(var > 1 / p4.BETA)


def test_posterior_mean_is_the_map_estimate():
    np.testing.assert_allclose(p4.posterior(PHI, T)[0], ridge(PHI, T, p4.LAM), atol=1e-12)
