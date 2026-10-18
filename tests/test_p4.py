"""P4's posterior and predictive distribution, against the textbook formulas with explicit
inverses (fine in a test) and an arbitrary fixed target."""

import numpy as np
import pytest

from src import p4
from src.common.bases import monomial
from src.common.linalg import ridge

X = np.linspace(0, 1, 10)
T = np.array([0.3, -1.2, 0.8, 0.1, -0.5, 1.1, -0.9, 0.4, 0.0, -0.2])
PHI = monomial(X, p4.D)
ZERO = np.zeros(p4.D + 1)  # any mean will do where only the variance is checked


def test_posterior_matches_the_formulas():
    m, L = p4.posterior(PHI, T)
    S = np.linalg.inv(p4.ALPHA * np.eye(p4.D + 1) + p4.BETA * PHI.T @ PHI)
    np.testing.assert_allclose(m, p4.BETA * S @ PHI.T @ T, rtol=1e-10)
    np.testing.assert_allclose(L @ L.T, np.linalg.inv(S), rtol=1e-10)

    x = np.linspace(0, 1, 7)
    mean, var, param_var = p4.predictive(x, m, L)
    phis = monomial(x, p4.D)
    np.testing.assert_allclose(mean, phis @ m, rtol=1e-12)
    np.testing.assert_allclose(param_var, np.einsum("ij,jk,ik->i", phis, S, phis), rtol=1e-10)
    np.testing.assert_allclose(var, 1 / p4.BETA + param_var)


def test_posterior_mean_is_the_map_estimate():
    m, _ = p4.posterior(PHI, T)
    np.testing.assert_allclose(m, ridge(PHI, T, p4.ALPHA / p4.BETA), atol=1e-12)


def test_variance_never_drops_below_the_noise():
    m, L = p4.posterior(PHI, T)
    _, var, param_var = p4.predictive(np.linspace(-0.5, 1.5, 50), m, L)
    assert np.all(param_var > 0)
    assert np.all(var >= 1 / p4.BETA)


def test_variance_ignores_the_targets():
    # s^2(x) depends only on where the inputs are, never on the t values
    _, L1 = p4.posterior(PHI, T)
    _, L2 = p4.posterior(PHI, -3 * T + 1)
    x = np.linspace(0, 1, 5)
    np.testing.assert_allclose(p4.predictive(x, ZERO, L1)[1], p4.predictive(x, ZERO, L2)[1])


@pytest.mark.parametrize(("small", "large"), [(5, 10), (10, 20), (20, 50)])
def test_more_evenly_spaced_points_shrink_the_band(small, large):
    x = np.linspace(0, 1, 101)
    widths = []
    for n in (small, large):
        _, L = p4.posterior(monomial(np.linspace(0, 1, n), p4.D), np.zeros(n))
        widths.append(p4.predictive(x, ZERO, L)[1])
    assert np.all(widths[1] < widths[0])
