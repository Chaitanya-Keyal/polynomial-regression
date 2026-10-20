"""Shared test inputs: the assignment's design (N = 10 evenly spaced x) and an arbitrary fixed
target, so that no test depends on the data."""

import numpy as np

X = np.linspace(0, 1, 10)
T = np.array([0.3, -1.2, 0.8, 0.1, -0.5, 1.1, -0.9, 0.4, 0.0, -0.2])
