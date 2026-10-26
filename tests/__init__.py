"""Shared test inputs: the provided training set of P1 to P3 (N = 10 evenly spaced x)."""

from src.common.data import TRAIN, load

X, T = load(TRAIN)
