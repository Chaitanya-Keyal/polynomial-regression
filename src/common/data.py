"""The six provided CSV files (columns x,y), loaded as given."""

from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
TRAIN, TEST = "poly_train.csv", "poly_test.csv"  # N = 10 (P1 to P3) and N = 100 (P2, P3)
P4_SIZES = (5, 10, 20, 50)  # p4_train_N{n}.csv


def load(name):
    """x and t from one provided file: a header line, then x,y per line."""
    return np.loadtxt(DATA_DIR / name, delimiter=",", skiprows=1, unpack=True)
