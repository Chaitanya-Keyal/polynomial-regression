"""The provided CSV files (columns x,y), loaded as given and never modified."""

from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

TRAIN = "poly_train.csv"  # N = 10, P1 to P3
TEST = "poly_test.csv"  # N = 100, P2 and P3
P4_SIZES = (5, 10, 20, 50)


def p4_train(n):
    return f"p4_train_N{n}.csv"


def load_xy(path):
    """x and t from a provided file: a header line `x,y`, then one point per line."""
    path = Path(path)
    with path.open(encoding="utf-8-sig") as f:
        header = f.readline().strip().replace(" ", "").replace('"', "").lower()
    if header != "x,y":
        raise ValueError(f"{path}: expected the header 'x,y', found {header!r}")
    xy = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2, encoding="utf-8-sig")
    return xy[:, 0], xy[:, 1]
