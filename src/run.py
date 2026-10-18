"""Runs the problems: results/<p>.json (every reported number), figures/<name>.pdf and .png,
and results/numbers.tex (the same numbers as LaTeX macros for the report).

python -m src.run                         every problem, reading data/, writing to the repo
python -m src.run p1 p3                   some of them
python -m src.run --data DIR --out DIR    other input and output directories
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src import p1, p2, p3, p4, report_numbers
from src.common.data import DATA_DIR

PROBLEMS = {"p1": p1, "p2": p2, "p3": p3, "p4": p4}
ROOT = Path(__file__).resolve().parents[1]


def _plain(obj):
    """json.dumps fallback for numpy scalars and arrays."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def save_figure(fig, stem):
    # no creation date in the PDF, so a rerun on the same data gives identical files
    fig.savefig(stem.with_suffix(".pdf"), metadata={"CreationDate": None})
    fig.savefig(stem.with_suffix(".png"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("problems", nargs="*", help="any of p1 p2 p3 p4 (default: all)")
    parser.add_argument("--data", type=Path, default=DATA_DIR, help="directory with the CSVs")
    parser.add_argument("--out", type=Path, default=ROOT, help="gets results/ and figures/")
    args = parser.parse_args()
    unknown = set(args.problems) - set(PROBLEMS)
    if unknown:
        parser.error(f"unknown problem(s): {', '.join(sorted(unknown))}")

    results, figures = args.out / "results", args.out / "figures"
    results.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    for name in args.problems or PROBLEMS:
        start = time.perf_counter()
        module = PROBLEMS[name]
        res = module.compute(args.data)
        text = json.dumps(res["numbers"], indent=2, default=_plain)
        (results / f"{name}.json").write_text(text + "\n")
        for fig_name, fig in module.figures(res).items():
            save_figure(fig, figures / fig_name)
        print(f"{name}: {time.perf_counter() - start:.1f} s")
    report_numbers.write(results)
    print(f"wrote {results} and {figures}")


if __name__ == "__main__":
    main()
