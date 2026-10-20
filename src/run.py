"""Runs the problems: results/<p>.json (every reported number), figures/<name>.pdf and
results/numbers.tex (the same numbers as LaTeX macros for the report).

python -m src.run              all four problems
python -m src.run p2 p3        some of them
python -m src.run --out DIR    write results/ and figures/ under DIR instead of the repo
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src import p1, p2, p3, p4, report_numbers

PROBLEMS = {"p1": p1, "p2": p2, "p3": p3, "p4": p4}


def main():
    parser = argparse.ArgumentParser(description="Run the problems and write their outputs.")
    parser.add_argument("problems", nargs="*", help="any of p1 p2 p3 p4 (default: all)")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    if unknown := set(args.problems) - set(PROBLEMS):
        parser.error(f"unknown problem(s): {', '.join(sorted(unknown))}")

    results, figures = args.out / "results", args.out / "figures"
    results.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    for name in args.problems or PROBLEMS:
        start = time.perf_counter()
        res = PROBLEMS[name].compute()
        (results / f"{name}.json").write_text(json.dumps(res["numbers"], indent=2) + "\n")
        for fig_name, fig in PROBLEMS[name].figures(res).items():
            # no creation date in the PDF, so a rerun gives byte-identical files
            fig.savefig(figures / f"{fig_name}.pdf", metadata={"CreationDate": None})
            plt.close(fig)
        print(f"{name}: {time.perf_counter() - start:.1f} s")
    report_numbers.write(results)


if __name__ == "__main__":
    main()
