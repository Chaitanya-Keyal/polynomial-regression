"""The number formatting behind results/numbers.tex, and that the report only asks for keys the
generator writes."""

import re
from pathlib import Path

import pytest

from src import report_numbers as rn

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, "0"),
        (0.1, "0.100"),
        (0.12015, "0.120"),
        (12.345, "12.3"),
        (28131.4, "28131"),
        (15193229.7, r"1.52 \times 10^{7}"),
        (8.26e-11, r"8.26 \times 10^{-11}"),
        (-13779.6, "-13780"),
    ],
)
def test_num(value, expected):
    assert rn.num(value) == expected


def test_lam_uses_powers_of_ten_on_the_grid():
    assert rn.lam(1e-10) == "10^{-10}"
    assert rn.lam(0.01) == "10^{-2}"
    assert rn.lam(5.62341325190349e-07) == r"5.62 \times 10^{-7}"
    assert rn.lam(0.15848931924611143) == r"1.58 \times 10^{-1}"


def test_vec_and_given():
    assert rn.vec([0, -0.928, 1.1171], fixed=3) == r"(0.000,\ -0.928,\ 1.117)"
    assert rn.given(0.05) == "0.05"
    assert rn.given(100.0) == "100"


def test_report_keys_are_all_generated():
    """Every \\res{key} in the report must be a key some builder writes (checked by name, using
    the results if present; skipped before the first run)."""
    report = ROOT / "report" / "report.tex"
    results = ROOT / "results"
    if not report.exists() or not (results / "p4.json").exists():
        pytest.skip("needs report/report.tex and a full run's results/")
    used = set(re.findall(r"\\res(?:rows)?\{([^}]+)\}", report.read_text()))
    assert used - set(rn.macros(results)) == set()
