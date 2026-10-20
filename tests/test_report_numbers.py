"""The number formatting behind results/numbers.tex, and that the report and the generator use
exactly the same keys."""

import re
from pathlib import Path

import pytest

from src import report_numbers as rn
from src.common.fmt import lam, num

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, "0"),
        (0.12015, "0.120"),
        (28131.4, "28131"),
        (15193229.7, r"1.52 \times 10^{7}"),
        (8.26e-11, r"8.26 \times 10^{-11}"),
    ],
)
def test_num(value, expected):
    assert num(value) == expected


def test_lam_given_and_rounding():
    assert lam(1e-10) == "10^{-10}"
    assert lam(1.7782794100389227e-07) == r"1.78 \times 10^{-7}"
    assert rn.given(0.05) == "0.05"
    assert rn.rounding(4.06e-6) == r"4 \times 10^{-6}"


def test_report_uses_every_key_and_only_generated_ones():
    report = (ROOT / "report" / "report.tex").read_text()
    used = set(re.findall(r"\\res(?:rows)?\{([^}]+)\}", report))
    assert used == set(rn.macros(ROOT / "results"))
