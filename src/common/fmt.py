r"""Numbers as TeX math, shared by the report's macros and the figure labels (mathtext)."""

import math


def num(v, sig=3):
    r"""Plain between 1e-3 and 1e5, otherwise m \times 10^{e}."""
    if v == 0:
        return "0"
    mantissa, exp = f"{v:.{sig - 1}e}".split("e")  # the exponent after rounding: 0.99996 -> 1.00
    if -3 <= int(exp) < 5:
        return f"{v:.{max(sig - 1 - int(exp), 0)}f}"
    return rf"{mantissa} \times 10^{{{int(exp)}}}"


def lam(v):
    r"""A grid value of lambda: 10^{k} for an exact power of ten, otherwise m \times 10^{k}."""
    k = math.log10(v)
    if abs(k - round(k)) < 1e-9:
        return f"10^{{{round(k)}}}"
    mantissa, exp = f"{v:.2e}".split("e")
    return rf"{mantissa} \times 10^{{{int(exp)}}}"
