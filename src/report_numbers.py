r"""results/*.json -> results/numbers.tex: one \resdef{key}{value} per number the report quotes.

The report reads them with \res{key}; a key that is not defined prints a bold ??. Values are
written for math mode.

python -m src.report_numbers
"""

import json
from pathlib import Path

from src import p1, p2, p3, p4
from src.common.fmt import lam, num

RESULTS = Path(__file__).resolve().parents[1] / "results"


def rounding(v):
    """A value set by rounding alone (the floating-point floor): one significant figure, since
    another linear-algebra library can change its leading digit."""
    return num(v, 1)


def given(v):
    """A parameter as the assignment states it (0.1, not 0.100)."""
    return f"{v:g}"


def vec(values, sig=3, fixed=None):
    """A weight vector: fixed decimals when given, otherwise significant digits."""
    items = [f"{v:.{fixed}f}" if fixed is not None else num(v, sig) for v in values]
    return "(" + r",\ ".join(items) + ")"


def thousands(k):
    return f"{k:,}".replace(",", "{,}")


def p1_macros(n):
    rows = []
    for d, c in n["closed"].items():
        f = rounding if int(d) == p1.D else num  # d = 9 interpolates, so its error is rounding
        rows.append(rf"{d} & ${f(c['train_E'])}$ & ${f(c['train_rmse'])}$ \\")
    m = {
        "p1-seed": str(p1.SEED),
        "p1-eta": given(p1.ETA),
        "p1-epochs": str(p1.EPOCHS),
        "p1-cond-phi": num(n["cond_phi"]),
        "p1-cond-q": num(n["cond_q"], 4),
        "p1-wclosed-norm": num(n["w_closed_norm"]),
        "p1-sgd-gap": rounding(n["sgd_batch_rel_gap"]),
        "p1-E0": num(n["E_0"], 4),
        "p1-closed-rows": " ".join(rows),
        "p1-w-rows": " ".join(
            rf"{j} & ${num(wc)}$ & ${num(w100)}$ \\"
            for j, (wc, w100) in enumerate(
                zip(n["closed"][str(p1.D)]["w"], n["w_100_too_large"], strict=True)
            )
        ),
        "p1-lr-small-eta": given(p1.LR_CASES["too small"]),
        "p1-lr-good-eta": given(p1.LR_CASES["well chosen"]),
        "p1-lr-large-eta": given(p1.LR_CASES["too large"]),
        "p1-lr-small-E100": num(n["E_100"]["too small"], 4),
        "p1-lr-good-E100": rounding(n["E_100"]["well chosen"]),  # at the floating-point floor
        "p1-lr-large-E100": num(n["E_100"]["too large"], 4),
        "p1-lr-ratio": num(n["w_100_over_w_closed"], 5),
    }
    for v, data in n["verification"].items():
        m[f"p1-{v}-abs"] = rounding(data["abs_err"])
        m[f"p1-{v}-rel"] = rounding(data["rel_err"])
        m[f"p1-{v}-final"] = rounding(data["final_E"])
        m[f"p1-{v}-factor"] = num(data["per_epoch_factor"], 4)
        m[f"p1-{v}-epochs12"] = str(data["epochs_to_1e-12"])
    return m


def p2_macros(n):
    gd, flat = n["gd"], n["flat"]
    m = {
        "p2-gd-lam": lam(p2.GD_LAM),
        "p2-gd-eta": given(p2.GD_ETA),
        "p2-gd-iters": thousands(gd["iterations"]),
        "p2-gd-obj": num(gd["objective"], 7),
        "p2-gd-obj-closed": num(gd["objective_closed"], 7),
        "p2-cond-before": num(gd["cond_PtP"]),
        "p2-cond-after": num(gd["cond_PtP_lam"]),
        "p2-flat-lo": lam(flat["lam_lo"]),
        "p2-flat-hi": lam(flat["lam_hi"]),
        "p2-flat-n": str(flat["n_run"]),
        "p2-flat-others": ", ".join(lam(v) for v in flat["others"]),
        "p2-flat-peak-lam": lam(flat["peak_lam"]),
        "p2-flat-peak-ratio": num(flat["peak_ratio"], 4),
    }
    for name, panel in n["panels"].items():
        m[f"p2-{name}-lam"] = lam(panel["lam"])
        m[f"p2-{name}-wnorm"] = num(panel["w_norm"])
        m[f"p2-{name}-train"] = num(panel["train_rmse"], 4)
        m[f"p2-{name}-test"] = num(panel["test_rmse"], 4)
    return m


def p3_macros(n):
    c, opt = n["check"], n["check"]["optimality"]
    m = {
        "p3-eta": given(p3.ETA),
        "p3-iters": thousands(p3.ITERS),
        "p3-cond": num(n["cond_PtP"]),
        "p3-check-lam": given(p3.CHECK_LAM),
        "p3-check-E": num(c["E_final"], 4),
        "p3-check-Eref": given(p3.E_REF),
        "p3-check-w": vec(c["w"], fixed=3),
        "p3-check-wref": vec(p3.W_REF, fixed=3),
        "p3-check-wdiff": num(c["w_minus_w_ref"]),
        "p3-opt-active": num(opt["max_active_violation"]),
        "p3-opt-zero": num(opt["max_zero_ratio"]),
        "p3-opt-nactive": str(opt["n_active"]),
        "p3-opt-nzero": str(opt["n_zero"]),
        "p3-nzero-star": str(len(n["zero_idx_star"])),
        "p3-zero-idx-star": ", ".join(str(j) for j in n["zero_idx_star"]),
        "p3-zero-rows": " ".join(
            rf"${lam(s['lam'])}$ & {len(s['zero_idx'])} & "
            rf"{', '.join(str(j) for j in s['zero_idx']) or '--'} & "
            rf"${num(s['train_rmse'])}$ & ${num(s['test_rmse'])}$ \\"
            for s in n["sweep"]
        ),
    }
    for name, row in n["table"].items():
        m[f"p3-{name}-lam"] = lam(row["lam"])
        m[f"p3-{name}-test"] = num(row["test_rmse"], 4)
        m[f"p3-{name}-wnorm"] = num(row["w_norm"])
        m[f"p3-{name}-nsmall"] = str(row["n_small"])
    return m


def p4_macros(n):
    m = {
        "p4-alpha": given(p4.ALPHA),
        "p4-beta": given(p4.BETA),
        "p4-lam": given(p4.LAM),
        "p4-mN": vec(n["m_N"], 4),
        "p4-wmap": vec(n["w_map"], 4),
        "p4-diff": rounding(n["w_map_minus_m_N"]),
        "p4-floor": num(p4.FLOOR),
    }
    for size, w in n["width"].items():
        m[f"p4-width-{size}-mean"] = num(w["mean"])
        m[f"p4-width-{size}-max"] = num(w["max"])
        m[f"p4-width-{size}-share"] = num(100 * w["param_share"], 2)
    return m


BUILDERS = {"p1": p1_macros, "p2": p2_macros, "p3": p3_macros, "p4": p4_macros}


def macros(results_dir=RESULTS):
    out = {}
    for name, build in BUILDERS.items():
        path = Path(results_dir) / f"{name}.json"
        if path.exists():
            out.update(build(json.loads(path.read_text())))
    return out


def write(results_dir=RESULTS):
    lines = [
        "% Generated by python -m src.report_numbers from results/*.json; do not edit.",
        *(rf"\resdef{{{key}}}{{{value}}}" for key, value in macros(results_dir).items()),
    ]
    (Path(results_dir) / "numbers.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    write()
