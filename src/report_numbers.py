r"""results/*.json -> results/numbers.tex: one \resdef{key}{value} per number the report quotes.

The report reads them with \res{key}, which prints a bold ?? for any key not yet defined, so
it compiles before the data exists. Values are written for math mode.

python -m src.report_numbers [RESULTS_DIR]
"""

import json
import math
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"


def num(v, sig=3):
    """A number for math mode: plain between 1e-3 and 1e5, otherwise m \times 10^{e}."""
    if v == 0:
        return "0"
    if 1e-3 <= abs(v) < 1e5:
        digits = max(sig - 1 - math.floor(math.log10(abs(v))), 0)
        return f"{v:.{digits}f}"
    mantissa, exp = f"{v:.{sig - 1}e}".split("e")
    return rf"{mantissa} \times 10^{{{int(exp)}}}"


def given(v):
    """A parameter as the assignment states it (0.1, not 0.100)."""
    return f"{v:g}"


def lam(v):
    """A grid value of lambda, as 10^{k} when it is an exact power of ten."""
    k = math.log10(v)
    if abs(k - round(k)) < 1e-9:
        return f"10^{{{round(k)}}}"
    return num(v)


def vec(values, sig=3, fixed=None):
    """A weight vector: fixed decimals when given, otherwise significant digits."""
    items = [f"{v:.{fixed}f}" if fixed is not None else num(v, sig) for v in values]
    return r"(" + r",\ ".join(items) + r")"


def p1_macros(n):
    m = {
        "p1-seed": str(n["seed"]),
        "p1-eta": given(n["eta"]),
        "p1-epochs": str(n["epochs"]),
        "p1-N": str(n["N"]),
        "p1-cond-phi": num(n["cond_phi"]),
        "p1-cond-q": num(n["cond_q"], 4),
        "p1-wclosed-norm": num(n["w_closed_norm"]),
        "p1-wclosed": vec(n["closed"]["9"]["w"]),
        "p1-sgd-gap-abs": num(n["sgd_vs_batch"]["max_abs_diff"], 2),
        "p1-sgd-gap-rel": num(n["sgd_vs_batch"]["max_rel_diff_while_E_above_1e-10"], 2),
        "p1-lr-ratio": num(n["divergent_w_over_w_closed"], 5),
        "p1-lr-large-w": vec(n["lr"]["too large"]["w_100"]),
    }
    for v, data in n["verification"].items():
        m[f"p1-{v}-abs"] = num(data["abs_err"])
        m[f"p1-{v}-rel"] = num(data["rel_err"])
        m[f"p1-{v}-final"] = num(data["final_E"], 2)
        m[f"p1-{v}-factor"] = num(data["per_epoch_factor"], 4)
        m[f"p1-{v}-epochs12"] = str(data["epochs_to_1e-12"])
    for label, key in (
        ("too small", "small"),
        ("well chosen", "good"),
        ("too large", "large"),
        ("edge", "edge"),
    ):
        data = n["lr"][label]
        m[f"p1-lr-{key}-eta"] = given(data["eta"])
        m[f"p1-lr-{key}-E0"] = num(data["E_0"], 4)
        m[f"p1-lr-{key}-E100"] = num(data["E_100"], 4)
    m["p1-closed-rows"] = " ".join(
        rf"{d} & ${num(c['train_E'])}$ & ${num(c['train_rmse'])}$ \\"
        for d, c in n["closed"].items()
    )
    return m


def p2_macros(n):
    gd, flat = n["gd"], n["flat"]
    m = {
        "p2-gd-lam": lam(gd["lam"]),
        "p2-gd-eta": given(gd["eta"]),
        "p2-gd-iters": f"{gd['iterations']:,}".replace(",", "{,}"),
        "p2-gd-relerr": num(gd["rel_err"]),
        "p2-gd-obj": num(gd["objective"], 7),
        "p2-gd-obj-closed": num(gd["objective_closed"], 7),
        "p2-cond-before": num(gd["cond_PtP"]),
        "p2-cond-after": num(gd["cond_PtP_lam"]),
        "p2-gd-limit": num(gd["stability_limit"]),
        "p2-lamstar": lam(n["lam_star"]),
        "p2-test-star": num(n["test_rmse_star"], 4),
        "p2-train-star": num(n["train_rmse_star"], 4),
        "p2-flat-lo": lam(flat["lam_lo"]),
        "p2-flat-hi": lam(flat["lam_hi"]),
        "p2-flat-n": str(flat["grid_points"]),
        "p2-flat-contig": "yes" if flat["contiguous"] else "no",
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
        "p3-eta": given(n["eta"]),
        "p3-iters": f"{n['iters']:,}".replace(",", "{,}"),
        "p3-cond": num(n["cond_PtP"]),
        "p3-limit": num(n["stability_limit"]),
        "p3-check-lam": given(c["lam"]),
        "p3-check-E": num(c["E_final"], 4),
        "p3-check-Ebest": num(c["E_best"], 4),
        "p3-check-Ecd": num(c["E_cd"], 4),
        "p3-check-Eref": num(c["E_ref"], 4),
        "p3-check-w": vec(c["w"], fixed=3),
        "p3-check-wcd": vec(c["w_cd"], fixed=3),
        "p3-check-wref": vec(c["w_ref"], fixed=3),
        "p3-check-wdiff": num(c["w_minus_w_ref"]),
        "p3-check-cddiff": num(c["w_cd_minus_w_ref"]),
        "p3-check-wcd-diff": num(c["w_minus_w_cd"]),
        "p3-opt-active": num(opt["max_active_violation"]),
        "p3-opt-zero": num(opt["max_zero_ratio"]),
        "p3-opt-nactive": str(opt["n_active"]),
        "p3-opt-nzero": str(opt["n_zero"]),
        "p3-lamstar": lam(n["lam_star"]),
        "p3-test-star": num(n["test_rmse_star"], 4),
        "p3-nzero-star": str(n["n_zero_star"]),
        "p3-zero-idx-star": ", ".join(str(j) for j in n["zero_idx_star"]) or "none",
        "p3-ridge-lam": lam(n["ridge"]["lam_best"]),
        "p3-zero-rows": " ".join(
            rf"${lam(s['lam'])}$ & {s['n_zero']} & "
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
        m[f"p3-{name}-w"] = vec(row["w"], fixed=3)
    return m


def p4_macros(n):
    m = {
        "p4-alpha": given(n["alpha"]),
        "p4-beta": given(n["beta"]),
        "p4-lam": given(n["lam"]),
        "p4-mN": vec(n["m_N"], 4),
        "p4-wmap": vec(n["w_map"], 4),
        "p4-diff": num(n["w_map_minus_m_N"], 2),
        "p4-curve-gap": num(n["max_curve_gap"], 2),
        "p4-floor": num(n["noise_floor_width"]),
    }
    for size, w in n["width"].items():
        m[f"p4-width-{size}-mean"] = num(w["mean"])
        m[f"p4-width-{size}-min"] = num(w["min"])
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
    write(sys.argv[1] if len(sys.argv) > 1 else RESULTS)
