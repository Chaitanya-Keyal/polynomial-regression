"""Shared figure style and helpers.

sin(2 pi x) is defined here and nowhere else, because it is only ever plotted, never fitted
(assignment, "Provided Files"). Importing this module sets the style for every figure.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Categorical slots, always taken in this order. Checked for colour-blind separation on a light
# surface; aqua and yellow are below 3:1 contrast, so figures that use them carry a legend or
# direct labels.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SERIES = (BLUE, ORANGE, AQUA, YELLOW)
# One hue, light to dark, for ordered series (P4's N = 5, 10, 20, 50).
RAMP = ("#86b6ef", "#3987e5", "#1c5cab", "#0d366b")
INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

mpl.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": AXIS,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.minor.size": 0,
        "ytick.minor.size": 0,
        "lines.linewidth": 1.6,
        "lines.markersize": 5,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
    }
)

X_DENSE = np.linspace(0, 1, 200)  # the plotting grid for every fitted curve


def true_function(x):
    return np.sin(2 * np.pi * x)


def truth(ax, x=X_DENSE):
    ax.plot(x, true_function(x), color=INK_2, linestyle="--", linewidth=1.2, label="sin(2πx)")


def points(ax, x, t, label="training data"):
    ax.scatter(x, t, s=22, color=INK, zorder=3, label=label)


def end_label(ax, x, y, text, dy=0.0):
    """A direct label in ink just right of a line's last point (identity never by colour alone)."""
    ax.annotate(
        text,
        (x[-1], y[-1]),
        xytext=(4, dy),
        textcoords="offset points",
        va="center",
        fontsize=8,
        color=INK,
    )


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
