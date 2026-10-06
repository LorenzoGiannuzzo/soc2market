"""Common figure style: units in square brackets, capitalised words, boxed legend at the top."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "savefig.dpi": 300,
    "axes.formatter.use_locale": False,
})


def legend_top(ax, ncol=2):
    """Legend centred at the top inside the axes, in a box with a black border."""
    leg = ax.legend(loc="upper center", ncol=ncol, frameon=True, fancybox=False, edgecolor="black")
    leg.get_frame().set_linewidth(0.8)
    return leg


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
