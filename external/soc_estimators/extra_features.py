"""
Three complementary figures for the SOC ensemble paper.

FIGURE 1 (figure_metrics_3panel.*) — per-scenario view
    Panel (a): RMSE vs MAE — scatter, marker area proportional to scenario duration.
    Panel (b): MaxAE vs execution time (log-log) — scatter + transparent
               confidence ellipses per model.
    Panel (c): Average RMSE vs average MaxAE — one square per model.

FIGURE 2 (figure_metrics_3panel_cluster.*) — cluster-aggregated view
    Same three panels, but each point is the mean over the scenarios of a
    given regime cluster (DWR-SOC, QSN, HSP) or the overall test-set average.
    Marker shape encodes the group (circle = DWR-SOC, diamond = QSN,
    triangle = HSP, square = Average); colour encodes the estimator.
    Panel (b) also shows per-model envelopes built on the 4 cluster points.
    The Average point of each model is highlighted with a halo of the same
    colour to make it stand out in all three panels.
    Legend handles for models are rectangles; cluster handles keep their
    marker shape to encode the group.

FIGURE 3 (figure_metrics_3panel_combined.*) — combined paper figure
    Panel (a): RMSE vs MAE per scenario (same as Figure 1 panel a).
    Panel (b): MaxAE vs execution time per scenario, log-log, with envelopes
               (same as Figure 1 panel b).
    Panel (c): RMSE vs MaxAE aggregated per cluster (same as Figure 2 panel c),
               with a dashed black Pareto front computed over all visible
               (model, cluster) points.

All six estimators are plotted in every panel: EKF+ANN, EKF+CNN, EKF+LSTM,
EKF+CatBoost, EKF+TFT, Ensemble metamodel.

Outputs:
    figure_metrics_3panel.pdf            (vector, paper-ready)
    figure_metrics_3panel.png            (raster, 500 dpi)
    figure_metrics_3panel_cluster.pdf    (vector, paper-ready)
    figure_metrics_3panel_cluster.png    (raster, 500 dpi)
    figure_metrics_3panel_combined.pdf   (vector, paper-ready)
    figure_metrics_3panel_combined.png   (raster, 500 dpi)

Run with:
    python3 figure_metrics_3panel.py
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ---------------------------------------------------------------------------
# Data (scenarios renumbered 1-12)
# ---------------------------------------------------------------------------
SCENARIOS = [f"S{i+1}" for i in range(12)]

COLORS = {
    "ANN":      "#1f77b4",
    "CNN":      "#7fcdff",
    "LSTM":     "#d35400",
    "CatBoost": "#f39c12",
    "TFT":      "#9b59b6",
    "Ensemble": "#1b5e20",
}
MODELS = ["ANN", "CNN", "LSTM", "CatBoost", "TFT", "Ensemble"]
LABELS = {m: ("Ensemble" if m == "Ensemble" else f"EKF+{m}") for m in MODELS}

DURATIONS = [
    168.00, 48.00, 96.00, 65.50, 48.00, 48.00,
    98.00, 65.50, 65.50, 144.00, 48.00, 48.00,
]

DATA = {
    "rmse": {
        "ANN":      [1.08,1.09,1.93,0.62,2.00,1.25,1.17,1.09,0.77,2.74,2.21,1.60],
        "CNN":      [1.25,1.87,1.75,1.07,1.80,1.57,1.73,1.73,0.77,2.55,1.75,1.65],
        "LSTM":     [1.09,1.27,1.30,0.99,2.08,1.37,0.67,0.74,0.98,2.82,2.29,1.79],
        "CatBoost": [1.04,1.14,1.77,0.71,1.37,1.09,1.14,0.89,0.64,2.08,1.40,1.20],
        "TFT":      [0.27,6.02,0.60,0.36,0.11,2.87,3.28,3.25,0.51,0.94,0.51,0.67],
        "Ensemble": [0.58,0.87,0.64,0.35,1.71,0.62,1.26,1.33,0.43,2.30,1.77,1.40],
    },
    "mae": {
        "ANN":      [0.83,0.57,1.61,0.55,1.96,1.02,0.97,0.94,0.57,2.64,2.13,1.56],
        "CNN":      [1.00,0.92,1.34,0.78,1.59,1.11,1.36,1.37,0.62,2.34,1.61,1.49],
        "LSTM":     [0.90,1.19,1.17,0.82,1.99,1.28,0.58,0.56,0.91,2.71,2.23,1.75],
        "CatBoost": [0.89,1.06,1.40,0.59,1.12,1.00,0.89,0.75,0.52,1.81,1.18,1.04],
        "TFT":      [0.18,0.79,0.37,0.19,0.04,0.69,0.68,0.60,0.22,0.19,0.15,0.13],
        "Ensemble": [0.38,0.30,0.35,0.23,1.56,0.23,0.81,0.89,0.23,2.13,1.65,1.29],
    },
    "maxae": {
        "ANN":      [5.91,14.16,7.13,2.66,2.88,11.93,4.54,3.16,2.64,3.98,3.29,2.32],
        "CNN":      [12.84,45.46,19.86,4.11,3.89,26.23,7.37,5.65,2.83,4.96,3.50,2.97],
        "LSTM":     [7.92,9.38,5.95,3.14,3.32,9.55,4.42,2.11,3.75,4.67,3.33,2.98],
        "CatBoost": [4.00,3.73,4.11,2.85,3.00,3.73,3.23,2.40,2.22,4.32,3.26,2.73],
        "TFT":      [3.40,93.46,3.05,1.76,2.93,22.76,37.63,36.63,2.49,8.61,6.43,7.14],
        "Ensemble": [12.84,45.46,19.86,4.11,3.32,26.23,7.37,5.65,2.62,4.60,3.29,2.81],
    },
    "time": {
        "ANN":      [31.1,6.61,16.74,11.03,7.31,7.32,8.34,21.23,8.12,23.5,6.89,11.8],
        "CNN":      [85.19,16.85,31.75,41.99,15.84,26.21,18.44,33.29,25.81,80.2,17.05,21.6],
        "LSTM":     [155.05,36.4,61.00,83.2,47.35,82.8,60.96,82.91,46.86,196.16,33.29,42.17],
        "CatBoost": [2.21,0.4,0.69,0.49,0.38,0.57,0.52,0.57,0.32,1.31,0.51,0.36],
        "TFT":      [1346.28,320.69,632.09,1157.92,327.66,210.3,284.29,509.47,294.28,995.8,322.15,347.93],
        "Ensemble": [3174.2,916.82,1613.21,1244.75,771.21,479.78,5213.68,1195.69,1055.33,2811.02,778.93,899.86],
    },
}

# ---------------------------------------------------------------------------
# Cluster definition (0-based scenario indices in the 12-element arrays)
# ---------------------------------------------------------------------------
CLUSTERS = {
    "DWR-SOC": [0, 1, 2, 5, 6, 7],
    "QSN":     [4, 9, 10, 11],
    "HSP":     [3, 8],
    "Average": list(range(12)),
}
CLUSTER_ORDER = ["DWR-SOC", "QSN", "HSP", "Average"]
CLUSTER_MARKERS = {
    "DWR-SOC": "o",   # circle
    "QSN":     "D",   # diamond
    "HSP":     "^",   # triangle
    "Average": "s",   # square
}

# Halo for the Average point: size in points^2, alpha, and no edge.
AVERAGE_HALO_SIZE = 750
AVERAGE_HALO_ALPHA = 0.22


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def duration_to_size(duration_h):
    """Linear map 48 h -> 30, 168 h -> 250 (units = matplotlib points squared)."""
    return 30.0 + (duration_h - 48.0) * (250.0 - 30.0) / (168.0 - 48.0)


SIZES = np.array([duration_to_size(d) for d in DURATIONS])


def cluster_mean(model, metric, cluster_name):
    indices = CLUSTERS[cluster_name]
    values = np.asarray(DATA[metric][model], dtype=float)[indices]
    return float(np.mean(values))


def confidence_ellipse(ax, x_data, y_data, color, alpha_fill=0.18,
                       alpha_edge=0.55, margin=1.05,
                       x_log=False, y_log=False):
    """Confidence ellipse on (linear or log) axes, sized to enclose all
    points + small margin."""
    x = np.asarray(x_data, dtype=float)
    y = np.asarray(y_data, dtype=float)

    x_proj = np.log10(x) if x_log else x
    y_proj = np.log10(y) if y_log else y

    if len(x_proj) < 3:
        return
    cov = np.cov(x_proj, y_proj)
    if not np.isfinite(np.linalg.det(cov)) or np.linalg.det(cov) <= 0:
        return

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
    cx = float(np.mean(x_proj))
    cy = float(np.mean(y_proj))

    cov_inv = np.linalg.inv(cov)
    diffs = np.column_stack([x_proj - cx, y_proj - cy])
    md = np.sqrt(np.einsum("ij,jk,ik->i", diffs, cov_inv, diffs))
    n_std = float(np.max(md)) * margin

    width = 2.0 * n_std * np.sqrt(eigvals[0])
    height = 2.0 * n_std * np.sqrt(eigvals[1])

    t = np.linspace(0.0, 2.0 * np.pi, 200)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    x_ell_proj = cx + (width / 2.0) * np.cos(t) * cos_a - (height / 2.0) * np.sin(t) * sin_a
    y_ell_proj = cy + (width / 2.0) * np.cos(t) * sin_a + (height / 2.0) * np.sin(t) * cos_a

    x_ell = 10.0 ** x_ell_proj if x_log else x_ell_proj
    y_ell = 10.0 ** y_ell_proj if y_log else y_ell_proj

    ax.fill(x_ell, y_ell, color=color, alpha=alpha_fill, zorder=1, linewidth=0)
    ax.plot(x_ell, y_ell, color=color, alpha=alpha_edge, linewidth=1.0, zorder=2)


def draw_average_halo(ax, x, y, color,
                      size=AVERAGE_HALO_SIZE, alpha=AVERAGE_HALO_ALPHA):
    """Soft circular halo around the Average point, in the model's colour."""
    ax.scatter(x, y, s=size, color=color, alpha=alpha,
               edgecolors="none", zorder=2)


def compute_pareto_front_2d(x, y):
    """Return Pareto-optimal points (minimizing both x and y), sorted by x.

    A point is Pareto-optimal if no other point has both lower x and lower y.
    Returns two lists (x_front, y_front) that can be plotted directly.
    """
    pts = sorted(zip(x, y), key=lambda p: (p[0], p[1]))
    front_x, front_y = [], []
    best_y = float("inf")
    for px, py in pts:
        if py < best_y:
            front_x.append(px)
            front_y.append(py)
            best_y = py
    return front_x, front_y


def make_combined_legend(ax, models, clusters, color_map, label_map,
                         marker_map, loc_models="upper left",
                         loc_clusters="lower right",
                         extra_handle=None):
    """Build two legends on the same axis: one for model colours (rectangles),
    one for cluster marker shapes. Optionally append an extra handle to the
    cluster legend (used for the Pareto-front entry)."""
    model_handles = [
        Patch(facecolor=color_map[m], edgecolor="black", linewidth=0.5,
              label=label_map[m])
        for m in models
    ]
    cluster_handles = [
        Line2D([0], [0], marker=marker_map[cl], color="white",
               markerfacecolor="lightgray", markeredgecolor="black",
               markeredgewidth=0.6, markersize=9,
               label=cl, linestyle="None")
        for cl in clusters
    ]
    if extra_handle is not None:
        cluster_handles.append(extra_handle)

    leg1 = ax.legend(handles=model_handles, loc=loc_models, ncol=2,
                     frameon=True, framealpha=0.95, edgecolor="lightgray")
    ax.add_artist(leg1)
    ax.legend(handles=cluster_handles, loc=loc_clusters, title="Group",
              frameon=True, framealpha=0.95, edgecolor="lightgray")


# ===========================================================================
# FIGURE 1 — per-scenario
# ===========================================================================
fig, (ax1, ax2, ax3) = plt.subplots(
    3, 1,
    figsize=(7.5, 13.5),
    gridspec_kw={"hspace": 0.30},
)

# Panel (a): RMSE vs MAE (per scenario)
xmax_a = max(max(v) for v in DATA["rmse"].values()) * 1.05
ymax_a = max(max(v) for v in DATA["mae"].values()) * 1.05

for m in MODELS:
    ax1.scatter(
        DATA["rmse"][m], DATA["mae"][m],
        s=SIZES, color=COLORS[m],
        edgecolors="black", linewidths=0.5,
        alpha=0.88, zorder=3,
        label=LABELS[m],
    )

ax1.set_xlim(0, xmax_a)
ax1.set_ylim(0, ymax_a)
ax1.set_xlabel("RMSE [%]")
ax1.set_ylabel("MAE [%]")
ax1.set_title("(a) Mean error metrics: RMSE vs MAE", loc="left", fontweight=500)
ax1.grid(True, linestyle=":", alpha=0.45, zorder=0)
ax1.set_axisbelow(True)
ax1.legend(loc="upper left", frameon=True, framealpha=0.95,
           edgecolor="lightgray", ncol=2)

# Panel (b): MaxAE vs execution time (log-log, per scenario)
for m in MODELS:
    confidence_ellipse(ax2, DATA["time"][m], DATA["maxae"][m], COLORS[m],
                       x_log=True, y_log=True)
for m in MODELS:
    ax2.scatter(
        DATA["time"][m], DATA["maxae"][m],
        s=SIZES, color=COLORS[m],
        edgecolors="black", linewidths=0.5,
        alpha=0.88, zorder=3,
    )

ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel("Execution time [s] (log)")
ax2.set_ylabel("MaxAE [%] (log)")
ax2.set_title(
    "(b) Worst-case error vs execution time (with per-model envelopes)",
    loc="left", fontweight=500,
)
ax2.grid(True, which="both", linestyle=":", alpha=0.4, zorder=0)
ax2.set_axisbelow(True)

size_handles = []
for d in [48, 96, 168]:
    s_pts = duration_to_size(d)
    size_handles.append(
        Line2D([0], [0], marker="o", color="white",
               markerfacecolor="lightgray", markeredgecolor="black",
               markeredgewidth=0.5, markersize=np.sqrt(s_pts),
               label=f"{d:.0f} h", linestyle="None")
    )
ax2.legend(handles=size_handles, loc="lower right",
           title="Scenario duration", frameon=True, framealpha=0.95,
           edgecolor="lightgray", labelspacing=1.2, borderpad=0.8)

# Panel (c): Average RMSE vs MaxAE
agg_rmse = np.array([np.mean(DATA["rmse"][m]) for m in MODELS])
agg_maxae = np.array([np.mean(DATA["maxae"][m]) for m in MODELS])

for i, m in enumerate(MODELS):
    ax3.scatter(
        agg_rmse[i], agg_maxae[i],
        s=200, color=COLORS[m], marker="s",
        edgecolors="black", linewidths=0.8,
        alpha=0.92, zorder=3,
        label=f"{LABELS[m]} (avg)",
    )

ax3.set_xlim(1.0, max(agg_rmse) * 1.10)
ax3.set_ylim(0, max(agg_maxae) * 1.12)
ax3.set_xlabel("RMSE [%]")
ax3.set_ylabel("MaxAE [%]")
ax3.set_title("(c) Average RMSE vs MaxAE", loc="left", fontweight=500)
ax3.grid(True, linestyle=":", alpha=0.4, zorder=0)
ax3.set_axisbelow(True)
ax3.legend(loc="upper right", frameon=True, framealpha=0.95,
           edgecolor="lightgray", ncol=1)

plt.savefig("figure_metrics_3panel.pdf", dpi=500, bbox_inches="tight")
plt.savefig("figure_metrics_3panel.png", dpi=500, bbox_inches="tight")


# ===========================================================================
# FIGURE 2 — cluster-aggregated view
# ===========================================================================
fig2, (bx1, bx2, bx3) = plt.subplots(
    3, 1,
    figsize=(7.5, 13.5),
    gridspec_kw={"hspace": 0.30},
)

# ---------------------------------------------------------------------------
# Panel (a): RMSE vs MAE per cluster + average
# ---------------------------------------------------------------------------
for m in MODELS:
    r_avg = cluster_mean(m, "rmse", "Average")
    ma_avg = cluster_mean(m, "mae", "Average")
    draw_average_halo(bx1, r_avg, ma_avg, COLORS[m])

for m in MODELS:
    for cl in CLUSTER_ORDER:
        r = cluster_mean(m, "rmse", cl)
        ma = cluster_mean(m, "mae", cl)
        bx1.scatter(
            r, ma,
            s=140, color=COLORS[m], marker=CLUSTER_MARKERS[cl],
            edgecolors="black", linewidths=0.6,
            alpha=0.90, zorder=3,
        )

all_r1 = [cluster_mean(m, "rmse", cl) for m in MODELS for cl in CLUSTER_ORDER]
all_m1 = [cluster_mean(m, "mae", cl) for m in MODELS for cl in CLUSTER_ORDER]
bx1.set_xlim(0, max(all_r1) * 1.10)
bx1.set_ylim(0, max(all_m1) * 1.10)
bx1.set_xlabel("RMSE [%]")
bx1.set_ylabel("MAE [%]")
bx1.set_title("(a) Mean error metrics aggregated per cluster",
              loc="left", fontweight=500)
bx1.grid(True, linestyle=":", alpha=0.45, zorder=0)
bx1.set_axisbelow(True)
make_combined_legend(bx1, MODELS, CLUSTER_ORDER, COLORS, LABELS,
                     CLUSTER_MARKERS,
                     loc_models="upper left", loc_clusters="lower right")

# ---------------------------------------------------------------------------
# Panel (b): MaxAE vs execution time per cluster + average (log-log)
# ---------------------------------------------------------------------------
for m in MODELS:
    t_cluster = [cluster_mean(m, "time", cl) for cl in CLUSTER_ORDER]
    mx_cluster = [cluster_mean(m, "maxae", cl) for cl in CLUSTER_ORDER]
    confidence_ellipse(bx2, t_cluster, mx_cluster, COLORS[m],
                       x_log=True, y_log=True)

for m in MODELS:
    t_avg = cluster_mean(m, "time", "Average")
    mx_avg = cluster_mean(m, "maxae", "Average")
    draw_average_halo(bx2, t_avg, mx_avg, COLORS[m])

for m in MODELS:
    for cl in CLUSTER_ORDER:
        t = cluster_mean(m, "time", cl)
        mx = cluster_mean(m, "maxae", cl)
        bx2.scatter(
            t, mx,
            s=140, color=COLORS[m], marker=CLUSTER_MARKERS[cl],
            edgecolors="black", linewidths=0.6,
            alpha=0.90, zorder=3,
        )

bx2.set_xscale("log")
bx2.set_yscale("log")
bx2.set_xlabel("Execution time [s] (log)")
bx2.set_ylabel("MaxAE [%] (log)")
bx2.set_title(
    "(b) Worst-case error vs execution time aggregated per cluster "
    "(with per-model envelopes)",
    loc="left", fontweight=500,
)
bx2.grid(True, which="both", linestyle=":", alpha=0.4, zorder=0)
bx2.set_axisbelow(True)
make_combined_legend(bx2, MODELS, CLUSTER_ORDER, COLORS, LABELS,
                     CLUSTER_MARKERS,
                     loc_models="upper left", loc_clusters="lower right")

# ---------------------------------------------------------------------------
# Panel (c): RMSE vs MaxAE per cluster + average
# ---------------------------------------------------------------------------
for m in MODELS:
    r_avg = cluster_mean(m, "rmse", "Average")
    mx_avg = cluster_mean(m, "maxae", "Average")
    draw_average_halo(bx3, r_avg, mx_avg, COLORS[m])

for m in MODELS:
    for cl in CLUSTER_ORDER:
        r = cluster_mean(m, "rmse", cl)
        mx = cluster_mean(m, "maxae", cl)
        bx3.scatter(
            r, mx,
            s=140, color=COLORS[m], marker=CLUSTER_MARKERS[cl],
            edgecolors="black", linewidths=0.6,
            alpha=0.90, zorder=3,
        )

all_r3 = [cluster_mean(m, "rmse", cl) for m in MODELS for cl in CLUSTER_ORDER]
all_x3 = [cluster_mean(m, "maxae", cl) for m in MODELS for cl in CLUSTER_ORDER]
bx3.set_xlim(0, max(all_r3) * 1.10)
bx3.set_ylim(0, max(all_x3) * 1.10)
bx3.set_xlabel("RMSE [%]")
bx3.set_ylabel("MaxAE [%]")
bx3.set_title("(c) RMSE vs MaxAE aggregated per cluster",
              loc="left", fontweight=500)
bx3.grid(True, linestyle=":", alpha=0.4, zorder=0)
bx3.set_axisbelow(True)
make_combined_legend(bx3, MODELS, CLUSTER_ORDER, COLORS, LABELS,
                     CLUSTER_MARKERS,
                     loc_models="upper left", loc_clusters="lower right")

plt.savefig("figure_metrics_3panel_cluster.pdf", dpi=500, bbox_inches="tight")
plt.savefig("figure_metrics_3panel_cluster.png", dpi=500, bbox_inches="tight")


# ===========================================================================
# FIGURE 3 — combined figure for the paper
#   (a) = Figure 1 panel (a)
#   (b) = Figure 1 panel (b)
#   (c) = Figure 2 panel (c) + dashed black Pareto front
# ===========================================================================
fig3, (cx1, cx2, cx3) = plt.subplots(
    3, 1,
    figsize=(7.5, 13.5),
    gridspec_kw={"hspace": 0.30},
)

# ---------------------------------------------------------------------------
# Panel (a): RMSE vs MAE (per scenario) — same as Figure 1 (a)
# ---------------------------------------------------------------------------
for m in MODELS:
    cx1.scatter(
        DATA["rmse"][m], DATA["mae"][m],
        s=SIZES, color=COLORS[m],
        edgecolors="black", linewidths=0.5,
        alpha=0.88, zorder=3,
        label=LABELS[m],
    )

cx1.set_xlim(0, xmax_a)
cx1.set_ylim(0, ymax_a)
cx1.set_xlabel("RMSE [%]")
cx1.set_ylabel("MAE [%]")
cx1.set_title("(a) Mean error metrics: RMSE vs MAE", loc="left", fontweight=500)
cx1.grid(True, linestyle=":", alpha=0.45, zorder=0)
cx1.set_axisbelow(True)
cx1.legend(loc="upper left", frameon=True, framealpha=0.95,
           edgecolor="lightgray", ncol=2)

# ---------------------------------------------------------------------------
# Panel (b): MaxAE vs execution time (log-log, per scenario) — same as Fig.1 (b)
# ---------------------------------------------------------------------------
for m in MODELS:
    confidence_ellipse(cx2, DATA["time"][m], DATA["maxae"][m], COLORS[m],
                       x_log=True, y_log=True)
for m in MODELS:
    cx2.scatter(
        DATA["time"][m], DATA["maxae"][m],
        s=SIZES, color=COLORS[m],
        edgecolors="black", linewidths=0.5,
        alpha=0.88, zorder=3,
    )

cx2.set_xscale("log")
cx2.set_yscale("log")
cx2.set_xlabel("Execution time [s] (log)")
cx2.set_ylabel("MaxAE [%] (log)")
cx2.set_title(
    "(b) Worst-case error vs execution time (with per-model envelopes)",
    loc="left", fontweight=500,
)
cx2.grid(True, which="both", linestyle=":", alpha=0.4, zorder=0)
cx2.set_axisbelow(True)

size_handles_c = []
for d in [48, 96, 168]:
    s_pts = duration_to_size(d)
    size_handles_c.append(
        Line2D([0], [0], marker="o", color="white",
               markerfacecolor="lightgray", markeredgecolor="black",
               markeredgewidth=0.5, markersize=np.sqrt(s_pts),
               label=f"{d:.0f} h", linestyle="None")
    )
cx2.legend(handles=size_handles_c, loc="lower right",
           title="Scenario duration", frameon=True, framealpha=0.95,
           edgecolor="lightgray", labelspacing=1.2, borderpad=0.8)

# ---------------------------------------------------------------------------
# Panel (c): RMSE vs MaxAE aggregated per cluster — same as Figure 2 (c)
#            + dashed black Pareto front
# ---------------------------------------------------------------------------
# Average halo (drawn first, behind the markers)
for m in MODELS:
    r_avg = cluster_mean(m, "rmse", "Average")
    mx_avg = cluster_mean(m, "maxae", "Average")
    draw_average_halo(cx3, r_avg, mx_avg, COLORS[m])

# All (model, cluster) points
all_r_c = []
all_mx_c = []
for m in MODELS:
    for cl in CLUSTER_ORDER:
        r = cluster_mean(m, "rmse", cl)
        mx = cluster_mean(m, "maxae", cl)
        all_r_c.append(r)
        all_mx_c.append(mx)
        cx3.scatter(
            r, mx,
            s=140, color=COLORS[m], marker=CLUSTER_MARKERS[cl],
            edgecolors="black", linewidths=0.6,
            alpha=0.90, zorder=3,
        )

# Pareto front on all displayed points (minimize both RMSE and MaxAE)
front_x, front_y = compute_pareto_front_2d(all_r_c, all_mx_c)
cx3.plot(front_x, front_y,
         color="black", linestyle="--", linewidth=1.4,
         alpha=0.85, zorder=2.5,
         label="Pareto front")

cx3.set_xlim(0, max(all_r_c) * 1.10)
cx3.set_ylim(0, max(all_mx_c) * 1.10)
cx3.set_xlabel("RMSE [%]")
cx3.set_ylabel("MaxAE [%]")
cx3.set_title("(c) RMSE vs MaxAE aggregated per cluster",
              loc="left", fontweight=500)
cx3.grid(True, linestyle=":", alpha=0.4, zorder=0)
cx3.set_axisbelow(True)

# Legend: models (rectangles), clusters (markers) + Pareto-front entry
pareto_handle = Line2D([0], [0], color="black", linestyle="--",
                       linewidth=1.4, label="Pareto front")
make_combined_legend(cx3, MODELS, CLUSTER_ORDER, COLORS, LABELS,
                     CLUSTER_MARKERS,
                     loc_models="upper left", loc_clusters="lower right",
                     extra_handle=pareto_handle)

plt.savefig("figure_metrics_3panel_combined.pdf", dpi=500, bbox_inches="tight")
plt.savefig("figure_metrics_3panel_combined.png", dpi=500, bbox_inches="tight")

plt.show()

print("Saved:")
print("  figure_metrics_3panel.pdf + .png            (per-scenario view)")
print("  figure_metrics_3panel_cluster.pdf + .png    (cluster-aggregated view)")
print("  figure_metrics_3panel_combined.pdf + .png   (combined paper figure)")

# ---------------------------------------------------------------------------
# Sanity prints
# ---------------------------------------------------------------------------
print("\nCluster aggregates (mean across scenarios in each cluster):\n")
header = f"  {'Estimator':>14s}  {'Cluster':>10s}  {'RMSE':>6s}  {'MAE':>6s}  {'MaxAE':>7s}  {'Time':>10s}"
print(header)
print("  " + "-" * (len(header) - 2))
for m in MODELS:
    for cl in CLUSTER_ORDER:
        r = cluster_mean(m, "rmse", cl)
        ma = cluster_mean(m, "mae", cl)
        mx = cluster_mean(m, "maxae", cl)
        tt = cluster_mean(m, "time", cl)
        print(f"  {LABELS[m]:>14s}  {cl:>10s}  {r:6.2f}  {ma:6.2f}  {mx:7.2f}  {tt:10.2f}")
    print()

print("\nPareto front in RMSE-MaxAE (cluster-aggregated, lower is better on both axes):")
print(f"  {'RMSE [%]':>10s}  {'MaxAE [%]':>10s}")
for fx, fy in zip(front_x, front_y):
    print(f"  {fx:10.3f}  {fy:10.3f}")