"""Compare one optimizer's no-blur baseline with its WINDOW_LENGTH-epoch blur windows. Set OPTIMIZER, then run it, once per optimizer:
    python graph_blur_windows.py

Writes results/graph_blur_windows_<optimizer>.png, with two rows:
1. Left, epoch CHECKPOINT_EPOCH networks on the test set at each blur sigma, mean and sample SD over seeds. Right, a table of
   mean ± sample SD over seeds. Area under the curve is per seed, over the plotted sigmas. Ranks are from the TensorBoard logs at
   CHECKPOINT_EPOCH, so they describe the same networks as the accuracies.
2. Per-layer Fisher information of seed FISHER_SEED, one panel per experiment.

Reads results/<experiment>/seed_<seed>/tensorboard/ and results/graph_report_blur_eval.csv, evaluating any runs missing from the
CSV (see graph_report.py).
"""
import os

from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from eval_blurred import CHECKPOINT_EPOCH, SD_DDOF
from graph_fisher import BLUR_COLOR, EXPERIMENTS, LAYER_COUNT, SEEDS, fit_fisher_axes, layer_color, plot_fisher, run_log_dir, style_axes
from graph_report import BLUR_SIGMAS, FISHER_SEED, load_blur_eval, plot_mean_sd
from graph_training import (FISHER_EXCLUDED_LAYERS, FISHER_PER_PARAM, FISHER_YLIM_PERCENTILE, GRID_COLOR, MUTED_COLOR, RESULTS_DIR,
                            TEXT_COLOR, X_RANGE, load_series, prepare_fisher)
from training_procedure import BLUR_SIGMA

OPTIMIZER = "SGD"  # "SGD" or "Adam"
WINDOW_LENGTH = 10  # blur windows of other lengths are left out
NO_BLUR_COLOR = "#7f7f7f"
VIRIDIS_MAX = 0.9  # the yellow end of viridis is too faint to read against a white background
SD_ALPHA = 0.15  # six overlapping curves need lighter shading
TITLE_BLOCK_FONT_SIZE = 16  # the colour block before each Fisher panel title
FIGURE_SIZE = (20.4, 9)
ROW_HEIGHT_RATIOS = [5, 4]  # blur test row, Fisher row
RANK_NAMES = {"Stable rank delta": "srank", "Effective rank": "erank"}
TABLE_COLUMNS = ["", "Experiment", "Blur epochs", "Acc., sigma 0", f"Acc., sigma {BLUR_SIGMA}", "AUC", "srank", "erank"]
TABLE_COLUMN_WIDTHS = [0.03, 0.10, 0.10, 0.17, 0.17, 0.15, 0.14, 0.14]
FIRST_METRIC_COLUMN = 3  # columns from here on are shaded with plasma, from each column's lowest mean to its highest
TABLE_BOTTOM = 0.12  # in axes coordinates; the table fills the axes above this
HIGHLIGHTED_BLUR = (0, 10)  # this blur window's table row gets a border
HIGHLIGHT_COLOR = "#b5403a"
LUMA_WEIGHTS = (0.299, 0.587, 0.114)
LIGHT_CELL_LUMA = 0.5  # cells lighter than this get dark text


def selected_experiments():
    """This optimizer's no-blur baseline and WINDOW_LENGTH-epoch windows, in order of window start."""
    selected = {name: blur for name, (optimizer, blur) in EXPERIMENTS.items()
                if optimizer == OPTIMIZER and (blur is None or blur[1] - blur[0] == WINDOW_LENGTH)}
    return dict(sorted(selected.items(), key=lambda item: -1 if item[1] is None else item[1][0]))


def experiment_colors(experiments):
    """Grey for no blur, then viridis by window start: the earliest window is dark purple."""
    starts = [blur[0] for blur in experiments.values() if blur is not None]
    return {name: NO_BLUR_COLOR if blur is None else plt.get_cmap("viridis")(VIRIDIS_MAX * blur[0] / max(starts))
            for name, blur in experiments.items()}


def blur_label(blur):
    return "No blur" if blur is None else f"{blur[0] + 1}-{blur[1]}"


def value_at_epoch(series, epoch):
    epochs, values = series
    index = int(np.argmin(np.abs(np.asarray(epochs) - epoch)))
    if not np.isclose(epochs[index], epoch):
        raise SystemExit(f"No logged value at epoch {epoch}: the closest is at {epochs[index]}.")
    return values[index]


def mean_sd(values, spec):
    return f"{np.mean(values):{spec}} ± {np.std(values, ddof=SD_DDOF):{spec}}"


def plot_fisher_row(fig, experiments, fisher):
    axes = fig.subplots(1, len(experiments), sharex=True)
    colors = experiment_colors(experiments)
    panels = {}
    for ax, (name, blur) in zip(axes, experiments.items()):
        panels[name, FISHER_SEED] = ax
        style_axes(ax, X_RANGE, blur)
        plot_fisher(ax, fisher[name])
        ax.set_title(f"{name}, {'no blur' if blur is None else f'blur epochs {blur_label(blur)}'}", color=TEXT_COLOR)
        ax.annotate("■", xy=(0, 0.5), xycoords=ax.title, xytext=(-3, 0), textcoords="offset points", ha="right", va="center",
                    color=colors[name], fontsize=TITLE_BLOCK_FONT_SIZE)
        ax.set_xlabel("Epoch", color=MUTED_COLOR)
    peak = fit_fisher_axes(OPTIMIZER, panels, {key: fisher[key[0]] for key in panels})
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    ylabel = "Fisher information per parameter" if FISHER_PER_PARAM else "Fisher information (trace)"
    axes[0].set_ylabel(ylabel, color=MUTED_COLOR)
    fig.suptitle(f"{OPTIMIZER}, seed {FISHER_SEED}: per-layer Fisher information, without {' and '.join(FISHER_EXCLUDED_LAYERS)}. "
                 f"Shared y-axis, clipped at the {FISHER_YLIM_PERCENTILE}th percentile (peak {peak.split()[-1]}).",
                 color=TEXT_COLOR, x=0.01, ha="left")
    excluded = [int(name.rsplit("_", 1)[1]) for name in FISHER_EXCLUDED_LAYERS]
    handles = [Line2D([], [], color=layer_color(f"layer_{i}"), linewidth=2, label=f"layer_{i}")
               for i in range(1, LAYER_COUNT + 1) if i not in excluded]
    fig.legend(handles=[*handles, Patch(color=BLUR_COLOR, label="Blurred epochs")], loc="outside right upper", frameon=False,
               labelcolor=TEXT_COLOR)


def plot_table(ax, experiments, colors, blur_eval, ranks):
    rows, means = [], []
    for name, blur in experiments.items():
        accuracy = np.array([[blur_eval[name, seed][sigma] for sigma in BLUR_SIGMAS] for seed in SEEDS])
        metrics = [(100 * accuracy[:, BLUR_SIGMAS.index(0.0)], ".1f"),
                   (100 * accuracy[:, BLUR_SIGMAS.index(BLUR_SIGMA)], ".1f"),
                   (np.trapezoid(accuracy, BLUR_SIGMAS, axis=1), ".3f"),
                   *[(ranks[name][rank_name], ".1f") for rank_name in RANK_NAMES]]
        rows.append(["", name, blur_label(blur), *[mean_sd(values, spec) for values, spec in metrics]])
        means.append([np.mean(values) for values, _ in metrics])
    means = np.array(means)
    shades = (means - means.min(axis=0)) / (means.max(axis=0) - means.min(axis=0))  # 0 to 1 within each column
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=TABLE_COLUMNS, colWidths=TABLE_COLUMN_WIDTHS, bbox=[0, TABLE_BOTTOM, 1, 1 - TABLE_BOTTOM], cellLoc="left",
                     colLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(GRID_COLOR)
        cell.visible_edges = "horizontal"
        cell.get_text().set_color(MUTED_COLOR if row == 0 else TEXT_COLOR)
        if row > 0 and col == 0:
            # A cell only fills when all its edges are drawn.
            cell.visible_edges = "closed"
            cell.set_facecolor(colors[list(experiments)[row - 1]])
            cell.set_edgecolor("white")
            cell.set_linewidth(3)
        if row > 0 and col >= FIRST_METRIC_COLUMN:
            color = plt.get_cmap("plasma")(shades[row - 1, col - FIRST_METRIC_COLUMN])
            cell.visible_edges = "closed"
            cell.set_facecolor(color)
            cell.set_edgecolor("white")
            cell.set_linewidth(1)
            cell.get_text().set_color(TEXT_COLOR if np.dot(color[:3], LUMA_WEIGHTS) > LIGHT_CELL_LUMA else "white")
    # All rows are the same height, and the header is one of them.
    row_height = (1 - TABLE_BOTTOM) / (len(rows) + 1)
    highlighted_row = list(experiments.values()).index(HIGHLIGHTED_BLUR)
    ax.add_patch(Rectangle((0, 1 - (highlighted_row + 2) * row_height), 1, row_height, transform=ax.transAxes, fill=False,
                           edgecolor=HIGHLIGHT_COLOR, linewidth=2.5, zorder=5, clip_on=False))
    ax.text(0, 0.08, f"Mean ± sample SD over seeds {SEEDS[0]}-{SEEDS[-1]}, at epoch {CHECKPOINT_EPOCH}. Accuracy in %. AUC: accuracy over "
                     f"blur sigma {BLUR_SIGMAS[0]:g}-{BLUR_SIGMAS[-1]:g}\n(accuracy x pixels, at most {BLUR_SIGMAS[-1]:g}). srank: stable rank "
                     "delta. erank: effective rank (penultimate features, first test batch).\n"
                     "Cell colour: plasma, from the lowest mean (dark blue) to the highest (yellow) within each column.",
            transform=ax.transAxes, va="top", color=MUTED_COLOR, fontsize=9)


def plot_blur_test_row(fig, experiments, blur_eval, images, ranks):
    colors = experiment_colors(experiments)
    ax_curves, ax_table = fig.subplots(1, 2, width_ratios=[1, 1.75])
    style_axes(ax_curves, (BLUR_SIGMAS[0], BLUR_SIGMAS[-1]))
    for name in experiments:
        accuracy = np.array([[blur_eval[name, seed][sigma] for sigma in BLUR_SIGMAS] for seed in SEEDS])
        plot_mean_sd(ax_curves, BLUR_SIGMAS, accuracy, colors[name], alpha=SD_ALPHA)
    ax_curves.axvline(BLUR_SIGMA, color=MUTED_COLOR, linewidth=1, linestyle="--", zorder=1)
    ax_curves.set_ylim(0, 1)
    ax_curves.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax_curves.set_xlabel("Test image blur sigma (pixels)", color=MUTED_COLOR)
    ax_curves.set_ylabel("Test accuracy", color=MUTED_COLOR)
    ax_curves.legend(handles=[*[Line2D([], [], color=colors[name], linewidth=2,
                                       label="No blur" if blur is None else f"Blur epochs {blur_label(blur)}")
                                for name, blur in experiments.items()],
                              Line2D([], [], color=MUTED_COLOR, linewidth=1, linestyle="--", label=f"Training blur sigma ({BLUR_SIGMA})"),
                              Patch(color=MUTED_COLOR, alpha=SD_ALPHA, linewidth=0, label="± 1 SD over seeds")],
                     loc="upper right", frameon=True, facecolor="white", edgecolor="none", framealpha=1, labelcolor=TEXT_COLOR)  # covers the sigma line
    plot_table(ax_table, experiments, colors, blur_eval, ranks)
    fig.suptitle(f"{OPTIMIZER}: epoch {CHECKPOINT_EPOCH} networks on {images} blurred test images", color=TEXT_COLOR, x=0.01, ha="left")


def main():
    experiments = selected_experiments()
    blur_eval, images = load_blur_eval()
    fisher, ranks = {}, {name: {rank_name: [] for rank_name in RANK_NAMES} for name in experiments}
    for name in experiments:
        for seed in SEEDS:
            log_dir = run_log_dir(name, seed)
            _, seed_fisher, _, _, seed_ranks = load_series(log_dir)
            if seed == FISHER_SEED:
                fisher[name] = prepare_fisher(seed_fisher, log_dir)
            for rank_name in RANK_NAMES:
                ranks[name][rank_name].append(value_at_epoch(seed_ranks[rank_name], CHECKPOINT_EPOCH))

    fig = plt.figure(figsize=FIGURE_SIZE, layout="constrained")
    blur_test_row, fisher_row = fig.subfigures(2, 1, height_ratios=ROW_HEIGHT_RATIOS)
    plot_blur_test_row(blur_test_row, experiments, blur_eval, images, ranks)
    plot_fisher_row(fisher_row, experiments, fisher)
    output_path = os.path.join(RESULTS_DIR, f"graph_blur_windows_{OPTIMIZER.lower()}.png")
    fig.savefig(output_path, dpi=150)
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
