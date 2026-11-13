"""A batch's final networks on test images blurred with sigma 0-3, every experiment in one panel, and a table of the results:
    python -m analysis.graph_blur_accuracy batch_01

Left: test accuracy against test image blur sigma, mean and sample SD over the batch's seeds. No blur is solid black, blur on
every epoch dotted black, and each blur window takes a viridis colour by its start epoch, darkest for the earliest.
Right: mean ± sample SD of accuracy on sharp images, at the training blur sigma, and the area under the accuracy curve (per
seed, over the plotted sigmas). The first column is each experiment's line, so the table doubles as the legend.

Reads results/<batch>/eval_blur.csv, evaluating any runs missing from it (see eval_blur.py), and writes
results/<batch>/graph_blur_accuracy.png.
"""
import argparse
import os

from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from analysis.common import (BLUR_SIGMAS, GRID_COLOR, MUTED_COLOR, REFERENCE_COLOR, REFERENCE_LINEWIDTH, REFERENCE_STYLES, SD_DDOF,
                             TEXT_COLOR, add_window_colorbar, batch_dir, blur_group, blur_label, experiment_styles, load_batch,
                             mean_sd, still_blurred, style_axes)
from analysis.eval_blur import load_eval
from training_procedure import BLUR_SIGMA

OUTPUT_NAME = "graph_blur_accuracy.png"
FIGURE_SIZE = (22, 13)
WIDTH_RATIOS = [1.25, 1]  # curves, table
SD_ALPHA = 0.02  # 31 overlapping bands add up, so each is very light
TABLE_COLUMNS = ["", "Experiment", "Blur epochs", "Acc., sigma 0", f"Acc., sigma {BLUR_SIGMA}", "AUC"]
TABLE_COLUMN_WIDTHS = [0.08, 0.13, 0.15, 0.21, 0.21, 0.22]
FIRST_METRIC_COLUMN = 3  # columns from here on are shaded with plasma, from each column's lowest mean to its highest
OUTLIER_HATCH = "////"
SWATCH_LINE_FRACTION = (0.15, 0.85)  # the line sample in the first table column, as fractions of its width
TABLE_BOTTOM = 0.08  # in axes coordinates; the table fills the axes above this
HIGHLIGHTED_BLUR = (0, 10)  # this blur window's table row gets a border
HIGHLIGHT_COLOR = TEXT_COLOR
LUMA_WEIGHTS = (0.299, 0.587, 0.114)
LIGHT_CELL_LUMA = 0.5  # cells lighter than this get dark text
TABLE_FONT_SIZE = 9


def accuracy_over_seeds(batch, results, name):
    return np.array([[results[name, seed]["accuracy"][sigma] for sigma in BLUR_SIGMAS] for seed in batch.SEEDS])


def plot_curves(fig, ax, batch, styles, results):
    style_axes(ax, (BLUR_SIGMAS[0], BLUR_SIGMAS[-1]))
    # Windows first and the references on top, so the black lines stay visible.
    order = sorted(batch.EXPERIMENTS, key=lambda name: blur_group(batch, batch.EXPERIMENTS[name]) != "window")
    for name in order:
        color, linestyle, linewidth = styles[name]
        accuracy = accuracy_over_seeds(batch, results, name)
        mean, sd = accuracy.mean(axis=0), accuracy.std(axis=0, ddof=SD_DDOF)
        ax.fill_between(BLUR_SIGMAS, mean - sd, mean + sd, color=color, alpha=SD_ALPHA, linewidth=0, zorder=1)
        ax.plot(BLUR_SIGMAS, mean, color=color, linestyle=linestyle, linewidth=linewidth, zorder=2)
    ax.axvline(BLUR_SIGMA, color=MUTED_COLOR, linewidth=1, linestyle="--", zorder=0)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlabel("Test image blur sigma (pixels)", color=MUTED_COLOR)
    ax.set_ylabel("Test accuracy", color=MUTED_COLOR)
    handles = [Line2D([], [], color=REFERENCE_COLOR, linestyle=REFERENCE_STYLES["none"], linewidth=REFERENCE_LINEWIDTH, label="No blur"),
               Line2D([], [], color=REFERENCE_COLOR, linestyle=REFERENCE_STYLES["full"], linewidth=REFERENCE_LINEWIDTH,
                      label="Blur on every epoch"),
               Line2D([], [], color=MUTED_COLOR, linewidth=1, linestyle="--", label=f"Training blur sigma ({BLUR_SIGMA})"),
               Patch(color=MUTED_COLOR, alpha=8 * SD_ALPHA, linewidth=0, label="± 1 SD over seeds")]
    ax.legend(handles=handles, loc="upper right", frameon=False, labelcolor=TEXT_COLOR)
    add_window_colorbar(fig, ax, batch, location="bottom", shrink=0.6, aspect=40, pad=0.02)


def plot_table(ax, batch, styles, results):
    experiments = batch.EXPERIMENTS
    rows, means = [], []
    for name, blur in experiments.items():
        accuracy = accuracy_over_seeds(batch, results, name)
        metrics = [(100 * accuracy[:, BLUR_SIGMAS.index(0.0)], ".1f"),
                   (100 * accuracy[:, BLUR_SIGMAS.index(BLUR_SIGMA)], ".1f"),
                   (np.trapezoid(accuracy, BLUR_SIGMAS, axis=1), ".3f")]
        rows.append(["", name, blur_label(batch, blur), *[mean_sd(values, spec) for values, spec in metrics]])
        means.append([np.mean(values) for values, _ in metrics])
    means = np.array(means)
    # Networks still blurred at the last epoch (full blur and the last window) are far from the rest, so they don't set the colour
    # range. Their values outside it take the nearest end colour and are hatched.
    scaled = np.array([not still_blurred(blur, batch.EPOCHS) for blur in experiments.values()])
    low, high = means[scaled].min(axis=0), means[scaled].max(axis=0)
    inliers = (means >= low) & (means <= high)
    shades = np.clip((means - low) / np.where(high > low, high - low, 1), 0, 1)  # 0 to 1 within each column
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=TABLE_COLUMNS, colWidths=TABLE_COLUMN_WIDTHS, bbox=[0, TABLE_BOTTOM, 1, 1 - TABLE_BOTTOM],
                     cellLoc="left", colLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(TABLE_FONT_SIZE)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(GRID_COLOR)
        cell.visible_edges = "horizontal"
        cell.get_text().set_color(MUTED_COLOR if row == 0 else TEXT_COLOR)
        if row > 0 and col >= FIRST_METRIC_COLUMN:
            # A cell only fills when all its edges are drawn.
            color = plt.get_cmap("plasma")(shades[row - 1, col - FIRST_METRIC_COLUMN])
            cell.visible_edges = "closed"
            cell.set_facecolor(color)
            cell.set_edgecolor("white")
            cell.set_linewidth(1)
            light = np.dot(color[:3], LUMA_WEIGHTS) > LIGHT_CELL_LUMA
            cell.get_text().set_color(TEXT_COLOR if light else "white")
            if not inliers[row - 1, col - FIRST_METRIC_COLUMN]:
                cell.set_hatch(OUTLIER_HATCH)
                cell.set_hatchcolor((0, 0, 0, 0.35) if light else (1, 1, 1, 0.35))
                cell.get_text().set_fontweight("bold")
    # All rows are the same height, and the header is one of them. The first column holds a sample of each experiment's line.
    row_height = (1 - TABLE_BOTTOM) / (len(rows) + 1)
    swatch_x = [TABLE_COLUMN_WIDTHS[0] * fraction for fraction in SWATCH_LINE_FRACTION]
    for index, name in enumerate(experiments):
        color, linestyle, linewidth = styles[name]
        y = 1 - (index + 1.5) * row_height
        ax.add_line(Line2D(swatch_x, [y, y], color=color, linestyle=linestyle, linewidth=2 * linewidth, transform=ax.transAxes,
                           solid_capstyle="butt"))
    if HIGHLIGHTED_BLUR in experiments.values():
        highlighted_row = list(experiments.values()).index(HIGHLIGHTED_BLUR)
        ax.add_patch(Rectangle((0, 1 - (highlighted_row + 2) * row_height), 1, row_height, transform=ax.transAxes, fill=False,
                               edgecolor=HIGHLIGHT_COLOR, linewidth=2, zorder=5, clip_on=False))
    ax.text(0, TABLE_BOTTOM - 0.01, f"Mean ± sample SD over seeds {batch.SEEDS[0]}-{batch.SEEDS[-1]}, at epoch {batch.EPOCHS}. "
                                    f"Accuracy in %. AUC: accuracy over blur sigma {BLUR_SIGMAS[0]:g}-{BLUR_SIGMAS[-1]:g} "
                                    f"(accuracy x pixels, at most {BLUR_SIGMAS[-1]:g}).\nCell colour: plasma, from the lowest mean "
                                    "(dark blue) to the highest (yellow) within each column, leaving out networks still blurred\n"
                                    f"at epoch {batch.EPOCHS}: their values outside that range are hatched, in bold and take the "
                                    "nearest end colour.",
            transform=ax.transAxes, va="top", color=MUTED_COLOR, fontsize=9)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("batch", help="batch module, e.g. batch_01")
    batch = load_batch(parser.parse_args().batch)
    results, images = load_eval(batch)
    styles = experiment_styles(batch)

    fig, (ax_curves, ax_table) = plt.subplots(1, 2, figsize=FIGURE_SIZE, width_ratios=WIDTH_RATIOS, layout="constrained")
    plot_curves(fig, ax_curves, batch, styles, results)
    plot_table(ax_table, batch, styles, results)
    fig.suptitle(f"{batch.BATCH} ({batch.OPTIMIZER}): epoch {batch.EPOCHS} networks on {images['accuracy_images']} blurred test "
                 "images", color=TEXT_COLOR, x=0.01, ha="left")
    output_path = os.path.join(batch_dir(batch), OUTPUT_NAME)
    fig.savefig(output_path, dpi=150)
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
