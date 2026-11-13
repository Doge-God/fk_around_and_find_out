"""Plot one run's training: train/test accuracy, per-layer Fisher information, weight norms, representation ranks and SIGReg.

Readings on blurred images (the same test or Fisher images with the training blur, logged throughout training) are solid lines
in a lighter shade of their sharp reading's colour. Fisher information gets two panels on the same y-axis, sharp and blurred.
Blurred epochs are shaded. Reads results/<batch>/experiments/<experiment>/seed_<seed>/tensorboard/ and writes
graph_training.png next to it:
    python -m analysis.graph_training batch_01 exp_03 --seed 42
"""
import argparse
import os

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from analysis.common import (BLUR_COLOR, FISHER_EXCLUDED_LAYERS, FISHER_LOG_Y, FISHER_PER_PARAM, FISHER_YLIM_PERCENTILE, LAYER_COUNT,
                             MUTED_COLOR, RAW_ALPHA, TEXT_COLOR, ema, fisher_label, layer_color, legend_label, load_batch,
                             load_series, prepare_fisher, run_dir, run_log_dir, style_axes)

DEFAULT_SEED = 42
OUTPUT_NAME = "graph_training.png"
FIGURE_SIZE = (26, 10)
ACCURACY_COLORS = {"Train": "#eb6834", "Test": "#2a78d6"}
RANK_COLORS = {"Effective rank": "#1a8f5a", "Stable rank delta": "#b84fc4"}
SIGREG_COLOR = "#2a78d6"
TOTAL_COLOR = TEXT_COLOR
BLURRED_LIGHTEN = 0.55  # blurred readings: this far from their sharp colour toward white


def lighter(color):
    return tuple(c + (1 - c) * BLURRED_LIGHTEN for c in to_rgb(color))


def plot_lines(ax, series, colors, smooth=False, blurred=False):
    """Lines of series in colors (name -> colour), or in lighter shades of them and under the sharp lines if blurred."""
    for name, (epochs, values) in series.items():
        color = lighter(colors[name]) if blurred else colors[name]
        label = f"{name}, blurred test images" if blurred else name
        zorder = 1 if blurred else 2
        if smooth:
            ax.plot(epochs, values, color=color, linewidth=1, alpha=RAW_ALPHA, zorder=zorder)
            ax.plot(epochs, ema(values), color=color, linewidth=2, label=label, zorder=zorder)
        else:
            ax.plot(epochs, values, color=color, linewidth=2, label=label, zorder=zorder)


def layer_colors(series):
    return {name: layer_color(name) for name in series}


def finish(ax, ylabel, title, legend=True, **legend_kwargs):
    ax.set_xlabel("Epoch", color=MUTED_COLOR)
    ax.set_ylabel(ylabel, color=MUTED_COLOR)
    ax.set_title(title, color=TEXT_COLOR, loc="left")
    if legend:
        ax.legend(frameon=False, labelcolor=TEXT_COLOR, **legend_kwargs)


def plot_fisher_pair(ax_sharp, ax_blurred, fisher, blurred_fisher):
    """Sharp and blurred Fisher on one y-axis, fit to both."""
    plot_lines(ax_sharp, fisher, layer_colors(fisher), smooth=True)
    plot_lines(ax_blurred, blurred_fisher, layer_colors(blurred_fisher), smooth=True)
    values = np.concatenate([values for run in (fisher, blurred_fisher) for _, values in run.values()])
    ax_blurred.sharey(ax_sharp)
    ax_sharp.set_yscale("log" if FISHER_LOG_Y else "linear")
    # A log axis can't start at 0, so it keeps its automatic lower limit.
    ax_sharp.set_ylim(None if FISHER_LOG_Y else 0, 1.1 * np.percentile(values, FISHER_YLIM_PERCENTILE))
    peak = f"{values.max():{'.3g' if FISHER_PER_PARAM else '.0f'}}"
    finish(ax_sharp, fisher_label(), f"{fisher_label()}, sharp images", legend=False)
    finish(ax_blurred, fisher_label(), f"Blurred images (same y-axis, clipped; peak of both {peak})", legend=False)
    if not blurred_fisher:  # a run logged before blurred Fisher existed
        ax_blurred.text(0.5, 0.5, "not logged", transform=ax_blurred.transAxes, ha="center", va="center", color=MUTED_COLOR)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("batch", help="batch module, e.g. batch_01")
    parser.add_argument("experiment", help="experiment in that batch, e.g. exp_03")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    batch = load_batch(args.batch)
    if args.experiment not in batch.EXPERIMENTS:
        raise SystemExit(f"{batch.BATCH} has no experiment {args.experiment}.")
    blur = batch.EXPERIMENTS[args.experiment]
    x_range = (0, batch.EPOCHS)

    log_dir = run_log_dir(batch, args.experiment, args.seed)
    series = load_series(log_dir)
    fisher = prepare_fisher(series["fisher"], log_dir)
    blurred_fisher = prepare_fisher(series["blurred_fisher"], log_dir) if series["blurred_fisher"] else {}

    fig, axes = plt.subplots(2, 4, figsize=FIGURE_SIZE, layout="constrained")
    (ax_accuracy, ax_fisher, ax_blurred_fisher, ax_weight_norm), (ax_ranks, ax_sigreg, ax_sigreg_standardized, ax_legend) = axes
    for ax in axes.flat:
        if ax is not ax_legend:
            style_axes(ax, x_range, blur)

    plot_lines(ax_accuracy, series["accuracy"], ACCURACY_COLORS)
    plot_lines(ax_accuracy, series["blurred_accuracy"], ACCURACY_COLORS, blurred=True)
    finish(ax_accuracy, "Accuracy", "Train and test accuracy")

    plot_fisher_pair(ax_fisher, ax_blurred_fisher, fisher, blurred_fisher)

    plot_lines(ax_weight_norm, series["weight_norm"], layer_colors(series["weight_norm"]))
    ax_weight_norm.plot(*series["weight_norm_total"], color=TOTAL_COLOR, linewidth=2, label="Total", zorder=3)
    finish(ax_weight_norm, "L2 norm", "Total and per-layer weight norm", legend=False)

    plot_lines(ax_ranks, series["ranks"], RANK_COLORS)
    plot_lines(ax_ranks, series["blurred_ranks"], RANK_COLORS, blurred=True)
    finish(ax_ranks, "Rank", "Representation rank (penultimate features, first test batch)")

    for ax, name in [(ax_sigreg, "SIGReg"), (ax_sigreg_standardized, "SIGReg, standardised features")]:
        plot_lines(ax, {name: series["sigreg"][name]}, {name: SIGREG_COLOR})
        if name in series["blurred_sigreg"]:
            plot_lines(ax, {name: series["blurred_sigreg"][name]}, {name: SIGREG_COLOR}, blurred=True)
        ax.set_ylim(bottom=0)
        finish(ax, "SIGReg", f"{name} (first test batch)", legend=False)

    # One legend for the panels without their own: layers (Fisher and weight norm), SIGReg, blurred epochs.
    ax_legend.axis("off")
    excluded = {int(name.rsplit("_", 1)[1]) for name in FISHER_EXCLUDED_LAYERS}
    handles = [Line2D([], [], color=layer_color(f"layer_{i}"), linewidth=2,
                      label=f"layer_{i}" + (" (weight norm only)" if i in excluded else "")) for i in range(1, LAYER_COUNT + 1)]
    handles += [Line2D([], [], color=TOTAL_COLOR, linewidth=2, label="Total weight norm"),
                Line2D([], [], color=SIGREG_COLOR, linewidth=2, label="SIGReg, sharp test images"),
                Line2D([], [], color=lighter(SIGREG_COLOR), linewidth=2, label="SIGReg, blurred test images")]
    if blur is not None:
        handles.append(Patch(color=BLUR_COLOR, label="Blurred training epochs"))
    ax_legend.legend(handles=handles, loc="center left", frameon=False, labelcolor=TEXT_COLOR)

    fig.suptitle(f"{batch.BATCH} ({batch.OPTIMIZER}) {args.experiment}, seed {args.seed}: {legend_label(batch, blur).lower()}. "
                 "Lighter lines: the same readings on images with the training blur.", color=TEXT_COLOR, x=0.01, ha="left")
    output_path = os.path.join(run_dir(batch, args.experiment, args.seed), OUTPUT_NAME)
    fig.savefig(output_path, dpi=150)
    print(f"read {log_dir}\nsaved {output_path}")


if __name__ == "__main__":
    main()
