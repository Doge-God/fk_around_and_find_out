"""Per-layer Fisher information of one seed (FISHER_SEED) for every experiment in a batch, one panel per experiment in sweep
order, blurred epochs shaded:
    python -m analysis.graph_fisher batch_01

Two grids: all epochs on top, and below, the same panels zoomed to the last FINAL_EPOCHS epochs. Within each grid, networks
still blurred at the grid's last epoch (blur on every epoch, the last window) can sit far above the rest, so they share one
y-axis and all the others share another. Each y-axis is clipped at the FISHER_YLIM_PERCENTILE of the values it shows.

Reads results/<batch>/experiments/<experiment>/seed_<FISHER_SEED>/tensorboard/ and writes results/<batch>/graph_fisher.png.
"""
import argparse
import math
import os

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from analysis.common import (BLUR_COLOR, FISHER_EXCLUDED_LAYERS, FISHER_LOG_Y, FISHER_YLIM_PERCENTILE, LAYER_COUNT, MUTED_COLOR,
                             TEXT_COLOR, batch_dir, experiment_styles, fisher_label, fit_fisher_axes, layer_color, legend_label,
                             load_batch, load_series, plot_fisher, prepare_fisher, run_log_dir, still_blurred, style_axes)

FISHER_SEED = 42
FINAL_EPOCHS = 10
COLUMNS = 8
PANEL_SIZE = (3.2, 2.3)  # inches per panel
TITLE_BLOCK_FONT_SIZE = 14  # the colour block before each panel title
OUTPUT_NAME = "graph_fisher.png"


def plot_fisher_grid(fig, batch, fisher, x_range):
    """One Fisher panel per experiment over x_range, then a legend in the first free slot."""
    experiments = batch.EXPERIMENTS
    rows = math.ceil((len(experiments) + 1) / COLUMNS)  # at least one free slot for the legend
    axes = fig.subplots(rows, COLUMNS, sharex=True).flatten()
    styles = experiment_styles(batch)
    panels = {}
    for index, (name, blur) in enumerate(experiments.items()):
        ax = axes[index]
        panels[name] = ax
        style_axes(ax, x_range, blur)
        plot_fisher(ax, fisher[name])
        ax.set_title(f"{name}, {legend_label(batch, blur).lower().replace('blur epochs', 'blur')}", color=TEXT_COLOR, fontsize=10)
        ax.annotate("■", xy=(0, 0.5), xycoords=ax.title, xytext=(-3, 0), textcoords="offset points", ha="right", va="center",
                    color=styles[name][0], fontsize=TITLE_BLOCK_FONT_SIZE)
        if index + COLUMNS >= len(experiments):  # nothing below it
            ax.set_xlabel("Epoch", color=MUTED_COLOR)
            ax.tick_params(labelbottom=True)  # sharex hides them on all but the bottom row
    for ax in axes[len(experiments):]:
        ax.axis("off")
    fig.supylabel(fisher_label(), color=MUTED_COLOR, fontsize=10)

    blurred = {name for name, blur in experiments.items() if still_blurred(blur, x_range[1])}
    peaks = {}
    for in_blurred, group in [(False, set(experiments) - blurred), (True, blurred)]:
        if group:
            names = [name for name in experiments if name in group]
            peaks[in_blurred] = fit_fisher_axes([panels[name] for name in names], [fisher[name] for name in names], x_range)
    # Plain tick labels, set after fit_fisher_axes because setting the scale resets them: the "1e-5" offset label that small values
    # get would run into the panel titles. Unless the axis is logarithmic, which formats its own.
    if not FISHER_LOG_Y:
        for ax in panels.values():
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2g}"))
    for row in range(rows):  # y tick labels on the first panel of each y-axis group in each row
        labelled = set()
        for name in list(experiments)[row * COLUMNS:(row + 1) * COLUMNS]:
            in_blurred = name in blurred
            panels[name].tick_params(labelleft=in_blurred not in labelled)
            labelled.add(in_blurred)

    note = (f"One y-axis for {', '.join(sorted(blurred))}, still blurred at epoch {x_range[1]} (peak {peaks[True]:.3g}), another "
            f"for the rest (peak {peaks[False]:.3g})") if blurred else f"Y-axis shared (peak {peaks[False]:.3g})"
    fig.suptitle(f"Epochs {x_range[0]}-{x_range[1]}. {note}, each clipped at the {FISHER_YLIM_PERCENTILE}th percentile.",
                 color=TEXT_COLOR, x=0.01, ha="left")
    excluded = [int(name.rsplit("_", 1)[1]) for name in FISHER_EXCLUDED_LAYERS]
    handles = [Line2D([], [], color=layer_color(f"layer_{i}"), linewidth=2, label=f"layer_{i}")
               for i in range(1, LAYER_COUNT + 1) if i not in excluded]
    legend_ax = axes[len(experiments)]
    legend_ax.legend(handles=[*handles, Patch(color=BLUR_COLOR, label="Blurred epochs")], loc="center", frameon=False,
                     labelcolor=TEXT_COLOR, ncols=2, title="Layer", title_fontproperties={"size": 10}, fontsize=9)
    legend_ax.get_legend().get_title().set_color(MUTED_COLOR)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("batch", help="batch module, e.g. batch_01")
    batch = load_batch(parser.parse_args().batch)
    fisher = {}
    for name in batch.EXPERIMENTS:
        log_dir = run_log_dir(batch, name, FISHER_SEED)
        fisher[name] = prepare_fisher(load_series(log_dir)["fisher"], log_dir)

    rows = math.ceil((len(batch.EXPERIMENTS) + 1) / COLUMNS)
    fig = plt.figure(figsize=(PANEL_SIZE[0] * COLUMNS, 2 * PANEL_SIZE[1] * rows + 1.2), layout="constrained")
    full_grid, final_grid = fig.subfigures(2, 1)
    plot_fisher_grid(full_grid, batch, fisher, (0, batch.EPOCHS))
    plot_fisher_grid(final_grid, batch, fisher, (max(0, batch.EPOCHS - FINAL_EPOCHS), batch.EPOCHS))
    fig.suptitle(f"{batch.BATCH} ({batch.OPTIMIZER}), seed {FISHER_SEED}: per-layer {fisher_label().lower()}, without "
                 f"{' and '.join(FISHER_EXCLUDED_LAYERS)}. Top: all epochs. Bottom: the last {FINAL_EPOCHS}.",
                 color=TEXT_COLOR, x=0.01, ha="left", fontsize=13)
    output_path = os.path.join(batch_dir(batch), OUTPUT_NAME)
    fig.savefig(output_path, dpi=150)
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
