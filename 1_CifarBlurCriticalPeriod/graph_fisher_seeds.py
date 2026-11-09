"""Plot per-layer Fisher information for every experiment and seed, one row per experiment and one column per seed.

Uses the same Fisher settings as graph_training.py (FISHER_PER_PARAM, FISHER_EXCLUDED_LAYERS, FISHER_LOG_Y, X_RANGE),
with the y-axis scale shared within each of Y_SCALE_GROUPS. Reads results/<experiment>/seed_<seed>/tensorboard/ and writes
one grid per entry of OUTPUTS to results/, e.g. results/graph_fisher_seeds.png:
    python graph_fisher_seeds.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np

from graph_training import (FISHER_LOG_Y, FISHER_PER_PARAM, FISHER_YLIM_PERCENTILE, GRID_COLOR, LAYER_COLORMAP,
                            MUTED_COLOR, RAW_ALPHA, RESULTS_DIR, TENSORBOARD_DIR, TEXT_COLOR, X_RANGE, ema,
                            load_series, prepare_fisher)

EXPERIMENTS = {"exp_01": "Adam", "exp_02": "SGD", "exp_03": "Adam + blur", "exp_04": "SGD + blur"}
SEEDS = [42, 43, 44, 45, 46]
Y_SCALE_GROUPS = [["exp_01", "exp_03"], ["exp_02", "exp_04"]]  # rows that share a y-axis scale: Adam, then SGD
BLUR_EPOCHS = {"exp_03": 10, "exp_04": 10}  # BLUR_EPOCHS in exp_03.py and exp_04.py, shaded on their rows
BLUR_COLOR = "#ebeae6"
OUTPUTS = {"graph_fisher_seeds.png": X_RANGE,  # file name -> epochs shown; y-limits fit the values in that range
           "graph_fisher_seeds_epoch_140_150.png": (140, 150)}


def main():
    runs = {}
    for experiment in EXPERIMENTS:
        for seed in SEEDS:
            log_dir = os.path.join(RESULTS_DIR, experiment, f"seed_{seed}", TENSORBOARD_DIR)
            if os.path.isdir(log_dir):
                runs[experiment, seed] = prepare_fisher(load_series(log_dir)[1], log_dir)
            else:
                print(f"skipping {log_dir}: not found")
    if not runs:
        raise SystemExit(f"No runs found under {RESULTS_DIR}/.")
    for output_name, x_range in OUTPUTS.items():
        plot_grid(runs, x_range, output_name)


def plot_grid(runs, x_range, output_name):
    def in_range(epochs, values):
        epochs = np.asarray(epochs)
        return np.asarray(values)[(epochs >= x_range[0]) & (epochs <= x_range[1])]

    group_values = {}  # first experiment in each group -> every Fisher value in that group's runs, within x_range
    for group in Y_SCALE_GROUPS:
        values = [in_range(*series) for (experiment, _), fisher in runs.items() if experiment in group
                  for series in fisher.values()]
        if values:
            group_values[group[0]] = np.concatenate(values)

    fig, axes = plt.subplots(len(EXPERIMENTS), len(SEEDS), figsize=(4.4 * len(SEEDS), 3.2 * len(EXPERIMENTS)),
                             sharex=True, layout="constrained", squeeze=False)
    for row, (experiment, description) in enumerate(EXPERIMENTS.items()):
        for col, seed in enumerate(SEEDS):
            ax = axes[row, col]
            if experiment in BLUR_EPOCHS:
                ax.axvspan(0, BLUR_EPOCHS[experiment], color=BLUR_COLOR, zorder=0)
            fisher = runs.get((experiment, seed))
            if fisher is None:
                ax.text(0.5, 0.5, "no run", transform=ax.transAxes, ha="center", va="center", color=MUTED_COLOR)
            else:
                for i, (name, (epochs, values)) in enumerate(fisher.items()):
                    color = LAYER_COLORMAP(i / max(len(fisher) - 1, 1))
                    ax.plot(epochs, values, color=color, linewidth=1, alpha=RAW_ALPHA, zorder=1)
                    ax.plot(epochs, ema(values), color=color, linewidth=1.5, label=name, zorder=2)

            ax.set_xlim(*x_range)
            ax.grid(True, which="major", color=GRID_COLOR, linewidth=0.8)
            ax.set_axisbelow(True)
            ax.tick_params(colors=MUTED_COLOR)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color(GRID_COLOR)
            if row == 0:
                ax.set_title(f"seed {seed}", color=TEXT_COLOR)
            if row == len(EXPERIMENTS) - 1:
                ax.set_xlabel("Epoch", color=MUTED_COLOR)
            if col == 0:
                ax.set_ylabel(f"{experiment}\n{description}", color=TEXT_COLOR)
            else:
                ax.tick_params(labelleft=False)

    rows = {experiment: row for row, experiment in enumerate(EXPERIMENTS)}
    peaks = []
    for group in Y_SCALE_GROUPS:
        if group[0] not in group_values:
            continue
        values = group_values[group[0]]
        group_axes = [ax for experiment in group if experiment in rows for ax in axes[rows[experiment]]]
        for ax in group_axes:
            ax.sharey(group_axes[0])
        # A log axis can't start at 0, so it keeps its automatic lower limit.
        group_axes[0].set_yscale("log" if FISHER_LOG_Y else "linear")
        group_axes[0].set_ylim(None if FISHER_LOG_Y else 0, 1.1 * np.percentile(values, FISHER_YLIM_PERCENTILE))
        peaks.append(f"{' and '.join(group)} {values.max():.3g}")

    ylabel = "Fisher information per parameter" if FISHER_PER_PARAM else "Fisher information (trace)"
    fig.suptitle(f"{ylabel}, per layer, epochs {x_range[0]}-{x_range[1]}. Y-axis shared within {' / '.join(' and '.join(g) for g in Y_SCALE_GROUPS)}, "
                 f"clipped at the {FISHER_YLIM_PERCENTILE}th percentile (peaks: {'; '.join(peaks)})."
                 + (" Shading is the blur." if x_range[0] < max(BLUR_EPOCHS.values()) else ""),
                 color=TEXT_COLOR, x=0.01, ha="left")
    handles, labels = next(ax for ax in axes.flat if ax.get_legend_handles_labels()[0]).get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside right center", frameon=False, labelcolor=TEXT_COLOR)

    output_path = os.path.join(RESULTS_DIR, output_name)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
