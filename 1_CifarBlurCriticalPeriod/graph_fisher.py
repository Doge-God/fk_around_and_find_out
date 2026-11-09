"""Plot per-layer Fisher information for all 16 experiments, one row per experiment and one column per seed.

Uses the same Fisher settings as graph_training.py (FISHER_PER_PARAM, FISHER_EXCLUDED_LAYERS, FISHER_LOG_Y, X_RANGE), with
the y-axis scale shared between rows with the same optimizer. Reads results/<experiment>/seed_<seed>/tensorboard/ and writes
one grid per entry of OUTPUTS to results/, e.g. results/graph_fisher.png:
    python graph_fisher.py
"""
import os

from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np

from graph_training import (FISHER_LOG_Y, FISHER_PER_PARAM, FISHER_YLIM_PERCENTILE, GRID_COLOR, LAYER_COLORMAP, MUTED_COLOR,
                            RAW_ALPHA, RESULTS_DIR, TENSORBOARD_DIR, TEXT_COLOR, X_RANGE, ema, load_series, prepare_fisher)

# Experiment -> optimizer and (BLUR_START_EPOCH, BLUR_END_EPOCH) from its script, None if it has no blur.
EXPERIMENTS = {"exp_01": ("Adam", None), "exp_02": ("SGD", None),
               "exp_03": ("Adam", (0, 10)), "exp_04": ("SGD", (0, 10)),
               "exp_05": ("Adam", (5, 15)), "exp_06": ("SGD", (5, 15)),
               "exp_07": ("Adam", (10, 20)), "exp_08": ("SGD", (10, 20)),
               "exp_09": ("Adam", (15, 25)), "exp_10": ("SGD", (15, 25)),
               "exp_11": ("Adam", (20, 30)), "exp_12": ("SGD", (20, 30)),
               "exp_13": ("Adam", (0, 15)), "exp_14": ("SGD", (0, 15)),
               "exp_15": ("Adam", (0, 20)), "exp_16": ("SGD", (0, 20))}
OPTIMIZERS = ("Adam", "SGD")
SEEDS = [42, 43, 44, 45, 46]
LAYER_COUNT = 10
BLUR_COLOR = "#dcd9d0"
OUTPUTS = {"graph_fisher.png": X_RANGE,  # file name -> epochs shown; y-limits fit the values in that range
           "graph_fisher_epoch_140_150.png": (140, 150)}


def run_log_dir(experiment, seed):
    log_dir = os.path.join(RESULTS_DIR, experiment, f"seed_{seed}", TENSORBOARD_DIR)
    if not os.path.isdir(log_dir):
        raise SystemExit(f"No run found at {log_dir}.")
    return log_dir


def in_range(epochs, values, x_range=X_RANGE):
    epochs = np.asarray(epochs)
    return np.asarray(values)[(epochs >= x_range[0]) & (epochs <= x_range[1])]


def layer_color(name):
    """Colour by layer number, so a layer looks the same in every panel even when others are left out. Layer 1 is red."""
    return LAYER_COLORMAP((int(name.rsplit("_", 1)[1]) - 1) / (LAYER_COUNT - 1))


def row_label(experiment):
    optimizer, blur = EXPERIMENTS[experiment]
    return f"{experiment}\n{optimizer}, {'no blur' if blur is None else f'blur epochs {blur[0] + 1}-{blur[1]}'}"


def style_axes(ax, x_range=X_RANGE, blur=None):
    if blur is not None:
        ax.axvspan(*blur, color=BLUR_COLOR, linewidth=0, zorder=0)
    ax.set_xlim(*x_range)
    ax.grid(True, which="major", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED_COLOR)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID_COLOR)


def plot_fisher(ax, fisher):
    for name, (epochs, values) in fisher.items():
        ax.plot(epochs, values, color=layer_color(name), linewidth=1, alpha=RAW_ALPHA, zorder=1)
        ax.plot(epochs, ema(values), color=layer_color(name), linewidth=1.5, label=name, zorder=2)


def fit_fisher_axes(optimizer, panels, fisher, x_range=X_RANGE):
    """Share a y-axis across this optimizer's Fisher panels, fit to their values within x_range. Returns a note on the peak.

    panels and fisher are keyed by (experiment, seed).
    """
    keys = [key for key in panels if EXPERIMENTS[key[0]][0] == optimizer]
    values = np.concatenate([in_range(*series, x_range) for key in keys for series in fisher[key].values()])
    group_axes = [panels[key] for key in keys]
    for ax in group_axes[1:]:
        ax.sharey(group_axes[0])
    # A log axis can't start at 0, so it keeps its automatic lower limit.
    group_axes[0].set_yscale("log" if FISHER_LOG_Y else "linear")
    group_axes[0].set_ylim(None if FISHER_LOG_Y else 0, 1.1 * np.percentile(values, FISHER_YLIM_PERCENTILE))
    return f"{optimizer} {values.max():.3g}"


def plot_grid(fisher, x_range, output_name):
    fig, axes = plt.subplots(len(EXPERIMENTS), len(SEEDS), figsize=(4.4 * len(SEEDS), 2.6 * len(EXPERIMENTS)),
                             sharex=True, layout="constrained", squeeze=False)
    panels = {}
    for row, (experiment, (_, blur)) in enumerate(EXPERIMENTS.items()):
        for col, seed in enumerate(SEEDS):
            ax = axes[row, col]
            panels[experiment, seed] = ax
            style_axes(ax, x_range, blur)
            plot_fisher(ax, fisher[experiment, seed])
            if row == 0:
                ax.set_title(f"seed {seed}", color=TEXT_COLOR)
            if row == len(EXPERIMENTS) - 1:
                ax.set_xlabel("Epoch", color=MUTED_COLOR)
            if col > 0:
                ax.tick_params(labelleft=False)
        axes[row, 0].set_ylabel(row_label(experiment), color=TEXT_COLOR)

    peaks = [fit_fisher_axes(optimizer, panels, fisher, x_range) for optimizer in OPTIMIZERS]
    ylabel = "Fisher information per parameter" if FISHER_PER_PARAM else "Fisher information (trace)"
    fig.suptitle(f"{ylabel}, per layer, epochs {x_range[0]}-{x_range[1]}. Y-axis shared within each optimizer, clipped at the "
                 f"{FISHER_YLIM_PERCENTILE}th percentile (peaks: {'; '.join(peaks)}).",
                 color=TEXT_COLOR, x=0.01, ha="left")
    handles = axes[0, 0].get_legend_handles_labels()[0]
    if x_range[0] < max(blur[1] for _, blur in EXPERIMENTS.values() if blur is not None):
        handles = [*handles, Patch(color=BLUR_COLOR, label="Blurred epochs")]
    fig.legend(handles=handles, loc="outside right upper", frameon=False, labelcolor=TEXT_COLOR)

    output_path = os.path.join(RESULTS_DIR, output_name)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"saved {output_path}")


def main():
    fisher = {}
    for experiment in EXPERIMENTS:
        for seed in SEEDS:
            log_dir = run_log_dir(experiment, seed)
            fisher[experiment, seed] = prepare_fisher(load_series(log_dir)[1], log_dir)
    for output_name, x_range in OUTPUTS.items():
        plot_grid(fisher, x_range, output_name)


if __name__ == "__main__":
    main()
