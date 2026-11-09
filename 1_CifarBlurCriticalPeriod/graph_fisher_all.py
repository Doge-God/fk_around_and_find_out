"""Plot per-layer Fisher information for all 16 experiments, one row per experiment and one column per seed, with a final
column of train/test accuracy aggregated over the seeds (mean line, SD shading, final test accuracy labelled) and a
horizontal line for the final saved networks' accuracy on blurred test images.

Uses the same Fisher settings as graph_training.py (FISHER_PER_PARAM, FISHER_EXCLUDED_LAYERS, FISHER_LOG_Y, X_RANGE), with
the y-axis scale shared between rows with the same optimizer. Reads results/<experiment>/seed_<seed>/tensorboard/ and
results/eval_blurred.csv (from eval_blurred.py), and writes results/graph_fisher_all.png:
    python graph_fisher_all.py
"""
import csv
import os

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from graph_training import (ACCURACY_TAGS, FISHER_LOG_Y, FISHER_PER_PARAM, FISHER_YLIM_PERCENTILE, GRID_COLOR, LAYER_COLORMAP,
                            MUTED_COLOR, RAW_ALPHA, RESULTS_DIR, TENSORBOARD_DIR, TEXT_COLOR, X_RANGE, ema,
                            load_series, prepare_fisher)

# Experiment -> optimizer and (BLUR_START_EPOCH, BLUR_END_EPOCH) from its script, None if it has no blur.
EXPERIMENTS = {"exp_01": ("Adam", None), "exp_02": ("SGD", None),
               "exp_03": ("Adam", (0, 10)), "exp_04": ("SGD", (0, 10)),
               "exp_05": ("Adam", (5, 15)), "exp_06": ("SGD", (5, 15)),
               "exp_07": ("Adam", (10, 20)), "exp_08": ("SGD", (10, 20)),
               "exp_09": ("Adam", (15, 25)), "exp_10": ("SGD", (15, 25)),
               "exp_11": ("Adam", (20, 30)), "exp_12": ("SGD", (20, 30)),
               "exp_13": ("Adam", (0, 15)), "exp_14": ("SGD", (0, 15)),
               "exp_15": ("Adam", (0, 20)), "exp_16": ("SGD", (0, 20))}
SEEDS = [42, 43, 44, 45, 46]
BLUR_COLOR = "#dcd9d0"
ACCURACY_COLORS = {"Train": "#eb6834", "Test": "#2a78d6"}
BLURRED_TEST_COLOR = "#7b4fc4"
SEED_SD_ALPHA = 0.3
SD_DDOF = 1  # sample SD, the same as eval_blurred.py
X_TICKS = range(0, X_RANGE[1] + 1, 25)
BLURRED_TEST_NAME = "eval_blurred.csv"
OUTPUT_NAME = "graph_fisher_all.png"


def in_range(epochs, values):
    epochs = np.asarray(epochs)
    return np.asarray(values)[(epochs >= X_RANGE[0]) & (epochs <= X_RANGE[1])]


def load_blurred_test():
    """(experiment, seed) -> (final checkpoint's epoch, its accuracy on blurred test images)."""
    path = os.path.join(RESULTS_DIR, BLURRED_TEST_NAME)
    if not os.path.exists(path):
        raise SystemExit(f"No {path}. Run eval_blurred.py first.")
    with open(path, newline="", encoding="utf-8") as f:
        return {(row["experiment"], int(row["seed"])): (int(row["epoch"]), float(row["blurred_accuracy"]))
                for row in csv.DictReader(f)}


def main():
    blurred_test = load_blurred_test()
    fisher, accuracy = {}, {}
    for experiment in EXPERIMENTS:
        for seed in SEEDS:
            log_dir = os.path.join(RESULTS_DIR, experiment, f"seed_{seed}", TENSORBOARD_DIR)
            if not os.path.isdir(log_dir):
                raise SystemExit(f"No run found at {log_dir}: the accuracy column aggregates all {len(SEEDS)} seeds.")
            series = load_series(log_dir)
            accuracy[experiment, seed] = series[0]
            fisher[experiment, seed] = prepare_fisher(series[1], log_dir)

    fig, axes = plt.subplots(len(EXPERIMENTS), len(SEEDS) + 1, figsize=(4.4 * (len(SEEDS) + 1), 2.6 * len(EXPERIMENTS)),
                             sharex=True, layout="constrained", squeeze=False)
    for row, (experiment, (optimizer, blur)) in enumerate(EXPERIMENTS.items()):
        for col, ax in enumerate(axes[row]):
            if blur is not None:
                ax.axvspan(*blur, color=BLUR_COLOR, linewidth=0, zorder=0)
            ax.set_xlim(*X_RANGE)
            ax.set_xticks(X_TICKS)
            ax.grid(True, which="major", color=GRID_COLOR, linewidth=0.8)
            ax.set_axisbelow(True)
            ax.tick_params(colors=MUTED_COLOR)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color(GRID_COLOR)
            if row == len(EXPERIMENTS) - 1:
                ax.set_xlabel("Epoch", color=MUTED_COLOR)
            if col > 0 and col < len(SEEDS):
                ax.tick_params(labelleft=False)
        for col, seed in enumerate(SEEDS):
            plot_fisher(axes[row, col], fisher[experiment, seed])
            if row == 0:
                axes[row, col].set_title(f"Fisher, seed {seed}", color=TEXT_COLOR)
        blur_label = "no blur" if blur is None else f"blur epochs {blur[0] + 1}-{blur[1]}"
        axes[row, 0].set_ylabel(f"{experiment}\n{optimizer}, {blur_label}", color=TEXT_COLOR)
        plot_accuracy(axes[row, -1], experiment, accuracy)
        plot_blurred_test(axes[row, -1], experiment, blurred_test)
    axes[0, -1].set_title(f"Accuracy, {len(SEEDS)} seeds", color=TEXT_COLOR)

    peaks = [fit_fisher_axes(optimizer, axes, fisher) for optimizer in ("Adam", "SGD")]
    accuracy_axes = axes[:, -1]
    for ax in accuracy_axes[1:]:
        ax.sharey(accuracy_axes[0])
    lowest = min(ax.dataLim.ymin for ax in accuracy_axes)
    accuracy_axes[0].set_ylim(np.floor(lowest * 10) / 10, 1)
    accuracy_axes[0].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))

    ylabel = "Fisher information per parameter" if FISHER_PER_PARAM else "Fisher information (trace)"
    fig.suptitle(f"{ylabel}, per layer, epochs {X_RANGE[0]}-{X_RANGE[1]}. Y-axis shared within each optimizer, clipped at the "
                 f"{FISHER_YLIM_PERCENTILE}th percentile (peaks: {'; '.join(peaks)}).\nAccuracy is the mean ± SD over seeds, "
                 f"with test labelled at epoch {X_RANGE[1]}. The dashed line is the final saved network on blurred test images.",
                 color=TEXT_COLOR, x=0.01, ha="left")
    handles = [*axes[0, 0].get_legend_handles_labels()[0], Patch(color=BLUR_COLOR, label="Blurred epochs")]
    fig.legend(handles=handles, loc="outside right upper", frameon=False, labelcolor=TEXT_COLOR)
    # exp_01 has no blur, so the middle of its accuracy panel is empty.
    axes[0, -1].legend(handles=[*[Line2D([], [], color=color, linewidth=2, label=name) for name, color in ACCURACY_COLORS.items()],
                                Line2D([], [], color=BLURRED_TEST_COLOR, linewidth=1.5, linestyle="--", label="Blurred test, final network"),
                                Patch(color=MUTED_COLOR, alpha=SEED_SD_ALPHA, linewidth=0, label="± 1 SD over seeds")],
                       loc="center right", bbox_to_anchor=(1, 0.5), frameon=False, labelcolor=TEXT_COLOR)

    output_path = os.path.join(RESULTS_DIR, OUTPUT_NAME)
    fig.savefig(output_path, dpi=150)
    print(f"saved {output_path}")


def plot_fisher(ax, fisher):
    for i, (name, (epochs, values)) in enumerate(fisher.items()):
        color = LAYER_COLORMAP(i / max(len(fisher) - 1, 1))
        ax.plot(epochs, values, color=color, linewidth=1, alpha=RAW_ALPHA, zorder=1)
        ax.plot(epochs, ema(values), color=color, linewidth=1.5, label=name, zorder=2)


def plot_accuracy(ax, experiment, accuracy):
    for name in ACCURACY_TAGS:
        epochs = accuracy[experiment, SEEDS[0]][name][0]
        epochs = in_range(epochs, epochs)
        values = np.stack([in_range(*accuracy[experiment, seed][name]) for seed in SEEDS])
        mean, sd = values.mean(axis=0), values.std(axis=0, ddof=SD_DDOF)
        ax.fill_between(epochs, mean - sd, mean + sd, color=ACCURACY_COLORS[name], alpha=SEED_SD_ALPHA, linewidth=0, zorder=1)
        ax.plot(epochs, mean, color=ACCURACY_COLORS[name], linewidth=1.5, zorder=2)
        if name == "Test":
            ax.plot(epochs[-1], mean[-1], "o", color=ACCURACY_COLORS[name], markersize=5, clip_on=False, zorder=3)
            ax.annotate(f"Test at {epochs[-1]:.0f}: {mean[-1]:.1%} ± {sd[-1]:.1%} SD",
                        (epochs[-1], mean[-1]), xytext=(0, -10), textcoords="offset points", ha="right", va="top",
                        color=TEXT_COLOR)
            print(f"{experiment}: test accuracy at epoch {epochs[-1]:.0f} {mean[-1]:.2%} ± {sd[-1]:.2%} SD")


def plot_blurred_test(ax, experiment, blurred_test):
    epoch = blurred_test[experiment, SEEDS[0]][0]  # the same for every seed of an experiment
    values = np.array([blurred_test[experiment, seed][1] for seed in SEEDS])
    mean, sd = values.mean(), values.std(ddof=SD_DDOF)
    ax.axhspan(mean - sd, mean + sd, color=BLURRED_TEST_COLOR, alpha=SEED_SD_ALPHA, linewidth=0, zorder=1)
    ax.axhline(mean, color=BLURRED_TEST_COLOR, linewidth=1.5, linestyle="--", zorder=2)
    ax.annotate(f"Blurred test at {epoch}: {mean:.1%} ± {sd:.1%} SD", (X_RANGE[1], mean), xytext=(0, 4),
                textcoords="offset points", ha="right", va="bottom", color=TEXT_COLOR)
    print(f"{experiment}: blurred test accuracy at epoch {epoch} {mean:.2%} ± {sd:.2%} SD")


def fit_fisher_axes(optimizer, axes, fisher):
    """Share a y-axis across the Fisher panels of this optimizer's rows, fit to their values within X_RANGE."""
    rows = [row for row, (_, (name, _)) in enumerate(EXPERIMENTS.items()) if name == optimizer]
    experiments = [experiment for experiment, (name, _) in EXPERIMENTS.items() if name == optimizer]
    values = np.concatenate([in_range(*series) for (experiment, _), layers in fisher.items() if experiment in experiments
                             for series in layers.values()])
    group_axes = [ax for row in rows for ax in axes[row, :len(SEEDS)]]
    for ax in group_axes[1:]:
        ax.sharey(group_axes[0])
    # A log axis can't start at 0, so it keeps its automatic lower limit.
    group_axes[0].set_yscale("log" if FISHER_LOG_Y else "linear")
    group_axes[0].set_ylim(None if FISHER_LOG_Y else 0, 1.1 * np.percentile(values, FISHER_YLIM_PERCENTILE))
    return f"{optimizer} {values.max():.3g}"


if __name__ == "__main__":
    main()
