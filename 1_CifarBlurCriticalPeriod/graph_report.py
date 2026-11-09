"""One summary figure for all 16 experiments, one row per experiment. Columns:

1. Per-layer Fisher information of seed FISHER_SEED, y-axis shared within each optimizer, as in graph_fisher.py.
2. Train and test accuracy over the seeds, with the final test mean and SD labelled.
3. Epoch CHECKPOINT_EPOCH checkpoints on the full test set blurred with sigma 0-3 in steps of 0.1, over the seeds.
4. Representation ranks (penultimate features, first test batch), over the seeds.
5. Total and per-layer weight norm, over the seeds, y-axis shared within each optimizer.

Lines are the mean over seeds and shading is +/- 1 sample SD. Columns other than 3 are read from the TensorBoard logs and show
X_RANGE epochs. Column 3 evaluates the checkpoints, best on a GPU, and saves the results to results/graph_report_blur_eval.csv.
Runs already there at CHECKPOINT_EPOCH are reused: delete that file to evaluate them all again. Writes results/graph_report.png:
    python graph_report.py
"""
import csv
from itertools import islice
import math
import os

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import torch
from torchvision.transforms import functional as TF

from eval_blurred import CHECKPOINT_EPOCH, DATA_ROOT, SD_DDOF, checkpoint_path
from graph_fisher import (BLUR_COLOR, EXPERIMENTS, LAYER_COUNT, OPTIMIZERS, SEEDS, fit_fisher_axes, in_range, layer_color,
                          plot_fisher, row_label, run_log_dir, style_axes)
from graph_training import (FISHER_PER_PARAM, FISHER_YLIM_PERCENTILE, MUTED_COLOR, RESULTS_DIR, TEXT_COLOR, X_RANGE, load_series,
                            prepare_fisher)
from networks import ResNet18
from training_procedure import BATCH_SIZE, BLUR_SIGMA, DATASET, evaluate
from util import get_loaders

FISHER_SEED = 42
BLUR_SIGMAS = [step / 10 for step in range(31)]  # in pixels
BLUR_EVAL_BATCHES = None  # test batches of 512 images to use; None is the full test set of 10,000
BLUR_EVAL_NAME = "graph_report_blur_eval.csv"
OUTPUT_NAME = "graph_report.png"
COLUMN_COUNT = 5
SD_ALPHA = 0.3
LAYER_SD_ALPHA = 0.2  # ten overlapping layers need lighter shading
ACCURACY_COLORS = {"Train": "#eb6834", "Test": "#2a78d6"}
RANK_COLORS = {"Effective rank": "#1a8f5a", "Stable rank delta": "#b84fc4"}
BLUR_EVAL_COLOR = "#7b4fc4"
TOTAL_COLOR = TEXT_COLOR


def blur_batches(batches, sigma):
    """The same reach as the training blur in training_procedure.py, 3 sigma either side. Sigma 0 leaves the images sharp."""
    if sigma == 0:
        return batches
    return [(TF.gaussian_blur(x, 2 * math.ceil(3 * sigma) + 1, sigma), y) for x, y in batches]


def run_blur_eval(runs):
    """Evaluate each (experiment, seed) in runs at every sigma, and return its rows for the CSV."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"evaluating {len(runs)} checkpoints from epoch {CHECKPOINT_EPOCH} on {device}")
    _, _, test_loader = get_loaders(DATASET, BATCH_SIZE, data_root=DATA_ROOT)
    batches = [(x.to(device), y.to(device)) for x, y in islice(test_loader, BLUR_EVAL_BATCHES)]
    images = sum(len(y) for _, y in batches)
    blurred = {sigma: blur_batches(batches, sigma) for sigma in BLUR_SIGMAS}

    model = ResNet18().to(device)
    rows = []
    for experiment, seed in runs:
        model.load_state_dict(torch.load(checkpoint_path(experiment, seed), map_location=device, weights_only=True))
        for sigma in BLUR_SIGMAS:
            _, accuracy = evaluate(model, blurred[sigma], device)
            rows.append({"experiment": experiment, "seed": seed, "epoch": CHECKPOINT_EPOCH, "sigma": sigma, "images": images,
                         "accuracy": accuracy})
        print(f"{experiment} seed {seed} done")
    return rows


def load_blur_eval():
    """Evaluate any runs missing from the CSV at CHECKPOINT_EPOCH. Returns (experiment, seed) -> {sigma: accuracy} and the number
    of test images."""
    path = os.path.join(RESULTS_DIR, BLUR_EVAL_NAME)
    rows = []
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            rows = [row for row in csv.DictReader(f) if int(row["epoch"]) == CHECKPOINT_EPOCH]
    done = {(row["experiment"], int(row["seed"])) for row in rows}
    missing = [(experiment, seed) for experiment in EXPERIMENTS for seed in SEEDS if (experiment, seed) not in done]
    if missing:
        rows += run_blur_eval(missing)
        rows.sort(key=lambda row: (row["experiment"], int(row["seed"]), float(row["sigma"])))
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"saved {path}")
    accuracy, images = {}, set()
    for row in rows:
        accuracy.setdefault((row["experiment"], int(row["seed"])), {})[float(row["sigma"])] = float(row["accuracy"])
        images.add(int(row["images"]))
    if len(images) != 1:
        raise SystemExit(f"{path} mixes runs on different numbers of test images ({sorted(images)}): delete it to evaluate again.")
    return accuracy, images.pop()


def plot_mean_sd(ax, epochs, values, color, alpha=SD_ALPHA, linewidth=1.5, zorder=2):
    """values has one row per seed. Returns the mean and SD."""
    mean, sd = values.mean(axis=0), values.std(axis=0, ddof=SD_DDOF)
    ax.fill_between(epochs, mean - sd, mean + sd, color=color, alpha=alpha, linewidth=0, zorder=zorder - 1)
    ax.plot(epochs, mean, color=color, linewidth=linewidth, zorder=zorder)
    return mean, sd


def plot_series_over_seeds(ax, runs, color, **kwargs):
    """runs has one (epochs, values) per seed, cut to X_RANGE here."""
    epochs = in_range(runs[0][0], runs[0][0])
    return (epochs, *plot_mean_sd(ax, epochs, np.stack([in_range(*run) for run in runs]), color, **kwargs))


def plot_accuracy(ax, experiment, runs):
    for name, color in ACCURACY_COLORS.items():
        epochs, mean, sd = plot_series_over_seeds(ax, [runs[experiment, seed]["accuracy"][name] for seed in SEEDS], color)
    # The loop ends on Test.
    ax.plot(epochs[-1], mean[-1], "o", color=color, markersize=5, clip_on=False, zorder=3)
    ax.annotate(f"Test at {epochs[-1]:.0f}: {mean[-1]:.1%} ± {sd[-1]:.1%} SD", (epochs[-1], mean[-1]), xytext=(0, -10),
                textcoords="offset points", ha="right", va="top", color=TEXT_COLOR)


def plot_blur_eval(ax, experiment, blur_eval):
    values = np.array([[blur_eval[experiment, seed][sigma] for sigma in BLUR_SIGMAS] for seed in SEEDS])
    plot_mean_sd(ax, BLUR_SIGMAS, values, BLUR_EVAL_COLOR)
    ax.axvline(BLUR_SIGMA, color=MUTED_COLOR, linewidth=1, linestyle="--", zorder=1)


def plot_ranks(ax, experiment, runs):
    for name, color in RANK_COLORS.items():
        plot_series_over_seeds(ax, [runs[experiment, seed]["ranks"][name] for seed in SEEDS], color)


def plot_weight_norm(ax, experiment, runs):
    for name in runs[experiment, SEEDS[0]]["weight_norm"]:
        plot_series_over_seeds(ax, [runs[experiment, seed]["weight_norm"][name] for seed in SEEDS], layer_color(name),
                               alpha=LAYER_SD_ALPHA, linewidth=1.2)
    plot_series_over_seeds(ax, [runs[experiment, seed]["weight_norm_total"] for seed in SEEDS], TOTAL_COLOR, zorder=4)


def share_y(axes):
    for ax in axes[1:]:
        ax.sharey(axes[0])


def main():
    blur_eval, blur_eval_images = load_blur_eval()
    runs, fisher = {}, {}
    for experiment in EXPERIMENTS:
        for seed in SEEDS:
            log_dir = run_log_dir(experiment, seed)
            accuracy, seed_fisher, weight_norm, weight_norm_total, ranks = load_series(log_dir)
            runs[experiment, seed] = {"accuracy": accuracy, "weight_norm": weight_norm, "weight_norm_total": weight_norm_total,
                                      "ranks": ranks}
            if seed == FISHER_SEED:
                fisher[experiment, seed] = prepare_fisher(seed_fisher, log_dir)

    fig, axes = plt.subplots(len(EXPERIMENTS), COLUMN_COUNT, figsize=(4.4 * COLUMN_COUNT, 2.6 * len(EXPERIMENTS)), sharex="col",
                             layout="constrained", squeeze=False)
    fisher_panels = {}
    for row, (experiment, (_, blur)) in enumerate(EXPERIMENTS.items()):
        ax_fisher, ax_accuracy, ax_blur_eval, ax_ranks, ax_weight_norm = axes[row]
        for ax in axes[row]:
            if ax is ax_blur_eval:
                style_axes(ax, (BLUR_SIGMAS[0], BLUR_SIGMAS[-1]))
            else:
                style_axes(ax, X_RANGE, blur)
        fisher_panels[experiment, FISHER_SEED] = ax_fisher
        plot_fisher(ax_fisher, fisher[experiment, FISHER_SEED])
        plot_accuracy(ax_accuracy, experiment, runs)
        plot_blur_eval(ax_blur_eval, experiment, blur_eval)
        plot_ranks(ax_ranks, experiment, runs)
        plot_weight_norm(ax_weight_norm, experiment, runs)
        ax_fisher.set_ylabel(row_label(experiment), color=TEXT_COLOR)

    titles = [f"Fisher information, seed {FISHER_SEED}", f"Accuracy, {len(SEEDS)} seeds",
              f"Epoch {CHECKPOINT_EPOCH} on blurred test set, {len(SEEDS)} seeds",
              f"Representation rank (test batch), {len(SEEDS)} seeds", f"Weight norm, {len(SEEDS)} seeds"]
    for ax, title in zip(axes[0], titles):
        ax.set_title(title, color=TEXT_COLOR)
    xlabels = ["Epoch", "Epoch", "Test image blur sigma (pixels)", "Epoch", "Epoch"]
    for ax, xlabel in zip(axes[-1], xlabels):
        ax.set_xlabel(xlabel, color=MUTED_COLOR)

    peaks = [fit_fisher_axes(optimizer, fisher_panels, fisher) for optimizer in OPTIMIZERS]
    share_y(axes[:, 1])
    lowest = min(ax.dataLim.ymin for ax in axes[:, 1])
    axes[0, 1].set_ylim(np.floor(lowest * 10) / 10, 1)
    axes[0, 1].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    share_y(axes[:, 2])
    axes[0, 2].set_ylim(0, 1)
    axes[0, 2].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    share_y(axes[:, 3])
    axes[0, 3].set_ylim(bottom=0)
    for optimizer in OPTIMIZERS:
        rows = [row for row, (name, _) in enumerate(EXPERIMENTS.values()) if name == optimizer]
        share_y(axes[rows, 4])
        axes[rows[0], 4].set_ylim(bottom=0)

    fisher_label = "Fisher information per parameter" if FISHER_PER_PARAM else "Fisher information (trace)"
    fig.suptitle(f"{fisher_label}, without the first convolution and classifier: y-axis shared within each optimizer, clipped at the "
                 f"{FISHER_YLIM_PERCENTILE}th percentile (peaks: {'; '.join(peaks)}).\n"
                 f"Other columns: mean ± sample SD over seeds {SEEDS[0]}-{SEEDS[-1]}. Weight norm y-axis shared within each optimizer.\n"
                 f"Epoch columns show epochs {X_RANGE[0]}-{X_RANGE[1]}. The blur evaluation uses {blur_eval_images} test images.",
                 color=TEXT_COLOR, x=0.01, ha="left")
    layer_handles = [Line2D([], [], color=layer_color(f"layer_{i}"), linewidth=2, label=f"layer_{i}")
                     for i in range(1, LAYER_COUNT + 1)]
    fig.legend(handles=[*layer_handles,
                        Line2D([], [], color=TOTAL_COLOR, linewidth=2, label="Total weight norm"),
                        Patch(color=BLUR_COLOR, label="Blurred epochs"),
                        *[Line2D([], [], color=color, linewidth=2, label=name)
                          for name, color in [*ACCURACY_COLORS.items(), *RANK_COLORS.items()]],
                        Line2D([], [], color=BLUR_EVAL_COLOR, linewidth=2, label="Blurred test accuracy"),
                        Line2D([], [], color=MUTED_COLOR, linewidth=1, linestyle="--", label=f"Training blur sigma ({BLUR_SIGMA})"),
                        Patch(color=MUTED_COLOR, alpha=SD_ALPHA, linewidth=0, label="± 1 SD over seeds")],
               loc="outside right upper", frameon=False, labelcolor=TEXT_COLOR)

    output_path = os.path.join(RESULTS_DIR, OUTPUT_NAME)
    fig.savefig(output_path, dpi=150)
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
