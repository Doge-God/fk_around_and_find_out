"""Plot train/test accuracy, per-layer Fisher information, weight norms and representation ranks, for one experiment's results.

Reads results/<experiment>/seed_<seed>/tensorboard/ and writes results/<experiment>/seed_<seed>/graph_training.png:
    python graph_training.py exp_01 --seed 42
"""
import argparse
import os

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

EMA_DECAY = 0.9  # per logged point: 0 is no smoothing, closer to 1 is smoother
RAW_ALPHA = 0.05
FISHER_PREFIX = "fisher information/"
WEIGHT_NORM_PREFIX = "weight norm/"
WEIGHT_NORM_TOTAL = "total"
RANK_TAGS = {"Effective rank": "representation/effective_rank",
             "Stable rank delta": "representation/stable_rank_delta"}
ACCURACY_TAGS = {"Train": "train/accuracy", "Test": "test/accuracy"}
RESULTS_DIR = "results"
DEFAULT_SEED = 42
TENSORBOARD_DIR = "tensorboard"
OUTPUT_NAME = "graph_training.png"
LAYER_COLORMAP = LinearSegmentedColormap.from_list("red_to_blue", ["#f4736b", "#6c9ff2"])  # layer 1 is red
BLUES_MIN = 0.35  # the pale start of Blues is too faint to read against a white background
X_RANGE = (0, 150)  # epochs shown on both plots
FISHER_LOG_Y = False  # log scale on the Fisher y-axis
FISHER_YLIM_PERCENTILE = 99  # fit the y-axis to the bulk of the values, not the early output layer spike
TEXT_COLOR = "#0b0b0b"
MUTED_COLOR = "#52514e"
GRID_COLOR = "#e4e3df"


def load_series(log_dir):
    events = EventAccumulator(log_dir, size_guidance={"scalars": 0})
    events.Reload()
    available = events.Tags()["scalars"]

    def read(tag):
        scalars = events.Scalars(tag)
        return [s.step / steps_per_epoch for s in scalars], [s.value for s in scalars]

    def layer_tags(prefix):
        tags = (tag for tag in available if tag.startswith(prefix) and "layer_" in tag)
        return sorted(tags, key=lambda tag: int(tag.rsplit("_", 1)[1]))  # numeric, so layer_10 comes after layer_9

    fisher_tags = layer_tags(FISHER_PREFIX)
    weight_norm_tags = layer_tags(WEIGHT_NORM_PREFIX)
    weight_norm_total_tag = WEIGHT_NORM_PREFIX + WEIGHT_NORM_TOTAL
    missing = [tag for tag in [*ACCURACY_TAGS.values(), *RANK_TAGS.values(), weight_norm_total_tag] if tag not in available]
    missing += [f"{prefix}*" for prefix, tags in [(FISHER_PREFIX, fisher_tags), (WEIGHT_NORM_PREFIX, weight_norm_tags)] if not tags]
    if missing:
        raise SystemExit(f"{log_dir} is missing scalars: {missing}.")
    steps_per_epoch = events.Scalars(ACCURACY_TAGS["Train"])[0].step  # accuracy is first logged at the end of epoch 1
    accuracy = {name: read(tag) for name, tag in ACCURACY_TAGS.items()}
    fisher = {tag[len(FISHER_PREFIX):]: read(tag) for tag in fisher_tags}
    weight_norm = {tag[len(WEIGHT_NORM_PREFIX):]: read(tag) for tag in weight_norm_tags}
    ranks = {name: read(tag) for name, tag in RANK_TAGS.items()}
    return accuracy, fisher, weight_norm, read(weight_norm_total_tag), ranks


def ema(values):
    smoothed = [values[0]]
    for value in values[1:]:
        smoothed.append(EMA_DECAY * smoothed[-1] + (1 - EMA_DECAY) * value)
    return smoothed


def plot_lines(ax, series, ylabel, title, colormap, colormap_min=0.0, smooth=True):
    for i, (name, (epochs, values)) in enumerate(series.items()):
        color = colormap(colormap_min + (1 - colormap_min) * i / max(len(series) - 1, 1))
        if smooth:
            ax.plot(epochs, values, color=color, linewidth=1, alpha=RAW_ALPHA, zorder=1)
            ax.plot(epochs, ema(values), color=color, linewidth=2, label=name, zorder=2)
        else:
            ax.plot(epochs, values, color=color, linewidth=2, label=name, zorder=2)

    ax.set_xlim(*X_RANGE)
    ax.set_xlabel("Epoch", color=MUTED_COLOR)
    ax.set_ylabel(ylabel, color=MUTED_COLOR)
    ax.set_title(title, color=TEXT_COLOR, loc="left")
    ax.grid(True, which="major", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED_COLOR)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID_COLOR)
    ax.legend(frameon=False, labelcolor=TEXT_COLOR)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", help=f"experiment name, a folder under {RESULTS_DIR}/ (e.g. exp_01)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    run_dir = os.path.join(RESULTS_DIR, args.experiment, f"seed_{args.seed}")
    log_dir = os.path.join(run_dir, TENSORBOARD_DIR)
    if not os.path.isdir(log_dir):
        raise SystemExit(f"No run found at {log_dir}. Run the experiment first.")
    accuracy, fisher, weight_norm, weight_norm_total, ranks = load_series(log_dir)

    fig, ((ax_accuracy, ax_fisher), (ax_weight_norm, ax_ranks)) = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")
    plot_lines(ax_accuracy, accuracy, "Accuracy", "Train and test accuracy", plt.get_cmap("Blues"), BLUES_MIN, smooth=False)
    fisher_values = np.concatenate([values for _, values in fisher.values()])
    plot_lines(ax_fisher, fisher, "Fisher information (trace)",
               f"Per-layer Fisher information (y-axis clipped, peak {fisher_values.max():.0f})", LAYER_COLORMAP)
    ax_fisher.set_yscale("log" if FISHER_LOG_Y else "linear")
    # A log axis can't start at 0, so it keeps its automatic lower limit.
    ax_fisher.set_ylim(None if FISHER_LOG_Y else 0, 1.1 * np.percentile(fisher_values, FISHER_YLIM_PERCENTILE))

    plot_lines(ax_weight_norm, weight_norm, "L2 norm", "Total and per-layer weight norm", LAYER_COLORMAP, smooth=False)
    ax_weight_norm.plot(*weight_norm_total, color=TEXT_COLOR, linewidth=2, label="Total", zorder=3)
    # Outside the axes: the lines span the whole height with SGD, so any position inside would cover them.
    ax_weight_norm.legend(frameon=False, labelcolor=TEXT_COLOR, loc="center left", bbox_to_anchor=(1, 0.5))

    plot_lines(ax_ranks, ranks, "Rank", "Representation rank (penultimate features, test batch)", LAYER_COLORMAP, smooth=False)

    output_path = os.path.join(run_dir, OUTPUT_NAME)
    fig.savefig(output_path, dpi=150)
    print(f"read {log_dir}\nsaved {output_path}")


if __name__ == "__main__":
    main()
