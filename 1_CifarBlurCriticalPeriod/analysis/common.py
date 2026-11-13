"""Helpers shared by the analysis scripts: finding a batch's runs, reading their TensorBoard logs, and the plot styles.

Every analysis script runs from the experiment folder, e.g. python -m analysis.graph_fisher batch_01, so the batch modules,
networks.py and results/ are found from there.
"""
import importlib
import os

from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from networks import ResNet18

RESULTS_DIR = "results"
EXPERIMENTS_DIR = "experiments"  # results/<batch>/experiments/<experiment>/seed_<seed>/
TENSORBOARD_DIR = "tensorboard"
CHECKPOINTS_DIR = "checkpoints"
SD_DDOF = 1  # sample SD, since the seeds are a sample of the possible runs
BLUR_SIGMAS = [step / 10 for step in range(31)]  # test image blur, in pixels

# TensorBoard tags. The blurred ones are only logged during blur epochs, so a run without blur has none.
ACCURACY_TAGS = {"Train": "train/accuracy", "Test": "test/accuracy"}
BLURRED_ACCURACY_TAGS = {"Test": "test_blurred/accuracy"}
RANK_TAGS = {"Effective rank": "representation/effective_rank", "Stable rank delta": "representation/stable_rank_delta"}
SIGREG_TAGS = {"SIGReg": "representation/sigreg", "SIGReg, standardised features": "representation/sigreg_standardized"}
BLURRED_PREFIX = {"test/": "test_blurred/", "representation/": "representation_blurred/"}
FISHER_PREFIX = "fisher information/"
BLURRED_FISHER_PREFIX = "fisher information_blurred/"
WEIGHT_NORM_PREFIX = "weight norm/"
WEIGHT_NORM_TOTAL = "total"

# Fisher information
EMA_DECAY = 0.9  # per logged point: 0 is no smoothing, closer to 1 is smoother
RAW_ALPHA = 0.05
FISHER_LOG_Y = False  # log scale on the Fisher y-axis
FISHER_PER_PARAM = True  # divide each layer's Fisher trace by its parameter count
FISHER_EXCLUDED_LAYERS = ("layer_1", "layer_10")  # first convolution and classifier: per parameter they dwarf the blocks
FISHER_YLIM_PERCENTILE = 99  # fit the y-axis to the bulk of the values, not the early output layer spike
LAYER_COUNT = 10

# Colours
TEXT_COLOR = "#0b0b0b"
MUTED_COLOR = "#52514e"
GRID_COLOR = "#e4e3df"
BLUR_COLOR = "#dcd9d0"  # shading over blurred epochs
LAYER_COLORMAP = LinearSegmentedColormap.from_list("red_to_blue", ["#f4736b", "#6c9ff2"])  # layer 1 is red
REFERENCE_COLOR = "#0b0b0b"  # no blur (solid) and full blur (dotted)
REFERENCE_STYLES = {"none": "-", "full": ":"}
REFERENCE_LINEWIDTH = 2.2  # a little heavier than the windows, so the dotted line reads as a line
WINDOW_LINEWIDTH = 1.4
VIRIDIS_MAX = 0.9  # the yellow end of viridis is too faint to read against a white background
WINDOW_COLORMAP = ListedColormap(plt.get_cmap("viridis")(np.linspace(0, VIRIDIS_MAX, 256)), name="viridis_to_0.9")


# ---- batches and runs ----

def load_batch(name):
    """The batch module, e.g. batch_01: BATCH, OPTIMIZER, EPOCHS, SEEDS and EXPERIMENTS (experiment -> blur window or None)."""
    return importlib.import_module(name)


def batch_dir(batch, root=RESULTS_DIR):
    """root is results/ by default; another root has the same layout, e.g. the HPC job's output or checkpoints dataset."""
    return os.path.join(root, batch.BATCH)


def run_dir(batch, experiment, seed, root=RESULTS_DIR):
    return os.path.join(batch_dir(batch, root), EXPERIMENTS_DIR, experiment, f"seed_{seed}")


def run_log_dir(batch, experiment, seed):
    log_dir = os.path.join(run_dir(batch, experiment, seed), TENSORBOARD_DIR)
    if not os.path.isdir(log_dir):
        raise SystemExit(f"No run found at {log_dir}.")
    return log_dir


def checkpoint_path(batch, experiment, seed, root=RESULTS_DIR):
    """The final network, which every experiment saves."""
    path = os.path.join(run_dir(batch, experiment, seed, root), CHECKPOINTS_DIR, f"epoch_{batch.EPOCHS:03d}.pt")
    if not os.path.exists(path):
        raise SystemExit(f"No checkpoint at {path}.")
    return path


def blur_group(batch, blur):
    """'none', 'full' (blur on every epoch) or 'window'."""
    return "none" if blur is None else "full" if blur == (0, batch.EPOCHS) else "window"


def blur_label(batch, blur):
    """Blurred epochs, counting from 1."""
    if blur is None:
        return "No blur"
    return f"{blur[0] + 1}-{blur[1]}" + (" (all)" if blur_group(batch, blur) == "full" else "")


def legend_label(batch, blur):
    return {"none": "No blur", "full": "Blur on every epoch"}.get(blur_group(batch, blur), f"Blur epochs {blur_label(batch, blur)}")


def still_blurred(blur, epoch):
    """Whether the network was still training on blurred images at epoch."""
    return blur is not None and blur[0] < epoch <= blur[1]


# ---- TensorBoard logs ----

def load_series(log_dir):
    """Each logged series as (epochs, values). Returns a dict with:
    accuracy, ranks, sigreg: name -> series (as in ACCURACY_TAGS, RANK_TAGS, SIGREG_TAGS),
    blurred_accuracy, blurred_ranks, blurred_sigreg: the same on blurred test images, for the names logged,
    fisher, blurred_fisher, weight_norm: layer name -> series (blurred_fisher empty if not logged), and weight_norm_total.
    """
    events = EventAccumulator(log_dir, size_guidance={"scalars": 0})
    events.Reload()
    available = events.Tags()["scalars"]

    def read(tag):
        scalars = events.Scalars(tag)
        return [s.step / steps_per_epoch for s in scalars], [s.value for s in scalars]

    def layer_tags(prefix):
        tags = (tag for tag in available if tag.startswith(prefix) and "layer_" in tag)
        return sorted(tags, key=lambda tag: int(tag.rsplit("_", 1)[1]))  # numeric, so layer_10 comes after layer_9

    def blurred(tag):
        prefix = next(prefix for prefix in BLURRED_PREFIX if tag.startswith(prefix))
        return BLURRED_PREFIX[prefix] + tag[len(prefix):]

    fisher_tags = layer_tags(FISHER_PREFIX)
    weight_norm_tags = layer_tags(WEIGHT_NORM_PREFIX)
    weight_norm_total_tag = WEIGHT_NORM_PREFIX + WEIGHT_NORM_TOTAL
    required = [*ACCURACY_TAGS.values(), *RANK_TAGS.values(), *SIGREG_TAGS.values(), weight_norm_total_tag]
    missing = [tag for tag in required if tag not in available]
    missing += [f"{prefix}*" for prefix, tags in [(FISHER_PREFIX, fisher_tags), (WEIGHT_NORM_PREFIX, weight_norm_tags)] if not tags]
    if missing:
        raise SystemExit(f"{log_dir} is missing scalars: {missing}.")
    steps_per_epoch = events.Scalars(ACCURACY_TAGS["Train"])[0].step  # accuracy is first logged at the end of epoch 1

    series = {"fisher": {tag[len(FISHER_PREFIX):]: read(tag) for tag in fisher_tags},
              "blurred_fisher": {tag[len(BLURRED_FISHER_PREFIX):]: read(tag) for tag in layer_tags(BLURRED_FISHER_PREFIX)},
              "weight_norm": {tag[len(WEIGHT_NORM_PREFIX):]: read(tag) for tag in weight_norm_tags},
              "weight_norm_total": read(weight_norm_total_tag)}
    for key, tags in [("accuracy", ACCURACY_TAGS), ("ranks", RANK_TAGS), ("sigreg", SIGREG_TAGS)]:
        series[key] = {name: read(tag) for name, tag in tags.items()}
        series[f"blurred_{key}"] = {name: read(blurred(tag)) for name, tag in tags.items()
                                    if tag.startswith(tuple(BLURRED_PREFIX)) and blurred(tag) in available}
    return series


def prepare_fisher(fisher, log_dir):
    """Divide each layer's Fisher by its parameter count if FISHER_PER_PARAM, and drop FISHER_EXCLUDED_LAYERS."""
    if FISHER_PER_PARAM:
        # Every experiment uses the default ResNet18, and util.compute_fisher groups parameters by model.layers too.
        param_counts = [sum(p.numel() for p in layer.parameters()) for layer in ResNet18().layers]
        if len(fisher) != len(param_counts):
            raise SystemExit(f"{log_dir} has {len(fisher)} Fisher layers, but ResNet18 has {len(param_counts)}.")
        fisher = {name: (epochs, [v / param_counts[int(name.rsplit("_", 1)[1]) - 1] for v in values])
                  for name, (epochs, values) in fisher.items()}
    return {name: series for name, series in fisher.items() if name not in FISHER_EXCLUDED_LAYERS}


def fisher_label():
    return "Fisher information per parameter" if FISHER_PER_PARAM else "Fisher information (trace)"


def value_at_epoch(series, epoch):
    epochs, values = series
    index = int(np.argmin(np.abs(np.asarray(epochs) - epoch)))
    if not np.isclose(epochs[index], epoch):
        raise SystemExit(f"No logged value at epoch {epoch}: the closest is at {epochs[index]}.")
    return values[index]


def in_range(epochs, values, x_range):
    epochs = np.asarray(epochs)
    return np.asarray(values)[(epochs >= x_range[0]) & (epochs <= x_range[1])]


def ema(values):
    smoothed = [values[0]]
    for value in values[1:]:
        smoothed.append(EMA_DECAY * smoothed[-1] + (1 - EMA_DECAY) * value)
    return smoothed


def mean_sd(values, spec):
    return f"{np.mean(values):{spec}} ± {np.std(values, ddof=SD_DDOF):{spec}}"


# ---- plotting ----

def style_axes(ax, x_range, blur=None):
    if blur is not None:
        ax.axvspan(*blur, color=BLUR_COLOR, linewidth=0, zorder=0)
    ax.set_xlim(*x_range)
    ax.grid(True, which="major", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED_COLOR)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID_COLOR)


def layer_color(name):
    """Colour by layer number, so a layer looks the same in every panel even when others are left out. Layer 1 is red."""
    return LAYER_COLORMAP((int(name.rsplit("_", 1)[1]) - 1) / (LAYER_COUNT - 1))


def plot_fisher(ax, fisher):
    for name, (epochs, values) in fisher.items():
        ax.plot(epochs, values, color=layer_color(name), linewidth=1, alpha=RAW_ALPHA, zorder=1)
        ax.plot(epochs, ema(values), color=layer_color(name), linewidth=1.5, label=name, zorder=2)


def fit_fisher_axes(axes, fisher, x_range):
    """Share one y-axis across axes (a list), fit to the values of fisher (one per axis) within x_range. Returns the peak."""
    values = np.concatenate([in_range(*series, x_range) for run in fisher for series in run.values()])
    for ax in axes[1:]:
        ax.sharey(axes[0])
    # A log axis can't start at 0, so it keeps its automatic lower limit.
    axes[0].set_yscale("log" if FISHER_LOG_Y else "linear")
    axes[0].set_ylim(None if FISHER_LOG_Y else 0, 1.1 * np.percentile(values, FISHER_YLIM_PERCENTILE))
    return values.max()


def window_norm(batch):
    """Window start epoch -> 0-1 position on WINDOW_COLORMAP."""
    starts = [blur[0] for blur in batch.EXPERIMENTS.values() if blur_group(batch, blur) == "window"]
    return plt.Normalize(min(starts), max(starts))


def experiment_styles(batch):
    """Experiment name -> (colour, linestyle, linewidth): no blur and full blur in black, solid and dotted; the windows on
    WINDOW_COLORMAP by start epoch, darkest for the earliest."""
    norm = window_norm(batch)
    styles = {}
    for name, blur in batch.EXPERIMENTS.items():
        group = blur_group(batch, blur)
        if group in REFERENCE_STYLES:
            styles[name] = (REFERENCE_COLOR, REFERENCE_STYLES[group], REFERENCE_LINEWIDTH)
        else:
            styles[name] = (WINDOW_COLORMAP(norm(blur[0])), "-", WINDOW_LINEWIDTH)
    return styles


def add_window_colorbar(fig, ax, batch, **kwargs):
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=window_norm(batch), cmap=WINDOW_COLORMAP), ax=ax, **kwargs)
    colorbar.set_label("Blur window start epoch", color=MUTED_COLOR)
    colorbar.ax.tick_params(colors=MUTED_COLOR)
    colorbar.outline.set_visible(False)
    return colorbar
