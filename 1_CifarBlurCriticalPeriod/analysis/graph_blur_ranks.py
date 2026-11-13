"""Representation measures of a batch's final networks on test images blurred with sigma 0-3, in the colours of
graph_blur_accuracy.py:
    python -m analysis.graph_blur_ranks batch_01

Four panels against test image blur sigma, mean and sample SD over the batch's seeds: stable rank delta, effective rank, SIGReg
and SIGReg on standardised features. All are of the penultimate features of the first test batch (no centering for the ranks),
as logged during training, so sigma 0 matches the TensorBoard values at the last epoch.

Reads results/<batch>/eval_blur.csv, evaluating any runs missing from it (see eval_blur.py), and writes
results/<batch>/graph_blur_ranks.png.
"""
import argparse
import os

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np

from analysis.common import (BLUR_SIGMAS, MUTED_COLOR, REFERENCE_COLOR, REFERENCE_LINEWIDTH, REFERENCE_STYLES, SD_DDOF, TEXT_COLOR,
                             add_window_colorbar, batch_dir, blur_group, experiment_styles, load_batch, style_axes)
from analysis.eval_blur import load_eval
from training_procedure import BLUR_SIGMA

METRICS = {"stable_rank_delta": "Stable rank delta", "effective_rank": "Effective rank", "sigreg": "SIGReg",
           "sigreg_standardized": "SIGReg, standardised features"}  # in panel order, left to right and top to bottom
SD_ALPHA = 0.02  # 31 overlapping bands add up, so each is very light
FIGURE_SIZE = (16, 11)
OUTPUT_NAME = "graph_blur_ranks.png"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("batch", help="batch module, e.g. batch_01")
    batch = load_batch(parser.parse_args().batch)
    results, images = load_eval(batch)
    styles = experiment_styles(batch)
    # Windows first and the references on top, so the black lines stay visible.
    order = sorted(batch.EXPERIMENTS, key=lambda name: blur_group(batch, batch.EXPERIMENTS[name]) != "window")

    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE, layout="constrained")
    for ax, (metric, label) in zip(axes.flat, METRICS.items()):
        style_axes(ax, (BLUR_SIGMAS[0], BLUR_SIGMAS[-1]))
        for name in order:
            values = np.array([[results[name, seed][metric][sigma] for sigma in BLUR_SIGMAS] for seed in batch.SEEDS])
            color, linestyle, linewidth = styles[name]
            mean, sd = values.mean(axis=0), values.std(axis=0, ddof=SD_DDOF)
            ax.fill_between(BLUR_SIGMAS, mean - sd, mean + sd, color=color, alpha=SD_ALPHA, linewidth=0, zorder=1)
            ax.plot(BLUR_SIGMAS, mean, color=color, linestyle=linestyle, linewidth=linewidth, zorder=2)
        ax.axvline(BLUR_SIGMA, color=MUTED_COLOR, linewidth=1, linestyle="--", zorder=1)
        ax.set_ylim(bottom=0)
        ax.set_title(label, color=TEXT_COLOR, loc="left")
        ax.set_xlabel("Test image blur sigma (pixels)", color=MUTED_COLOR)
        ax.set_ylabel(label, color=MUTED_COLOR)

    handles = [Line2D([], [], color=REFERENCE_COLOR, linestyle=REFERENCE_STYLES["none"], linewidth=REFERENCE_LINEWIDTH, label="No blur"),
               Line2D([], [], color=REFERENCE_COLOR, linestyle=REFERENCE_STYLES["full"], linewidth=REFERENCE_LINEWIDTH,
                      label="Blur on every epoch"),
               Line2D([], [], color=MUTED_COLOR, linewidth=1, linestyle="--", label=f"Training blur sigma ({BLUR_SIGMA})"),
               Patch(color=MUTED_COLOR, alpha=8 * SD_ALPHA, linewidth=0, label="± 1 SD over seeds")]
    fig.legend(handles=handles, loc="outside right upper", frameon=False, labelcolor=TEXT_COLOR)
    add_window_colorbar(fig, axes, batch, location="right", shrink=0.5, aspect=30)
    fig.suptitle(f"{batch.BATCH} ({batch.OPTIMIZER}): representation of epoch {batch.EPOCHS} networks on blurred test images\n"
                 f"Penultimate features of the first {images['representation_images']} test images, mean over seeds "
                 f"{batch.SEEDS[0]}-{batch.SEEDS[-1]}", color=TEXT_COLOR, x=0.01, ha="left")
    output_path = os.path.join(batch_dir(batch), OUTPUT_NAME)
    fig.savefig(output_path, dpi=150)
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
