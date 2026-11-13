"""Show one CIFAR-10 test image at each blur sigma from 0 to MAX_SIGMA in steps of SIGMA_STEP, blurred the way the
experiments blur (kernel reaching 3 sigma either side, see training_procedure.py). Run it from the experiment folder with:
    python -m analysis.graph_blur_examples

Writes results/graph_blur_examples.png.
"""
import os

import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from analysis.common import RESULTS_DIR, TEXT_COLOR
from analysis.eval_blur import blur
from training_procedure import BLUR_SIGMA

SAMPLE_INDEX = 0  # test set image to show
MAX_SIGMA = 3.0  # in pixels
SIGMA_STEP = 0.5
UPSCALE = 8  # nearest-neighbour upscaling, so matplotlib doesn't smooth the 32x32 pixels itself
OUTPUT_NAME = "graph_blur_examples.png"


def main():
    test_set = datasets.CIFAR10(root="../datasets", train=False, download=True, transform=transforms.ToTensor())
    x, label = test_set[SAMPLE_INDEX]
    sigmas = [step * SIGMA_STEP for step in range(round(MAX_SIGMA / SIGMA_STEP) + 1)]

    fig, axes = plt.subplots(1, len(sigmas), figsize=(1.8 * len(sigmas), 2.3))
    for ax, sigma in zip(axes, sigmas):
        image = blur(x, sigma).clamp(0, 1).permute(1, 2, 0).repeat_interleave(UPSCALE, 0).repeat_interleave(UPSCALE, 1)
        ax.imshow(image.numpy(), interpolation="nearest")
        ax.set_title(f"σ = {sigma:g}", color=TEXT_COLOR, fontsize=10)
        ax.set_axis_off()
    fig.suptitle(f"CIFAR-10 test image {SAMPLE_INDEX} ({test_set.classes[label]}), Gaussian blur by sigma in pixels (training uses {BLUR_SIGMA:g})",
                 color=TEXT_COLOR, fontsize=11)
    fig.tight_layout()
    output_path = os.path.join(RESULTS_DIR, OUTPUT_NAME)
    fig.savefig(output_path, dpi=150)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
