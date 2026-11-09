"""Test every experiment's network from epoch CHECKPOINT_EPOCH on sharp and on blurred test images, and print the mean and SD
over seeds. Every experiment has this checkpoint, so they are all compared after the same amount of training.

The blur is the one the experiments train with (training_procedure.py), and batch norm uses its running statistics, as in
every other test here. Reads results/<experiment>/seed_<seed>/checkpoints/ and writes results/eval_blurred.csv:
    python eval_blurred.py
"""
import csv
import os

import numpy as np
import torch
from torchvision.transforms import functional as TF

from networks import ResNet18
from training_procedure import BATCH_SIZE, BLUR_KERNEL_SIZE, BLUR_SIGMA, DATASET, evaluate
from util import get_loaders

EXPERIMENTS = [f"exp_{number:02d}" for number in range(1, 17)]
SEEDS = [42, 43, 44, 45, 46]
CHECKPOINT_EPOCH = 150
DATA_ROOT = "../datasets"
RESULTS_DIR = "results"
OUTPUT_NAME = "eval_blurred.csv"
SD_DDOF = 1  # sample SD, since the seeds are a sample of the possible runs


def checkpoint_path(experiment, seed):
    path = os.path.join(RESULTS_DIR, experiment, f"seed_{seed}", "checkpoints", f"epoch_{CHECKPOINT_EPOCH:03d}.pt")
    if not os.path.exists(path):
        raise SystemExit(f"No checkpoint at {path}.")
    return path


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, test_loader = get_loaders(DATASET, BATCH_SIZE, data_root=DATA_ROOT)
    sharp = [(x.to(device), y.to(device)) for x, y in test_loader]
    blurred = [(TF.gaussian_blur(x, BLUR_KERNEL_SIZE, BLUR_SIGMA), y) for x, y in sharp]

    model = ResNet18().to(device)
    rows = []
    for experiment in EXPERIMENTS:
        for seed in SEEDS:
            model.load_state_dict(torch.load(checkpoint_path(experiment, seed), map_location=device, weights_only=True))
            sharp_loss, sharp_accuracy = evaluate(model, sharp, device)
            blurred_loss, blurred_accuracy = evaluate(model, blurred, device)
            rows.append({"experiment": experiment, "seed": seed, "epoch": CHECKPOINT_EPOCH, "sharp_accuracy": sharp_accuracy,
                         "blurred_accuracy": blurred_accuracy, "sharp_loss": sharp_loss, "blurred_loss": blurred_loss})
        values = [row for row in rows if row["experiment"] == experiment]
        sharp_values, blurred_values = ([row[name] for row in values] for name in ("sharp_accuracy", "blurred_accuracy"))
        print(f"{experiment}, checkpoint at epoch {CHECKPOINT_EPOCH}: "
              f"sharp {np.mean(sharp_values):.2%} ± {np.std(sharp_values, ddof=SD_DDOF):.2%} SD, "
              f"blurred {np.mean(blurred_values):.2%} ± {np.std(blurred_values, ddof=SD_DDOF):.2%} SD")

    output_path = os.path.join(RESULTS_DIR, OUTPUT_NAME)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
