"""Test every final network of a batch on test images blurred with each sigma in BLUR_SIGMAS (0-3 pixels):
- accuracy, on the full test set,
- representation ranks and SIGReg (raw and standardised), on the penultimate features of the first test batch, the batch the
  training logs use, so sigma 0 matches the TensorBoard values at the last epoch.
Batch norm uses its running statistics, as in training's test readings.

Saves the results to results/<batch>/eval_blur.csv. Runs already there are reused, so only new runs are evaluated: delete the
file to evaluate them all again. Best on a GPU. The graph scripts call this themselves; to fill the file on its own:
    python -m analysis.eval_blur batch_01
    python -m analysis.eval_blur batch_01 --experiment exp_03 --experiment exp_04   # only some experiments
On the HPC (hpc_eval_blur.py) each job evaluates one experiment, reading the checkpoints from a NAS dataset laid out like
results/ and writing its CSV to the job's output, which is then merged into results/<batch>/eval_blur.csv:
    python -m analysis.eval_blur batch_01 --experiment exp_03 --data-root /workspace/datasets/fzho027_cifar10 \
        --checkpoints-root /workspace/datasets/fzho027_cifar_blur_checkpoints --results-root /workspace/output
"""
import argparse
import csv
import math
import os
import time

import torch
from torchvision.transforms import functional as TF

from analysis.common import BLUR_SIGMAS, RESULTS_DIR, batch_dir, checkpoint_path, load_batch
from networks import ResNet18
from training_procedure import BATCH_SIZE, DATASET, evaluate
from util import compute_ranks, compute_sigreg, get_loaders

DATA_ROOT = "../datasets"
OUTPUT_NAME = "eval_blur.csv"
METRICS = ["accuracy", "effective_rank", "stable_rank_delta", "sigreg", "sigreg_standardized"]


def blur(x, sigma):
    """The training blur's reach, 3 sigma either side. Sigma 0 leaves the images sharp."""
    if sigma == 0:
        return x
    return TF.gaussian_blur(x, 2 * math.ceil(3 * sigma) + 1, sigma)


class BlurredLoader:
    """Iterates over (x, y) batches already on the device, blurring each x by sigma."""

    def __init__(self, batches, sigma):
        self.batches, self.sigma = batches, sigma

    def __iter__(self):
        return ((blur(x, self.sigma), y) for x, y in self.batches)


def format_duration(seconds):
    hours, rest = divmod(int(seconds), 3600)
    return f"{hours}h{rest // 60:02d}m" if hours else f"{rest // 60}m{rest % 60:02d}s"


def run_eval(batch, runs, checkpoints_root=RESULTS_DIR, data_root=DATA_ROOT):
    """Rows for the CSV, one per (experiment, seed) in runs and sigma. Checkpoints are read from under checkpoints_root."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{batch.BATCH}: evaluating {len(runs)} epoch {batch.EPOCHS} checkpoints at {len(BLUR_SIGMAS)} blur sigmas on "
          f"{device}", flush=True)
    _, _, test_loader = get_loaders(DATASET, BATCH_SIZE, data_root=data_root)
    batches = [(x.to(device), y.to(device)) for x, y in test_loader]  # the test loader doesn't shuffle
    accuracy_images = sum(len(y) for _, y in batches)
    representation_x = batches[0][0]

    model = ResNet18().to(device)
    rows = []
    start = time.time()
    for i, (experiment, seed) in enumerate(runs, 1):
        model.load_state_dict(torch.load(checkpoint_path(batch, experiment, seed, checkpoints_root), map_location=device, weights_only=True))
        for sigma in BLUR_SIGMAS:
            _, accuracy = evaluate(model, BlurredLoader(batches, sigma), device)  # sets eval mode
            with torch.no_grad():
                features = model.features(blur(representation_x, sigma))
            rows.append({"experiment": experiment, "seed": seed, "epoch": batch.EPOCHS, "sigma": sigma,
                         "accuracy_images": accuracy_images, "representation_images": len(representation_x),
                         "accuracy": accuracy, **compute_ranks(features), **compute_sigreg(features)})
        elapsed = time.time() - start
        remaining = elapsed / i * (len(runs) - i)
        print(f"[{i}/{len(runs)}] {experiment} seed {seed} done, accuracy {rows[-len(BLUR_SIGMAS)]['accuracy']:.3f} sharp, "
              f"elapsed {format_duration(elapsed)}, about {format_duration(remaining)} left", flush=True)
    return rows


def output_path(batch, results_root=RESULTS_DIR):
    return os.path.join(batch_dir(batch, results_root), OUTPUT_NAME)


def read_rows(batch, path):
    """The CSV's rows for the batch's final epoch, or none if there is no file."""
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [row for row in csv.DictReader(f) if int(row["epoch"]) == batch.EPOCHS]


def write_rows(rows, path):
    rows = sorted(rows, key=lambda row: (row["experiment"], int(row["seed"]), float(row["sigma"])))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp"  # so an interrupted write never leaves a half-written CSV
    with open(temporary, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)
    print(f"saved {path}", flush=True)


def runs_in(rows):
    return {(row["experiment"], int(row["seed"])) for row in rows}


def image_counts(rows, path):
    """{"accuracy_images": int, "representation_images": int}, the same for every row."""
    counts = {name: {int(row[name]) for row in rows} for name in ("accuracy_images", "representation_images")}
    if any(len(values) > 1 for values in counts.values()):
        raise SystemExit(f"{path} mixes runs on different numbers of test images: delete it to evaluate again.")
    return {name: next(iter(values)) for name, values in counts.items() if values}


def merge_rows(batch, rows, results_root=RESULTS_DIR):
    """Add rows (from another CSV, e.g. an HPC job's) for runs not yet in <results_root>/<batch>/eval_blur.csv. Returns
    the runs added."""
    path = output_path(batch, results_root)
    existing = read_rows(batch, path)
    added = runs_in(rows) - runs_in(existing)
    if added:
        merged = existing + [row for row in rows if (row["experiment"], int(row["seed"])) in added]
        image_counts(merged, path)  # refuse to mix runs on different test sets
        write_rows(merged, path)
    return added


def load_eval(batch, experiments=None, checkpoints_root=RESULTS_DIR, results_root=RESULTS_DIR, data_root=DATA_ROOT):
    """Evaluate any runs of experiments (default: all of the batch's) missing from <results_root>/<batch>/eval_blur.csv.
    Returns (experiment, seed) -> metric -> {sigma: value} for every run in the CSV, and
    {"accuracy_images": int, "representation_images": int}."""
    path = output_path(batch, results_root)
    rows = read_rows(batch, path)
    done = runs_in(rows)
    names = batch.EXPERIMENTS if experiments is None else experiments
    missing = [(name, seed) for name in names for seed in batch.SEEDS if (name, seed) not in done]
    if missing:
        rows += run_eval(batch, missing, checkpoints_root, data_root)
        write_rows(rows, path)
    results = {}
    for row in rows:
        run = results.setdefault((row["experiment"], int(row["seed"])), {metric: {} for metric in METRICS})
        for metric in METRICS:
            run[metric][float(row["sigma"])] = float(row[metric])
    return results, image_counts(rows, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("batch", help="batch module, e.g. batch_01")
    parser.add_argument("--experiment", action="append", help="only this experiment; repeat for more (default: all)")
    parser.add_argument("--data-root", default=DATA_ROOT, help="folder holding the dataset")
    parser.add_argument("--checkpoints-root", default=RESULTS_DIR,
                        help="checkpoints are read from <checkpoints-root>/<batch>/experiments/<experiment>/seed_<seed>/")
    parser.add_argument("--results-root", default=RESULTS_DIR, help=f"the CSV goes to <results-root>/<batch>/{OUTPUT_NAME}")
    args = parser.parse_args()
    batch = load_batch(args.batch)
    unknown = [name for name in args.experiment or [] if name not in batch.EXPERIMENTS]
    if unknown:
        raise SystemExit(f"{args.batch} has no experiment {', '.join(unknown)}.")
    load_eval(batch, args.experiment, args.checkpoints_root, args.results_root, args.data_root)


if __name__ == "__main__":
    main()
