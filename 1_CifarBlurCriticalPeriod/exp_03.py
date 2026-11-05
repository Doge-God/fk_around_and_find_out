"""Deficit: ResNet-18 on CIFAR-10, trained with Adam, with Gaussian blur on training images for the first epochs.

Logs train/test metrics, per-layer Fisher information, weight norms and representation ranks to TensorBoard,
and saves the network every CHECKPOINT_EVERY_EPOCHS epochs. Results go to results/exp_03/seed_<seed>/:
    python exp_03.py --seed 42
"""
import argparse
import math
import os
import shutil

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.transforms import functional as TF

from networks import ResNet18
from util import get_loaders, log_fisher, log_representation, log_weight_norm

RESULTS_DIR = os.path.join("results", "exp_03")
DEFAULT_SEED = 42
DATASET = "cifar10"
EPOCHS = 200
BATCH_SIZE = 128
LR = 0.005
FISHER_EVERY_STEPS = 100
FISHER_SAMPLES = 1000
WEIGHT_NORM_EVERY_STEPS = 100
CHECKPOINT_EVERY_EPOCHS = 10
# Random translation of up to 4 pixels and random horizontal flip.
TRAIN_AUGMENTATION = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
BLUR_EPOCHS = 10
BLUR_SIGMA = 2.4  # in pixels
BLUR_KERNEL_SIZE = 2 * math.ceil(3 * BLUR_SIGMA) + 1  # reaches 3 sigma either side


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += len(y)
    return loss_sum / total, correct / total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fisher_set, train_loader, test_loader = get_loaders(DATASET, BATCH_SIZE, TRAIN_AUGMENTATION)
    fisher_indices = torch.randperm(len(fisher_set))[:FISHER_SAMPLES].tolist()
    fisher_x = next(iter(DataLoader(Subset(fisher_set, fisher_indices), batch_size=FISHER_SAMPLES)))[0].to(device)

    # The test loader doesn't shuffle, so this is the same batch at every epoch.
    representation_x = next(iter(test_loader))[0].to(device)

    model = ResNet18().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    results_dir = os.path.join(RESULTS_DIR, f"seed_{args.seed}")
    log_dir = os.path.join(results_dir, "tensorboard")
    shutil.rmtree(log_dir, ignore_errors=True)  # a repeated run replaces the previous one
    writer = SummaryWriter(log_dir)
    checkpoint_dir = os.path.join(results_dir, "checkpoints")
    shutil.rmtree(checkpoint_dir, ignore_errors=True)
    os.makedirs(checkpoint_dir)

    step = 0
    log_fisher(writer, model, fisher_x, step)
    log_weight_norm(writer, model, step)
    log_representation(writer, model, representation_x, step)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if epoch <= BLUR_EPOCHS:
                x = TF.gaussian_blur(x, BLUR_KERNEL_SIZE, BLUR_SIGMA)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += len(y)

            if step % FISHER_EVERY_STEPS == 0:
                log_fisher(writer, model, fisher_x, step)
            if step % WEIGHT_NORM_EVERY_STEPS == 0:
                log_weight_norm(writer, model, step)

        train_loss, train_acc = loss_sum / total, correct / total
        test_loss, test_acc = evaluate(model, test_loader, device)
        writer.add_scalar("train/loss", train_loss, step)
        writer.add_scalar("train/accuracy", train_acc, step)
        writer.add_scalar("test/loss", test_loss, step)
        writer.add_scalar("test/accuracy", test_acc, step)
        log_representation(writer, model, representation_x, step)
        if epoch % CHECKPOINT_EVERY_EPOCHS == 0:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pt"))
        print(f"epoch {epoch:2d}/{EPOCHS}  train loss {train_loss:.4f} acc {train_acc:.4f}  "
              f"test loss {test_loss:.4f} acc {test_acc:.4f}")

    writer.close()
    print(f"logs written to {log_dir}")


if __name__ == "__main__":
    main()
