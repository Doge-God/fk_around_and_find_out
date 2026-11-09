"""Baseline: ResNet-18 on CIFAR-10, trained with Adam, with no deficit applied.

The training loop is in training_procedure.py. Logs train/test metrics, per-layer Fisher information, weight norms and
representation ranks to TensorBoard, and saves the network every CHECKPOINT_EVERY_EPOCHS epochs.
Results go to results/exp_01/seed_<seed>/:
    python exp_01.py --seed 42
"""
import torch

from training_procedure import parse_args, train

EXPERIMENT = "exp_01"
EPOCHS = 200
LR = 0.005
CHECKPOINT_EVERY_EPOCHS = 10


def make_optimizer(params):
    return torch.optim.Adam(params, lr=LR)


if __name__ == "__main__":
    train(EXPERIMENT, parse_args(__doc__), make_optimizer, EPOCHS, CHECKPOINT_EVERY_EPOCHS)
