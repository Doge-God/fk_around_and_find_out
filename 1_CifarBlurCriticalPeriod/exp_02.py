"""Baseline: ResNet-18 on CIFAR-10, trained with SGD, with no deficit applied.

The training loop is in training_procedure.py. Logs train/test metrics, per-layer Fisher information, weight norms and
representation ranks to TensorBoard, and saves the network every CHECKPOINT_EVERY_EPOCHS epochs.
Results go to results/exp_02/seed_<seed>/:
    python exp_02.py --seed 42
"""
import torch

from training_procedure import parse_args, train

EXPERIMENT = "exp_02"
EPOCHS = 200
LR = 0.01
MOMENTUM = 0.9
CHECKPOINT_EVERY_EPOCHS = 10


def make_optimizer(params):
    return torch.optim.SGD(params, lr=LR, momentum=MOMENTUM)


if __name__ == "__main__":
    train(EXPERIMENT, parse_args(__doc__), make_optimizer, EPOCHS, CHECKPOINT_EVERY_EPOCHS)
