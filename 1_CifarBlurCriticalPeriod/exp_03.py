"""Deficit: ResNet-18 on CIFAR-10, trained with Adam, with Gaussian blur on training images for the first 10 epochs.

The training loop is in training_procedure.py. Logs train/test metrics, per-layer Fisher information, weight norms and
representation ranks to TensorBoard, and saves the network every CHECKPOINT_EVERY_EPOCHS epochs.
Results go to results/exp_03/seed_<seed>/:
    python exp_03.py --seed 42
"""
import torch

from training_procedure import parse_args, train

EXPERIMENT = "exp_03"
EPOCHS = 200
LR = 0.005
CHECKPOINT_EVERY_EPOCHS = 10
BLUR_START_EPOCH = 0  # blurs epochs BLUR_START_EPOCH + 1 to BLUR_END_EPOCH, counting from 1
BLUR_END_EPOCH = 10


def make_optimizer(params):
    return torch.optim.Adam(params, lr=LR)


if __name__ == "__main__":
    train(EXPERIMENT, parse_args(__doc__), make_optimizer, EPOCHS, CHECKPOINT_EVERY_EPOCHS,
          blur_epochs=(BLUR_START_EPOCH, BLUR_END_EPOCH))
