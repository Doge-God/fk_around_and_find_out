"""Batch 01: ResNet-18 on CIFAR-10, trained with Adam for 150 epochs, sweeping where a 10-epoch blur window falls.

Experiments (sweep() below): no blur, blur on every epoch, and a WINDOW_LENGTH-epoch blur window starting at epoch 0 and moved
WINDOW_STEP epochs later each time, up to the window ending at the last epoch. The training loop is in training_procedure.py.
Results go to results/batch_01/experiments/<experiment>/seed_<seed>/:
    python batch_01.py                                # list the experiments
    python batch_01.py --experiment exp_03 --seed 42
"""
import torch

from training_procedure import parse_args, print_sweep, train

BATCH = "batch_01"
OPTIMIZER = "Adam"
LR = 0.005
EPOCHS = 150
CHECKPOINT_EVERY_EPOCHS = EPOCHS  # only the final network
WINDOW_LENGTH = 10
WINDOW_STEP = 5
SEEDS = [42, 43, 44, 45, 46]  # the seeds hpc_run.py and the analysis scripts use


def make_optimizer(params):
    return torch.optim.Adam(params, lr=LR)


def sweep():
    """Experiment name -> blur window (start, end), blurring epochs start + 1 to end, or None for no blur."""
    windows = [(start, start + WINDOW_LENGTH) for start in range(0, EPOCHS - WINDOW_LENGTH + 1, WINDOW_STEP)]
    return {f"exp_{number:02d}": blur for number, blur in enumerate([None, (0, EPOCHS), *windows], start=1)}


EXPERIMENTS = sweep()


if __name__ == "__main__":
    args = parse_args(__doc__, EXPERIMENTS)
    if args.experiment is None:
        print_sweep(BATCH, EXPERIMENTS)
    else:
        train(BATCH, args.experiment, args, make_optimizer, EPOCHS, CHECKPOINT_EVERY_EPOCHS,
              blur_epochs=EXPERIMENTS[args.experiment])
