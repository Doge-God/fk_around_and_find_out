# Experiments

Every experiment trains a CIFAR-style ResNet-18 on CIFAR-10 for 150 epochs with batch size 128, random 4-pixel translations
and horizontal flips, no weight decay and a constant learning rate (`training_procedure.py`), and saves only the final
network (`checkpoints/epoch_150.pt`). Experiments are grouped into batches: a batch file holds the settings its experiments
share and a sweep that generates them.

| Batch | Optimizer | Seeds |
|---|---|---|
| `batch_01` | Adam, learning rate 0.005 | 42-46 |
| `batch_02` | SGD, learning rate 0.01, momentum 0.9 | 42-46 |

- **Blur:** Gaussian, sigma 2.4 pixels, on training images only.
- **Blur window "a-b":** epochs count from 1, and the blur covers epochs a+1 to b. So 0-10 is the first 10 epochs.
- **Readings:** test accuracy, Fisher information, ranks and SIGReg are measured on sharp images, and again on the same images
  with the training blur, throughout training in every experiment (blurred under their own TensorBoard tags).

Both batches run the same sweep (`sweep()` in each batch file): no blur, blur on every epoch, and a 10-epoch window starting at
epoch 0 and moved 5 epochs later each time, up to 140-150.

| Experiment | Blur window | Blurred epochs |
|---|---|---|
| `exp_01` | none | none |
| `exp_02` | 0-150 | 1-150 (all) |
| `exp_03` | 0-10 | 1-10 |
| `exp_04` | 5-15 | 6-15 |
| `exp_05` | 10-20 | 11-20 |
| `exp_06` | 15-25 | 16-25 |
| `exp_07` | 20-30 | 21-30 |
| `exp_08` | 25-35 | 26-35 |
| `exp_09` | 30-40 | 31-40 |
| `exp_10` | 35-45 | 36-45 |
| `exp_11` | 40-50 | 41-50 |
| `exp_12` | 45-55 | 46-55 |
| `exp_13` | 50-60 | 51-60 |
| `exp_14` | 55-65 | 56-65 |
| `exp_15` | 60-70 | 61-70 |
| `exp_16` | 65-75 | 66-75 |
| `exp_17` | 70-80 | 71-80 |
| `exp_18` | 75-85 | 76-85 |
| `exp_19` | 80-90 | 81-90 |
| `exp_20` | 85-95 | 86-95 |
| `exp_21` | 90-100 | 91-100 |
| `exp_22` | 95-105 | 96-105 |
| `exp_23` | 100-110 | 101-110 |
| `exp_24` | 105-115 | 106-115 |
| `exp_25` | 110-120 | 111-120 |
| `exp_26` | 115-125 | 116-125 |
| `exp_27` | 120-130 | 121-130 |
| `exp_28` | 125-135 | 126-135 |
| `exp_29` | 130-140 | 131-140 |
| `exp_30` | 135-145 | 136-145 |
| `exp_31` | 140-150 | 141-150 |

Results: `results/<batch>/experiments/<experiment>/seed_<seed>/`.

## Earlier experiments
Before the batches, each experiment was its own script, `exp_01.py`-`exp_30.py`, with the same training setup and seeds 42-46.
`exp_01`-`exp_16` are in git history (commit `86651a8`); `exp_17`-`exp_30` were never committed, but differed only in the
settings below. Their results were deleted apart from the summary graphs in `results/archive/`. "Final only" saved
`checkpoints/epoch_150.pt` only.

| Experiment | Optimizer | Blur window | Epochs | Checkpoints | What it tested |
|---|---|---|---|---|---|
| `exp_01` | Adam | none | 200 | every 10 epochs | Baseline |
| `exp_02` | SGD | none | 200 | every 10 epochs | Baseline |
| `exp_03` | Adam | 0-10 | 200 | every 10 epochs | Blur at the start of training |
| `exp_04` | SGD | 0-10 | 200 | every 10 epochs | Blur at the start of training |
| `exp_05` | Adam | 5-15 | 150 | final only | 10-epoch window shifted by 5 |
| `exp_06` | SGD | 5-15 | 150 | final only | 10-epoch window shifted by 5 |
| `exp_07` | Adam | 10-20 | 150 | final only | 10-epoch window shifted by 10 |
| `exp_08` | SGD | 10-20 | 150 | final only | 10-epoch window shifted by 10 |
| `exp_09` | Adam | 15-25 | 150 | final only | 10-epoch window shifted by 15 |
| `exp_10` | SGD | 15-25 | 150 | final only | 10-epoch window shifted by 15 |
| `exp_11` | Adam | 20-30 | 150 | final only | 10-epoch window shifted by 20 |
| `exp_12` | SGD | 20-30 | 150 | final only | 10-epoch window shifted by 20 |
| `exp_13` | Adam | 0-15 | 150 | final only | Longer blur from the start |
| `exp_14` | SGD | 0-15 | 150 | final only | Longer blur from the start |
| `exp_15` | Adam | 0-20 | 150 | final only | Longer blur from the start |
| `exp_16` | SGD | 0-20 | 150 | final only | Longer blur from the start |
| `exp_17` | Adam | 0-150 | 150 | final only | Blur on every epoch |
| `exp_18` | SGD | 0-150 | 150 | final only | Blur on every epoch |
| `exp_19` | Adam | 140-150 | 150 | final only | Blur in the last 10 epochs |
| `exp_20` | SGD | 140-150 | 150 | final only | Blur in the last 10 epochs |
| `exp_21` | Adam | 135-145 | 150 | final only | Last-10 window shifted back by 5 |
| `exp_22` | SGD | 135-145 | 150 | final only | Last-10 window shifted back by 5 |
| `exp_23` | Adam | 130-140 | 150 | final only | Last-10 window shifted back by 10 |
| `exp_24` | SGD | 130-140 | 150 | final only | Last-10 window shifted back by 10 |
| `exp_25` | Adam | 125-135 | 150 | final only | Last-10 window shifted back by 15 |
| `exp_26` | SGD | 125-135 | 150 | final only | Last-10 window shifted back by 15 |
| `exp_27` | Adam | 120-130 | 150 | final only | Last-10 window shifted back by 20 |
| `exp_28` | SGD | 120-130 | 150 | final only | Last-10 window shifted back by 20 |
| `exp_29` | Adam | 70-80 | 150 | final only | Blur in the middle of training |
| `exp_30` | SGD | 70-80 | 150 | final only | Blur in the middle of training |
