# Experiments

Every experiment trains a CIFAR-style ResNet-18 on CIFAR-10 with batch size 128, random 4-pixel translations and
horizontal flips, no weight decay and a constant learning rate (`training_procedure.py`). They differ only in the
optimizer, the blur window, the number of epochs and how often the network is saved.

- **Adam:** learning rate 0.005.
- **SGD:** learning rate 0.01, momentum 0.9.
- **Blur:** Gaussian, sigma 2.4 pixels, on training images only. Test accuracy, Fisher information and representation
  ranks are always measured on sharp images.
- **Blur window "a-b":** epochs count from 1, and the blur covers epochs a+1 to b. So 0-10 is the first 10 epochs, and
  5-15 is epochs 6-15.

| Experiment | Optimizer | Blur window | Blurred epochs | Epochs | Checkpoints | What it tests |
|---|---|---|---|---|---|---|
| `exp_01` | Adam | none | none | 200 | every 10 epochs | Baseline |
| `exp_02` | SGD | none | none | 200 | every 10 epochs | Baseline |
| `exp_03` | Adam | 0-10 | 1-10 | 200 | every 10 epochs | Blur at the start of training |
| `exp_04` | SGD | 0-10 | 1-10 | 200 | every 10 epochs | Blur at the start of training |
| `exp_05` | Adam | 5-15 | 6-15 | 150 | final only | 10-epoch window shifted by 5 |
| `exp_06` | SGD | 5-15 | 6-15 | 150 | final only | 10-epoch window shifted by 5 |
| `exp_07` | Adam | 10-20 | 11-20 | 150 | final only | 10-epoch window shifted by 10 |
| `exp_08` | SGD | 10-20 | 11-20 | 150 | final only | 10-epoch window shifted by 10 |
| `exp_09` | Adam | 15-25 | 16-25 | 150 | final only | 10-epoch window shifted by 15 |
| `exp_10` | SGD | 15-25 | 16-25 | 150 | final only | 10-epoch window shifted by 15 |
| `exp_11` | Adam | 20-30 | 21-30 | 150 | final only | 10-epoch window shifted by 20 |
| `exp_12` | SGD | 20-30 | 21-30 | 150 | final only | 10-epoch window shifted by 20 |
| `exp_13` | Adam | 0-15 | 1-15 | 150 | final only | Longer blur from the start |
| `exp_14` | SGD | 0-15 | 1-15 | 150 | final only | Longer blur from the start |
| `exp_15` | Adam | 0-20 | 1-20 | 150 | final only | Longer blur from the start |
| `exp_16` | SGD | 0-20 | 1-20 | 150 | final only | Longer blur from the start |

"Final only" saves `checkpoints/epoch_150.pt` and nothing else. `exp_01`-`exp_04` ran as the first HPC sweep (seeds
42-46); `exp_05`-`exp_16` are the `blur_window` sweep in `hpc_run.py`. That sweep ran `exp_11` and `exp_12` with a
20-25 window by mistake, so the `blur_window_20_30` sweep reran them.

`eval_blurred.py` tests each run's final saved network (epoch 200 for `exp_01`-`exp_04`, epoch 150 for the rest) on test
images with the training blur.
