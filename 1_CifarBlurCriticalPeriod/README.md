# Per-layer Fisher information in a ResNet-18 on CIFAR-10

**Question:** how do each layer's Fisher information, the weight norms and the rank of the representation change
while a ResNet-18 trains, and does a blur deficit in the first epochs change them?

## Setup
- **Model:** CIFAR-style ResNet-18 (3x3 first convolution, no max pool).
- **Training:** batch size 128, no weight decay, constant learning rate. Adam uses learning rate 0.005; SGD uses
  learning rate 0.01 with momentum 0.9. The blur is Gaussian, sigma 2.4 pixels, on training images only. Epoch numbers
  count from 1, and a blur window "a-b" covers epochs a+1 to b.

| | Adam | SGD | Blur window | Epochs | Checkpoints |
|---|---|---|---|---|---|
| Baseline | `exp_01` | `exp_02` | none | 200 | every 10 epochs |
| Blur at the start | `exp_03` | `exp_04` | 0-10 | 200 | every 10 epochs |
| Window shifted by 5 | `exp_05` | `exp_06` | 5-15 | 150 | final only |
| Shifted by 10 | `exp_07` | `exp_08` | 10-20 | 150 | final only |
| Shifted by 15 | `exp_09` | `exp_10` | 15-25 | 150 | final only |
| Shifted by 20 | `exp_11` | `exp_12` | 20-30 | 150 | final only |
| Longer, from the start | `exp_13` | `exp_14` | 0-15 | 150 | final only |
| Longer, from the start | `exp_15` | `exp_16` | 0-20 | 150 | final only |

- **Augmentation:** random translations of up to 4 pixels and random horizontal flips, on training images only.
- **Fisher information:** trace of the true Fisher per layer, where the 10 "layers" are the first convolution,
  the 8 residual blocks and the classifier, with labels sampled from the model's predictions.
  It is measured every 100 steps on 1000 fixed, unaugmented training images.
- **Weight norm:** L2 norm of the parameters in total and per layer (batch norm parameters included), every 100 steps.
- **Representation:** singular values of the penultimate features (512 pooled values, before the classifier) of the
  first test batch (512 images, the same ones every time, no centering), at step 0 and after each epoch. With the
  singular values normalised by their sum into p:
  - effective rank = exp(-sum p_i log p_i).
  - stable rank delta = the smallest k whose top-k values of p cover at least 99% of the total.

The logs hold Fisher totals per layer. `graph_training.py` plots them divided by each layer's parameter count
(`FISHER_PER_PARAM`), since the layers range from 1,856 parameters (first convolution) to 4.7 million (last block).
The Fisher panel leaves out the first convolution and the classifier (`FISHER_EXCLUDED_LAYERS`), whose much higher
values per parameter would flatten the residual blocks.
They use batch norm's running statistics. These are still at their initial values at
step 0, so that point is less comparable with the rest of the run.

Train accuracy is measured on augmented images during training, so it can sit below test accuracy.

The blur applies to training images only. Test accuracy, Fisher information and representation ranks are always
measured on sharp images, so during the blur they describe images the model is not training on.

## Files
- `experiments.md`: what each experiment does, in one table.
- `exp_01.py` ... `exp_16.py`: one per row of the table above, holding that experiment's settings (optimizer, epochs,
  checkpoints, blur window).
- `training_procedure.py`: the training loop they all call, and the settings they share (batch size, augmentation, blur
  sigma, how often Fisher and weight norms are logged).
- `networks.py`: the MLP and ResNet-18.
- `util.py`: data loading, Fisher information, weight norm and representation ranks.
- `graph_training.py`: plots a TensorBoard run.
- `graph_fisher.py`: plots Fisher information for all 16 experiments (rows) and 5 seeds (columns), with blur windows shaded and the y-axis shared by the Adam rows and by the SGD rows.
- `graph_report.py`: one row per experiment, with seed 42's Fisher information, then over the 5 seeds (mean, SD shading): train/test accuracy, epoch 150 networks on the full test set at blur sigma 0-3, representation ranks, and total and per-layer weight norm.
- `graph_blur_windows.py`: for one optimizer (`OPTIMIZER`), the no-blur baseline against the 10-epoch blur windows: seed 42's Fisher information, and epoch 150 accuracy across blur sigmas with a table of accuracy, area under the curve and ranks over the 5 seeds.
- `eval_blurred.py`: tests each run's epoch 150 network on the full sharp and training-blurred test sets, and prints the mean and SD over seeds.
- `hpc_run.py`: runs a sweep of experiments and seeds on the HPC, collects the results and plots them.
- `Dockerfile`: the training image `hpc_run.py` builds and submits.

## Run
```
python exp_01.py --seed 42                     # logs to results/exp_01/seed_42/tensorboard/, replacing the previous run with that seed
tensorboard --logdir results/exp_01           # one run per seed; see the "fisher information", "weight norm" and "representation" sections
python graph_training.py exp_01 --seed 42     # -> results/exp_01/seed_42/graph_training.png
python graph_fisher.py                        # -> results/graph_fisher.png, all experiments and seeds,
                                              #    and graph_fisher_epoch_140_150.png, zoomed to epochs 140-150
python eval_blurred.py                        # -> results/eval_blurred.csv, epoch 150 networks on sharp and blurred test images
python graph_report.py                        # -> results/graph_report.png, the summary of all experiments. Saves the blur
                                              #    evaluation to results/graph_report_blur_eval.csv; delete it to evaluate again
python graph_blur_windows.py                  # -> results/graph_blur_windows_<optimizer>.png, for OPTIMIZER
```
`--seed` defaults to 42 in both scripts.
Each run also saves the network's `state_dict` to `results/exp_01/seed_42/checkpoints/epoch_XXX.pt` (about 45 MB each,
git-ignored): every 10 epochs for `exp_01`-`exp_04`, and only after the last epoch (`epoch_150.pt`) for `exp_05`-`exp_16`.
A repeated run with the same seed replaces them.
The other experiments run the same way: replace `exp_01` with any of `exp_02`-`exp_16` in the commands above.

## Run on the HPC
```
python hpc_run.py            # exp_11 and exp_12, seeds 42-46, 10 runs on the HPC
python hpc_run.py --local    # this PC also takes one run at a time, cancelling a queued HPC job if needed
```
Needs `~/hpc_outputs` mounted, and either a valid `hpc-client login` or `HPC_PASSWORD` set. CIFAR-10 must be on the NAS
as `fzho027_cifar10/cifar-10-batches-py`.

At most 30 jobs wait in the HPC queue at once (`MAX_QUEUED`); the rest are submitted as queued jobs start.
The dashboard shows each run's status and refreshes from the scheduler every 30 seconds. Finished runs are copied to
`results/<experiment>/seed_<seed>/`, with the job's log as `run.log`, and plotted to `graph_training.png`. A run whose
folder already exists is skipped, so nothing is overwritten.

`EXPERIMENTS` and `SEEDS` in `hpc_run.py` set the sweep, and `SWEEP` names its state file,
`results/hpc_run_state_<SWEEP>.json` (the current sweep is `blur_window_20_30`, which reruns `exp_11` and `exp_12` after
`blur_window`, `exp_05`-`exp_16`, ran them with a 20-25 window; the first sweep, `exp_01`-`exp_04`, used `results/hpc_run_state.json`). Events and job ids from every sweep go to `results/hpc_run.log`. Ctrl-C only stops the
dashboard: the jobs keep running, and running the script again picks them back up. The sweep stays on the image it
started with, so editing the code doesn't change it. To start a new sweep with the current code, delete the state file, or change `SWEEP` to keep the old one.
`--retry-failed` runs failed and cancelled runs again.

## Findings
### Epoch 150 networks on blurred test images
Five seeds per experiment, mean ± sample SD, from `results/eval_blurred.csv`. The blur is the training one. Every experiment
uses its epoch 150 checkpoint, including `exp_01`-`exp_04`, which trained on to epoch 200. `results/graph_report.png` shows the
same networks across blur sigmas 0-3.

| Blur window | Adam | Blurred test | Sharp test | SGD | Blurred test | Sharp test |
|---|---|---|---|---|---|---|
| none | `exp_01` | 17.3% ± 2.6% | 92.2% ± 0.4% | `exp_02` | 16.4% ± 2.4% | 92.6% ± 0.2% |
| 0-10 | `exp_03` | 21.9% ± 1.6% | 89.9% ± 0.7% | `exp_04` | 23.8% ± 0.7% | 91.1% ± 0.3% |
| 5-15 | `exp_05` | 16.5% ± 1.2% | 92.0% ± 0.2% | `exp_06` | 20.6% ± 2.2% | 92.3% ± 0.2% |
| 10-20 | `exp_07` | 17.4% ± 1.4% | 92.3% ± 0.5% | `exp_08` | 19.6% ± 1.7% | 92.8% ± 0.2% |
| 15-25 | `exp_09` | 18.2% ± 0.6% | 92.2% ± 0.4% | `exp_10` | 19.5% ± 1.8% | 92.8% ± 0.2% |
| 20-30 | `exp_11` | 17.7% ± 0.5% | 92.5% ± 0.4% | `exp_12` | 18.6% ± 2.0% | 92.7% ± 0.3% |
| 0-15 | `exp_13` | 23.9% ± 1.7% | 88.9% ± 0.5% | `exp_14` | 27.5% ± 1.7% | 90.2% ± 0.2% |
| 0-20 | `exp_15` | 24.6% ± 2.8% | 88.1% ± 0.4% | `exp_16` | 30.4% ± 1.8% | 89.6% ± 0.6% |

- **Blurred images are mostly forgotten.** Right after a 0-10 blur, the networks score about 72-74% on blurred test images
  (`exp_03` and `exp_04`, seeds 42-43, epoch 10), and by epoch 150 22-24%.
- **Blur from the start keeps the most, and costs the most on sharp images.** Windows starting at epoch 0 end highest on
  blurred images and lowest on sharp ones. Windows starting later end close to the no-blur baselines on both.

### Blur at the start, first runs
One 200-epoch run per configuration, all with seed 0. Graphs: `results/exp_01/seed_0/graph_training.png`, and the same path
for `exp_02`, `exp_03` and `exp_04`. They show epochs up to 150 (`X_RANGE` in `graph_training.py`); the table is from
the full logs. `exp_03` and `exp_04` use blur sigma 2.4. `exp_01` and `exp_02` are older runs, from before checkpoints
were added and before the last edit to `util.py`, and have no saved networks.

| | `exp_01` Adam | `exp_02` SGD | `exp_03` Adam + blur | `exp_04` SGD + blur |
|---|---|---|---|---|
| Test accuracy, mean of last 20 epochs | 0.923 | 0.930 | 0.897 | 0.911 |
| Best test accuracy | 0.930 | 0.935 | 0.905 | 0.915 |
| Total weight norm, epoch 200 (80.0 at start) | 4273 | 91.2 | 4368 | 93.6 |
| Effective rank, epoch 200 | 243 | 126 | 267 | 144 |
| Stable rank delta, epoch 200 | 428 | 425 | 433 | 429 |
| Fisher peak, any layer (raw total) | 266 | 1089 | 98 | 3703 |

- **Ten blurred epochs leave a lasting accuracy deficit.** Final test accuracy is 2.6 points lower under Adam and 1.9
  points lower under SGD, and the gap is still there 190 epochs after the blur ends. For scale, earlier runs with a
  negligible blur (sigma 0.2) differed from these baselines by 0.3 points. This is one seed per configuration, so more seeds
  are needed to confirm the size of the deficit.
- **Recovery is fast at first, then stalls.** Test accuracy on sharp images is only 0.14-0.34 during the blur, then
  reaches about 0.78 one epoch after it ends. Train accuracy still ends above 0.99 in both blurred runs, so the deficit
  shows up as a generalisation gap and not as a failure to fit the training set.
- **Blur raises final effective rank.** It ends about 10% higher under Adam (267 against 243) and about 14% higher
  under SGD (144 against 126). Stable rank delta barely moves: it stays between 425 and 437 after epoch 20 in all four
  runs.
- **Fisher information responds to the blur, in opposite directions.** Under SGD, total Fisher around epochs 5-10 is
  about twice the baseline, and the peak is 3.4 times the baseline, at epoch 9.5 in layer 8. Under Adam it is lower
  than the baseline until about epoch 30. In both, it drops sharply when the blur ends (SGD: about 6500 to 2500; Adam:
  about 49 to 23). Fisher is measured on sharp images, so these values may partly reflect the model's behaviour on
  images it isn't training on. Single-measurement peaks are noisy.
- **Weight norms follow the optimizer, not the blur.** Under Adam the total weight norm grows about 50-fold and is
  still rising at epoch 200; under SGD it grows by about 15% and levels off, with or without blur. There is no weight
  decay in either.
- **The optimizer also sets effective rank and the scale of Fisher.** Effective rank is about 240-270 under Adam and
  about 125-145 under SGD. Fisher values are far larger under SGD. Adam's much larger weights may explain this, since
  batch norm makes convolution weights scale-invariant and larger weights then mean smaller gradients, but these runs
  don't test that.
