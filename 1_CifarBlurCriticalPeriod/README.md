# Blur critical period in a ResNet-18 on CIFAR-10

**Question:** does a ResNet-18 have a critical period for blurred input? How does what a 10-epoch blur window costs the final
network depend on where in training the window falls, and how do per-layer Fisher information, representation rank and
SIGReg respond to the blur?

## Setup
- **Model:** CIFAR-style ResNet-18 (3x3 first convolution, no max pool).
- **Training:** 150 epochs, batch size 128, no weight decay, constant learning rate, random translations of up to 4 pixels and
  horizontal flips. Only the final network is saved.
- **Blur:** Gaussian, sigma 2.4 pixels, on training images only. Epoch numbers count from 1, and a blur window "a-b" covers
  epochs a+1 to b.
- **Batches** (`experiments.md` lists every experiment):

| Batch | Optimizer | Experiments | Seeds |
|---|---|---|---|
| `batch_01` | Adam, learning rate 0.005 | 31: no blur, blur on every epoch, 10-epoch windows 0-10, 5-15, ..., 140-150 | 42-46 |
| `batch_02` | SGD, learning rate 0.01, momentum 0.9 | the same 31 | 42-46 |

Logged to TensorBoard:
- **Fisher information:** trace of the true Fisher per layer, where the 10 "layers" are the first convolution, the 8 residual
  blocks and the classifier, with labels sampled from the model's predictions. Every 100 steps, on 1000 fixed, unaugmented
  training images.
- **Weight norm:** L2 norm of the parameters in total and per layer (batch norm parameters included), every 100 steps.
- **Representation** (`representation/`): of the penultimate features (512 pooled values, before the classifier) of the first
  test batch (512 images, the same every time), at step 0 and after each epoch:
  - effective rank = exp(-sum p_i log p_i), with p the singular values normalised by their sum (no centering),
  - stable rank delta = the smallest k whose top-k values of p cover at least 99% of the total,
  - SIGReg, from [LeJEPA](https://github.com/galilai-group/lejepa/blob/main/MINIMAL.md): how far 256 random 1-D projections
    of the features are from a standard normal, by their characteristic functions. 0 is Gaussian. `sigreg` is on the
    features as they are, so it also counts their mean and scale (they are non-negative, after a ReLU);
    `sigreg_standardized` standardises each feature over the batch first, so it measures shape only. The projections are
    drawn once from a fixed seed, so every reading uses the same ones.
- **Accuracy and loss:** train, on augmented (and during the blur, blurred) images as it trains; test, on the full sharp
  test set after each epoch.
- **Blurred readings:** every reading above that depends on input images is taken again on the same images with the
  training blur, throughout training and in every experiment, including no blur: test accuracy and loss
  (`test_blurred/`), the representation measures (`representation_blurred/`) and Fisher information
  (`fisher information_blurred/`). They show how the network treats blurred images before, during and after a blur
  window. This roughly doubles the cost of Fisher, the slowest reading.

Fisher, ranks and SIGReg use batch norm's running statistics. These are still at their initial values at step 0, so that
point is less comparable with the rest of the run. `analysis/graph_training.py` plots Fisher divided by each layer's
parameter count, leaving out the first convolution and the classifier, whose much higher values per parameter would flatten
the residual blocks.

## Files
- `batch_01.py`, `batch_02.py`: one per batch, holding the settings its experiments share (optimizer, epochs, seeds) and the
  sweep that generates them. Each runs one experiment of its batch.
- `training_procedure.py`: the training loop, and the settings every batch shares (batch size, augmentation, blur sigma, how
  often Fisher and weight norms are logged).
- `networks.py`: the MLP and ResNet-18.
- `util.py`: data loading, Fisher information, weight norm, representation ranks and SIGReg.
- `analysis/`: graphs and evaluation, run from this folder as `python -m analysis.<script>`:
  - `common.py`: what the scripts below share: reading a batch's runs and logs, and the plot styles.
  - `graph_training.py`: one run's accuracy, Fisher information, weight norms, ranks and SIGReg over training, with the
    blurred readings as lighter lines, and Fisher on sharp and on blurred images in two panels on the same y-axis.
  - `eval_blur.py`: tests a batch's final networks on test images blurred with sigma 0-3 (accuracy on the full test set, ranks
    and SIGReg on the first test batch), cached in `results/<batch>/eval_blur.csv`. The graphs below call it for any runs
    missing from it.
  - `graph_blur_accuracy.py`: accuracy of a batch's final networks against test image blur sigma, all experiments in one panel
    (no blur and blur on every epoch in black, the windows in viridis by start epoch), and a table of accuracy on sharp and
    training-blurred images and area under the curve, mean ± SD over seeds.
  - `graph_fisher.py`: seed 42's per-layer Fisher information for every experiment of a batch, over all epochs and over the
    last 10.
  - `graph_blur_ranks.py`: stable rank delta, effective rank and SIGReg (raw and standardised) of a batch's final networks
    against test image blur sigma, in the colours of `graph_blur_accuracy.py`.
  - `graph_blur_examples.py`: one test image at blur sigma 0-3.
- `hpc_run.py`: runs both batches on the HPC, collects the results and plots each run.
- `hpc_eval_blur.py`: runs `analysis/eval_blur.py` on the HPC, one job per experiment, and merges the rows into
  `results/<batch>/eval_blur.csv`.
- `Dockerfile`: the image `hpc_run.py` and `hpc_eval_blur.py` build and submit, with the training and analysis code.
- `experiments.md`: every experiment, and the earlier `exp_XX` experiments.

## Run
```
python batch_01.py                                   # list the experiments
python batch_01.py --experiment exp_03 --seed 42     # -> results/batch_01/experiments/exp_03/seed_42/, replacing an earlier run
tensorboard --logdir results/batch_01/experiments/exp_03
python -m analysis.graph_training batch_01 exp_03 --seed 42   # -> .../exp_03/seed_42/graph_training.png
python -m analysis.eval_blur batch_01                # -> results/batch_01/eval_blur.csv; delete it to evaluate again
python -m analysis.graph_blur_accuracy batch_01      # -> results/batch_01/graph_blur_accuracy.png
python -m analysis.graph_fisher batch_01             # -> results/batch_01/graph_fisher.png
python -m analysis.graph_blur_ranks batch_01         # -> results/batch_01/graph_blur_ranks.png
python -m analysis.graph_blur_examples               # -> results/graph_blur_examples.png
```
`--seed` defaults to 42. Each run saves its final network to `checkpoints/epoch_150.pt` (about 45 MB, git-ignored).
The batch graphs need every experiment and seed of the batch.

## Run on the HPC
```
python hpc_run.py            # batch_01 and batch_02, 31 experiments x 5 seeds each, 310 runs on the HPC
python hpc_run.py --local    # this PC also takes one run at a time, cancelling a queued HPC job if needed
```
Needs `~/hpc_outputs` mounted, and either a valid `hpc-client login` or `HPC_PASSWORD` set. CIFAR-10 must be on the NAS
as `fzho027_cifar10/cifar-10-batches-py`.

At most 30 jobs wait in the HPC queue at once (`MAX_QUEUED`); the rest are submitted as queued jobs start.
The dashboard shows each run's status, one row per experiment and one column per batch and seed, and refreshes from the
scheduler every 30 seconds. Finished runs are copied to `results/<batch>/experiments/<experiment>/seed_<seed>/`, with the
job's log as `run.log`, and plotted to `graph_training.png`. A run whose folder already exists is skipped, so nothing is
overwritten.

`BATCHES` in `hpc_run.py` sets the sweep, with each batch file's `EXPERIMENTS` and `SEEDS`, and `SWEEP` names its state file,
`results/hpc_run_state_<SWEEP>.json` (currently `batch_01_02`). Events and job ids go to `results/hpc_run.log`. Ctrl-C only
stops the dashboard: the jobs keep running, and running the script again picks them back up. The sweep stays on the image it
started with, so editing the code doesn't change it. To start a new sweep with the current code, delete the state file, or
change `SWEEP` to keep the old one. `--retry-failed` runs failed and cancelled runs again.
When the sweep is done, run the batch graphs above for `batch_01` and `batch_02`.

## Evaluate on the HPC
`eval_blur.py` takes about 37 s per checkpoint on this PC's RTX 3080, so about 3 hours for both batches. `hpc_eval_blur.py`
spreads the work over the HPC instead, one job per experiment (62 jobs, each 5 seeds x 31 sigmas).
Jobs can only read NAS datasets, not `~/hpc_outputs`, so the final checkpoints go up first as the dataset
`fzho027_cifar_blur_checkpoints`, laid out like `results/` (about 13 GB; each worker caches its copy). Mount the NAS datasets
share and upload them. Running the rsync again only sends new or changed files:
```
mkdir -p ~/cares_datasets
sudo mount -t cifs //130.216.238.2/datasets ~/cares_datasets -o username=<upi>,uid=$(id -u),gid=$(id -g)
rsync -avh --prune-empty-dirs --include='*/' --include='checkpoints/epoch_*.pt' --exclude='*' \
    results/batch_01 results/batch_02 ~/cares_datasets/fzho027_cifar_blur_checkpoints/
```
Then:
```
python hpc_eval_blur.py --check    # every needed checkpoint is on the NAS at its local size; lists the jobs, submits nothing
python hpc_eval_blur.py            # build and push the image, submit, poll every 30 s and merge each finished job's CSV
```
Experiments whose seeds are all in `results/<batch>/eval_blur.csv` already are skipped. Each job runs
`python -m analysis.eval_blur <batch> --experiment <exp> --checkpoints-root /workspace/datasets/fzho027_cifar_blur_checkpoints
--results-root /workspace/output ...`. Its container log is kept in `results/<batch>/eval_blur_logs/<exp>.log`. Progress is kept in
`results/hpc_eval_blur_state.json`. Ctrl-C leaves the jobs running, running the script again picks them back up, and
`--retry-failed` resubmits failed jobs. Then run `graph_blur_accuracy` and `graph_blur_ranks` for both batches: everything is
in the CSVs, so they only plot.

## Findings
### Batches 01 and 02
Not run yet.

### Earlier experiments
From `exp_01`-`exp_30` (see `experiments.md`), whose results were deleted. Their summary graphs are in `results/archive/`;
the files named below are gone.

#### Epoch 150 networks on blurred test images
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

#### Blur at the start, first runs
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
