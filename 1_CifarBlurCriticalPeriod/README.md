# Per-layer Fisher information in a ResNet-18 on CIFAR-10

**Question:** how do each layer's Fisher information, the weight norms and the rank of the representation change
while a ResNet-18 trains, and does a blur deficit in the first epochs change them?

## Setup
- **Model:** CIFAR-style ResNet-18 (3x3 first convolution, no max pool).
- **Training:** batch size 128, 200 epochs, no weight decay, constant learning rate.
  - `exp_01`: Adam, learning rate 0.005.
  - `exp_02`: SGD, learning rate 0.01, momentum 0.9.
  - `exp_03`: as `exp_01`, with Gaussian blur (sigma 2.4 pixels) on training images for the first 10 epochs.
  - `exp_04`: as `exp_02`, with the same blur.
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

Fisher values are totals per layer, not normalised by parameter count, so compare layers of different
sizes with care. They use batch norm's running statistics. These are still at their initial values at
step 0, so that point is less comparable with the rest of the run.

Train accuracy is measured on augmented images during training, so it can sit below test accuracy.

The blur applies to training images only. Test accuracy, Fisher information and representation ranks are always
measured on sharp images, so during the blur they describe images the model is not training on.

## Files
- `exp_01.py`: settings and training loop for the baseline ResNet-18 on CIFAR-10, with Adam.
- `exp_02.py`: the same baseline, with SGD.
- `exp_03.py`: `exp_01` with the blur deficit.
- `exp_04.py`: `exp_02` with the blur deficit.
- `networks.py`: the MLP and ResNet-18.
- `util.py`: data loading, Fisher information, weight norm and representation ranks.
- `graph_training.py`: plots a TensorBoard run.

## Run
```
python exp_01.py --seed 42                     # logs to results/exp_01/seed_42/tensorboard/, replacing the previous run with that seed
tensorboard --logdir results/exp_01           # one run per seed; see the "fisher information", "weight norm" and "representation" sections
python graph_training.py exp_01 --seed 42     # -> results/exp_01/seed_42/graph_training.png
```
`--seed` defaults to 42 in both scripts.
Each run also saves the network's `state_dict` every 10 epochs to `results/exp_01/seed_42/checkpoints/epoch_XXX.pt`
(about 45 MB each, git-ignored). A repeated run with the same seed replaces them.
The other experiments run the same way: replace `exp_01` with `exp_02`, `exp_03` or `exp_04` in the commands above.

## Findings
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
| Fisher peak, any layer | 266 | 1089 | 98 | 3703 |

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
