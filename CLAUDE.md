# fk_around_and_find_out

A scratch repo for quick, throwaway experiments (mostly Python / ML). Each experiment is a
standalone folder at the repo root. There is no shared library and no top-level package;
the only shared resource is the `datasets/` download cache.

## Layout

```
fk_around_and_find_out/
├── CLAUDE.md
├── .gitignore
├── environment.yml        # the shared `experiments` conda env
├── datasets/              # shared download cache, git-ignored (e.g. datasets/mnist/)
├── CnnExperimentOne/
│   ├── README.md          # what is being tested, how to run, findings
│   ├── network.py
│   ├── training_procedure.py
│   ├── batch_01.py        # entry point, one per batch: shared settings + the sweep of experiments
│   ├── batch_02.py
│   ├── analysis/          # graphing and evaluation scripts, run with `python -m analysis.<script>`
│   │   ├── __init__.py
│   │   └── graph_xxx.py
│   └── results/
│       ├── archive/       # optional: old outputs worth keeping
│       ├── batch_01/      # batch-level outputs: graphs, evaluation CSVs
│       │   └── experiments/
│       │       ├── exp_01/
│       │       │   └── seed_42/   # logs / checkpoints / per-run plots for this experiment and seed
│       │       └── exp_02/
│       │           └── seed_42/
│       └── batch_02/
└── <NextExperiment>/
    └── ...
```

## Rules

- **One experiment per folder.** Name folders in PascalCase describing the idea,
  e.g. `CnnExperimentOne`, `AttentionVsConvSmallData`.
- **Fully self-contained.** An experiment must never import from another experiment's folder
  or from a shared module at the repo root. Everything it needs lives in its own folder.
  The one exception is data: see `datasets/` below.
- **Shared datasets.** Datasets are downloaded once into the root `datasets/` folder and
  reused across experiments, one subfolder per dataset (e.g. `datasets/mnist/`,
  `datasets/cifar10/`). Each experiment's data-loading code checks `../datasets/<name>/`
  first and only downloads if it is missing, saving the download there (e.g. pass
  `root="../datasets"` to torchvision). Treat `datasets/` as read-only after download:
  never modify or preprocess files in place; write derived data to the experiment's own
  folder instead.
- **Duplication is fine and expected.** If two experiments need the same helper (plotting,
  data loading, training loops), copy it into each folder. Do not refactor it into a shared
  module. Changing one experiment must never break another.
- **Run from inside the folder.** Use relative imports/paths that assume the working
  directory is the experiment folder, e.g.
  `cd CnnExperimentOne && conda run -n experiments python batch_01.py --experiment exp_01`.
- **Results stay local.** Write generated images, plots and logs to that experiment's
  `results/` folder (create it if missing). Never write output outside the experiment folder
  (downloads into `datasets/` are the only exception).
- **One script per batch.** Experiments are grouped into batches, one script each (`batch_01.py`,
  `batch_02.py`, ...). A batch script holds the settings its experiments share as constants in the
  file (not command-line options), plus a sweep that generates its experiments (`exp_01`, `exp_02`,
  ... -> what varies between them). It runs one experiment per call, chosen with `--experiment`,
  and takes a `--seed` int argument, defaulting to 42. A new set of ablations is a new batch;
  don't edit a batch whose results you keep. Batch scripts are near-duplicates by design: don't
  import one batch from another.
- **Results folder structure.** Each run writes to `results/<batch>/experiments/<experiment>/seed_<seed>/`,
  e.g. `results/batch_01/experiments/exp_01/seed_42/`, so runs with different seeds never overwrite
  each other. Outputs about a whole batch (comparison graphs, evaluation CSVs) go in `results/<batch>/`.
  Old outputs worth keeping can go in `results/archive/`.
- **Analysis in its own folder.** Graphing and evaluation scripts live in the experiment's
  `analysis/` folder (with an empty `__init__.py`) and run from the experiment folder as
  `python -m analysis.<script> <batch>`, so they can import the experiment's modules and batch
  files. Helpers shared between them go in `analysis/common.py`.
- **One shared environment.** All experiments run in the `experiments` conda env
  (`conda activate experiments`, or `conda run -n experiments python exp_01.py`).
  Don't create per-experiment envs or venvs. The env is defined by the root
  `environment.yml` (`conda env create -f environment.yml` on a new machine,
  `conda env update -n experiments -f environment.yml` after changes). If an experiment
  needs a new package, add it to `environment.yml` (conda-forge where possible, the `pip:`
  section otherwise), update the env, and list it in the experiment's `requirements.txt`
  for reference. Keep `environment.yml` cross-platform: no OS-specific builds, CUDA
  index URLs or pinned build strings.
- **pyenv gotcha.** On this machine pyenv shims can shadow the conda env's `python`/`pip`,
  even under `conda run`. Check with `which python`; if it isn't under `.../envs/experiments/`,
  call the interpreter directly: `~/anaconda3/envs/experiments/bin/python exp_01.py`.
- **Keep it quick.** Favour simple, readable scripts over abstractions, configs or frameworks.
  No tests or packaging unless an experiment specifically calls for them.
- **Document briefly.** Each experiment gets a short `README.md`: the question/hypothesis,
  how to run it, and what was found (with references to images in `results/`).

## When working in this repo

- Starting a new experiment: create a new folder; you may copy useful files from an existing
  experiment as a starting point, but adapt the copy rather than linking to the original.
- Modifying an experiment: only touch files inside that experiment's folder.
- Large artifacts should not be committed. Datasets go in the git-ignored `datasets/`
  folder; model checkpoints and other large outputs go in the experiment's own `.gitignore`.
