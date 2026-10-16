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
│   ├── experiment.py      # entry point
│   ├── graph_utils.py
│   └── results/           # generated plots / images
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
  `cd CnnExperimentOne && conda run -n experiments python experiment.py`.
- **Results stay local.** Write generated images, plots and logs to that experiment's
  `results/` folder (create it if missing). Never write output outside the experiment folder
  (downloads into `datasets/` are the only exception).
- **One shared environment.** All experiments run in the `experiments` conda env
  (`conda activate experiments`, or `conda run -n experiments python experiment.py`).
  Don't create per-experiment envs or venvs. The env is defined by the root
  `environment.yml` (`conda env create -f environment.yml` on a new machine,
  `conda env update -n experiments -f environment.yml` after changes). If an experiment
  needs a new package, add it to `environment.yml` (conda-forge where possible, the `pip:`
  section otherwise), update the env, and list it in the experiment's `requirements.txt`
  for reference. Keep `environment.yml` cross-platform: no OS-specific builds, CUDA
  index URLs or pinned build strings.
- **pyenv gotcha.** On this machine pyenv shims can shadow the conda env's `python`/`pip`,
  even under `conda run`. Check with `which python`; if it isn't under `.../envs/experiments/`,
  call the interpreter directly: `~/anaconda3/envs/experiments/bin/python experiment.py`.
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
