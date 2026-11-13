"""Run analysis/eval_blur.py on the HPC for every experiment of the batches in BATCHES, one job per experiment (its seeds
together), and merge each job's rows into results/<batch>/eval_blur.csv, where the graph scripts find them.

The jobs read the checkpoints from the NAS dataset CHECKPOINTS_DATASET, a copy of results/ holding only the final
checkpoints. Upload it first, with the datasets share mounted at NAS_DATASETS (see the README):
    rsync -avh --prune-empty-dirs --include='*/' --include='checkpoints/epoch_150.pt' --exclude='*' \\
        results/batch_01 results/batch_02 ~/cares_datasets/fzho027_cifar_blur_checkpoints/
Then:
    python hpc_eval_blur.py --check   # only check the upload and list the jobs
    python hpc_eval_blur.py
Experiments whose seeds are all in the CSV already are skipped. Progress is kept in results/hpc_eval_blur_state.json:
Ctrl-C leaves the jobs running, and running the script again picks them back up.
"""
import argparse
import glob
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime

from hpc_client import HPCClient
from hpc_client.config import HPCConfig
from hpc_client.errors import HPCAuthenticationError, HPCRequestError

from analysis.common import BLUR_SIGMAS, CHECKPOINTS_DIR, RESULTS_DIR, run_dir
from analysis.eval_blur import merge_rows, output_path, read_rows, runs_in

BATCHES = ["batch_01", "batch_02"]
BATCH_MODULES = {name: importlib.import_module(name) for name in BATCHES}
DATASET_NAME = "fzho027_cifar10"
CHECKPOINTS_DATASET = "fzho027_cifar_blur_checkpoints"
NAS_DATASETS = os.path.expanduser("~/cares_datasets")  # the NAS datasets share, mounted here to upload CHECKPOINTS_DATASET
CONTAINER_DATASETS = "/workspace/datasets"
CONTAINER_OUTPUT = "/workspace/output"
REGISTRY_IMAGE = "130.216.238.2:5500/fzho027_cifar-blur-critical-period"
IMAGE_FILES = ["Dockerfile", *(f"{batch}.py" for batch in BATCHES), "training_procedure.py", "networks.py", "util.py",
               *sorted(glob.glob(os.path.join("analysis", "*.py")))]
MAX_QUEUED = 30  # HPC jobs waiting in the queue at once; more are submitted as these start
POLL_SECONDS = 30
MAX_RUNTIME_HOURS = 1.0  # 5 checkpoints x 31 sigmas, a few minutes on a GPU
OUTPUTS_TIMEOUT_SECONDS = 1800  # how long a completed job's outputs may take to appear on the NAS
MAX_COLLECT_ATTEMPTS = 5  # the NAS is a soft mount, so reads can fail and need retrying
HPC_OUTPUTS = os.path.expanduser("~/hpc_outputs")
STATE_PATH = os.path.join(RESULTS_DIR, "hpc_eval_blur_state.json")
LOG_PATH = os.path.join(RESULTS_DIR, "hpc_eval_blur.log")
LOGS_DIR = "eval_blur_logs"  # results/<batch>/eval_blur_logs/<experiment>.log, each job's container log
PASSWORD_ENV = "HPC_PASSWORD"

# Scheduler statuses are taken as they come; one not listed here counts as active.
PENDING = "pending"
QUEUED = "queued"
RUNNING = "running"
COMPLETED = "completed"
COLLECTED = "collected"
FAILED = "failed"
MISSING = "missing"
TERMINAL = {COMPLETED, "failed", "cancelled", "timeout", "deleted"}
FAILURES = (TERMINAL - {COMPLETED}) | {MISSING}
FINISHED = FAILURES | {COLLECTED}


# ---- the jobs ----

def job_key(batch, experiment):
    return f"{batch}/{experiment}"


def job_label(job):
    return f"{job['batch']} {job['experiment']}"


def checkpoint_files(batch, experiment):
    """Each seed's final checkpoint, relative to results/ (and so to the NAS dataset)."""
    module = BATCH_MODULES[batch]
    return [os.path.relpath(os.path.join(run_dir(module, experiment, seed), CHECKPOINTS_DIR, f"epoch_{module.EPOCHS:03d}.pt"),
                            RESULTS_DIR)
            for seed in module.SEEDS]


def needed_jobs():
    """(batch, experiment) for every experiment with a seed missing from its batch's CSV."""
    jobs = []
    for batch, module in BATCH_MODULES.items():
        done = runs_in(read_rows(module, output_path(module)))
        jobs += [(batch, name) for name in module.EXPERIMENTS
                 if any((name, seed) not in done for seed in module.SEEDS)]
    return jobs


def job_command(job):
    return (f"python -m analysis.eval_blur {job['batch']} --experiment {job['experiment']} "
            f"--data-root {CONTAINER_DATASETS}/{DATASET_NAME} --checkpoints-root {CONTAINER_DATASETS}/{CHECKPOINTS_DATASET} "
            f"--results-root {CONTAINER_OUTPUT}")


def job_name(job, image):
    return f"cifar_eval_blur_b{job['batch'].removeprefix('batch_')}_{job['experiment']}_{image.rsplit(':', 1)[1]}"


def job_root(job):
    return os.path.join(HPC_OUTPUTS, job["job_id"])


def job_csv(job):
    return os.path.join(job_root(job), "output", job["batch"], "eval_blur.csv")


def job_log(job):
    return os.path.join(job_root(job), "logs", "container.log")


# ---- checks ----

def upload_hint():
    return ("Upload the checkpoints first, with the NAS datasets share mounted:\n"
            f"    mkdir -p {NAS_DATASETS}\n"
            f"    sudo mount -t cifs //130.216.238.2/datasets {NAS_DATASETS} -o username=<upi>,uid=$(id -u),gid=$(id -g)\n"
            f"    rsync -avh --prune-empty-dirs --include='*/' --include='checkpoints/epoch_*.pt' --exclude='*' "
            f"{' '.join(os.path.join(RESULTS_DIR, batch) for batch in BATCHES)} {NAS_DATASETS}/{CHECKPOINTS_DATASET}/")


def check_checkpoints(jobs):
    """Every checkpoint the jobs need is here and on the NAS, at the same size."""
    if not os.path.ismount(NAS_DATASETS):
        raise SystemExit(f"{NAS_DATASETS} is not mounted, so the checkpoint upload can't be checked.\n{upload_hint()}")
    missing, different = [], []
    for batch, experiment in jobs:
        for path in checkpoint_files(batch, experiment):
            local = os.path.join(RESULTS_DIR, path)
            remote = os.path.join(NAS_DATASETS, CHECKPOINTS_DATASET, path)
            if not os.path.exists(local):
                raise SystemExit(f"No checkpoint at {local}: the training run is missing.")
            if not os.path.exists(remote):
                missing.append(remote)
            elif os.path.getsize(remote) != os.path.getsize(local):
                different.append(remote)
    if missing or different:
        lines = [f"missing: {path}" for path in missing[:5]] + [f"different size: {path}" for path in different[:5]]
        raise SystemExit(f"{len(missing)} checkpoints missing from the NAS and {len(different)} of a different size, e.g.\n"
                         + "\n".join(f"    {line}" for line in lines) + f"\n{upload_hint()}")
    print(f"All {sum(len(checkpoint_files(*job)) for job in jobs)} checkpoints are on the NAS in {CHECKPOINTS_DATASET}.")


def check_ready():
    if not os.path.ismount(HPC_OUTPUTS):
        raise SystemExit(f"{HPC_OUTPUTS} is not mounted, so results couldn't be collected.")
    if not shutil.which("docker"):
        raise SystemExit("docker is not installed.")
    if not os.path.isdir(os.path.join(NAS_DATASETS, DATASET_NAME)):
        raise SystemExit(f"{DATASET_NAME} is not on the NAS at {NAS_DATASETS}/{DATASET_NAME}.")


# ---- image and scheduler ----

def image_name():
    digest = hashlib.sha256()
    for path in IMAGE_FILES:
        with open(path, "rb") as f:
            digest.update(f.read())
    return f"{REGISTRY_IMAGE}:{digest.hexdigest()[:12]}"


def build_and_push(image):
    subprocess.run(["docker", "build", "-t", image, "."], check=True)
    subprocess.run(["docker", "push", image], check=True)


def logged_in_client():
    """Log in with a new client: one holding an expired cookie can't read back the new one."""
    config = HPCConfig.load()
    password = os.environ.get(PASSWORD_ENV)
    if not password or not config.username:
        raise RuntimeError(f"HPC session expired. Set {PASSWORD_ENV} or run: hpc-client login")
    client = HPCClient(config.scheduler_url)
    client.login(config.username, password, save=True)
    return client


def connect():
    client = HPCClient.from_config()
    try:
        client.me()
    except HPCAuthenticationError:
        client = logged_in_client()
    return client


# ---- the sweep ----

class Sweep:
    def __init__(self, client, state):
        self.client = client
        self.state = state
        self.jobs = state["jobs"]

    def save(self):
        temporary = f"{STATE_PATH}.tmp"
        with open(temporary, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=1)
        os.replace(temporary, STATE_PATH)

    def log(self, message):
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {message}\n")

    def error(self, message):
        print(f"{datetime.now():%H:%M:%S} {message}", flush=True)
        self.log(f"ERROR {message}")

    def update(self, job, **fields):
        changed = "status" in fields and fields["status"] != job["status"]
        job.update(fields)
        self.save()
        if changed:
            self.log(f"{job_label(job)}: {job['status']}, job {job['job_id']} {job['detail']}".rstrip())

    def select(self, predicate):
        return [job for job in self.jobs.values() if predicate(job)]

    def call(self, method, *args):
        try:
            return getattr(self.client, method)(*args)
        except HPCAuthenticationError:
            self.client = logged_in_client()
            return getattr(self.client, method)(*args)

    def sync(self):
        data = self.call("jobs")
        scheduled = data.get("jobs", data) if isinstance(data, dict) else data
        by_id = {job.get("job_id"): job for job in scheduled}

        # A submission whose response was lost shows up here under its name.
        by_name = {job.get("job_name"): job for job in scheduled if job.get("status") not in TERMINAL}
        for job in self.select(lambda job: job["status"] == PENDING):
            found = by_name.get(job_name(job, self.state["image"]))
            if found:
                self.update(job, job_id=found["job_id"], status=found.get("status", QUEUED))

        for job in self.select(lambda job: job["job_id"] and job["status"] not in TERMINAL | FINISHED):
            found = by_id.get(job["job_id"])
            if found is None:
                try:
                    data = self.call("job", job["job_id"])
                    found = data.get("job", data)
                except HPCRequestError as error:
                    if str(error).startswith("404"):
                        self.update(job, status=MISSING, detail="the scheduler no longer has this job")
                        continue
                    raise
            status = found.get("status", "unknown")
            fields = {"status": status}
            if status == COMPLETED:
                fields["done_at"] = time.time()
            elif status in FAILURES:
                fields["detail"] = f"{found.get('message') or ''} see {job_log(job)}".strip()
            self.update(job, **fields)

    def submit_pending(self):
        waiting = self.select(lambda job: job["job_id"] and job["status"] not in TERMINAL | FINISHED | {RUNNING})
        for job in self.select(lambda job: job["status"] == PENDING)[:max(0, MAX_QUEUED - len(waiting))]:
            spec = {"job_name": job_name(job, self.state["image"]), "image": self.state["image"],
                    "command": job_command(job), "max_runtime_hours": MAX_RUNTIME_HOURS,
                    "required_datasets": [DATASET_NAME, CHECKPOINTS_DATASET], "required_worker_ids": []}
            try:
                data = self.call("submit", spec)
            except HPCRequestError as error:  # probably the active job limit, so try again next poll
                self.error(f"submit rejected: {error}")
                return
            submitted = data.get("job", data) if isinstance(data, dict) else {"job_id": data}
            if submitted.get("job_id"):
                self.update(job, job_id=submitted["job_id"], status=submitted.get("status", QUEUED))
            else:  # sync adopts it by name on the next poll
                self.error(f"no job id in submit response: {data}")

    def outputs_ready(self, job):
        """The NAS copy can lag the status, so wait for the job's own status file as well as the CSV."""
        try:
            with open(os.path.join(job_root(job), "status.json"), encoding="utf-8") as f:
                return json.load(f).get("status") == COMPLETED and os.path.exists(job_csv(job))
        except (OSError, ValueError):
            return False

    def collect(self, job):
        module = BATCH_MODULES[job["batch"]]
        rows = read_rows(module, job_csv(job))
        expected = {(job["experiment"], seed) for seed in module.SEEDS}
        if runs_in(rows) != expected or len(rows) != len(expected) * len(BLUR_SIGMAS):
            raise ValueError(f"{job_csv(job)} has {len(rows)} rows for runs {sorted(runs_in(rows))}, expected "
                             f"{len(expected) * len(BLUR_SIGMAS)} for {sorted(expected)}")
        added = merge_rows(module, rows)
        logs_dir = os.path.join(RESULTS_DIR, job["batch"], LOGS_DIR)
        os.makedirs(logs_dir, exist_ok=True)
        if os.path.exists(job_log(job)):
            shutil.copyfile(job_log(job), os.path.join(logs_dir, f"{job['experiment']}.log"))
        self.update(job, status=COLLECTED, detail=f"{len(added)} runs added")

    def collect_finished(self):
        for job in self.select(lambda job: job["status"] == COMPLETED):
            if not self.outputs_ready(job):
                if time.time() - job["done_at"] > OUTPUTS_TIMEOUT_SECONDS:
                    self.update(job, status=FAILED, detail=f"no outputs at {job_csv(job)}, see {job_log(job)}")
                continue
            try:
                self.collect(job)
            except (OSError, ValueError) as error:
                attempts = job["collect_attempts"] + 1
                self.error(f"collecting {job_label(job)}, attempt {attempts}: {error}")
                if attempts >= MAX_COLLECT_ATTEMPTS or isinstance(error, ValueError):
                    self.update(job, status=FAILED, collect_attempts=attempts, detail=f"could not collect: {error}")
                else:
                    self.update(job, collect_attempts=attempts)

    def poll(self):
        try:
            self.sync()
        except Exception as error:  # our view of the jobs is stale, so don't submit on it
            self.error(f"poll: {error}")
            return
        for step in (self.collect_finished, self.submit_pending):
            try:
                step()
            except Exception as error:
                self.error(f"{step.__name__}: {error}")

    def summary(self):
        counts = Counter(job["status"] for job in self.jobs.values())
        return "  ".join(f"{status}: {count}" for status, count in sorted(counts.items()))

    def finished(self):
        return not self.select(lambda job: job["status"] not in FINISHED)

    def run(self):
        try:
            while True:
                self.poll()
                print(f"{datetime.now():%H:%M:%S}  {self.summary()}", flush=True)
                if self.finished():
                    return True
                time.sleep(POLL_SECONDS)
        except KeyboardInterrupt:
            self.save()
            print("Stopped watching. Jobs keep running, run this script again to pick them back up.")
            return False


def new_state(image, jobs):
    return {"image": image, "jobs": {
        job_key(batch, experiment): {"batch": batch, "experiment": experiment, "status": PENDING, "job_id": None,
                                     "done_at": None, "collect_attempts": 0, "detail": ""}
        for batch, experiment in jobs}}


def load_state(retry_failed):
    with open(STATE_PATH, encoding="utf-8") as f:
        state = json.load(f)
    if retry_failed:
        for job in state["jobs"].values():
            if job["status"] in FAILURES:
                job.update(status=PENDING, job_id=None, done_at=None, collect_attempts=0, detail="")
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="only check the checkpoint upload and list the jobs")
    parser.add_argument("--retry-failed", action="store_true", help="submit failed or cancelled jobs again")
    args = parser.parse_args()

    jobs = needed_jobs()
    for batch in BATCHES:
        print(f"{batch}: {sum(job[0] == batch for job in jobs)} of {len(BATCH_MODULES[batch].EXPERIMENTS)} experiments "
              f"to evaluate")
    if os.path.exists(STATE_PATH):  # a sweep in progress still needs its own jobs' checkpoints
        with open(STATE_PATH, encoding="utf-8") as f:
            jobs = sorted(set(jobs) | {(job["batch"], job["experiment"]) for job in json.load(f)["jobs"].values()
                                       if job["status"] not in FINISHED})
    if not jobs and not os.path.exists(STATE_PATH):
        print("Every run is in the CSVs already, nothing to do.")
        return
    check_checkpoints(jobs)
    if args.check:
        return

    check_ready()
    client = connect()
    image = image_name()
    if os.path.exists(STATE_PATH):
        state = load_state(args.retry_failed)
        if state["image"] != image:
            print(f"The code has changed since this sweep started, so it stays on {state['image']}. "
                  f"Delete {STATE_PATH} to start a new sweep with the current code.")
        for key, job in new_state(state["image"], jobs)["jobs"].items():
            state["jobs"].setdefault(key, job)  # e.g. a run removed from the CSV since
    else:
        build_and_push(image)
        state = new_state(image, jobs)

    sweep = Sweep(client, state)
    if not sweep.run():
        return
    failed = sweep.select(lambda job: job["status"] in FAILURES)
    for job in failed:
        print(f"{job_label(job)} {job['status']}: {job['detail']}")
    print("Graph the results with:")
    for batch in BATCHES:
        print(f"    python -m analysis.graph_blur_accuracy {batch}\n    python -m analysis.graph_blur_ranks {batch}")
    if failed:
        print(f"{len(failed)} jobs failed: run again with --retry-failed to resubmit them.")
        sys.exit(1)


if __name__ == "__main__":
    main()
