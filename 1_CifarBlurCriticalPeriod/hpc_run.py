"""Run every experiment and seed of the batches in BATCHES on the HPC, collect the results into results/ and plot them, with a
live dashboard.

Builds and pushes the image, submits the jobs and polls them. Each batch module's EXPERIMENTS and SEEDS set its part of the
sweep. Finished runs are copied to results/<batch>/experiments/<experiment>/seed_<seed>/. With --local this PC
takes one run at a time as well, cancelling a queued HPC job to do so. Progress is kept in
results/hpc_run_state_<SWEEP>.json: Ctrl-C leaves the jobs running, and running the script again picks the sweep up
where it was.
    python hpc_run.py --local
"""
import argparse
import glob
import hashlib
import importlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from hpc_client import HPCClient
from hpc_client.config import HPCConfig
from hpc_client.errors import HPCAuthenticationError, HPCRequestError
from rich.console import Group
from rich.live import Live
from rich.table import Table
from rich.text import Text

# Each sweep keeps its own state file, so a new one starts fresh.
SWEEP = "batch_01_02"
BATCHES = ["batch_01", "batch_02"]  # Adam and SGD, each 31 experiments x 5 seeds
BATCH_MODULES = {name: importlib.import_module(name) for name in BATCHES}
MAX_QUEUED = 30  # HPC jobs waiting in the queue at once; more are submitted as these start
POLL_SECONDS = 30
REFRESH_SECONDS = 0.5
MAX_RUNTIME_HOURS = 6.0  # past 200-epoch runs took 1.2-2.2 hours on this PC
OUTPUTS_TIMEOUT_SECONDS = 1800  # how long a completed job's outputs may take to appear on the NAS
MAX_COLLECT_ATTEMPTS = 5  # the NAS is a soft mount, so reads can fail and need retrying
COLLECT_WORKERS = 2
REGISTRY_IMAGE = "130.216.238.2:5500/fzho027_cifar-blur-critical-period"
IMAGE_FILES = ["Dockerfile", *(f"{batch}.py" for batch in BATCHES), "training_procedure.py", "networks.py", "util.py"]
DATASET_NAME = "fzho027_cifar10"
CONTAINER_DATA_ROOT = f"/workspace/datasets/{DATASET_NAME}"
CONTAINER_OUTPUT = "/workspace/output"
LOCAL_DATA_ROOT = os.path.abspath(os.path.join("..", "datasets"))
HPC_OUTPUTS = os.path.expanduser("~/hpc_outputs")
RESULTS_DIR = "results"
LOCAL_RUNS_DIR = os.path.join(RESULTS_DIR, ".local_runs")
STATE_PATH = os.path.join(RESULTS_DIR, f"hpc_run_state_{SWEEP}.json")
LOG_PATH = os.path.join(RESULTS_DIR, "hpc_run.log")
PASSWORD_ENV = "HPC_PASSWORD"

HPC = "hpc"
LOCAL = "local"
# Scheduler statuses are shown as they come, so one not listed here still displays and counts as active.
QUEUED = "queued"
RUNNING = "running"
COMPLETED = "completed"
CANCELLED = "cancelled"
TERMINAL = {COMPLETED, "failed", CANCELLED, "timeout", "deleted"}
# Statuses this script adds around the scheduler's.
PENDING = "pending"
COLLECTING = "collecting"
PLOTTING = "plotting"
COLLECTED = "collected"
EXISTING = "existing"
MISSING = "missing"
FAILED = "failed"
FAILURES = (TERMINAL - {COMPLETED}) | {MISSING}
FINISHED = FAILURES | {COLLECTED, EXISTING}
# Short, so the dashboard's batch x seed columns fit a terminal.
LABELS = {PENDING: "-", QUEUED: "queued", RUNNING: "run", COMPLETED: "done", COLLECTING: "copy", PLOTTING: "plot",
          COLLECTED: "ok", EXISTING: "exists"}
MAX_FAILURE_LINES = 10
STYLES = {
    PENDING: "dim", QUEUED: "yellow", RUNNING: "blue", COMPLETED: "cyan", COLLECTING: "magenta", PLOTTING: "magenta",
    COLLECTED: "green", EXISTING: "dim green", **{status: "bold red" for status in FAILURES},
}


def run_key(batch, experiment, seed):
    return f"{batch}/{experiment}/{seed}"


def run_name(run):
    return f"{run['batch']} {run['experiment']} seed {run['seed']}"


def run_path(run):
    """Where a run's results go, under the results root: <batch>/experiments/<experiment>/seed_<seed>."""
    return os.path.join(run["batch"], "experiments", run["experiment"], f"seed_{run['seed']}")


def target_dir(run):
    return os.path.join(RESULTS_DIR, run_path(run))


def job_root(run):
    return os.path.join(LOCAL_RUNS_DIR if run["backend"] == LOCAL else HPC_OUTPUTS, run["job_id"])


def output_dir(run):
    return os.path.join(job_root(run), "output", run_path(run))


def job_command(run):
    return (f"python {run['batch']}.py --experiment {run['experiment']} --seed {run['seed']} "
            f"--data-root {CONTAINER_DATA_ROOT} --results-root {CONTAINER_OUTPUT}")


def short_batch(run):
    return f"b{run['batch'].removeprefix('batch_')}"


def parse_time(value):
    try:
        return datetime.fromisoformat(value).timestamp()
    except (TypeError, ValueError):
        return None


def format_duration(seconds):
    minutes = int(seconds) // 60
    return f"{minutes // 60}:{minutes % 60:02d}"


def image_name():
    digest = hashlib.sha256()
    for path in IMAGE_FILES:
        with open(path, "rb") as f:
            digest.update(f.read())
    return f"{REGISTRY_IMAGE}:{digest.hexdigest()[:12]}"


def run_command(command, **kwargs):
    # In its own session, so Ctrl-C on the dashboard doesn't interrupt a docker call or a plot halfway through.
    return subprocess.run(command, capture_output=True, text=True, start_new_session=True, **kwargs)


def build_and_push(image):
    subprocess.run(["docker", "build", "-t", image, "."], check=True)
    subprocess.run(["docker", "push", image], check=True)


def new_state(image):
    runs = {}
    for batch, module in BATCH_MODULES.items():
        for experiment in module.EXPERIMENTS:
            for seed in module.SEEDS:
                run = {"batch": batch, "experiment": experiment, "seed": seed, "status": PENDING, "backend": None,
                       "job_id": None, "submitted_at": None, "started_at": None, "finished_at": None, "done_at": None,
                       "stealing": False, "collect_attempts": 0, "plot_failed": False, "detail": ""}
                if os.path.isdir(target_dir(run)):  # never overwrite an earlier run
                    run["status"] = EXISTING
                runs[run_key(batch, experiment, seed)] = run
    return {"image": image, "runs": runs}


def load_state(retry_failed):
    with open(STATE_PATH, encoding="utf-8") as f:
        state = json.load(f)
    for run in state["runs"].values():
        # Whatever was copying or plotting died with the previous session.
        if run["status"] in (COLLECTING, PLOTTING):
            run.update(status=COMPLETED, done_at=time.time(), collect_attempts=0)
        if retry_failed and run["status"] in FAILURES:
            run.update(status=PENDING, backend=None, job_id=None, submitted_at=None, started_at=None, finished_at=None,
                       done_at=None, stealing=False, collect_attempts=0, detail="")
    return state


class Sweep:
    def __init__(self, client, state, use_local):
        self.client = client
        self.state = state
        self.runs = state["runs"]
        self.use_local = use_local
        self.local_broken = False  # set if docker can't start a run, so we stop cancelling jobs for it
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.pool = ThreadPoolExecutor(COLLECT_WORKERS)
        self.errors = deque(maxlen=3)
        self.last_poll = None
        self.next_poll = None
        self.save()

    # ---- state ----

    def save(self):
        with self.lock:
            temporary = f"{STATE_PATH}.tmp"
            with open(temporary, "w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=1)
            os.replace(temporary, STATE_PATH)

    def log(self, message):
        with self.lock, open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {message}\n")

    def record_error(self, message):
        with self.lock:
            self.errors.append(f"{datetime.now():%H:%M:%S} {message}")
        self.log(f"ERROR {message}")

    def update(self, run, **fields):
        with self.lock:
            changed = "status" in fields and fields["status"] != run["status"]
            run.update(fields)
            self.save()
        if changed:
            self.log(f"{run_name(run)}: {run['status']} on {run['backend']}, "
                     f"job {run['job_id']} {run['detail']}".rstrip())

    def select(self, predicate):
        with self.lock:
            return [run for run in self.runs.values() if predicate(run)]

    def finished(self):
        return not self.select(lambda run: run["status"] not in FINISHED)

    # ---- scheduler ----

    def call(self, method, *args):
        try:
            return getattr(self.client, method)(*args)
        except HPCAuthenticationError:
            self.client = logged_in_client()
            return getattr(self.client, method)(*args)

    def fetch_job(self, job_id):
        data = self.call("job", job_id)
        return data.get("job", data)

    def sync_hpc(self):
        data = self.call("jobs")
        jobs = data.get("jobs", data) if isinstance(data, dict) else data
        by_id = {job.get("job_id"): job for job in jobs}

        # A submission whose response was lost, or whose id wasn't saved, shows up here under its name.
        by_name = {job.get("job_name"): job for job in jobs if job.get("status") not in TERMINAL}
        for run in self.select(lambda run: run["status"] == PENDING):
            job = by_name.get(self.job_name(run))
            if job:
                self.update(run, backend=HPC, job_id=job["job_id"], status=job.get("status"), submitted_at=time.time())

        active = lambda run: run["backend"] == HPC and run["status"] not in TERMINAL | FINISHED | {COLLECTING, PLOTTING}
        for run in self.select(active):
            job = by_id.get(run["job_id"])
            if job is None:
                try:
                    job = self.fetch_job(run["job_id"])
                except HPCRequestError as error:
                    if str(error).startswith("404"):
                        self.update(run, status=MISSING, stealing=False, detail="the scheduler no longer has this job")
                        continue
                    raise
            self.apply_job(run, job)

    def apply_job(self, run, job):
        status = job.get("status", "unknown")
        if run["stealing"]:
            if status == CANCELLED:
                self.start_local(run)
                return
            if status in TERMINAL:  # it finished before the cancel landed, so it counts as a normal HPC run
                self.update(run, stealing=False)

        fields = {"status": status}
        if status == RUNNING:
            if run["started_at"] is None:
                fields["started_at"] = parse_time(job.get("started_at")) or time.time()
        elif status not in TERMINAL:  # includes a preempted job going back in the queue
            fields["started_at"] = None
        if status in TERMINAL and run["finished_at"] is None:
            fields["finished_at"] = parse_time(job.get("finished_at")) or time.time()
        if status == COMPLETED:
            fields["done_at"] = time.time()
        elif status in FAILURES:
            log_path = os.path.join(job_root(run), "logs", "container.log")
            fields["detail"] = f"{job.get('message') or ''} see {log_path}".strip()  # a job can fail before it has logs
        self.update(run, **fields)

    def job_name(self, run):
        tag = self.state["image"].rsplit(":", 1)[1]
        return f"cifar_blur_{short_batch(run)}_{run['experiment']}_s{run['seed']}_{tag}"

    def submit_pending(self):
        waiting = self.select(lambda run: run["backend"] == HPC and run["status"] not in
                              TERMINAL | FINISHED | {RUNNING, COLLECTING, PLOTTING})
        for run in self.select(lambda run: run["status"] == PENDING)[:max(0, MAX_QUEUED - len(waiting))]:
            spec = {"job_name": self.job_name(run), "image": self.state["image"], "command": job_command(run),
                    "max_runtime_hours": MAX_RUNTIME_HOURS, "required_datasets": [DATASET_NAME],
                    "required_worker_ids": []}
            try:
                data = self.call("submit", spec)
            except HPCRequestError as error:  # probably the active job limit, so try again next poll
                self.record_error(f"submit rejected: {error}")
                return
            job = data.get("job", data) if isinstance(data, dict) else {"job_id": data}
            if job.get("job_id"):
                self.update(run, backend=HPC, job_id=job["job_id"], status=job.get("status", QUEUED),
                            submitted_at=time.time())
            else:  # sync_hpc adopts it by name on the next poll
                self.record_error(f"no job id in submit response: {data}")

    # ---- this PC ----

    def local_busy(self):
        return bool(self.select(lambda run: run["stealing"] or (
            run["backend"] == LOCAL and run["status"] not in TERMINAL | FINISHED | {COLLECTING, PLOTTING})))

    def take_over(self):
        """Give this PC a run if it's idle: one not submitted yet, otherwise the newest queued HPC job."""
        if not self.use_local or self.local_broken or self.local_busy():
            return
        pending = self.select(lambda run: run["status"] == PENDING)
        if pending:
            self.start_local(pending[-1])
            return
        queued = self.select(lambda run: run["backend"] == HPC and run["status"] == QUEUED)
        queued.sort(key=lambda run: run["submitted_at"] or 0, reverse=True)  # the last one the scheduler would dispatch
        for run in queued:
            self.update(run, stealing=True)
            try:
                self.call("cancel", run["job_id"])
            except HPCRequestError as error:  # most likely dispatched just now, so leave it on the HPC
                self.update(run, stealing=False)
                self.record_error(f"could not cancel {run['job_id']}: {error}")
                continue
            # If this doesn't read cancelled yet, sync_hpc finishes the move on a later poll.
            self.apply_job(run, self.fetch_job(run["job_id"]))
            return

    def start_local(self, run):
        job_id = f"cifar_blur_{short_batch(run)}_{run['experiment']}_s{run['seed']}_{datetime.now():%y%m%d%H%M%S}"
        output = os.path.abspath(os.path.join(LOCAL_RUNS_DIR, job_id, "output"))
        os.makedirs(output)
        result = run_command(
            ["docker", "run", "-d", "--gpus", "all", "--name", job_id, "--user", f"{os.getuid()}:{os.getgid()}",
             "-e", "HOME=/tmp", "-v", f"{LOCAL_DATA_ROOT}:{CONTAINER_DATA_ROOT}:ro", "-v", f"{output}:{CONTAINER_OUTPUT}",
             self.state["image"], *shlex.split(job_command(run))])
        if result.returncode != 0:
            # The run goes back to the HPC, and stays there, since retrying would cancel a job every poll.
            self.local_broken = True
            run_command(["docker", "rm", "-f", job_id])
            shutil.rmtree(os.path.dirname(output), ignore_errors=True)
            self.update(run, status=PENDING, backend=None, job_id=None, stealing=False)
            self.record_error(f"local run disabled, docker failed: {result.stderr.strip()}")
            return
        self.update(run, status=RUNNING, backend=LOCAL, job_id=job_id, stealing=False, started_at=time.time(),
                    finished_at=None, submitted_at=time.time())

    def sync_local(self):
        for run in self.select(lambda run: run["backend"] == LOCAL and run["status"] == RUNNING):
            result = run_command(["docker", "inspect", "-f", "{{.State.Status}} {{.State.ExitCode}}", run["job_id"]])
            if result.returncode != 0:
                if "No such" in result.stderr:
                    self.update(run, status=FAILED, finished_at=time.time(), detail="the container disappeared")
                else:
                    self.record_error(f"docker inspect: {result.stderr.strip()}")
                continue
            container_status, exit_code = result.stdout.split()
            if container_status not in ("exited", "dead"):
                continue
            logs_dir = os.path.join(job_root(run), "logs")
            os.makedirs(logs_dir, exist_ok=True)
            logs = run_command(["docker", "logs", run["job_id"]])
            with open(os.path.join(logs_dir, "container.log"), "w", encoding="utf-8") as f:
                f.write(logs.stdout + logs.stderr)
            run_command(["docker", "rm", run["job_id"]])
            if exit_code == "0":
                self.update(run, status=COMPLETED, finished_at=time.time(), done_at=time.time())
            else:
                self.update(run, status=FAILED, finished_at=time.time(),
                            detail=f"exit code {exit_code}, see {os.path.join(logs_dir, 'container.log')}")

    # ---- collecting ----

    def outputs_ready(self, run):
        """The NAS copy can lag the status, so wait for its own status file as well as the logs."""
        try:
            if run["backend"] == HPC:
                with open(os.path.join(job_root(run), "status.json"), encoding="utf-8") as f:
                    if json.load(f).get("status") != COMPLETED:
                        return False
            return bool(glob.glob(os.path.join(output_dir(run), "tensorboard", "events*")))
        except (OSError, ValueError):
            return False

    def collect_finished(self):
        for run in self.select(lambda run: run["status"] == COMPLETED):
            if os.path.isdir(target_dir(run)) or self.outputs_ready(run):
                self.update(run, status=COLLECTING)
                self.pool.submit(self.collect, run)
            elif time.time() - run["done_at"] > OUTPUTS_TIMEOUT_SECONDS:
                self.update(run, status=FAILED, detail=f"no outputs at {output_dir(run)}")

    def copy_results(self, run):
        target = target_dir(run)
        temporary = f"{target}.tmp"
        shutil.rmtree(temporary, ignore_errors=True)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        source = output_dir(run)
        # copyfile, since the NAS files belong to root and their metadata means nothing here.
        shutil.copytree(source, temporary, copy_function=shutil.copyfile)
        if file_sizes(source) != file_sizes(temporary):
            raise OSError(f"copy of {source} doesn't match the original")
        log_path = os.path.join(job_root(run), "logs", "container.log")
        status_path = os.path.join(job_root(run), "status.json")
        if os.path.exists(log_path):
            shutil.copyfile(log_path, os.path.join(temporary, "run.log"))
        if os.path.exists(status_path):
            shutil.copyfile(status_path, os.path.join(temporary, "hpc_status.json"))
        os.rename(temporary, target)
        if run["backend"] == LOCAL:
            shutil.rmtree(job_root(run), ignore_errors=True)

    def collect(self, run):
        try:
            if not os.path.isdir(target_dir(run)):
                self.copy_results(run)
        except Exception as error:
            attempts = run["collect_attempts"] + 1
            self.record_error(f"collecting {run_name(run)}, attempt {attempts}: {error}")
            if attempts >= MAX_COLLECT_ATTEMPTS:
                self.update(run, status=FAILED, collect_attempts=attempts, detail=f"could not collect: {error}")
            else:
                self.update(run, status=COMPLETED, collect_attempts=attempts)
            return

        self.update(run, status=PLOTTING)
        result = run_command([sys.executable, "-m", "analysis.graph_training", run["batch"], run["experiment"],
                              "--seed", str(run["seed"])], env={**os.environ, "MPLBACKEND": "Agg"})
        if result.returncode != 0:
            self.record_error(f"plotting {run_name(run)}: {result.stderr.strip()[-300:]}")
        self.update(run, status=COLLECTED, plot_failed=result.returncode != 0)

    # ---- polling ----

    def poll(self):
        hpc_ok = True
        try:
            self.sync_hpc()
        except Exception as error:
            hpc_ok = False  # our view of the jobs is stale, so don't act on it
            self.record_error(f"poll: {error}")
        steps = [self.sync_local, self.collect_finished]
        if hpc_ok:
            steps += [self.take_over, self.submit_pending]  # in this order, so a run can start here without being queued
        for step in steps:
            try:
                step()
            except Exception as error:
                self.record_error(f"{step.__name__}: {error}")

    def poll_loop(self):
        while not self.stop.is_set():
            self.next_poll = None
            self.poll()
            self.last_poll = time.time()
            self.next_poll = self.last_poll + POLL_SECONDS
            self.stop.wait(POLL_SECONDS)

    # ---- dashboard ----

    def cell(self, run):
        status = run["status"]
        label = "moving" if run["stealing"] else LABELS.get(status, status)
        parts = [label]
        if run["backend"] == LOCAL:  # most runs are on the HPC, so only this PC's are marked
            parts.append("here")
        if run["started_at"]:
            parts.append(format_duration((run["finished_at"] or time.time()) - run["started_at"]))
        if run["plot_failed"]:
            parts.append("no plot")
        return Text(" ".join(parts), style=STYLES.get(status, "white"))

    def render(self):
        with self.lock:
            # One row per experiment name, and a column per batch and seed.
            table = Table(expand=True, padding=(0, 1))
            table.add_column("")
            columns = [(batch, seed) for batch, module in BATCH_MODULES.items() for seed in module.SEEDS]
            for batch, seed in columns:
                table.add_column(f"{batch.removeprefix('batch_')}/{seed}", no_wrap=True)
            experiments = dict.fromkeys(name for module in BATCH_MODULES.values() for name in module.EXPERIMENTS)
            for experiment in experiments:
                cells = [self.runs.get(run_key(batch, experiment, seed)) for batch, seed in columns]
                table.add_row(experiment, *(self.cell(run) if run else Text("") for run in cells))

            counts = Counter(LABELS.get(run["status"], run["status"]) for run in self.runs.values())
            last_poll = f"{datetime.fromtimestamp(self.last_poll):%H:%M:%S}" if self.last_poll else "-"
            next_poll = f"{max(0, self.next_poll - time.time()):.0f}s" if self.next_poll else "polling now"
            local = self.select(lambda run: run["backend"] == LOCAL and run["status"] == RUNNING)
            local_text = ("off" if not self.use_local else "disabled after a docker failure" if self.local_broken
                          else run_name(local[0]) if local else "idle")
            lines = [
                Text("Columns are batch/seed. " + "  ".join(f"{batch}: {module.OPTIMIZER}" for batch, module in BATCH_MODULES.items())),
                Text(f"Last poll {last_poll}   next poll in {next_poll}"),
                Text("  ".join(f"{name}: {count}" for name, count in sorted(counts.items()))),
                Text(f"This PC: {local_text}"),
            ]
            failed = [run for run in self.runs.values() if run["status"] in FAILURES]
            for run in failed[:MAX_FAILURE_LINES]:
                lines.append(Text(f"{run_name(run)} {run['status']}: {run['detail']}", style="red"))
            if len(failed) > MAX_FAILURE_LINES:
                lines.append(Text(f"... and {len(failed) - MAX_FAILURE_LINES} more failed, see {LOG_PATH}", style="red"))
            lines += [Text(error, style="yellow", overflow="fold") for error in self.errors]
        return Group(table, *lines)

    def run(self):
        threading.Thread(target=self.poll_loop, daemon=True).start()
        interrupted = False
        with Live(self.render(), auto_refresh=False) as live:
            try:
                while not self.finished():
                    live.update(self.render(), refresh=True)
                    time.sleep(REFRESH_SECONDS)
                live.update(self.render(), refresh=True)
            except KeyboardInterrupt:
                interrupted = True
        self.stop.set()
        if interrupted:
            print("Stopped watching. Jobs keep running, run this script again to pick them back up.")
            print("Finishing any copy in progress...")
        self.pool.shutdown(wait=True, cancel_futures=True)
        return interrupted


def file_sizes(root):
    sizes = {}
    for folder, _, names in os.walk(root):
        for name in names:
            path = os.path.join(folder, name)
            sizes[os.path.relpath(path, root)] = os.path.getsize(path)
    return sizes


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


def check_ready(use_local):
    if not os.path.ismount(HPC_OUTPUTS):
        raise SystemExit(f"{HPC_OUTPUTS} is not mounted, so results couldn't be collected.")
    if not shutil.which("docker"):
        raise SystemExit("docker is not installed.")
    if use_local and not os.path.isdir(os.path.join(LOCAL_DATA_ROOT, "cifar-10-batches-py")):
        raise SystemExit(f"--local needs CIFAR-10 in {LOCAL_DATA_ROOT}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--local", action="store_true", help="also run one job at a time on this PC")
    parser.add_argument("--retry-failed", action="store_true", help="run failed or cancelled runs again")
    args = parser.parse_args()

    check_ready(args.local)
    client = connect()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    image = image_name()
    if os.path.exists(STATE_PATH):
        state = load_state(args.retry_failed)
        if state["image"] != image:
            print(f"The code has changed since this sweep started, so it stays on {state['image']}. "
                  f"Delete {STATE_PATH} to start a new sweep with the current code.")
    else:
        build_and_push(image)
        state = new_state(image)

    sweep = Sweep(client, state, args.local)
    interrupted = sweep.run()
    if not interrupted and sweep.select(lambda run: run["status"] in FAILURES):
        sys.exit(1)


if __name__ == "__main__":
    main()
