"""
Runs many HOOMD-engine stages at once in this job: a pool of stage-runner subprocesses spread over the available
GPUs, sharing each GPU through a dedicated NVIDIA MPS daemon (HOOMDEngine.run_batch()).

Each worker is `python -m pypatchy.patchy.hoomd_backend.run_stage <stage dir>`, with CUDA_VISIBLE_DEVICES set to
the UUID of the one GPU it should use. UUIDs name the same GPU whether or not MPS is in use, and whatever
CUDA_VISIBLE_DEVICES this job was given (e.g. by slurm). Stages whose `device` is "cpu" just run concurrently.
"""
from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
import time
import uuid
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Union

from .run_stage import LOG_FILE, read_spec

# NVIDIA MPS accepts a limited number of client contexts per GPU (48 on Volta and newer)
MPS_MAX_CLIENTS_PER_GPU = 48


# ---------------------------------------------------------------------------------------------- GPUs
def query_gpus() -> list[tuple[str, str]]:
    """
    :return: (index, UUID) of every GPU on this node, from nvidia-smi; empty if there's no nvidia-smi or no GPU
    """
    if shutil.which("nvidia-smi") is None:
        return []
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                            capture_output=True, text=True)
    if result.returncode != 0:
        return []
    gpus = []
    for line in result.stdout.splitlines():
        if line.strip():
            index, gpu_uuid = (s.strip() for s in line.split(",", 1))
            gpus.append((index, gpu_uuid))
    return gpus


def visible_gpu_uuids(gpus: list[tuple[str, str]], cuda_visible_devices: Union[str, None]) -> list[str]:
    """
    :param gpus: (index, UUID) of the node's GPUs (see query_gpus)
    :param cuda_visible_devices: this job's CUDA_VISIBLE_DEVICES, or None if unset
    :return: UUIDs of the GPUs this job may use, in CUDA's order. Entries can be indices or (abbreviated) UUIDs;
        as in CUDA, the list stops at the first entry that names no GPU
    """
    if cuda_visible_devices is None:
        return [gpu_uuid for _, gpu_uuid in gpus]
    visible = []
    for entry in (e.strip() for e in cuda_visible_devices.split(",")):
        match = next((gpu_uuid for index, gpu_uuid in gpus
                      if entry == index or (entry and gpu_uuid.startswith(entry))), None)
        if match is None:
            break
        visible.append(match)
    return visible


# ---------------------------------------------------------------------------------------------- MPS
class MPSDaemon:
    """
    Context manager for an nvidia-cuda-mps-control daemon. Starts a dedicated one (unique pipe and log
    directories, so batches never share daemons), unless CUDA_MPS_PIPE_DIRECTORY already names a daemon that
    answers, which is used as-is and left running. `env` holds the variables workers need to use it; it is empty
    if MPS couldn't be set up (workers then just share the GPUs by time-slicing).
    """

    def __init__(self):
        self.env: dict[str, str] = {}
        self.owned_daemon: Union[None, tuple[str, str]] = None

    @staticmethod
    def control(command: Union[str, None], pipe_dir: str, log_dir: str, **kwargs) -> subprocess.CompletedProcess:
        env = {**os.environ, "CUDA_MPS_PIPE_DIRECTORY": pipe_dir, "CUDA_MPS_LOG_DIRECTORY": log_dir}
        if command is None:
            return subprocess.run(["nvidia-cuda-mps-control", "-d"], env=env, capture_output=True, text=True,
                                  **kwargs)
        return subprocess.run(["nvidia-cuda-mps-control"], input=command + "\n", env=env, capture_output=True,
                              text=True, **kwargs)

    @classmethod
    def responds(cls, pipe_dir: str, log_dir: str) -> bool:
        try:
            return cls.control("get_default_active_thread_percentage", pipe_dir, log_dir, timeout=10).returncode == 0
        except (subprocess.SubprocessError, OSError):
            return False

    def __enter__(self) -> MPSDaemon:
        if shutil.which("nvidia-cuda-mps-control") is None:
            warnings.warn("nvidia-cuda-mps-control isn't on PATH; running HOOMD workers without MPS")
            return self
        existing = os.environ.get("CUDA_MPS_PIPE_DIRECTORY")
        if existing:
            log_dir = os.environ.get("CUDA_MPS_LOG_DIRECTORY", existing)
            if self.responds(existing, log_dir):
                self.env = {"CUDA_MPS_PIPE_DIRECTORY": existing, "CUDA_MPS_LOG_DIRECTORY": log_dir}
                return self
        tag = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
        pipe_dir, log_dir = f"/tmp/mps-pipe_{tag}", f"/tmp/mps-log_{tag}"
        os.makedirs(pipe_dir)
        os.makedirs(log_dir)
        result = self.control(None, pipe_dir, log_dir)
        # `-d` returns as soon as the daemon forks: check it's actually answering
        if result.returncode != 0 or not self.responds(pipe_dir, log_dir):
            warnings.warn(f"Couldn't start nvidia-cuda-mps-control ({result.stderr.strip() or 'not responding'}; "
                          f"see {log_dir}); running HOOMD workers without MPS")
            self.control("quit", pipe_dir, log_dir, timeout=30)
            shutil.rmtree(pipe_dir, ignore_errors=True)
            return self
        self.owned_daemon = (pipe_dir, log_dir)
        self.env = {"CUDA_MPS_PIPE_DIRECTORY": pipe_dir, "CUDA_MPS_LOG_DIRECTORY": log_dir}
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.owned_daemon is not None:
            pipe_dir, log_dir = self.owned_daemon
            try:
                self.control("quit", pipe_dir, log_dir, timeout=60)
            except (subprocess.SubprocessError, OSError) as e:
                warnings.warn(f"Failed to stop the MPS daemon in {pipe_dir}: {e}")
            shutil.rmtree(pipe_dir, ignore_errors=True)
            self.owned_daemon = None
        return False


# ---------------------------------------------------------------------------------------------- the pool
@dataclass
class StageJob:
    stage_dir: Path
    device: str  # "gpu" or "cpu", from the stage's parameters
    gpu: Union[str, None] = None  # UUID of the GPU it runs on
    process: Union[subprocess.Popen, None] = field(default=None, repr=False)


def run_stage_dirs(stage_dirs: list[Path],
                   max_concurrent: Union[int, None] = None,
                   max_per_gpu: Union[int, None] = None,
                   use_mps: bool = True,
                   launch_interval: float = 0.5,
                   log: Callable[[str], None] = print,
                   gpus: Union[list[str], None] = None) -> dict[Path, int]:
    """
    Runs HOOMD stages (built stage directories) concurrently, blocking until all of them finish.

    :param stage_dirs: stage directories to run
    :param max_concurrent: most stages to run at once; None: the number of CPUs available to this process
    :param max_per_gpu: most stages to run at once on one GPU; None: the MPS client limit when MPS is in use,
        otherwise no limit beyond max_concurrent
    :param use_mps: share each GPU through an MPS daemon (see MPSDaemon)
    :param launch_interval: seconds between launching GPU workers, so their CUDA contexts aren't all created at once
    :param log: where to report progress
    :param gpus: UUIDs of the GPUs to use; None: every GPU visible to this job
    :return: each stage directory's runner exit code (0 = the runner finished)
    """
    jobs = [StageJob(Path(d), read_spec(d)["params"]["device"]) for d in stage_dirs]
    if max_concurrent is None:
        max_concurrent = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    gpu_jobs = [j for j in jobs if j.device == "gpu"]
    if gpu_jobs:
        if gpus is None:
            gpus = visible_gpu_uuids(query_gpus(), os.environ.get("CUDA_VISIBLE_DEVICES"))
        if not gpus:
            raise RuntimeError(f"{len(gpu_jobs)} HOOMD stages run on the GPU, but no GPU is visible to this job")

    exit_codes: dict[Path, int] = {}
    mps = MPSDaemon() if (gpu_jobs and use_mps) else None
    with (mps if mps is not None else contextlib.nullcontext()):
        mps_env = mps.env if mps is not None else {}
        if max_per_gpu is None and mps_env:
            max_per_gpu = MPS_MAX_CLIENTS_PER_GPU
        log(f"Running {len(jobs)} HOOMD stages: at most {max_concurrent} at once"
            + (f", on {len(gpus)} GPU(s) (at most {max_per_gpu or 'unlimited'} per GPU, "
               f"{'with' if mps_env else 'without'} MPS)" if gpu_jobs else ""))
        queue = list(jobs)
        running: list[StageJob] = []
        try:
            while queue or running:
                for job in [j for j in running if j.process.poll() is not None]:
                    running.remove(job)
                    exit_codes[job.stage_dir] = job.process.returncode
                    if job.process.returncode != 0:
                        log(f"HOOMD stage {job.stage_dir} failed (exit code {job.process.returncode}); "
                            f"see {job.stage_dir / LOG_FILE}")
                launched_gpu = False
                while queue and len(running) < max_concurrent:
                    job = queue[0]
                    if job.device == "gpu":
                        load = {g: sum(1 for j in running if j.gpu == g) for g in gpus}
                        gpu = min(gpus, key=lambda g: load[g])
                        if max_per_gpu is not None and load[gpu] >= max_per_gpu:
                            break  # every GPU is full; wait
                        job.gpu = gpu
                    queue.pop(0)
                    launch_job(job, mps_env if job.device == "gpu" else {})
                    running.append(job)
                    if job.device == "gpu":
                        launched_gpu = True
                        break  # stagger GPU launches
                time.sleep(launch_interval if launched_gpu else 0.2)
        finally:
            # interrupted (KeyboardInterrupt, SIGTERM-as-exception, ...): don't leave workers on the GPUs
            for job in running:
                if job.process.poll() is None:
                    job.process.terminate()
            for job in running:
                try:
                    job.process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    job.process.kill()
                    job.process.wait()
                exit_codes.setdefault(job.stage_dir, job.process.returncode)
    failed = [d for d, code in exit_codes.items() if code != 0]
    log(f"HOOMD batch finished: {len(exit_codes) - len(failed)} of {len(jobs)} stages ran to completion")
    return exit_codes


def launch_job(job: StageJob, extra_env: dict[str, str]):
    env = {**os.environ, **extra_env}
    if job.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = job.gpu
    with (job.stage_dir / LOG_FILE).open("w") as log:
        job.process = subprocess.Popen([sys.executable, "-m", "pypatchy.patchy.hoomd_backend.run_stage",
                                        str(job.stage_dir)],
                                       stdout=log, stderr=subprocess.STDOUT, cwd=job.stage_dir, env=env)
