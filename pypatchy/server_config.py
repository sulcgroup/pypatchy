from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, field
from typing import IO, Any, Union

from .patchy.ensemble_parameter import parameter_value
from .patchy.simulation_specification import ParamSet
from .util import cfg, get_local_dir, get_spec_json


def list_server_configs() -> Generator[PatchyServerConfig]:
    """
    iterates files in the server config directory
    """
    server_config_path = get_local_dir() / "spec_files" / "server_configs"
    assert server_config_path.exists()
    # iter files in directory
    for f in server_config_path.rglob("*.json"):
        yield load_server_settings(f.stem)


def load_server_settings(settings_name: str) -> PatchyServerConfig:
    return PatchyServerConfig(
        name=settings_name,
        **get_spec_json(settings_name, "server_configs")
    )

@dataclass
class PatchyServerConfig:
    """
    dataclass for "server" configuration. includes path to oxDNA directory, the appropriate
    interaction to use for patchy simulations, whether to run as slurm jobs, whether to check
    confs with oxpy before run....
    """
    name: str = field()
    oxdna_path: str = field()
    patchy_format: str = field()
    slurm_bash_flags: dict[str, Any] = field(default_factory=dict)
    slurm_includes: list[str] = field(default_factory=list)
    input_file_params: ParamSet = field(default_factory=list)
    absolute_paths: bool = False
    # despite the name, this does NOT mean "runs on a slurm cluster" - plenty of ensembles
    # with is_slurm=False still run inside a slurm allocation, they just run all of their
    # simulations directly in that one job/process (now that oxpy supports contrib/romano,
    # this is the common case) rather than each simulation submitting its own nested sbatch
    # job. See submits_per_sim_slurm_job().
    is_slurm: bool = False
    # run all of an ensemble's simulations as concurrent oxpy worker processes in this job,
    # spread over the available GPUs, with an NVIDIA MPS daemon letting them share each GPU
    # (see PatchySimulationEnsemble.start_simulations()). for CPU backends the workers run
    # concurrently without MPS
    cuda_mps: bool = False
    # most simulations to run at once on any one GPU. None: the MPS client limit
    # (SimulationManager.MPS_MAX_CLIENTS_PER_GPU)
    max_sims_per_gpu: int | None = None
    # most simulations to run at once in total. None: the number of CPUs available to this process
    max_concurrent_sims: int | None = None
    no_oxpy_check: bool = True
    # the simulation program to build, run and read simulations with (see pypatchy.patchy.engines)
    engine: str = "oxdna"

    # post_init function courtesy of chatGPT
    def __post_init__(self):
        if not isinstance(self.input_file_params, ParamSet):
            self.input_file_params = ParamSet([
                parameter_value(pkey, pval)
                for pkey, pval in (self.input_file_params.items() if isinstance(self.input_file_params, dict)
                                   else self.input_file_params)
            ])
        self.cuda_mps = bool(self.cuda_mps)

    def write_sbatch_params(self, job_name: str, slurm_file: IO):
        slurm_file.write("#!/bin/bash\n") # TODO: other shells?

        # slurm flags
        for flag_key in self.slurm_bash_flags:
            if len(flag_key) > 1:
                slurm_file.write(f"#SBATCH --{flag_key}=\"{self.slurm_bash_flags[flag_key]}\"\n")
            else:
                slurm_file.write(f"#SBATCH -{flag_key} {self.slurm_bash_flags[flag_key]}\n")
        slurm_file.write(f"#SBATCH --job-name=\"{job_name}\"\n")
        slurm_file.write(f"#SBATCH -o run%j.out\n")
        # slurm_file.write(f"#SBATCH -e run{run_oxdna_counter}_%j.err\n")

        # slurm includes ("module load xyz" and the like)
        slurm_file.writelines(line + "\n" for line in self.slurm_includes)

    def is_batched(self) -> bool:
        """
        Whether simulations are run together by a SimulationManager (concurrent worker processes
        in this job) rather than one at a time
        """
        return self.cuda_mps

    def submits_per_sim_slurm_job(self) -> bool:
        """
        Returns whether each simulation should be submitted as its own individual `sbatch`
        job (True), as opposed to running all simulations for this ensemble directly in
        whatever job/process is already running (False) - which is the common case, and is
        orthogonal to whether that job/process itself happens to be inside a slurm allocation.
        """
        return self.is_slurm

    # TODO: slurm library? this project is experiancing mission creep
    def get_slurm_bash_flags(self) -> dict[str, Any]:
        assert self.submits_per_sim_slurm_job(), "Trying to get slurm bash flags for a config that doesn't submit per-sim slurm jobs!"
        return self.slurm_bash_flags

    def get_slurm_n_tasks(self) -> int:
        bashflags = self.get_slurm_bash_flags()
        if "n" in bashflags:
            return bashflags["n"]
        if "ntasks" in bashflags:
            return bashflags["ntasks"]
        return 1  # default value

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "oxdna_path": self.oxdna_path,
            "patchy_format": self.patchy_format,
            "slurm_bash_flags": self.slurm_bash_flags,
            "slurm_includes": self.slurm_includes,
            "input_file_params": self.input_file_params,
            "absolute_paths": self.absolute_paths,
            "is_slurm": bool(self.is_slurm),
            "cuda_mps": bool(self.cuda_mps),
            "max_sims_per_gpu": self.max_sims_per_gpu,
            "max_concurrent_sims": self.max_concurrent_sims,
            "no_oxpy_check": self.no_oxpy_check,
            "engine": self.engine
        }
def get_server_config() -> PatchyServerConfig:
    return load_server_settings(cfg["SETUP"]["server_config"])