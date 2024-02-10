from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, IO, Union

from .patchy.ensemble_parameter import parameter_value
from .patchy.simulation_specification import ParamSet
from .util import get_spec_json, cfg


def load_server_settings(settings_name: str) -> PatchyServerConfig:
    return PatchyServerConfig(
        **get_spec_json(settings_name, "server_configs")
    )


@dataclass
class PatchyServerConfig:
    oxdna_path: str
    patchy_format: str
    slurm_bash_flags: dict[str, Any] = field(default_factory=dict)
    slurm_includes: list[str] = field(default_factory=list)
    input_file_params: ParamSet = field(default_factory=ParamSet)
    absolute_paths: bool = False
    is_slurm: bool = False
    cuda_mps: bool = False

    # post_init function courtesy of chatGPT
    def __post_init__(self):
        if not isinstance(self.input_file_params, ParamSet):
            self.input_file_params = ParamSet([
                parameter_value(pkey, pval)
                for pkey, pval in (self.input_file_params.items() if isinstance(self.input_file_params, dict)
                                   else self.input_file_params)
            ])


    def write_sbatch_params(self, job_name: str, slurm_file: IO):
        slurm_file.write("#!/bin/bash\n") # TODO: other shells?

        # slurm flags
        for flag_key in self.slurm_bash_flags:
            if len(flag_key) > 1:
                slurm_file.write(f"#SBATCH --{flag_key}=\"{self.slurm_bash_flags[flag_key]}\"\n")
            else:
                slurm_file.write(f"#SBATCH -{flag_key} {self.slurm_bash_flags[flag_key]}\n")
        run_oxdna_counter = 1
        slurm_file.write(f"#SBATCH --job-name=\"{job_name}\"\n")
        slurm_file.write(f"#SBATCH -o run%j.out\n")
        # slurm_file.write(f"#SBATCH -e run{run_oxdna_counter}_%j.err\n")

        # slurm includes ("module load xyz" and the like)
        for line in self.slurm_includes:
            slurm_file.write(line + "\n")

    def is_batched(self) -> bool:
        batch = self.cuda_mps
        assert not batch or self.absolute_paths, "Cannot run using MPS without absolute paths!!!"
        return batch

    def is_server_slurm(self) -> bool:
        """
        Returns whether the server is a slurm server. Defaults to true for legacy reasons.
        """
        return len(self.slurm_bash_flags) > 0

    # TODO: slurm library? this project is experiancing mission creep
    def get_slurm_bash_flags(self) -> dict[str, Any]:
        assert self.is_server_slurm(), "Trying to get slurm bash flags for a non-slurm setup!"
        return self.slurm_bash_flags

    def get_slurm_n_tasks(self) -> int:
        bashflags = self.get_slurm_bash_flags()
        if "n" in bashflags:
            return bashflags["n"]
        if "ntasks" in bashflags:
            return bashflags["ntasks"]
        return 1  # default value


def get_server_config() -> PatchyServerConfig:
    return load_server_settings(cfg["SETUP"]["server_config"])


