from __future__ import annotations

import pickle
import subprocess
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Union, IO, Any

import pandas as pd

from .ensemble_parameter import EnsembleParameter
from .simulation_specification import PatchySimulation


class PipelineDataTypeEnum(Enum):
    # raw data from trajectory.dat - currently not used
    PIPELINE_DATATYPE_RAWDATA = 0
    # data from an observable
    PIPELINE_DATATYPE_OBSERVABLE = 1
    # data from a pandas dataframe
    PIPELINE_DATATYPE_DATAFRAME = 2
    # list of graphs
    PIPELINE_DATATYPE_GRAPH = 3


PipelineDataType = Any  # todo: detail!


class AnalysisPipelineStep(ABC):
    # the name of this step on the analysis pipeline
    name: str

    # specifier for data that will be taken as input by this object
    input_data: PipelineDataTypeEnum

    # specifier for output data.
    output_data: PipelineDataTypeEnum

    # steps immediately feeding into this step
    previous_steps: tuple[AnalysisPipelineStep]

    # interval in timesteps between input data points
    input_tstep: int

    # interval in timesteps between input data points
    output_tstep: int

    def __init__(self,
                 step_name: str,
                 input_tstep: int,
                 output_tstep: int,
                 previous_steps: tuple[AnalysisPipelineStep]):
        self.name = step_name  # unique name, not class name
        self.idx = -1
        self.input_tstep = input_tstep
        self.output_tstep = output_tstep
        self.previous_steps = previous_steps

    @abstractmethod
    def load_cached_files(self, f: IO) -> PipelineDataType:
        """
        loads data from a file object (from `open(filepath)`)
        """
        pass

    def exec_step_slurm(self,
                        script_dir_path: Path,
                        slurm_bash_flags: dict[str: str],
                        slurm_includes: list[str],
                        data_sources: Union[tuple[Path], list[Path]],
                        cache_file: Path) -> int:
        assert script_dir_path.exists()
        script_path = script_dir_path / f"{self.name}.sh"
        with open(script_path, "w") as f:
            f.write("!/bin/bash\n")
            # write slurm flags (server-specific)
            for flag_key in slurm_bash_flags:
                if len(flag_key) > 1:
                    f.write(f"#SBATCH --{flag_key}=\"{slurm_bash_flags[flag_key]}\"\n")
                else:
                    f.write(f"#SBATCH -{flag_key} {slurm_bash_flags[flag_key]}\n")
            for incl in slurm_includes:
                f.write(f"{incl}\n")
            f.write("source activate polycubes\n")
            f.write("python <<EOF\n")
            self.write_steps_slurm(f, data_sources, cache_file)
            f.write("EOF\n")

        # submit slurm job
        result = subprocess.run(['sbatch', script_path], stdout=subprocess.PIPE)
        # get job id
        job_id = int(result.stdout.decode().split(' ')[-1].strip())  # extract the job id from output
        return job_id

    @abstractmethod
    def can_parallelize(self):
        pass

    @abstractmethod
    def write_steps_slurm(self,
                          f: IO,
                          data_sources: tuple[Path],
                          cache_file: Path):
        pass

    @abstractmethod
    def data_matches_trange(self, data: PipelineDataType, trange: range) -> bool:
        pass

    @abstractmethod
    def exec(self, din: Union[PipelineDataType, AnalysisPipelineStep]) -> PipelineDataType:
        pass

    @abstractmethod
    def get_cache_file_name(self) -> str:
        pass

    def cache_data(self, data: PipelineDataType, file_path: Path):
        if self.get_output_data_type() == PipelineDataTypeEnum.PIPELINE_DATATYPE_DATAFRAME:
            data.to_csv(file_path)
        elif self.get_output_data_type() == PipelineDataTypeEnum.PIPELINE_DATATYPE_GRAPH:
            with open(file_path, "w") as f:
                pickle.dump(data, f)
        else:
            assert False

    @abstractmethod
    def get_input_data_type(self) -> PipelineDataTypeEnum:
        pass

    @abstractmethod
    def get_output_data_type(self) -> PipelineDataTypeEnum:
        pass
