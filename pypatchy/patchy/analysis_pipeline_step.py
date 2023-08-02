from __future__ import annotations

import pickle
import subprocess
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Union, IO, Any

import pandas as pd

from .simulation_specification import PatchySimulation
from .ensemble_parameter import EnsembleParameter, ParameterValue


class PipelineDataType(Enum):
    # raw data from trajectory.dat - currently not used
    PIPELINE_DATATYPE_RAWDATA = 0
    # data from an observable
    PIPELINE_DATATYPE_OBSERVABLE = 1
    # data from a pandas dataframe
    PIPELINE_DATATYPE_DATAFRAME = 2
    # list of graphs
    PIPELINE_DATATYPE_GRAPH = 3


PipelineData = Any  # todo: detail!


class AnalysisPipelineStep(ABC):
    # unique ID
    # idx: int

    # the name of this step on the analysis pipeline
    name: str

    # specifier for data that will be taken as input by this object
    input_data: PipelineDataType

    # specifier for output data.
    output_data: PipelineDataType

    # steps immediately feeding into this step

    # interval in timesteps between input data points
    input_tstep: int

    # interval in timesteps between input data points
    output_tstep: int

    def __init__(self,
                 step_name: str,
                 input_tstep: Union[int, None] = None,
                 output_tstep: Union[int, None] = None):
        self.name = step_name  # unique name, not class name
        # self.idx = -1
        self.input_tstep = input_tstep
        self.output_tstep = output_tstep
        self.config_io(input_tstep=self.input_tstep, output_tstep=self.output_tstep)

    def __str__(self):
        return self.name

    @abstractmethod
    def load_cached_files(self, f: Path) -> PipelineData:
        """
        loads data from a file object (from `open(filepath)`)
        """
        pass

    # def exec_step_slurm(self,
    #                     script_dir_path: Path,
    #                     slurm_bash_flags: dict[str: str],
    #                     slurm_includes: list[str],
    #                     data_sources: Union[tuple[Path], list[Path]],
    #                     cache_file: Path) -> int:
    #     """
    #     executes this step in the analysis pipeline,
    #     using slurm
    #     Parameters:
    #         :param script_dir_path a directory to put a temporary bash script (
    #         :param slurm_bash_flags
    #         :param slurm_includes
    #         :param data_sources
    #         :param cache_file
    #     """
    #     assert script_dir_path.exists()
    #     script_path = script_dir_path / f"{self.name}.sh"
    #     with open(script_path, "w") as f:
    #         f.write("!/bin/bash\n")
    #         # write slurm flags (server-specific)
    #         for flag_key in slurm_bash_flags:
    #             if len(flag_key) > 1:
    #                 f.write(f"#SBATCH --{flag_key}=\"{slurm_bash_flags[flag_key]}\"\n")
    #             else:
    #                 f.write(f"#SBATCH -{flag_key} {slurm_bash_flags[flag_key]}\n")
    #         for incl in slurm_includes:
    #             f.write(f"{incl}\n")
    #         f.write("source activate polycubes\n")
    #         f.write("python <<EOF\n")
    #         self.write_steps_slurm(f, data_sources, cache_file)
    #         f.write("EOF\n")
    #
    #     # submit slurm job
    #     result = subprocess.run(['sbatch', script_path], stdout=subprocess.PIPE)
    #     # get job id
    #     job_id = int(result.stdout.decode().split(' ')[-1].strip())  # extract the job id from output
    #     return job_id

    # def write_steps_slurm(self,
    #                       f: IO,
    #                       data_sources: tuple[Path],
    #                       cache_file: Path):
    #     f.write(self.get_py_steps_slurm(data_sources, cache_file))

    # @abstractmethod
    # def get_py_steps_slurm(self, data_sources: tuple[Path], cache_file: Path):
    #     pass

    @abstractmethod
    def data_matches_trange(self, data: PipelineData, trange: range) -> bool:
        pass

    @abstractmethod
    def exec(self, *args: Union[PipelineData, AnalysisPipelineStep]) -> PipelineData:
        pass

    def get_cache_file_name(self) -> str:
        if self.get_output_data_type() == PipelineDataType.PIPELINE_DATATYPE_GRAPH:
            return f"{self.name}.pickle"
        else:
            return f"{self.name}.csv"

    def cache_data(self, data: PipelineData, file_path: Path):
        if self.get_output_data_type() == PipelineDataType.PIPELINE_DATATYPE_DATAFRAME:
            data.to_csv(file_path)
        elif self.get_output_data_type() == PipelineDataType.PIPELINE_DATATYPE_GRAPH:
            with open(file_path, "wb") as f:
                pickle.dump(data, f)
        else:
            assert False

    @abstractmethod
    def get_input_data_type(self) -> PipelineDataType:
        pass

    @abstractmethod
    def get_output_data_type(self) -> PipelineDataType:
        pass

    def config_io(self, input_tstep=None, output_tstep=None):
        """
        Configures the input and output time intervals
        """
        if input_tstep is not None:
            self.input_tstep = input_tstep
        if output_tstep is not None:
            self.output_tstep = output_tstep

        # if there's a specified input tstep but not output tstep, assume they're the same
        if self.output_tstep is None and self.input_tstep is not None:
            self.output_tstep = self.input_tstep

        # if the user specifies a smaller output timestep than input timestep,
        # assume it's an interval
        if self.input_tstep is not None and self.output_tstep is not None:
            if self.output_tstep < self.input_tstep:
                self.output_tstep *= self.input_tstep
            assert self.output_tstep % self.input_tstep == 0


class AnalysisPipelineHead(AnalysisPipelineStep, ABC):
    """
    Class for any "head" node of the analysis pipeline.
    Note that there's nothing stopping even a connected graph
    of the pipeline from having multiple "heads"
    """
    def __init__(self,
                 step_name: str,
                 input_tstep: int,
                 output_tstep: Union[int, None] = None):
        super().__init__(step_name, input_tstep, output_tstep)
        if output_tstep is None:
            self.output_tstep = input_tstep

    @abstractmethod
    def get_data_in_filenames(self) -> list[str]:
        pass


class AggregateAnalysisPipelineStep(AnalysisPipelineStep, ABC):
    """
    Class for analysis pipeline steps that aggregate data from multiple
    simulations, e.g. average yield over duplicates
    """

    def __init__(self, step_name: str,
                 input_tstep: int,
                 output_tstep: int,
                 aggregate_over: tuple[EnsembleParameter, ...]):
        super().__init__(step_name, input_tstep, output_tstep)
        self.params_aggregate_over = aggregate_over

    def get_input_data_params(self, sim):
        if isinstance(sim, PatchySimulation):
            this_step_param_specs: tuple[ParameterValue] = tuple(sim.param_vals)
        else:
            this_step_param_specs = sim
        return (param for param in this_step_param_specs if param not in self.params_aggregate_over)


PipelineStepDescriptor = Union[AnalysisPipelineStep, int, str]
