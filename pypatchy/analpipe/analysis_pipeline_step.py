from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from pypatchy.patchy.simulation_specification import PatchySimulation
from pypatchy.patchy.ensemble_parameter import EnsembleParameter, ParameterValue
from .analysis_data import PipelineData, PipelineDataType

import drawsvg as draw


class AnalysisPipelineStep(ABC):
    """
    Base class for a step in a Patchy Particle Data analysis pipeline
    """
    # the name of this step on the analpipe pipeline
    name: str

    # steps immediately feeding into this step

    # interval in timesteps between input data points
    input_tstep: int

    # interval in timesteps between input data points
    output_tstep: int

    # flag which allows the user to temporarily disable loading cached data
    force_recompute: bool

    def __init__(self,
                 step_name: str,
                 input_tstep: Union[int, None] = None,
                 output_tstep: Union[int, None] = None):
        self.name = step_name  # unique name, not class name
        # self.idx = -1
        self.input_tstep = input_tstep
        self.output_tstep = output_tstep
        self.config_io(input_tstep=self.input_tstep, output_tstep=self.output_tstep)
        self.force_recompute = False

    def __str__(self):
        return self.name

    @abstractmethod
    def load_cached_files(self, f: Path) -> PipelineData:
        """
        loads data from a file object (from `open(filepath)`)
        """
        pass

    @abstractmethod
    def exec(self, *args: Union[PipelineData, AnalysisPipelineStep]) -> PipelineData:
        """
        Executes this analysis step

        Returns:
            data! (in a PipelineData wrapper object)
        """
        pass

    def get_cache_file_name(self) -> str:
        """
        Returns:
            the filename for where this step will cache data
        """
        if self.get_output_data_type() == PipelineDataType.PIPELINE_DATATYPE_GRAPH:
            return f"{self.name}.pickle"
        else:
            return f"{self.name}.h5"

    def cache_data(self, data: PipelineData, file_path: Path):
        if self.get_output_data_type() == PipelineDataType.PIPELINE_DATATYPE_DATAFRAME:
            with pd.HDFStore(str(file_path)) as f:
                f["data"] = data.get()
                f["trange"] = pd.Series(data.trange())
        elif self.get_output_data_type() in [PipelineDataType.PIPELINE_DATATYPE_GRAPH,
                                             PipelineDataType.PIPELINE_DATATYPE_RAWDATA]:
            with open(file_path, "wb") as f:
                pickle.dump(data, f)
        else:
            raise Exception("Invalid data type!!!!")

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

    @abstractmethod
    def get_output_data_type(self) -> PipelineDataType:
        """
        Override this method to tell stuff what kind of data this step will spit out
        It's assumed that whatever it does will be constant

        Returns:
            the data type produced by this pipeline, represented by an enum
        """
        pass

    def draw(self) -> tuple[tuple[int, int], draw.Group]:
        """
        This method isn't abstract but it's highly recommended that you override it
        """
        g = draw.Group()
        w = 180
        y = 0
        # draw step name
        g.append(draw.Rectangle(0, y, w, 16, stroke="black", stroke_width=1, fill="tan"))
        g.append(draw.Text(f"Name: {self.name}", text_anchor='middle', font_size=12, x=w/2, y=14))
        y += 16
        # draw step input
        g.append(draw.Rectangle(0, y, w/2, 16, stroke="black", stroke_width=1, fill="beige"))
        tstep = "{:e}".format(self.input_tstep)
        g.append(draw.Text(f"Input Freq: {tstep}", text_anchor='begin', font_size=7, x=1, y=y+7))
        # if not isinstance(self, AnalysisPipelineHead):
        #     input_dt = ["Raw", "Observable", "DataFrame", "Graph"][self.input_data.value]
        #     g.append(draw.Text(f"Input Data Type: {input_dt}", text_anchor='begin', font_size="14", x=0, y=28))
        # draw step output
        g.append(draw.Rectangle(w/2, y, w/2, 16, stroke="black", stroke_width=1, fill="beige"))
        tstep = "{:e}".format(self.output_tstep)
        g.append(draw.Text(f"Output Freq: {tstep}", text_anchor='end', font_size=7, x=w-1, y=y+7))
        output_dt = ["Raw", "Observable", "DataFrame", "Graph"][self.get_output_data_type().value]
        g.append(draw.Text(f"Output DT: {output_dt}", text_anchor='end', font_size=7, x=w-1, y=y+15))
        y += 16
        return (w, y), g

class AnalysisPipelineHead(AnalysisPipelineStep, ABC):
    """
    Class for any "head" node of the analpipe pipeline.
    Note that there's nothing stopping even a connected graph
    of the pipeline from having multiple "heads"
    Note that while I haven't explicitly stated it here, AnalysisPipelineHead.exec
    the first two positional arguements to be a PatchySimulationEnsemble and a PatchySimulation
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
        """
        Returns:
            a list of files containing the raw data that will be used by this step
        """
        pass


class AggregateAnalysisPipelineStep(AnalysisPipelineStep, ABC):
    """
    Class for analpipe pipeline steps that aggregate data from multiple
    simulations, e.g. average yield over duplicates
    """

    def __init__(self, step_name: str,
                 input_tstep: int,
                 output_tstep: int,
                 aggregate_over: tuple[EnsembleParameter, ...]):
        super().__init__(step_name, input_tstep, output_tstep)
        self.params_aggregate_over = aggregate_over

    def get_input_data_params(self, sim) -> tuple[ParameterValue, ...]:
        """
        .........
        """
        if isinstance(sim, PatchySimulation):
            this_step_param_specs: tuple[ParameterValue] = tuple(sim.param_vals)
        else:
            this_step_param_specs = sim
        return tuple(param for param in this_step_param_specs if param not in self.params_aggregate_over)


PipelineStepDescriptor = Union[AnalysisPipelineStep, int, str]
