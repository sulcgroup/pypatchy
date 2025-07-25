"""
Abstract base class for things that can be used for analysis pipelines
Mostly meant to be extended by `PatchySimulationEnsemble`
"""
from __future__ import annotations
import logging
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import pandas as pd

from .analysis_pipeline import AnalysisPipeline
from .analysis_pipeline_step import AnalysisPipelineStep
from .analysis_data import PipelineData
from ..patchy.simulation_specification import ParamSet
from ..util import get_input_dir


class Analyzable(ABC):
    # -------------- STUFF SPECIFIC TO ANALYSIS ------------- #

    # each "node" of the analysis pipeline graph is a step in the analysis pipeline
    analysis_pipeline: AnalysisPipeline

    # dict to store loaded analysis data
    analysis_data: dict[tuple[AnalysisPipelineStep, ParamSet], PipelineData]
    analysis_file: str

    def __init__(self,
                 analysis_file: str,
                 analysis_pipeline: AnalysisPipeline):
        self.analysis_file = analysis_file
        self.analysis_pipeline = analysis_pipeline
        # construct analysis data dict in case we need it
        self.analysis_data = dict()

    def set_analysis_pipeline(self,
                              src: Union[str, Path],
                              clear_old_pipeline=True,
                              link_file=True):
        """
        Sets the ensemble's analysis pipeline from an existing analysis pipeline file
        Args:
            src: a string or path object indicating the file to set from
            clear_old_pipeline: if set to True, the existing analysis pipleine will be replaced with the pipeline from the file. otherwise, the new pipeline will be appended
            link_file: if set to True, the provided source file will be linked to this ensemble, so changes made to this analysis pipeline will apply to all ensembles that source from that file
        """
        if isinstance(src, str):
            if src.endswith(".pickle"):
                src = get_input_dir() / src
            else:
                src = get_input_dir() / (src + ".pickle")
        if clear_old_pipeline:
            self.analysis_pipeline = AnalysisPipeline()
        try:
            with open(src, "rb") as f:
                self.analysis_pipeline = self.analysis_pipeline + pickle.load(f)
            if link_file:
                self.analysis_file = src
        except FileNotFoundError:
            logging.error(f"No analysis pipeline found at {str(src)}.")
        self.save_pipeline_data()

    def get_analysis_step(self, step: Union[str, AnalysisPipelineStep]) -> AnalysisPipelineStep:
        """
        Alias for PatchySimulationEnsemble.get_pipeline_step
        """
        return self.get_pipeline_step(step)

    def get_pipeline_step(self, step: Union[str, AnalysisPipelineStep]) -> AnalysisPipelineStep:
        """
        Returns a step in the analysis pipeline
        """
        return self.analysis_pipeline.get_pipeline_step(step)

    def has_pipeline(self) -> bool:
        return len(self.analysis_pipeline) != 0

    def show_analysis_pipeline(self):
        return self.analysis_pipeline.draw_pipeline()

    def add_analysis_steps(self, *args):
        """
        Adds steps to the analysis pipeline
        """
        if isinstance(args[0], AnalysisPipeline):
            new_steps = args[0]
        else:
            new_steps = AnalysisPipeline(args[0], *args[1:])
            # if the above line didn't work
        newPipes = [a for a in args if isinstance(a, tuple)]
        newSteps = [a for a in args if issubclass(type(a), AnalysisPipelineStep)]
        if new_steps.num_pipes() != len(newPipes) or new_steps.num_pipeline_steps() != len(newSteps):
            self.analysis_pipeline = self.analysis_pipeline.extend(newSteps, newPipes)
            self.save_pipeline_data()
        elif new_steps not in self.analysis_pipeline:
            self.get_logger().info(f"Adding {len(new_steps)} steps "
                                   f"and {len(new_steps.pipeline_graph.edges)} pipes to the analysis pipeline")
            self.analysis_pipeline = self.analysis_pipeline + new_steps
            self.save_pipeline_data()
        else:
            self.get_logger().info("The analysis pipeline you passed is already present")

    def link_analysis_pipeline(self, other: Analyzable):
        """
        links this ensembles's analysis pipeline to another ensemble
        """
        if len(self.analysis_pipeline) != 0:
            self.get_logger().warning("Error: should not link from existing analysis pipeline! "
                                      "Use `clear_analysis_pipeline() to clear pipeline and try again.")
            return
        self.analysis_pipeline = other.analysis_pipeline
        self.analysis_file = other.analysis_file
        self.save_pipeline_data()

    def missing_analysis_data(self,
                              step: Union[AnalysisPipelineStep,
                                          str]) -> pd.DataFrame:
        """
        Returns: a Pandas dataframe showing which analysis data are missing
        """
        if isinstance(step, str):
            return self.missing_analysis_data(self.analysis_pipeline.name_map[step])
        else:
            return ~self.analysis_status().loc[~self.analysis_status()[step.name]]

    def is_data_loaded(self,
                       sim: ParamSet,
                       step: AnalysisPipelineStep,
                       time_steps) -> bool:
        """
        Returns: true if we have data cached for the given simulation and step, false otherwise
        """
        return not self.is_nocache() and (step, sim,) in self.analysis_data and self.analysis_data[
            (step, sim)].matches_trange(time_steps)

    def get_cached_analysis_data(self,
                                 sim: ParamSet,
                                 step: AnalysisPipelineStep) -> PipelineData:
        """
        Returns: cached pipeline data for the given step
        """
        assert (step, sim) in self.analysis_data, f"No cached data for {sim} step {step}"
        self.get_logger().info("Data already loaded!")
        return self.analysis_data[(step, sim)]  # i don't care enough to load partial data

    @abstractmethod
    def analysis_status(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def save_pipeline_data(self):
        pass

    @abstractmethod
    def get_logger(self) -> logging.Logger:
        pass

    @abstractmethod
    def is_nocache(self) -> bool:
        pass

    def clear_pipeline(self):
        """
        deletes all steps from the analysis pipeline
        """
        self.analysis_pipeline = AnalysisPipeline()
