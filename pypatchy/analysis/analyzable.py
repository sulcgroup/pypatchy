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
                 analysis_pipeline: AnalysisPipeline):
        """
        :param analysis_file: path to analysis file
        :param analysis_pipeline: analysis pipeline to use for this dataset
        """
        self.analysis_pipeline = analysis_pipeline
        # construct analysis data dict in case we need it
        self.analysis_data = dict()

    def set_analysis_pipeline(self,
                              src: Union[str, Path],
                              clear_old_pipeline=True,
                              link_file=True):
        """
        Sets the ensemble's analysis pipeline from an existing analysis pipeline file
        :param src: a string or path object indicating the file to set from
        :param clear_old_pipeline: if set to True, the existing analysis pipleine will be replaced with the pipeline from the file. otherwise, the new pipeline will be appended
        :param link_file: if set to True, the provided source file will be linked to this ensemble, so changes made to this analysis pipeline will apply to all ensembles that source from that file
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
        Alias for get_pipeline_step
        :param step: step to access
        :return: a step in the analysis pipeline
        """
        return self.get_pipeline_step(step)

    def get_pipeline_step(self, step: Union[str, AnalysisPipelineStep]) -> AnalysisPipelineStep:
        """
        Returns a step in the analysis pipeline
        :param step: step to access
        :return: a step in the analysis pipeline
        """
        return self.analysis_pipeline.get_pipeline_step(step)

    def has_pipeline(self) -> bool:
        """
        check if the pipeline has any steps
        :return: True if the pipeline has any steps, False otherwise
        """
        return len(self.analysis_pipeline) != 0

    def show_analysis_pipeline(self):
        """
        Draw the analysis pipeline, by calling draw_pipeline on the analysis pipeline associated with these data
        :return: a draw.Drawing object showing the steps on the pipeline
        """
        return self.analysis_pipeline.draw_pipeline()

    def add_analysis_steps(self, *args: Union[AnalysisPipeline, tuple[str, str], AnalysisPipelineStep]):
        """
        Adds steps to the analysis pipeline. Can pass an existing AnalysisPipeline object or a list of Steps
        and links. In the latter case, the steps and links are passed in the same args list
        part of the same args list
        :param args: steps to add
        """
        # if we've passed an existing pipeline, use it as a starting point
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
        :param other: another set of analyzable data to link with this Analyzable's pipeline
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
        Figure out which data are missing for a given step.
        :param step: an AnalysisPipelineStep object or string indicating a step in the pipeline
        :returns: a Pandas dataframe showing which analysis data are missing
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
        Checks is data are already loaded for a given subset of the analyziable data and step in the pipeline
        :param sim: a set of parameters identifying s aubset of the analyzable data
        :param step: the analysis step we're loading
        :param time_steps: timesteps to search for data
        :return: true if we have data cached for the given simulation and step, false otherwise
        """
        return not self.is_nocache() and (step, sim,) in self.analysis_data and self.analysis_data[
            (step, sim)].matches_trange(time_steps)

    def get_cached_analysis_data(self,
                                 sim: ParamSet,
                                 step: AnalysisPipelineStep) -> PipelineData:
        """
        Load cached data for a subset of the analyzable system.
        :param sim: subset of the analyzable to load cache for
        :param step: step for which to load cached data
        :return: cached pipeline data for the given step
        """
        assert (step, sim) in self.analysis_data, f"No cached data for {sim} step {step}"
        self.get_logger().info("Data already loaded!")
        return self.analysis_data[(step, sim)]  # i don't care enough to load partial data

    @abstractmethod
    def analysis_status(self) -> pd.DataFrame:
        """
        to be implemented in subclasses
        :return: the status of the analysis, as a Pandas dataframe
        """
        pass

    @abstractmethod
    def save_pipeline_data(self):
        """
        to be implemented in subclasses
        """
        pass

    @abstractmethod
    def get_logger(self) -> logging.Logger:
        """
        to be implemented in subclasses
        :return: a logging.Logger object to which to log stuff
        """
        pass

    @abstractmethod
    def is_nocache(self) -> bool:
        """
        to be implemented in subclasses
        :return: whether we're skipping caching
        """
        pass

    def clear_pipeline(self):
        """
        deletes all steps from the analysis pipeline
        """
        self.analysis_pipeline = AnalysisPipeline()
