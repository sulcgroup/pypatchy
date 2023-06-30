from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Union

import networkx as nx

from pypatchy.patchy.analysis_pipeline_step import AnalysisPipelineStep


def analysis_step_idx(step: Union[int, AnalysisPipelineStep]) -> int:
    return step if isinstance(step, int) else step.idx

class AnalysisPipeline:
    pipeline_graph = nx.DiGraph
    pipeline_steps: list[AnalysisPipelineStep]
    name_map: dict[str: AnalysisPipelineStep]

    def __init__(self, path: Path):
        self.pipeline_graph = nx.DiGraph()
        self.pipeline_steps = []
        self.name_map = {}

        self.file_path = path

    def add_step(self, new_step: AnalysisPipelineStep):
        """
        Adds a step to this analysis pipeline
        """
        new_step.idx = len(self.pipeline_steps)
        self.pipeline_graph.add_node(new_step.idx)
        self.pipeline_steps.append(new_step)

    def add_step_dependant(self,
                           step: Union[int, AnalysisPipelineStep],
                           dependant_step: Union[int, AnalysisPipelineStep]):
        self.pipeline_graph.add_edge(analysis_step_idx(dependant_step), analysis_step_idx(step))
        self.get_pipeline_step(step).previous_steps.append(dependant_step)

    def num_pipeline_steps(self) -> int:
        """
        Returns the total number of analysis steps in this pipeline.

        """
        return len(self.pipeline_steps)

    def get_pipeline_step(self, step: Union[int, AnalysisPipelineStep]) -> AnalysisPipelineStep:
        return step if isinstance(step, AnalysisPipelineStep) else self.pipeline_steps[step]

    def __add__(self, other: AnalysisPipeline):
        id_remap: dict[int: int] = {}
        for node in other.pipeline_graph.nodes:
            assert other.pipeline_steps[node].name not in self.name_map
            n = copy.deepcopy(node)
            id_remap[n] = self.num_pipeline_steps()
            n.idx = self.num_pipeline_steps()
            self.name_map[n.name] = n
            self.pipeline_steps.append(n)
            self.pipeline_graph.add_node(n.idx)

        for n1, n2 in other.pipeline_graph.edges:
            self.pipeline_graph.add_edge(id_remap[n1], id_remap[n2])
