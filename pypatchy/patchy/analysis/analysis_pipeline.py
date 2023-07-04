from __future__ import annotations

import copy
from typing import Union

import networkx as nx

from pypatchy.patchy.analysis_pipeline_step import AnalysisPipelineStep


def analysis_step_idx(step: Union[int, AnalysisPipelineStep]) -> int:
    return step if isinstance(step, int) else step.idx

class AnalysisPipeline:
    pipeline_graph = nx.DiGraph
    pipeline_steps: list[AnalysisPipelineStep]
    name_map: dict[str: AnalysisPipelineStep]

    def __init__(self, pathway: list[tuple[str: str]] = [], *args: AnalysisPipelineStep):
        self.pipeline_graph = nx.DiGraph()
        self.pipeline_steps = []
        for i, step in enumerate(args):
            step = copy.deepcopy(step)
            step.idx = i
            self.pipeline_steps.append(step)
        self.name_map = {step.name: step for step in self.pipeline_steps}
        for begin, end in pathway:
            assert begin in self.name_map, f"No step in pathway with name {begin}."
            assert end in self.name_map, f"No step in pathway with name {end}."
            self.pipeline_graph.add_edge(self.name_map[begin].idx,
                                         self.name_map[end].idx)

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

    def num_pipeline_steps(self) -> int:
        """
        Returns the total number of analysis steps in this pipeline.

        """
        return len(self.pipeline_steps)

    def get_pipeline_step(self, step: Union[int, str, AnalysisPipelineStep]) -> AnalysisPipelineStep:
        return step if isinstance(step, AnalysisPipelineStep) else\
            self.name_map[step] if isinstance(step, str) else self.pipeline_steps[step]

    def steps_before(self, step: AnalysisPipelineStep) -> list[int]:
        return self.pipeline_graph.in_edges(step.idx).keys()

    def __add__(self, other: AnalysisPipeline):
        id_remap: dict[int: int] = {}
        new_pipeline = copy.deepcopy(self)
        for node in other.pipeline_graph.nodes:
            assert other.pipeline_steps[node].name not in new_pipeline.name_map
            n = copy.deepcopy(other.pipeline_steps[node])
            id_remap[node] = new_pipeline.num_pipeline_steps()
            n.idx = new_pipeline.num_pipeline_steps()
            new_pipeline.name_map[n.name] = n
            new_pipeline.pipeline_steps.append(n)
            new_pipeline.pipeline_graph.add_node(n.idx)

        for n1, n2 in other.pipeline_graph.edges:
            new_pipeline.pipeline_graph.add_edge(id_remap[n1], id_remap[n2])

        return new_pipeline
