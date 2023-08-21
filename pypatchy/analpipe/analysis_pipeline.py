from __future__ import annotations

import copy
from typing import Union, Generator

import networkx as nx

from pypatchy.analpipe.analysis_pipeline_step import AnalysisPipelineStep, AnalysisPipelineHead


# def analysis_step_idx(step: Union[int, AnalysisPipelineStep]) -> int:
#     return step if isinstance(step, int) else step.idx


class AnalysisPipeline:
    """
    A pipeline for analyzing patchy particle data.
    the pipeline consists of multiple steps (graph nodes) connected by pipes (directional graph edges)
    The pipeline graph does not have to be connected but must not be cyclic for reasons i hope are obvious
    """
    pipeline_graph = nx.DiGraph
    # pipeline_steps: list[AnalysisPipelineStep]
    name_map: dict[str: AnalysisPipelineStep]

    def __init__(self, *args: Union[AnalysisPipelineStep, tuple[str, str]]):
        # assert all([isinstance(s, AnalysisPipelineStep) for s in args]), "Invalid arguement passed to pipeline constructor!"
        # construct new directed graph for the pipeline
        self.pipeline_graph: nx.DiGraph = nx.DiGraph()
        # construct dict mapping names of nodes to steps in the pipeline
        self.name_map: dict[str: AnalysisPipelineStep] = dict()

        # constructor takes edge and node info together, must seperate!
        # sort steps in pipeline so head nodes will be in front
        steps = sorted([s for s in args if isinstance(s, AnalysisPipelineStep)],
                       key=lambda s: isinstance(s, AnalysisPipelineHead), reverse=True)
        pipes = [p for p in args if isinstance(p, tuple)]
        # loop nodes
        for step in steps:
            parent_steps = [self.name_map[a] for a, b in pipes if a in self.name_map and b == step.name]
            if len(parent_steps) > 0:
                self.add_step(parent_steps[0], step)
                for parent in parent_steps[1:]:
                    self._add_pipe_between(parent, step)
            else:
                # if node is pipeline head
                self.add_step(None, step)

        assert len(list(nx.simple_cycles(self.pipeline_graph))) == 0, "Analysis pipeline is cyclic"

    def _add_pipe_between(self,
                          first: AnalysisPipelineStep,
                          second: AnalysisPipelineStep):
        """
        Adds a pipe from one node to another node

        Args:
            first: the origin of the pipe. this analysis step will provide data to the second node
            second: the destination of the pipe. this step will recieve data from the first node

        """
        self.pipeline_graph.add_edge(first.name, second.name)
        # if the child node we're adding has no specified input timestep
        if second.input_tstep is None:
            assert first.output_tstep is not None
            second.config_io(input_tstep=first.output_tstep)

    def num_pipeline_steps(self) -> int:
        """
        Returns:
             the total number of analysis steps in this pipeline.
        """
        return len(self.name_map)

    def add_step(self, origin_step: Union[AnalysisPipelineStep, None],
                 newStep: AnalysisPipelineStep) -> AnalysisPipelineStep:
        """
        Adds a step to the analysis pipeline.
        Args:
            origin_step: an existing step in the pipeline that serves as a data source for the new step, or None if the new step is a head node
            newStep: new step to add to the pipeline
        """
        if origin_step is not None:
            assert origin_step.name in self.name_map
            # assert origin_step.idx <= len(self.name_map)
            # assert self.pipeline_steps[origin_step.idx] == self.name_map[origin_step.name]
        # newStep.idx = len(self.name_map)
        self.name_map[newStep.name] = newStep
        self.pipeline_graph.add_node(newStep.name)
        # self.pipeline_steps.append(newStep)
        if origin_step is not None:
            self._add_pipe_between(origin_step, newStep)
        return newStep

    def get_pipeline_step(self, step: Union[str, AnalysisPipelineStep]) -> AnalysisPipelineStep:
        """
        Method mainly for standardizing object types. If provided with an AnalysisPipelineStep object, it
        will just return the object. if provided with a string it will return the step in the pipeline with the provided name

        Args:
            step: identifier for step being gotten

        Returns:
              an analysis pipeline step
        """
        assert isinstance(step,AnalysisPipelineStep) or step in self.name_map,\
            f"Pipeline has no step called {step}. Pipeline steps: {', '.join(self.name_map.keys())}"
        return step if isinstance(step, AnalysisPipelineStep) else self.name_map[step]

    def steps_before(self, step: AnalysisPipelineStep) -> list[int]:
        """
        Args:
            step: an analyis pipeline step to get preceding steps for
        Returns:
            a list of steps that provide data to the step passed
        """
        # if the pipeline data is expected to be in raw form
        assert not isinstance(step, AnalysisPipelineHead)
        return [u for u, v in self.pipeline_graph.in_edges(step.name)]

    def head_nodes(self) -> list[AnalysisPipelineHead]:
        """
        Returns:
            a list of steps in the pipeline that don't recieve data from other steps
        """
        return [self.name_map[str(n)] for n in self.pipeline_graph.nodes
                if isinstance(self.name_map[str(n)], AnalysisPipelineHead)]

    def num_distinct_pipelines(self) -> int:
        """
        Returns:
            the number of connected components of the overall analysis graph
        """
        return len([n for n in self.get_distinct_pipelines()])

    def get_distinct_pipelines(self) -> Generator[nx.DiGraph]:
        """
        Returns:
            generator for connected components of the pipeline graph
        """
        return nx.weakly_connected_components(self.pipeline_graph)

    def validate(self):
        """
        Checks if the pipeline is okay, makes it everyone's problem if not
        """
        assert len(list(nx.simple_cycles(self.pipeline_graph))) == 0, "Analysis pipeline is cyclic"
        for pipe_start, pipe_end in self.pipeline_graph.edges:
            start_tstep = self.name_map[pipe_start].output_tstep
            end_tstep = self.name_map[pipe_end].input_tstep
            assert end_tstep % start_tstep == 0, "Inconsistant pipeline step time intervals between" \
                                                 f"{start_tstep} and {end_tstep}!"

    def _add_recursive(self,
                       other_graph: AnalysisPipeline,
                       other_start_node: AnalysisPipelineStep,
                       other_prev_node: Union[AnalysisPipelineStep, None] = None):
        """
        Recursively joins two pipelines using an algorithm i am franly not awake enough right now to explain
        """
        new_node = self.add_step(other_prev_node, copy.deepcopy(other_start_node))
        pipes = list(other_graph.pipeline_graph.successors(other_start_node.name))
        for destination_node in pipes:
            # check by name so we don't run into indexing problems
            dest_name = other_graph[destination_node].name
            if dest_name not in self:
                # node we just added -> child node
                self._add_recursive(other_graph, other_graph[dest_name], new_node)
            else:
                # reverse order but the same operation
                self._add_pipe_between(new_node, other_graph[dest_name])

    def __add__(self, other: AnalysisPipeline) -> AnalysisPipeline:
        """
        Joins this pipeline and the other pipeline and returns the merged pipelines

        Args:
            other: another analyiss pipeline

        Returns:
            the analyis pipeline formed by joining self and other pipeline

        """
        new_pipeline = copy.deepcopy(self)
        # add nodes recursively from head
        for node in other.head_nodes():
            new_pipeline._add_recursive(other, node)

        assert len(new_pipeline.pipeline_graph) == len(self.pipeline_graph) + len(other.pipeline_graph)
        assert len(list(nx.simple_cycles(new_pipeline.pipeline_graph))) == 0, "Analysis pipeline is cyclic"
        return new_pipeline

    def __contains__(self, key: Union[str, AnalysisPipelineStep, AnalysisPipeline]):
        if isinstance(key, AnalysisPipelineStep):
            return key.name in self
        elif isinstance(key, AnalysisPipeline):
            # test if all analysis steps and pipes in our key are also present in this
            for node in key.pipeline_graph.nodes():
                if not self.pipeline_graph.has_node(node):
                    return False
            for edge in key.pipeline_graph.edges():
                if not self.pipeline_graph.has_edge(*edge):
                    return False
            return True
        else:
            return key in self.name_map

    def __getitem__(self, item: str) -> AnalysisPipelineStep:
        """
        Args:
            item: the name of a step in the pipeline
        Returns:
            pipeline step with the given name
        """
        return self.name_map[item]

    def __len__(self):
        """
        Returns:
            the number of steps in the pipeline
        """
        return len(self.pipeline_graph)

    def __getstate__(self) -> dict:
        """
        Pickles pipeline
        """
        return {
            "nodes": self.name_map,
            "edges": [
                (u, v) for u, v in self.pipeline_graph.edges
            ]
        }

    def __setstate__(self, state: dict):
        """
        Unpickles pipeline
        """
        self.pipeline_graph = nx.DiGraph()
        self.name_map = state['nodes']
        for name in self.name_map:
            self.pipeline_graph.add_node(name)
        for u, v in state["edges"]:
            self._add_pipe_between(self.name_map[u], self.name_map[v])

