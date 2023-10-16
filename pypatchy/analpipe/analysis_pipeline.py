from __future__ import annotations

import copy
import math
import drawsvg as draw
from typing import Union, Generator

import networkx as nx

from pypatchy.analpipe.analysis_pipeline_step import AnalysisPipelineStep, AnalysisPipelineHead, \
    AggregateAnalysisPipelineStep


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
        # print(f"Adding step {newStep.name}")
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
        assert isinstance(step, AnalysisPipelineStep) or step in self.name_map, \
            f"Pipeline has no step called {step}. Pipeline steps: {', '.join(self.name_map.keys())}"
        return step if isinstance(step, AnalysisPipelineStep) else self.name_map[step]

    def steps_before(self, step: AnalysisPipelineStep) -> list[AnalysisPipelineStep]:
        """
        Args:
            step: an analyis pipeline step to get directly preceding steps for
        Returns:
            a list of steps that provide data to the step passed
        """
        # if the pipeline data is expected to be in raw form
        assert not isinstance(step, AnalysisPipelineHead)
        return [u for u, v in self.pipeline_graph.in_edges(step.name)]

    def ancestors(self, step: AnalysisPipelineStep) -> list[AnalysisPipelineStep]:
        """
        Returns:
            a list of steps that provide data - perhaps through intermediaries - to the passed step
        """
        return [self.name_map[n] for n in nx.ancestors(self.pipeline_graph, step.name)]

    def descendants(self, step: AnalysisPipelineStep) -> list[AnalysisPipelineStep]:
        return [self.name_map[n] for n in nx.descendants(self.pipeline_graph, step.name)]

    def is_force_recompute(self, step: Union[AnalysisPipelineStep, str]) -> bool:
        """
        Returns:
            true if the passed step requires a recompute, false otherwise
        """
        if isinstance(step, str):
            return self.is_force_recompute(self.get_pipeline_step(step))
        else:
            return step.force_recompute or any([s.force_recompute for s in self.ancestors(step)])

    def head_nodes(self) -> list[AnalysisPipelineHead]:
        """
        Returns:
            a list of steps in the pipeline that don't recieve data from other steps
        """
        return [self.name_map[n] for n in self.pipeline_graph.nodes
                if isinstance(self.name_map[n], AnalysisPipelineHead)]

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

    def extend(self, newSteps: list[AnalysisPipelineStep], newPipes: list[tuple[str, str]]) -> AnalysisPipeline:
        """
        Extends the analysis pipeline by adding the new steps and pipes within
        """
        stepqueue = [*newSteps]
        edgeset = [*newPipes]
        count = 0
        newpipe = copy.deepcopy(self)
        # test that all steps either have a source node or are pipeline head nodes
        for step in stepqueue:
            assert issubclass(type(step), AnalysisPipelineHead) or any(
                [v == step.name for _, v in edgeset]), f"Missing data source for step {step.name}"

        while len(stepqueue) > 0 and count < math.pow(len(newSteps), 2):
            step = stepqueue[0]
            if issubclass(type(step), AnalysisPipelineHead):
                newpipe.add_step(None, step)
                stepqueue = stepqueue[1:]
            else:
                found_pipe = False
                for u, v in newPipes:
                    if v == step.name and u in newpipe:
                        newpipe.add_step(newpipe[u], step)
                        stepqueue = stepqueue[1:]
                        found_pipe = True
                        break
                if not found_pipe:
                    stepqueue.append(stepqueue.pop(0))
            count += 1
        assert len(
            stepqueue) == 0, f"Malformed steps!!! {len(stepqueue)} extraneous steps, starting with {stepqueue[0].name}"
        for u, v in edgeset:
            if (u, v) not in newpipe.pipeline_graph.edges:
                assert u in newpipe, f"{u} not in pipeline {str(newpipe)}!"
                assert v in newpipe, f"{v} not in pipeline {str(newpipe)}!"
                newpipe._add_pipe_between(newpipe[u], newpipe[v])
        return newpipe

    def set_input_tstep(self,
                        step: Union[str, AnalysisPipelineStep],
                        propegate_fwd: bool,
                        propegate_back: bool,
                        new_val: int = 0,
                        factor: float = 0,
                        visited_nodes: Union[set[str], None] = None):
        if visited_nodes is None:
            visited_nodes = set()
        if isinstance(step, str):
            self.set_input_tstep(self.get_pipeline_step(step), propegate_fwd, propegate_back, new_val, factor)
        else:
            assert not factor or not new_val, "Specify either a scale factor or a new tstep value, not both"
            if new_val:
                step.input_tstep = new_val
            else:
                step.input_tstep = int(step.input_tstep * factor)
            # do backwards propegation - depth-first search halting at aggregate or head nodes
            if propegate_back:
                # loop predecessors
                for predecessor_name in self.pipeline_graph.predecessors(step.name):
                    # avoid double-counting
                    if predecessor_name not in visited_nodes:
                        visited_nodes.add(predecessor_name)
                        n = self.name_map[predecessor_name]
                        self.set_output_tstep(n, False, False, new_val, factor)
                        if not issubclass(type(n), AnalysisPipelineHead) and not issubclass(type(n), AggregateAnalysisPipelineStep):
                            self.set_input_tstep(n, True, False, new_val, factor, visited_nodes)
            # do forwards propegation - depth-first search halting at aggregate nodes
            if propegate_fwd:
                for successor_name in self.pipeline_graph.successors(step):
                    # avoid double-counting
                    if successor_name not in visited_nodes:
                        visited_nodes.add(successor_name)
                        n = self.name_map[successor_name]

    def set_output_tstep(self,
                         step: Union[str, AnalysisPipelineStep],
                         propegate_fwd: bool,
                         propegate_back: bool,
                         new_val: int = 0,
                         factor: float = 0,
                         visited_nodes: Union[set[str], None] = None):
        if visited_nodes is None:
            visited_nodes = set()
        if isinstance(step, str):
            self.set_output_tstep(self.get_pipeline_step(step), propegate_fwd, propegate_back, new_val, factor)
        else:
            assert not factor or not new_val, "Specify either a scale factor or a new tstep value, not both"
            assert factor or new_val, "Either factor or new_val must be specified otherwise we aren't doing anything"
            if new_val:
                step.output_tstep = new_val
            else:
                step.output_tstep = int(step.output_tstep * factor)
            # do backwards propegation - depth-first search halting at aggregate or head nodes
            if propegate_back:
                # set input freq for this step
                self.set_input_tstep(step, False, False, new_val, factor)
                # loop immediate predecessors
                for ancestor_name in self.pipeline_graph.predecessors(step.name):
                    # skip nodes we've already handled (important for recursion)
                    if ancestor_name not in visited_nodes:
                        # flag visited
                        visited_nodes.add(ancestor_name)
                        # get node object
                        a = self.name_map[ancestor_name]
                        # check if step is an aggregate step or a head step
                        if issubclass(type(a), AggregateAnalysisPipelineStep) or issubclass(type(a), AnalysisPipelineHead):
                            # set the output step, but stop otherwise
                            self.set_output_tstep(a, False, False, new_val, factor)
                        else:
                            # set node input timestep
                            self.set_input_tstep(a, False, False, new_val, factor)
                            # set node output timestep, continue propegating backwards
                            self.set_output_tstep(a, False, True, new_val, factor, visited_nodes)
            # do forward propegation - depth-first search halting at aggregate nodes
            if propegate_fwd:
                # loop immediate successors
                for descendent_name in self.pipeline_graph.successors(step.name):
                    # avoid double-counting
                    if descendent_name not in visited_nodes:
                        # flag descendant as visited
                        visited_nodes.add(descendent_name)
                        # get node object
                        a = self.name_map[descendent_name]
                        # if node is an aggregate
                        self.set_input_tstep(a, False, False, new_val, factor)
                        # if the node is not an aggregation step
                        if not issubclass(type(a), AggregateAnalysisPipelineStep):
                            # continue propegating
                            self.set_output_tstep(a, True, False, new_val, factor, visited_nodes)

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

    def step_exists(self, step_name: str) -> bool:
        """
        Args:
            step_name: a name to test for is in pipeline
        Returns:
            True if the analysis pipeline contains a step with the provided name, false otherwise
        """
        return step_name in self.name_map

    def __str__(self):
        return f"Analysis pipeline with {self.num_pipes()} pipes and {len(self)} steps."

    def num_pipes(self) -> int:
        return len(self.pipeline_graph.edges)

    def draw_pipeline(self, scale=120) -> draw.Drawing:
        dw = scale * self.num_pipeline_steps()
        dh = scale * .6 * self.num_pipeline_steps()
        drawing = draw.Drawing(width=dw,
                               height=dh,
                               origin=(-120, -dh / 2))
        levels = {}
        level = 0
        levelpops = {}
        ys = {}
        to_visit = [n.name for n in self.head_nodes()]
        lvlidx = 0
        x_max = 0
        y_max = 0
        x_min = math.inf
        y_min = math.inf
        while to_visit:
            next_level = []
            for node in to_visit:
                levels[node] = level
                lvlidx += 1
                if level not in levelpops:
                    levelpops[level] = 1
                else:
                    levelpops[level] += 1
                if len(list(self.pipeline_graph.predecessors(node))) > 0:
                    parent = list(self.pipeline_graph.predecessors(node))[0]
                    sibs = list(self.pipeline_graph.successors(parent))
                    ys[node] = (sibs.index(node) - (len(sibs) - 1) / 2) + ys[parent]
                else:
                    ys[node] = 0
                next_level.extend([n for n in self.pipeline_graph.successors(node) if n not in levels])
            to_visit = next_level
            level += 1
            lvlidx = 0
        pos = {}
        for node in self.pipeline_graph.nodes():
            x = levels[node]
            pos[node] = (levels[node], ys[node])

        # pos = nx.spring_layout(self.pipeline_graph, pos=pos, iterations=40) # k=1/self.num_pipeline_steps())
        ws = {}
        hs = {}
        # draw steps
        for step_name in self.pipeline_graph.nodes:
            step = self.name_map[step_name]
            (w, h), g = step.draw()
            ws[step_name] = w
            hs[step_name] = h
            x, y = pos[step_name]
            x *= w * 1.25
            y *= h * 1.25
            pos[step_name] = (x, y)
            gg = draw.Group(transform=f"translate({x - w / 2}, {y - h / 2})")
            gg.append(g)
            drawing.append(gg)
            # update drawing mins and maxs
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)
        arrow = draw.Marker(-0.8, -0.51, 0.2, 0.5, scale=4, orient='auto')
        arrow.append(draw.Lines(-0.8, 0.5, -1.0, -0.5, 0.2, 0, fill='black', close=True))
        drawing.append(arrow)
        # draw pipes
        for u, v in self.pipeline_graph.edges:
            # start of path
            x0, y0 = pos[u]
            x0 = x0 + ws[u] / 2
            y0 = y0 - hs[u] / 2 + 0.1 * ws[u]
            # first crtl pt
            cx1, cy1 = x0 + 50, y0
            # end of path
            ex, ey = pos[v]
            ex = ex - ws[v] / 2 - 2
            ey = ey - hs[v] / 2 + 0.1 * ws[v]
            # second ctrl pt
            cx2, cy2 = ex - 50, ey
            # draw
            p = draw.Path(stroke="black", stroke_width=1.8, fill='none', marker_end=arrow)
            p.M(x=x0, y=y0)
            p.C(cx1, cy1, cx2, cy2, ex, ey)
            drawing.append(p)
        drawing.width = x_max - x_min + 40
        drawing.height = y_max - y_min + 40
        return drawing
