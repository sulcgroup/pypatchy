import itertools
import math
from typing import Union, Any

from abc import ABC

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from pypatchy.patchy.analysis_lib import GraphsFromClusterTxt, ClassifyClusters, YIELD_KEY

from pypatchy.patchy.simulation_specification import PatchySimulation

from ..analpipe.analysis_data import TIMEPOINT_KEY
from .ensemble_parameter import EnsembleParameter, ParameterValue
from .simulation_ensemble import PatchySimulationEnsemble, PipelineStepDescriptor, shared_ensemble


import seaborn as sb

from ..vis_util import get_particle_color

DEFAULT_SB_ARGS = {
    "kind": "line",
    "errorbar": "sd"  # standard deviation
}

def plot_analysis_data(e: PatchySimulationEnsemble,
                       analysis_data_source: PipelineStepDescriptor,
                       data_source_key: str = YIELD_KEY,
                       other_spec: Union[None, list[ParameterValue], list[tuple]] = None,
                       cols: Union[None, str, EnsembleParameter] = None,
                       rows: Union[None, str, EnsembleParameter] = None,
                       color: Union[None, str, EnsembleParameter] = None,
                       stroke: Union[None, str, EnsembleParameter] = None,
                       trange: Union[None, tuple[int, int]] = None,
                       norm: Union[None, str, int, float] = None
                       ) -> sb.FacetGrid:
    """
    Uses seaborn to construct a plot of data provided with the output values on the y axis
    and the time on the x axis
    This method will plot the output of a single analysis pipeline step

    Args:
        e: the dataset to draw data from
        analysis_data_source: the pipeline step (can be str or object) to draw data from. the step output datatype should be a pandas DataFrame
        data_source_key: the key in the step output dataframe to use for data
        other_spec:  ensemble parameter values that will be constant across the figure
        stroke: ensemble parameter to use for the plot line stroke
        rows: ensemble parameter to use for the plot grid rows
        color: ensemble parameter to use for the plot line
        cols: ensemble parameter to use for the plot grid cols
        norm: simulation parameter to use to normalize the data, or none for no data normalization

    """

    # validate inputs
    if other_spec is None:
        other_spec = list()
    plt_args = DEFAULT_SB_ARGS.copy()
    if isinstance(cols, str):
        plt_args["col"] = cols
    if isinstance(rows, str):
        plt_args["row"] = rows
    if isinstance(color, str):
        plt_args["hue"] = color
    if isinstance(stroke, str):
        plt_args["style"] = stroke

    other_spec = [ParameterValue(spec[0], spec[1]) if isinstance(spec, tuple) else spec for spec in other_spec]

    data_source = e.get_data(analysis_data_source, tuple(other_spec))
    data: pd.DataFrame = data_source.get().copy()
    # could put this in get_data but frankly idk if i trust it
    if trange is not None:
        start, end = trange
        data = data[data[TIMEPOINT_KEY] >= start]
        data = data[data[TIMEPOINT_KEY] <= end]
    if len(data_source.trange()) == 1:
        raise Exception("Error: only one timepoint included in data range! Check your analysis pipeline tsteps and/or data completeness.")
    elif len(data_source.trange()) < 10:
        print(f"Warning: only {len(data_source.trange())} timepoints in data range! You can continue I guess but it's not GREAT.")
    for col in data.columns:
        if col not in ["duplicate", TIMEPOINT_KEY, data_source_key] and col not in plt_args.values():
            if len(data[col].unique()) != 1:
                print(f"Warning: ensemble parameter {col} not accounted for in visualization! problems may arise!")

    if norm:
        if isinstance(norm, int) or isinstance(norm, float):
            def normalize_row(row):
                sim = row.drop([TIMEPOINT_KEY, data_source_key]).to_dict()
                # sim = data_source.get().iloc[row].drop([TIMEPOINT_KEY, data_source_key]).to_dict()
                sim = e.get_simulation(**sim)
                return row[data_source_key] / norm
                # data.loc[row, data_source_key] /= e.sim_get_param(sim, norm)

        else:
            def normalize_row(row):
                sim = row.drop([TIMEPOINT_KEY, data_source_key]).to_dict()
                # sim = data_source.get().iloc[row].drop([TIMEPOINT_KEY, data_source_key]).to_dict()
                sim = e.get_simulation(**sim)
                return row[data_source_key] / e.sim_get_param(sim, norm)
                # data.loc[row, data_source_key] /= e.sim_get_param(sim, norm)

        data[data_source_key] = data.apply(normalize_row, axis=1)

    data.rename(mapper={TIMEPOINT_KEY: "steps"}, axis="columns", inplace=True)

    fig = sb.relplot(data,
                     x="steps",
                     y=data_source_key,
                     **plt_args)
    if norm:
        fig.set(ylim=(0.0, 1.0))
    fig.fig.suptitle(f"{e.export_name} - {analysis_data_source}", y=1)
    return fig


def plot_compare_ensembles(es: list[PatchySimulationEnsemble],
                           analysis_data_source: str,
                           data_source_key: str,
                           other_spec: Union[None, list[ParameterValue]] = None,
                           rows: Union[None, str, EnsembleParameter] = None,
                           cols: Union[None, str, EnsembleParameter] = None,
                           color: Union[None, str, EnsembleParameter] = None,
                           stroke: Union[None, str, EnsembleParameter] = None,
                           norm: Union[None, str] = None,
                           trange: Union[range, None] = None,
                           ignores: Union[set[str], None] = None
                           ) -> Union[sb.FacetGrid, bool]:
    """
    Compares data from different ensembles
    """
    assert all([
        e.analysis_pipeline.step_exists(analysis_data_source)
        for e in es
    ]), f"Not all provided ensembles have analysis pipelines with step named {analysis_data_source}"

    plt_args = DEFAULT_SB_ARGS.copy()

    if isinstance(rows, str):
        plt_args["row"] = rows
    if isinstance(cols, str):
        plt_args["col"] = cols
    if isinstance(color, str):
        plt_args["hue"] = color
    if isinstance(stroke, str):
        plt_args["style"] = stroke

    all_data: list[pd.DataFrame] = []
    # get sim specs shared among all ensembles
    shared_sims: list[list[PatchySimulation]] = shared_ensemble(es, ignores)
    if shared_sims is None:
        return False
    if trange is not None:
        data_sources = [e.get_data(analysis_data_source, sims, trange) for e, sims in zip(es, shared_sims)]
    else:
        data_sources = [e.get_data(analysis_data_source, sims) for e, sims in zip(es, shared_sims)]
    for sims, e, data_source in zip(shared_sims, es, data_sources):
        if other_spec is None:  # unlikely
            other_spec = list()

        some_data = []  # avoid reusing name all_data
        for sim, sim_data in zip(sims, data_source):
            data = sim_data.get()
            for param in sim:
                data.insert(0, param.param_name, param.value_name)
            some_data.append(data)
        data = pd.concat(some_data, ignore_index=True)
        if norm:
            def normalize_row(row):
                s = row.drop([TIMEPOINT_KEY, data_source_key]).to_dict()
                s = e.get_simulation(**s)
                return row[data_source_key] / e.sim_get_param(s, norm)

            data[data_source_key] = data.apply(normalize_row, axis=1)
        data["ensemble"] = e.export_name
        all_data.append(data)
    # compute timepoints shared between all ensembles
    shared_timepoints = set.intersection(*[set(d.trange()) for d in itertools.chain.from_iterable(data_sources)])
    data = pd.concat(all_data, ignore_index=True)

    data.set_index(TIMEPOINT_KEY, inplace=True)
    data = data.loc[shared_timepoints]
    data.reset_index(inplace=True)
    data.rename(mapper={TIMEPOINT_KEY: "steps"}, axis="columns", inplace=True)

    fig = sb.relplot(data,
                     x="steps",
                     y=data_source_key,
                     **plt_args)
    fig.fig.suptitle(f"Comparison of {analysis_data_source} Data", y=1)
    if norm:
        fig.set(ylim=(0.0, 1.0))
    return fig


def show_clusters(e: PatchySimulationEnsemble,
                  sim: PatchySimulation,
                  analysis_step: Union[GraphsFromClusterTxt,str],
                  timepoint: int = -1,
                  step: int = -1,
                  figsize=4
                  ) -> Union[plt.Figure, None]:
    # load particle id data from top file
    # todo: automate more?
    if isinstance(analysis_step, str):
        analysis_step = e.get_analysis_step(analysis_step)
    with (e.folder_path(sim) / "init.top").open('r') as f:
        f.readline()  # clear first line
        particle_types = [int(p) for p in f.readline().split()]

    # default to last step
    if step == -1 and timepoint == -1:
        timepoint = e.get_data(analysis_step, sim).trange()[-1]

    elif timepoint == -1:  # timepoint from step
        timepoint = step * analysis_step.output_tstep

    tr = range(int(timepoint),
               int(timepoint + analysis_step.output_tstep),
               int(analysis_step.output_tstep))
    graphs: list[nx.Graph] = e.get_data(analysis_step, sim, tr).get()[timepoint]
    nclusters = len(graphs)
    if nclusters == 0:
        print(f"No clusters at step {timepoint*analysis_step.output_tstep}")
        return None
    r = math.ceil(math.sqrt(nclusters))
    fig, axs = plt.subplots(nrows=r, ncols=r, figsize=(r * figsize, r * figsize))

    for i, cluster in enumerate(graphs):
        # axs coords
        x = i % r
        y = int(i / r)

        ptypemap = [get_particle_color(particle_types[j]) for j in cluster.nodes]

        nx.draw(cluster,
                ax=axs[x, y],
                with_labels=True,
                node_color=ptypemap)
    # clear remaining axes for style reasons
    for i in range(len(graphs), r ** 2):
        x = i % r
        y = int(i / r)
        axs[x, y].remove()

    return fig


def plot_total_graph(e: PatchySimulationEnsemble,
                     analysis_data_source: PipelineStepDescriptor,
                     grid_rows: Union[None, str] = None,
                     grid_cols: Union[None, str] = None,
                     line_color: Union[None, str] = None):
    """
    Plots the total size of all the graphs in the simulation over time
    """
    assert isinstance(analysis_data_source, ClassifyClusters)
    # get data all at once to standardize timeframe
    raw_data = e.get_data(analysis_data_source, ()).get()
    # if no_overreach:
    #     raw_data = raw_data.loc[
    #         raw_data[ClassifyClusters.CLUSTER_CATEGORY_KEY].isin([ClusterCategory.MATCH.value, ClusterCategory.SUBSET.value])]

    data = []
    for sim in e.ensemble():
        sim_data: pd.DataFrame = raw_data.loc[np.all([
            raw_data[param.param_name] == param.value_name for param in sim
        ], axis=0)]
        sim_data = sim_data.groupby(TIMEPOINT_KEY)["sizeratio"].sum().reset_index()
        for param in sim:
            # sim_data.insert(len(sim_data.columns) - 1, param.param_name, param.value_name)
            sim_data[param.param_name] = [param.value_name] * len(sim_data.index)
        data.append(sim_data)
    data = pd.concat(data)
    data.rename(mapper={TIMEPOINT_KEY: "steps", "sizeratio": "size"}, axis="columns", inplace=True)

    plt_args = DEFAULT_SB_ARGS.copy()

    if isinstance(grid_rows, str):
        plt_args["row"] = grid_rows
    if isinstance(grid_cols, str):
        plt_args["col"] = grid_cols
    if isinstance(line_color, str):
        plt_args["hue"] = line_color

    fig = sb.relplot(data,
                     x="steps",
                     y="size",
                     **plt_args)
    return fig


def plot_compare_analysis_outputs(e: PatchySimulationEnsemble,
                                  sources: list[PipelineStepDescriptor],
                                  data_source_key: str,
                                  orientation="col",
                                  **kwargs):
    plt_args = {
        **DEFAULT_SB_ARGS,
        orientation: "data_source"
    }

    if "col" in kwargs:
        assert orientation != "col", "Trying to specify two different variables for columns!"
        plt_args["col"] = kwargs["col"]
    if "row" in kwargs:
        assert orientation != "row", "Trying to specify two different variables for rows!"
        plt_args["row"] = kwargs["row"]
    if "color" in kwargs:
        plt_args["hue"] = kwargs["color"]
    if "stroke" in kwargs:
        plt_args["style"] = kwargs["stroke"]

    other_spec = kwargs["other_spec"] if "other_spec" in kwargs else list()
    norm = kwargs["norm"] if "norm" in kwargs else None

    data_big = []
    for analysis_data_source in sources:
        data_source = e.get_data(analysis_data_source, tuple(other_spec))
        data = data_source.get().copy()

        if norm:
            def normalize_row(row):
                sim = row.drop([TIMEPOINT_KEY, data_source_key]).to_dict()
                # sim = data_source.get().iloc[row].drop([TIMEPOINT_KEY, data_source_key]).to_dict()
                sim = e.get_simulation(**sim)
                return row[data_source_key] / e.sim_get_param(sim, norm)
                # data.loc[row, data_source_key] /= e.sim_get_param(sim, norm)

            data[data_source_key] = data.apply(normalize_row, axis=1)

        data.rename(mapper={TIMEPOINT_KEY: "steps"}, axis="columns", inplace=True)
        data.insert(1, "data_source", e.get_pipeline_step(analysis_data_source).name)
        data_big.append(data)
    data = pd.concat(data_big, ignore_index=True, axis=0)
    fig = sb.relplot(data,
                     x="steps",
                     y=data_source_key,
                     **plt_args)
    if norm:
        fig.set(ylim=(0.0, 1.0))
    fig.fig.suptitle(f"{e.export_name} Data", y=1.0)
    return fig

class PolycubesFigure (ABC):
    """
    Wrapper class to facilitate creating and showing figures. Extend to make specific figures
    """
    fig: sb.FacetGrid
    def __init__(self,
                 e: Union[PatchySimulationEnsemble, list[PatchySimulationEnsemble]],
                 **kwargs):
        pass

    def __repr__(self):
        return self.fig.fig

class BaseYieldCurveFigure(PolycubesFigure, ABC):
    """
    Abstract base class. Wrapper class for figures that measure yield (or some other quantity) over a time period
    """
    plt_args: dict[str, Any]
    def __init__(self,
                 e: Union[PatchySimulationEnsemble,
                                list[PatchySimulationEnsemble]],
                 analysis_data_source: PipelineStepDescriptor,
                 data_source_key: str = YIELD_KEY,
                 **kwargs):
        super().__init__(e, **kwargs)
        self.plt_args = {}
        # validate inputs

        plt_args = DEFAULT_SB_ARGS.copy()
        if "cols" in kwargs:
            plt_args["col"] = kwargs["cols"]
        elif "col" in kwargs:
            plt_args["col"] = kwargs["cols"]
        if "rows" in kwargs:
            plt_args["row"] = kwargs["rows"]
        elif "row" in kwargs:
            plt_args["row"] = kwargs["row"]
        if "color" in kwargs:
            plt_args["hue"] = kwargs["color"]
        if "stroke" in kwargs:
            plt_args["style"] = kwargs["stroke"]


class YieldCurveFigure(BaseYieldCurveFigure):
    def __init__(self, e: PatchySimulationEnsemble,
                 analysis_data_source: PipelineStepDescriptor,
                 data_source_key: str = YIELD_KEY,
                 other_spec: Union[None, list[Union[ParameterValue, tuple[str, Any]]]] = None,
                 norm=None,
                 **kwargs):
        """
        Uses seaborn to construct a plot of data provided with the output values on the y axis
        and the time on the x axis
        This method will plot the output of a single analysis pipeline step

        Args:
            e: the dataset to draw data from
            analysis_data_source: the pipeline step (can be str or object) to draw data from. the step output datatype should be a pandas DataFrame
            data_source_key: the key in the step output dataframe to use for data
            other_spec:  ensemble parameter values that will be constant across the figure
            stroke: ensemble parameter to use for the plot line stroke
            rows: ensemble parameter to use for the plot grid rows
            color: ensemble parameter to use for the plot line
            cols: ensemble parameter to use for the plot grid cols
            norm: simulation parameter to use to normalize the data, or none for no data normalization

        """
        super().__init__(e, analysis_data_source, data_source_key, **kwargs)
        if other_spec is None:
            other_spec = list()

        data_source = e.get_data(analysis_data_source, tuple(other_spec))
        data = data_source.get().copy()
        if len(data_source.trange()) == 1:
            raise Exception(
                "Error: only one timepoint included in data range! Check your analysis pipeline tsteps and/or data completeness.")
        elif len(data_source.trange()) < 10:
            print(f"Warning: only {len(data_source.trange())} timepoints in data range! You can continue I guess but it's not GREAT.")

        if norm:
            def normalize_row(row):
                sim = row.drop([TIMEPOINT_KEY, data_source_key]).to_dict()
                # sim = data_source.get().iloc[row].drop([TIMEPOINT_KEY, data_source_key]).to_dict()
                sim = e.get_simulation(**sim)
                return row[data_source_key] / e.sim_get_param(sim, norm)
                # data.loc[row, data_source_key] /= e.sim_get_param(sim, norm)

            data[data_source_key] = data.apply(normalize_row, axis=1)

        data.rename(mapper={TIMEPOINT_KEY: "steps"}, axis="columns", inplace=True)

        self.fig = sb.relplot(data,
                         x="steps",
                         y=data_source_key,
                         **self.plt_args)
        if norm:
            self.fig.set(ylim=(0.0, 1.0))
        self.fig.fig.suptitle(f"{e.export_name} - {analysis_data_source}", y=1)
