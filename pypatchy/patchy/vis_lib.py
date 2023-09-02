import math
from typing import Union

import matplotlib.colors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pylab as p

from pypatchy.patchy.analysis_lib import GraphsFromClusterTxt, ClassifyClusters

from pypatchy.patchy.simulation_specification import PatchySimulation
from .analysis.analyseClusters import ClusterCategory

from ..analpipe.analysis_data import TIMEPOINT_KEY
from .ensemble_parameter import EnsembleParameter, ParameterValue
from .simulation_ensemble import PatchySimulationEnsemble, PipelineStepDescriptor, describe_param_vals, shared_ensemble

import seaborn as sb


def plot_analysis_data(e: PatchySimulationEnsemble,
                       analysis_data_source: PipelineStepDescriptor,
                       data_source_key: str,
                       other_spec: Union[None, list[ParameterValue]] = None,
                       plot_grid_cols: Union[None, str, EnsembleParameter] = None,
                       plot_grid_rows: Union[None, str, EnsembleParameter] = None,
                       plot_line_color: Union[None, str, EnsembleParameter] = None,
                       plot_line_stroke: Union[None, str, EnsembleParameter] = None,
                       norm: Union[None, str] = None
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
        plot_line_stroke: ensemble parameter to use for the plot line stroke
        plot_grid_rows: ensemble parameter to use for the plot grid rows
        plot_line_color: ensemble parameter to use for the plot line
        plot_grid_cols: ensemble parameter to use for the plot grid cols
        norm: simulation parameter to use to normalize the data, or none for no data normalization

    """

    # validate inputs
    if other_spec is None:
        other_spec = list()
    plt_args = {
        "kind": "line",
        "errorbar": "sd"
    }
    if isinstance(plot_grid_cols, str):
        plt_args["col"] = plot_grid_cols
    if isinstance(plot_grid_rows, str):
        plt_args["row"] = plot_grid_rows
    if isinstance(plot_line_color, str):
        plt_args["hue"] = plot_line_color
    if isinstance(plot_line_stroke, str):
        plt_args["style"] = plot_line_stroke

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

    fig = sb.relplot(data,
                     x="steps",
                     y=data_source_key,
                     **plt_args)
    if norm:
        fig.set(ylim=(0.0, 1.0))
    fig.fig.suptitle(f"{e.export_name} - {analysis_data_source}", y=0.8)
    return fig


def compare_ensembles(es: list[PatchySimulationEnsemble],
                      analysis_data_source: str,
                      data_source_key: str,
                      other_spec: Union[None, list[ParameterValue]] = None,
                      grid_cols: Union[None, EnsembleParameter] = None,
                      plot_line_color: Union[None, str, EnsembleParameter] = None,
                      plot_line_stroke: Union[None, str, EnsembleParameter] = None,
                      norm: Union[None, str] = None
                      ) -> sb.FacetGrid:
    """
    Compares data from different ensembles
    """
    assert all([
        e.analysis_pipeline.step_exists(analysis_data_source)
        for e in es
    ]), f"Not all provided ensembles have analysis pipelines with step named {analysis_data_source}"

    plt_args = {
        "row": "ensemble",
        "kind": "line",
        "errorbar": "sd"
    }
    if isinstance(grid_cols, str):
        plt_args["col"] = grid_cols
    if isinstance(plot_line_color, str):
        plt_args["hue"] = plot_line_color
    if isinstance(plot_line_stroke, str):
        plt_args["style"] = plot_line_stroke

    all_data: list[pd.DataFrame] = []
    # get sim specs shared among all ensembles
    shared_sims: list[list[PatchySimulation]] = shared_ensemble(es)
    for sims, e in zip(shared_sims, es):
        if other_spec is None:  # unlikely
            other_spec = list()

        data_source = e.get_data(analysis_data_source, shared_sims[i])
        data = data_source.get().copy()
        if norm:
            def normalize_row(row):
                sim = row.drop([TIMEPOINT_KEY, data_source_key]).to_dict()
                sim = e.get_simulation(**sim)
                return row[data_source_key] / e.sim_get_param(sim, norm)

            data[data_source_key] = data.apply(normalize_row, axis=1)
        data.rename(mapper={TIMEPOINT_KEY: "steps"}, axis="columns", inplace=True)
        data["ensemble"] = e.export_name
        all_data.append(data)
    # compute timepoints shared between all ensembles
    shared_timepoints = set.intersection(*[set(d.timepoint.data) for d in all_data])
    data = pd.concat(all_data, ignore_index=True)

    data.set_index(TIMEPOINT_KEY)
    data = data.loc[shared_timepoints]
    data.reset_index()
    fig = sb.relplot(data,
                     x="steps",
                     y=data_source_key,
                     **plt_args)
    fig.fig.suptitle(f"Comparison of {analysis_data_source} Data", y=1)
    return fig


def get_particle_color(ptypeidx: int):
    """
    returns an rgb color consistant with the usage in polycubes
    """
    hue = ptypeidx * 137.508
    return matplotlib.colors.hsv_to_rgb((hue / 360, .5, .5))


def show_clusters(e: PatchySimulationEnsemble,
                  sim: PatchySimulation,
                  timepoint: int,
                  analysis_step: GraphsFromClusterTxt,
                  figsize=4
                  ) -> plt.Figure:
    # load particle id data from top file
    # todo: automate more?
    with (e.folder_path(sim) / "init.top").open('r') as f:
        f.readline()  # clear first line
        particle_types = [int(p) for p in f.readline().split()]

    tr = range(timepoint * analysis_step.output_tstep,
               (timepoint + 1) * analysis_step.output_tstep,
               analysis_step.output_tstep)
    graphs: list[nx.Graph] = e.get_data(analysis_step, sim, tr).get()[timepoint * analysis_step.output_tstep]
    nclusters = len(graphs)
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

    plt_args = {
        "kind": "line",
        "errorbar": "sd"
    }

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
        "kind": "line",
        "errorbar": "sd",
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

