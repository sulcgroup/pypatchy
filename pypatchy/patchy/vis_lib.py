import itertools
import math
from typing import Union, Any, Optional, Iterable, Tuple

from abc import ABC

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from pypatchy.patchy.analysis_lib import GraphsFromClusterTxt, ClassifyClusters, YIELD_KEY

from pypatchy.patchy.simulation_specification import PatchySimulation

from ..analysis.analysis_data import TIMEPOINT_KEY, PipelineData, PDPipelineData
from .base_param_val import ParameterValue
from .ensemble_parameter import EnsembleParameter
from .simulation_ensemble import PatchySimulationEnsemble, PipelineStepDescriptor, shared_ensemble, PatchySimDescriptor

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
        raise Exception(
            "Error: only one timepoint included in data range! Check your analysis pipeline tsteps and/or data completeness.")
    elif len(data_source.trange()) < 10:
        print(
            f"Warning: only {len(data_source.trange())} timepoints in data range! You can continue I guess but it's not GREAT.")
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

# def plot_curve(
#         e: PatchySimulationEnsemble,
#         ax: plt.Axes,
#         analysis_data_source: str,
#         plot_params: dict[str, Any],
#         trange: Optional[Tuple[float, float]] = None,
#         data_params: Optional[tuple[ParameterValue]]=None,
#         data_source_key: str = YIELD_KEY,
#         norm: Optional[Union[str, int, float]] = None
#         ):
#     # lookup simulations involved in this simulation
#     data = e.get_data(analysis_data_source, data_params, trange).get()
#
#     if norm:
#         if isinstance(norm, int) or isinstance(norm, float):
#             def normalize_row(row):
#                 return row[data_source_key] / norm
#
#         else:
#             def normalize_row(row):
#                 sim = row.drop([TIMEPOINT_KEY, data_source_key]).to_dict()
#                 sim = e.get_simulation(**sim)
#                 return row[data_source_key] / e.sim_get_param(sim, norm)
#         data[data_source_key] = data.apply(normalize_row, axis=1)
#
#     ax.plot(TIMEPOINT_KEY,
#             data_source_key,
#             data=data)

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
                data.insert(0, param.param_name, param.value_name())
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


def show_clusters(e: PatchySimulationEnsemble, sim: PatchySimulation, analysis_step: Union[GraphsFromClusterTxt, str],
                  timepoint: int = -1, step: int = -1, figsize=4, min_cluster_size=2) -> Union[plt.Figure, None]:
    """

    """
    import matplotlib.pyplot as plt

    if isinstance(analysis_step, str):
        analysis_step = e.get_analysis_step(analysis_step)

    if step == -1 and timepoint == -1:
        timepoint = e.get_data(analysis_step, sim).trange()[-1]
    elif timepoint == -1:
        timepoint = step * analysis_step.output_tstep

    tr = range(int(timepoint),
               int(timepoint + analysis_step.output_tstep),
               int(analysis_step.output_tstep))
    graphs: list[nx.Graph] = [g for g in e.get_data(analysis_step, sim, tr).get()[timepoint] if len(g) > min_cluster_size]


    nclusters = len(graphs)
    if nclusters == 0:
        print(f"No clusters at step {timepoint * analysis_step.output_tstep}")
        return None

    r = math.ceil(math.sqrt(nclusters))
    fig, axs = plt.subplots(nrows=r, ncols=r, figsize=(r * figsize, r * figsize))

    axs = axs.flatten() if isinstance(axs, np.ndarray) else [axs]

    for i, cluster in enumerate(graphs):
        ax = axs[i]
        k_val = 2 / np.sqrt(len(cluster))
        try:
            pos = nx.planar_layout(cluster)
        except nx.NetworkXException:
            pos = nx.kamada_kawai_layout(cluster)
        node_colors = [get_particle_color(cluster.nodes[j]["particle_type"]) for j in cluster.nodes]
        nx.draw_networkx_nodes(cluster, pos=pos, node_color=node_colors, ax=ax)
        nx.draw_networkx_labels(cluster, pos=pos, ax=ax, font_size=8)
        # Extract edge colors if available
        try:
            edge_colors = [get_particle_color(abs(cluster.edges[edge].get("patch").color())) for edge in cluster.edges]

            # Draw edges with curved style to show direction and prevent overlap
            nx.draw_networkx_edges(cluster, pos=pos, ax=ax,
                                   edge_color=edge_colors,
                                   connectionstyle="arc3,rad=0.2",
                                   arrows=True,
                                   arrowsize=10)
        except AttributeError:

            # Draw edges with curved style to show direction and prevent overlap
            nx.draw_networkx_edges(cluster, pos=pos, ax=ax,
                                   connectionstyle="arc3,rad=0.2",
                                   arrows=True,
                                   arrowsize=10)


        ax.set_axis_off()

    # Remove unused axes
    for i in range(len(graphs), len(axs)):
        fig.delaxes(axs[i])

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

    data = []
    for sim in e.ensemble():
        sim_data: pd.DataFrame = raw_data.loc[np.all([
            raw_data[param.param_name] == param.value_name() for param in sim
        ], axis=0)]
        sim_data = sim_data.groupby(TIMEPOINT_KEY)["sizeratio"].sum().reset_index()
        for param in sim:
            # sim_data.insert(len(sim_data.columns) - 1, param.param_name, param.value_name)
            sim_data[param.param_name] = [param.value_name()] * len(sim_data.index)
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
    """
    Parameters:
        e the ensemble to plot data from
        sources names of analysis pipeline nodes to plot data from

    """
    plt_args = {
        **DEFAULT_SB_ARGS,
        orientation: "data_source"
    }

    # include info in kwargs, some aliases to make my life easier
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

    # read other specs to select simulations
    other_spec = kwargs["other_spec"] if "other_spec" in kwargs else list()
    # verify other spec
    e.get_simulation(*other_spec)

    norm = kwargs["norm"] if "norm" in kwargs else None

    data_big = []
    # loop data source nodes
    for analysis_data_source in sources:
        # load data
        data_source: PipelineData = e.get_data(analysis_data_source, tuple(other_spec))
        # todo: assertions
        assert isinstance(data_source, PDPipelineData), f"Analysis pipeline node {analysis_data_source} does not produce data in the form of a Pandas dataframe!"
        data: pd.DataFrame = data_source.get().copy()

        # if we have specified that the data should be normalized
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
    # merge pandas dataframes
    data = pd.concat(data_big, ignore_index=True, axis=0)
    # use seaborne (ugh why) to plot data
    fig = sb.relplot(data,
                     x="steps",
                     y=data_source_key,
                     **plt_args)
    # if the data are normalized, set y axis range to be 0 - 1.0
    if norm:
        fig.set(ylim=(0.0, 1.0))
    fig.fig.suptitle(f"{e.export_name} Data", y=1.0)
    return fig


def plot_energy(e: PatchySimulationEnsemble,
                data_identifier: PatchySimDescriptor = tuple(),
                curve_color_param: str = "T",
                curve_stroke_w_param: Union[str, None] = None,
                curve_stroke_style_param: Union[str, None] = None,
                load_energy_step_name: str = "load_energies",
                e_type: str = "te",
                t_range: Union[tuple[int, int], None] = None,
                log_x: bool = False,
                curve_alpha: float = 1.
                ):
    """
    quick function to plot energies
    """

    load_energy_tstep = e.get_pipeline_step(load_energy_step_name).output_tstep
    if t_range is not None:
        data = e.get_data(load_energy_step_name, data_identifier,
                          time_steps=range(*t_range, load_energy_tstep))
    else:
        data = e.get_data(load_energy_step_name, data_identifier)
    df = data.get()

    if t_range is not None:
        df = df[(t_range[0] <= df["timepoint"]) & (df["timepoint"] <= t_range[1])]

    sortby = ['timepoint']
    if e.is_multiselect(data_identifier):
        sortby = [curve_color_param, 'timepoint']
    if curve_stroke_w_param:
        sortby.insert(0, curve_stroke_w_param)
    if curve_stroke_style_param:
        sortby.insert(0, curve_stroke_style_param)
    df = df.sort_values(by=sortby)

    if curve_stroke_w_param:
        unique_w = np.unique(df[curve_stroke_w_param])
        line_widths = np.linspace(1, 3, len(unique_w))
        width_map = dict(zip(unique_w, line_widths))
    if curve_stroke_style_param:
        unique_styles = np.unique(df[curve_stroke_style_param])
        style_list = ['-', '--', '-.', ':']  # fallback if >4 styles needed
        style_map = dict(zip(unique_styles, style_list * ((len(unique_styles) // 4) + 1)))

    plt.figure(figsize=(10, 6))
    if curve_color_param in df:
        colors = plt.cm.tab10(np.linspace(0, 1, len(np.unique(df[curve_color_param]))))
        color_map = dict(zip(np.unique(df[curve_color_param]), colors))

        for param1 in np.unique(df[curve_color_param]):
            subset_param1 = df[df[curve_color_param] == param1]
            groupby_keys = []
            if curve_stroke_w_param:
                groupby_keys.append(curve_stroke_w_param)
            if curve_stroke_style_param:
                groupby_keys.append(curve_stroke_style_param)
            # plot duplicates seperately
            if "duplicate" in e.ensemble_params:
                groupby_keys.append("duplicate")

            # If no groupby keys, wrap in a list with a dummy key to still iterate once
            group_iter = subset_param1.groupby(groupby_keys) if groupby_keys else [(None, subset_param1)]

            for _, subset in group_iter:
                lw = width_map[subset[curve_stroke_w_param].iloc[0]] if curve_stroke_w_param else 1.5
                ls = style_map[subset[curve_stroke_style_param].iloc[0]] if curve_stroke_style_param else '-'
                label_parts = [f'{curve_color_param}={param1}']
                if curve_stroke_w_param:
                    label_parts.append(f'{curve_stroke_w_param}={subset[curve_stroke_w_param].iloc[0]}')
                if curve_stroke_style_param:
                    label_parts.append(f'{curve_stroke_style_param}={subset[curve_stroke_style_param].iloc[0]}')
                plt.plot(subset['timepoint'], subset[e_type], label=', '.join(label_parts),
                         color=color_map[param1], linewidth=lw, linestyle=ls, alpha=curve_alpha)
    else:
        plt.plot(df["timepoint"], df[e_type])

    plt.tight_layout(rect=[0, 0, 0.8, 1])  # leave space on right for legend
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)
    plt.title('Energy vs Timepoint')
    if e_type == "te":
        y_label = 'Total Energy'
    elif e_type == "pe":
        y_label = 'Potential Energy'
    elif e_type == "ke":
        y_label = "Kinetic Energy"
    else:
        raise Exception(f"Unrecognized energy identifier {e_type}")
    if log_x:
        plt.xlabel("log Timepoint")
        plt.xscale("log")
    else:
        plt.xlabel(f'Timepoint (x {e.get_pipeline_step(load_energy_step_name).input_tstep * e.get_param("print_energy_every")})')
    plt.ylabel(y_label)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_energies(e: PatchySimulationEnsemble, sim: PatchySimulation,
                  load_energy_step_name: str = "load_energies"):
    data = e.get_data(load_energy_step_name, sim)
    df = data.get()
    plt.figure(figsize=(10, 6))
    colors = {'pe': 'red', 'ke': 'blue', 'te': 'green'}

    for e_type in ("pe", "ke", "te"):
        plt.plot(df["timepoint"], df[e_type], label=e_type, color=colors[e_type])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.title(f"Energy vs Timepoint for {str(sim)} of {e.long_name()}")
    plt.show()


class PolycubesFigure(ABC):
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
            print(
                f"Warning: only {len(data_source.trange())} timepoints in data range! You can continue I guess but it's not GREAT.")

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

def plot_last_conf_cluster_size_histograms(e: PatchySimulationEnsemble,
                                           hist_ax: plt.Axes,
                                           color_term: str,
                                           x_step: int
                                           ):
    T_vals: EnsembleParameter = e.ensemble_params[color_term]
    num_color_term_params = len(T_vals)
    total_group_height = 0.9 * x_step
    bar_height = total_group_height / max(num_color_term_params, 1)
    offsets_y = (np.arange(num_color_term_params) - (num_color_term_params - 1) / 2.0) * bar_height

    for i, T in enumerate(T_vals):
        df = e.get_data("cluster_sizes", sim=(nt, T), time_steps=e.time_length()).get()
        if df.empty:
            continue
        last_tp = df[TIMEPOINT_KEY].max()
        sizes = df.loc[df[TIMEPOINT_KEY] == last_tp, "cluster_size"].astype(float).values

        # warn if any out of range
        too_large = sizes[sizes > 400]
        if len(too_large):
            print(f"Large Clusters for {nt}, T={T.value_name()}: {too_large}")

        counts, _ = np.histogram(sizes, bins=edges)

        y = centers + offsets_y[i]
        hist_ax.barh(y, counts, height=bar_height, align="center", label=T.value_name())

    hist_ax.set_title(str(nt))
    hist_ax.set_xlabel("Count")
    hist_ax.set_ylabel("Cluster size")
    hist_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    hist_ax.grid(True, axis="x", linestyle=":", linewidth=0.8)
