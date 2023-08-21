from typing import Union

import pandas as pd

from ..analpipe.analysis_data import TIMEPOINT_KEY
from .ensemble_parameter import EnsembleParameter, ParameterValue
from .simulation_ensemble import PatchySimulationEnsemble, PipelineStepDescriptor, describe_param_vals

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
        for row in data.index:
            sim = data_source.get().iloc[row].drop([TIMEPOINT_KEY, data_source_key]).to_dict()
            sim = e.get_simulation(**sim)
            data.loc[row, data_source_key] /= e.sim_get_param(sim, norm)
    data.rename(mapper={TIMEPOINT_KEY: "steps"}, axis="columns", inplace=True)

    fig = sb.relplot(data,
                     x="steps",
                     y=data_source_key,
                     **plt_args)
    fig.set(title=f"{e.export_name} - {analysis_data_source}")
    return fig
