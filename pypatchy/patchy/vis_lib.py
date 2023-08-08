from typing import Union

import pandas as pd

from build.lib.pypatchy.patchy.analysis_lib import TIMEPOINT_KEY
from pypatchy.patchy.ensemble_parameter import EnsembleParameter, ParameterValue
from simulation_ensemble import PatchySimulationEnsemble, PipelineStepDescriptor, describe_param_vals

import seaborn as sb
import matplotlib.pyplot as plt


def plot_analysis_data(e: PatchySimulationEnsemble,
                       analysis_data_source: PipelineStepDescriptor,
                       data_source_key: str,
                       # data_source_key_upper: Union[None, str],
                       # data_source_key_lower: Union[None, str],
                       # data_source_key_range: Union[None, str],
                       other_spec: Union[None, list[ParameterValue]] = None,
                       plot_grid_h_axis: Union[None, str, EnsembleParameter] = None,
                       plot_grid_v_axis: Union[None, str, EnsembleParameter] = None,
                       plot_line_color: Union[None, str, EnsembleParameter] = None,
                       plot_line_stroke: Union[None, str, EnsembleParameter] = None,
                       ) -> plt.Figure:
    """
    Uses matplotlib to construct a plot of data provided with the output values on the y axis
    and the time on the x axis
    This method will plot the output of a single analysis pipeline step
    Parameters:

    """
    # validate inputs
    if other_spec is None:
        other_spec = list()
    # # every parameter in the simumulation must either be one of the ranges specified
    # # or be specified in `other_spec`
    # for param in e.ensemble_params:
    #     assert any([
    #         plot_grid_h_axis == param,
    #         plot_grid_v_axis == param,
    #         plot_line_color == param,
    #         plot_line_stroke == param,
    #         *[p in param for p in other_spec]
    #     ]), f"Parameter {str(param)} unspecified!"
    if isinstance(plot_grid_h_axis, str):
        plot_grid_h_axis = e.get_ensemble_parameter(plot_grid_h_axis)
    if isinstance(plot_grid_v_axis, str):
        plot_grid_v_axis = e.get_ensemble_parameter(plot_grid_v_axis)
    if isinstance(plot_line_color, str):
        plot_line_color = e.get_ensemble_parameter(plot_line_color)
    if isinstance(plot_line_stroke, str):
        plot_line_stroke = e.get_ensemble_parameter(plot_line_stroke)

    nx = len(plot_grid_h_axis) if plot_grid_h_axis is not None else 1
    ny = len(plot_grid_v_axis) if plot_grid_v_axis is not None else 1
    fig, axs = plt.subplots(ny, nx, squeeze=False, sharex='all', sharey='all')

    for x in range(nx):
        for y in range(ny):
            selector = [*other_spec]
            if plot_grid_h_axis:
                selector.append(plot_grid_h_axis[x])
            if plot_grid_v_axis:
                selector.append(plot_grid_v_axis[y])
            ax: plt.Axes = axs[y, x]
            ax.set_title(describe_param_vals(selector))
            data: pd.DataFrame = e.get_data(analysis_data_source, tuple(selector))
            plt_args = {
                "kind": "line",
                "errorbar": "sd"
            }
            if plot_line_color:
                plt_args["hue"] = plot_line_color.param_key
            if plot_line_stroke:
                plt_args["style"] = plot_line_stroke.param_key

            # if data_source_key_range or data_source_key_upper:
            #     plt_args["err_style"] = "band"
            #     if data_source_key_range:
            #         data["lower"] = data[data_source_key] - data[data_source_key_range]
            #         data["upper"] = data[data_source_key] + data[data_source_key_range]
            #         plt_args["errobar"]
            #
            #     else:
            #         assert data_source_key_lower
            #     ax.fill_between(data[TIMEPOINT_KEY],
            #                     data[data_source_key_lower],
            #                     data[data_source_key_upper],
            #                     c=data["DISPLAY_color"],
            #                     **plt_args)
            # plot data
            sb.relplot(data,
                       x=TIMEPOINT_KEY,
                       y=data_source_key,
                       ax=ax,
                       **plt_args)

    return fig
