from __future__ import annotations

import os
from pathlib import Path
from typing import Union, Any, Iterable

from .ensemble_parameter import ParameterValue
from ..slurm_log_entry import LogEntryObject


class PatchySimulation(LogEntryObject):
    """
    Specifies a execution of a Patchy Particle simulation run with the specific set of parameters

    provides methods for storing, construction, and managing stuff

    parameter_values is a list of (key,value) tuples where the key is a string identifier
    and the value is a ParameterValue object
    """
    param_vals: list[ParameterValue]
    parameter_dict: list[str, Any]

    def __init__(self, parameter_values: Iterable[ParameterValue]):
        self.param_vals = list(parameter_values)
        self.parameter_dict = {}
        for _, val in self.param_vals:
            if val.is_grouped_params():
                for param_name in val.group_params_names():
                    self.parameter_dict[param_name] = val[param_name]
            else:
                self.parameter_dict[val.name] = val.value

    def __contains__(self, parameter_name: str) -> bool:
        return parameter_name in self.parameter_dict

    def __getitem__(self, parameter_name: str):
        return self.parameter_dict[parameter_name]

    def var_names(self) -> list[str]:
        return list(self.parameter_dict.keys())

    def __str__(self) -> str:
        return "_".join([f"{key}-{self[key]}" for key in self.var_names()])

    def __iter__(self):
        return iter(self.param_vals)

    def get_folder_path(self) -> Path:
        return Path(os.sep.join([f"{key}_{str(val)}" for key, val in self.param_vals]))

    def __repr__(self) -> str:
        return ", ".join([f"{key}={self[key]}" for key in self.var_names()])

    def to_dict(self) -> dict[str, Union[str, int, float]]:
        return {
            p.name: p.value if not p.is_grouped_params() else p.value["value"]
            for p in self.param_vals
        }
