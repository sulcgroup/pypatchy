from __future__ import annotations

import itertools
import os
from pathlib import Path
from typing import Union, Any, Iterable


from .ensemble_parameter import ParameterValue, ParamValueGroup, parameter_value
from ..slurm_log_entry import LogEntryObject
from ..util import get_spec_json


class ParamSet(LogEntryObject):
    """
    Specifies a execution of a Patchy Particle simulation run with the specific set of parameters

    provides methods for storing, construction, and managing stuff

    parameter_values is a list of (key,value) tuples where the key is a string identifier
    and the value is a ParameterValue object
    """
    # ordered list of parameter values that specify this simulation
    param_vals: list[ParameterValue]

    parameter_dict: dict[str, Any] # for easy access

    def __init__(self, parameter_values: Iterable[ParameterValue]):
        self.param_vals = list(parameter_values)
        self.parameter_dict = {}
        for val in self.param_vals:
            if isinstance(val, ParamValueGroup):
                for param_name in val.group_params_names():
                    self.parameter_dict[param_name] = val[param_name]
            else:
                self.parameter_dict[val.param_name] = val.param_value
            # assert not isinstance(self.parameter_dict[val.param_name], ParameterValue)

    def __contains__(self, parameter_name: str) -> bool:
        return parameter_name in self.parameter_dict

    def __getitem__(self, parameter_name: str) -> Any:
        return self.parameter_dict[parameter_name]

    def var_names(self) -> list[str]:
        """

        """
        return list(self.parameter_dict.keys())

    def __str__(self) -> str:
        return "_".join([f"{key}-{self[key]}" for key in self.var_names()])

    def __iter__(self):
        return iter(self.param_vals)

    def get_folder_path(self) -> Path:
        return Path(os.sep.join([f"{param.param_name}_{str(param.value_name())}" for param in self.param_vals]))

    def __repr__(self) -> str:
        return ", ".join([f"{pv.param_name}={pv.value_name()}" for pv in self.param_vals])

    def to_dict(self) -> dict[str, Union[str, int, float]]:
        return {
            p.param_name: p.value_name()
            for p in self.param_vals
        }

    def __hash__(self) -> int:
        return hash(str(self))

    def equivelant(self, other: PatchySimulation) -> bool:
        """
        order-independant version of __equals__
        probably a better way to write this but frankly i am operating on 3 hours of sleep
        """
        for p1 in other.param_vals:
            found_param = False
            for p2 in self.param_vals:
                if p1.param_name == p2.param_name:
                    if p1.value_name() != p2.value_name():
                        return False
                    else:
                        found_param = True
                        break
            if not found_param:
                return False

        return True

    def __eq__(self, other: ParamSet) -> bool:
        return repr(self) == repr(other)


PatchySimulation = ParamSet  # alias for backwards compatibility


class NoSuchParamError(Exception):
    _ensemble: Any
    _sim: Union[None, PatchySimulation]
    _param_name: str

    def __init__(self, e: Any, param_name: str, sim: Union[PatchySimulation, None] = None):
        self._ensemble = e
        self._sim = sim
        self._param_name = param_name

    def set_sim(self, sim: Union[PatchySimulation, None]):
        self._sim = sim

    def __str__(self):
        errstr = f"No value specified for parameter \"{self._param_name}\""
        if self._sim is not None:
            errstr += f"simulation {repr(self._sim)} in "
        errstr += f"ensemble {self._ensemble.long_name()}"
        return errstr

    def param_name(self) -> str:
        return self._param_name


def get_param_set(filename: str) -> ParamSet:
    param_set = get_spec_json(filename, "input_files")
    # backwards compatibility
    # hardcode these stupid headers, was such a mistake
    if "PROGRAM_PARAMETERS" in param_set["input"]:
        param_set.update({
            key: val for key, val
            in itertools.chain.from_iterable([
                subgroup.items() for subgroup
                in param_set["input"].values()])
        })
    else:
        param_set.update(param_set["input"])
    del param_set["input"]
    return ParamSet([parameter_value(key, val) for key, val in param_set.items()])
