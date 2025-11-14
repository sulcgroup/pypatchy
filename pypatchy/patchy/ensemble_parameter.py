from __future__ import annotations

import copy
from typing import Union

from .base_param_val import ParameterValue, ParamValueGroup
from .param_val_types import PARTICLE_TYPES_KEY, MDT_CONVERT_KEY, STAGES_KEY, ParticleSetParam, MDTConvertParams, \
    StagedAssemblyParam
from .pl.plparticleset import MultidentateConvertSettings, decode_particles_dict

def parameter_value(key: str, val: Union[dict, str, int, float, bool, list]) -> ParameterValue:
    """
    Constructs a ParameterValue object
    """
    if isinstance(val, dict):
        if "name" in val:
            param_name = val["name"]
        else:
            param_name = key  # acceptable for const params, catastrophic for ensemble params
        # if type key is present, paramater is a particle set or mdt convert settings or something
        if "type" in val:
            if "value" in val:  # *sirens* BACKWARDS COMPATIBILITY DANGER ZONE
                data = val["value"]
            else:
                data = {k: val[k] for k in val if k not in ("type", "name")}
            if val["type"] == MDT_CONVERT_KEY:
                return MDTConvertParams(MultidentateConvertSettings(**data), param_name)
            elif val["type"] == PARTICLE_TYPES_KEY:

                return ParticleSetParam(decode_particles_dict(**data))
            elif val["type"] == STAGES_KEY:
                if not "stages" in data:
                    raise TypeError(f"Missing key 'stages' in staging info param {data['name']}")
                return StagedAssemblyParam(data["stages"], param_name, key)
            else:
                raise Exception(f"Invalid object-parameter type {val['type']}")
        else:
            # if no type is specified this is a parameter group
            return ParamValueGroup(param_name=key, param_value={pkey: parameter_value(pkey, pval)
                                                                for pkey, pval in val.items() if pkey != "name"}, valname=param_name)
    else:
        return ParameterValue(key, val)


class EnsembleParameter:
    """
    Class for a varialbe parameter in a simulation ensemble
    """
    param_key: str
    param_value_set: list[ParameterValue]  # sorry
    param_value_map: dict[str, ParameterValue]

    def __init__(self, key: str, paramData: list[ParameterValue]):
        self.param_key = key
        self.param_value_set = paramData
        self.param_value_map = {
            p.value_name(): p for p in self.param_value_set
        }
        assert len({p.value_name() for p in self.param_value_set}) == len(
            self.param_value_set), "Duplicate param value(s)!"

    def dir_names(self) -> list[str]:
        return [f"{key}_{str(val)}" for key, val in self]

    def is_grouped_params(self) -> bool:
        """
        Returns true if the parameter is grouped, false otherwises
        """
        assert any(isinstance(p, ParamValueGroup) for p in self.param_value_set) == all(
            isinstance(p, ParamValueGroup) for p in self.param_value_set)
        return any(isinstance(p, ParamValueGroup) for p in self.param_value_set)

    def lookup(self, key: str) -> ParameterValue:
        if not isinstance(key, str):
            key = str(key)
        return self.param_value_map[key]

    def __getitem__(self, item) -> ParameterValue:
        if isinstance(item, int):
            return self.param_value_set[item]
        else:
            # assert isinstance(item, str)
            return self.lookup(item)

    """
    ChatGPT wrote this method so use with caution
    """

    def __iter__(self):
        return iter(self.param_value_set)

    def __str__(self) -> str:
        return f"{self.param_key}: [{','.join([p.value_name() for p in self.param_value_set])}]"

    def __len__(self):
        return len(self.param_value_set)

    def __contains__(self, item: ParameterValue):
        return item in self.param_value_set

class EnsembleParameters:
    """
    class for managing a group of ensemble parameters
    """
    __ensemble_parameters: list[EnsembleParameter]
    # map to indexes in ensemble_parameters to avoid duplicate object refs
    __name_map: dict[str, int]

    def __init__(self, param_list: list[EnsembleParameter]):
        self.__ensemble_parameters = param_list
        self.__name_map = {p.param_key: i for i, p in enumerate(self.__ensemble_parameters)}
        assert len(self.__name_map) == len(self.__ensemble_parameters), "Duplicate ensemble parameter names!"

    def __getitem__(self, item: Union[str, int]) -> EnsembleParameter:
        if isinstance(item, int):
            return self.__ensemble_parameters[item]
        else:
            return self.__ensemble_parameters[self.__name_map[item]]

    def set_param(self, param: EnsembleParameter):
        """
        sets or adds an ensemble parameter
        """
        if param.param_key in self.__name_map:
            self.__ensemble_parameters[self.__name_map[param.param_key]] = param
        else:
            self.__name_map[param.param_key] = len(self.__ensemble_parameters)
            self.__ensemble_parameters.append(param)

    def __contains__(self, item: Union[str, EnsembleParameter]) -> bool:
        if isinstance(item, str):
            return item in self.__name_map
        else:
            return item in self.__ensemble_parameters

    def __delitem__(self, key: str):
        if key in self.__name_map:
            idx = self.__name_map[key]
            del self.__ensemble_parameters[idx]
            del self.__name_map[key]
            # update name map indexes
            for k, v in self.__name_map.items():
                if v > idx:
                    self.__name_map[k] = v - 1
        else:
            raise KeyError(f"Ensemble parameter {key} not found!")

    def __len__(self):
        return len(self.__ensemble_parameters)

    def __iter__(self):
        yield from self.__ensemble_parameters

    def to_list(self) -> list[EnsembleParameter]:
        return copy.deepcopy(self.__ensemble_parameters)