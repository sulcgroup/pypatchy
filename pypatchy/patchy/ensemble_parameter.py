from __future__ import annotations
import itertools
import os
import pathlib
from typing import Union


class ParameterValue:
    """
    A simple ParameterValue has a key that's a parameter (T, narrow_type, density, etc.) and
    values that are ints, strs, or floats
    A more complex ParameterValue consists of a named group of multiple parameters
    """
    param_name: str
    value_name: str
    param_value: Union[str, bool, float, dict, int]

    def __init__(self, key, val):
        # values for the parameter can be either simple types (int, str, or float) which are
        # pretty simple, or object, which are really really not
        self.param_name = key
        # if the parameter is grouped, the value name and the value are different
        if isinstance(val, dict):
            self.value_name = val['name']
            self.param_value = val['value']
        else:  # otherwise the value name and the value are the same
            self.value_name = str(val)
            self.param_value = val

    def is_grouped_params(self) -> bool:
        return isinstance(self.param_value, dict)

    """
    Return a list of names of parameters which have values specified in this ParameterValue object
    """

    def group_params_names(self) -> list[str]:
        return list(self.param_value.keys())

    """
    Returns a true if this ParameterValue object specifies a value for the parameter
    with name param_name
    """

    def has_param(self, param_name: str) -> bool:
        if not self.is_grouped_params():
            return self.param_name == param_name
        else:
            return param_name in self.group_params_names()

    def __getitem__(self, key: str):
        assert self.is_grouped_params()
        return self.param_value[key]

    def __str__(self) -> str:
        if not self.is_grouped_params():
            return str(self.param_value)
        else:
            return self.param_name

    def __eq__(self, other: ParameterValue) -> bool:
        return self.param_name == other.param_name and self.value_name == other.value_name
    
    def __hash__(self) -> int:
        return hash(str(self))


class EnsembleParameter:
    """
    Class for a varialbe parameter in a simulation ensemble
    """
    param_key: str
    param_value_set: list[ParameterValue]  # sorry
    param_value_map: dict[str, ParameterValue]

    def __init__(self, key: str, paramData):
        self.param_key = key
        self.param_value_set = [ParameterValue(key, val) for val in paramData]
        self.param_value_map = {
            p.value_name: p for p in self.param_value_set
        }

    def dir_names(self) -> list[str]:
        return [f"{key}_{str(val)}" for key, val in self]

    def is_grouped_params(self) -> bool:
        """
        Returns true if the parameter is grouped, false otherwise
        """
        assert any(p.is_grouped_params() for p in self.param_value_set) == all(
            p.is_grouped_params() for p in self.param_value_set)
        return any(p.is_grouped_params() for p in self.param_value_set)

    def lookup(self, key: str) -> ParameterValue:
        return self.param_value_map[key]

    def __getitem__(self, item) -> ParameterValue:
        if isinstance(item, int):
            return self.param_value_set[item]
        else:
            assert isinstance(item, str)
            return self.lookup(item)

    """
    ChatGPT wrote this method so use with caution
    """

    def __iter__(self):
        return iter(self.param_value_set)

    def __str__(self) -> str:
        return f"{self.param_key}: [{','.join([p.value_name for p in self.param_value_set])}]"

    def __len__(self):
        return len(self.param_value_set)

    def __contains__(self, item: ParameterValue):
        if isinstance(item, ParameterValue):
            return item in self.param_value_set
        else:
            return ParameterValue(*item) in self
