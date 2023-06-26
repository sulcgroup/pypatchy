import itertools
import os
import pathlib


class ParameterValue:
    """
    A simple ParameterValue has a key that's a parameter (T, narrow_type, density, etc.) and
    values that are ints, strs, or floats
    A more complex ParameterValue consists of a named group of multiple parameters
    """

    def __init__(self, key, val):
        # values for the parameter can be either simple types (int, str, or float) which are
        # pretty simple, or object, which are really really not
        if isinstance(val, dict):
            self.name = val['name']
            self.value = val['value']
        else:
            self.name = key
            self.value = val

    def is_grouped_params(self) -> bool:
        return isinstance(self.value, dict)

    """
    Return a list of names of parameters which have values specified in this ParameterValue object
    """

    def group_params_names(self) -> list[str]:
        assert self.is_grouped_params()
        return list(self.value.keys())

    """
    Returns a true if this ParameterValue object specifies a value for the parameter
    with name param_name
    """

    def has_param(self, param_name: str) -> bool:
        if not self.is_grouped_params():
            return self.name == param_name
        else:
            return param_name in self.group_params_names()

    def __getitem__(self, key: str):
        assert self.is_grouped_params()
        return self.value[key]

    def __str__(self) -> str:
        if not self.is_grouped_params():
            return str(self.value)
        else:
            return self.name


class EnsembleParameter:
    def __init__(self, key: str, paramData):
        self.param_key = key
        self.param_value_set = [ParameterValue(key, val) for val in paramData]
        names = itertools.chain.from_iterable(
            [p.group_params_names() if p.is_grouped_params() else self.param_key for p in self.param_value_set])
        self.parameter_names = set(names)

    def dir_names(self) -> list[str]:
        return [f"{key}_{str(val)}" for key, val in self]

    def param_names(self) -> set[str]:
        return self.parameter_names

    """
    ChatGPT wrote this method so use with caution
    """

    def __iter__(self):
        return iter([(self.param_key, val) for val in self.param_value_set])


class SimulationSpecification:
    """
    parameter_values is a list of (key,value) tuples where the key is a string identifier
    and the value is a ParameterValue object
    """

    def __init__(self, parameter_values):
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

    def get_folder_path(self) -> pathlib.Path:
        return pathlib.Path(os.sep.join([f"{key}_{str(val)}" for key, val in self.param_vals]))

    def __repr__(self) -> str:
        return ", ".join([f"{key}={self[key]}" for key in self.var_names()])
