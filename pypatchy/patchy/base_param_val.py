from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union, Any, Iterator


@dataclass
class ParameterValue:
    """
    A simple ParameterValue has a key that's a parameter (T, narrow_type, density, etc.) and
    values that are ints, strs, or floats
    Base class is only applicable to basic types (int, float, str, bool). groups of params need ParamValueGroup
    others are also included
    A more complex ParameterValue consists of a named group of multiple parameters
    """
    param_name: str = field()
    param_value: Union[str, bool, float, int] = field()  # only basic types allowed here

    def __post_init__(self):
        if isinstance(self.param_value, ParameterValue):
            raise InvalidParameterError(self.param_name, self.param_value, True)

    def value_name(self):
        return str(self.param_value)

    def __eq__(self, other: Union[ParameterValue, Any]) -> bool:
        """
        Tests equality. If arguement is a ParameterValue, tests if name and value are equal. Otherwise tests
        if self.value == other
        """
        if not isinstance(other, ParameterValue):
            return self.param_value == other
        else:
            return self.param_name == other.param_name and self.value_name() == other.value_name()

    def str_verbose(self) -> str:
        """
        Returns: a string describing the parameter value
        """
        return f"{self.param_name}: {self.value_name()}"

    def __str__(self):
        return f"{self.param_name}_{str(self.value_name())}"

    def __hash__(self) -> int:
        return hash((self.param_name, self.value_name(),))


class InvalidParameterError(BaseException):
    _param_name: str
    _param_value: Any
    _custom_msg: str

    def __init__(self, param_name: str, param_value: Any, custom: Union[str, None] = None):
        """
        :param param_name: name of param with invalid value
        :param param_value: value of param with invalid value. given the nature of this exception this won't
        always be stringifyable
        """
        self._param_name = param_name
        self._param_value = param_value

    def msg(self) -> str:
        return f"Parameter {self._param_name} has invalid value {self._param_value}. Value type is {type(self._param_value)}"

    def __str__(self):
        return self.msg() + self._custom_msg if self._custom_msg else self.msg()


class InvalidParameterTypeError(InvalidParameterError, TypeError):
    def __init__(self, param_name: str, param_value: Any, custom: Union[str, None] = None):
        InvalidParameterError.__init__(param_name, param_value, custom)

    def msg(self):
        return f"Parameter {self._param_name} has invalid value type {type(self._param_value)}. Value is {self._param_value}"


@dataclass
class ParamValueGroup(ParameterValue):
    """
    grouped params
    """
    param_value: dict[str, ParameterValue]
    valname: str

    def value_name(self):
        return self.valname

    def group_params_names(self) -> list[str]:
        return list(self.param_value.keys())

    def __getitem__(self, key: str):
        return self.param_value[key].param_value

    def has_param(self, param_name: str) -> bool:
        return param_name in self.group_params_names()

    def __contains__(self, val: Union[str, ParameterValue]):
        if isinstance(val, str):
            return val in self.param_value
        else:
            return val.param_name in self.param_value and self.param_value[val.param_name]

    def __eq__(self, other: ParameterValue):
        if self.param_name == other.param_name:
            return self.value_name() == other.value_name()
        else:
            return isinstance(other, ParamValueGroup) and all([key in self and self[key] == val
                                                           for key, val in other.param_value.items()])

    def str_verbose(self) -> str:
        return f"{self.param_name}: {self.value_name()}\n" + \
               "\n".join([v.str_verbose().replace("\n", "\n\t") for v in self.param_value.values()])

    def __iter__(self) -> Iterator[ParameterValue]:
        for v in self.param_value.values():
            if isinstance(v, ParamValueGroup):
                yield from v
            else:
                yield v


