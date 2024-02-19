from __future__ import annotations
from dataclasses import dataclass, field
from typing import Union

from .pl.plparticleset import PLParticleSet, MultidentateConvertSettings
# from ..patchyio import get_writer # TODO: sort out this spaghtii

PARTICLE_TYPES_KEY = "particle_types"
MDT_CONVERT_KEY = "mdt_convert"


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
        assert not isinstance(self.param_value, ParameterValue)

    def value_name(self):
        return self.param_value

    def __eq__(self, other: ParameterValue) -> bool:
        return self.param_name == other.param_name and self.value_name() == other.value_name()

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
        return isinstance(other, ParamValueGroup) and all([key in self and self[key] == val
                                                           for key, val in other.param_value.items()])


# probably a shorter, worse way to write this
class EnsembleParameter:
    """
    Class for a varialbe parameter in a simulation ensemble
    """
    param_key: str
    param_value_set: list[ParameterValue]  # sorry
    param_value_map: dict[str, ParameterValue]

    def __init__(self, key: str, paramData: list[EnsembleParameter]):
        self.param_key = key
        self.param_value_set = paramData
        self.param_value_map = {
            p.value_name(): p for p in self.param_value_set
        }
        assert len({p.value_name() for p in self.param_value_set}) == len(self.param_value_set), "Duplicate param value(s)!"

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


class ParticleSetParam(ParameterValue, PLParticleSet):
    set_name: str = field(init=False)

    def __init__(self, particles: PLParticleSet, name=PARTICLE_TYPES_KEY):
        ParameterValue.__init__(self, PARTICLE_TYPES_KEY, particles)
        PLParticleSet.__init__(self, particles.particles())
        self.set_name = name

    def value_name(self):
        return self.set_name


class MDTConvertParams(ParameterValue):
    convert_params_name: str

    def __init__(self, cvt_settings: MultidentateConvertSettings, convert_params_name: str = MDT_CONVERT_KEY):
        ParameterValue.__init__(self, MDT_CONVERT_KEY, cvt_settings)
        self.convert_params_name = convert_params_name
    


    def value_name(self) -> str:
        return self.convert_params_name


def parameter_value(key: str, val: Union[dict, str, int, float, bool]) -> ParameterValue:
    """
    Constructs a ParameterValue object
    """
    if isinstance(val, dict):
        param_name = val["name"]
        # if type key is present, paramater is a particle set or mdt convert settings or something
        if "type" in val:
            if "value" in val: # *sirens* BACKWARDS COMPATIBILITY DANGER ZONE
                data = val["value"]
            else:
                data = {k: val[k] for k in val if k not in ("type", "name")}
            if val["type"] == MDT_CONVERT_KEY:
                return MDTConvertParams(MultidentateConvertSettings(**data), param_name)
            elif val["type"] == PARTICLE_TYPES_KEY:
                raise Exception("Particle type parameters not currently supported!")
                # get_writer().set_directory(get_input_dir())
                # return ParticleSetParam(get_writer().read_particle_types(**data)) # TODO: TEST
            else:
                raise Exception(f"Invalid object-parameter type {val['type']}")
        else:
            # if no type is specified this is a parameter group
            return ParamValueGroup(param_name=key, param_value={pkey: parameter_value(pkey, pval)
                                     for pkey, pval in val.items()}, valname=val["name"])
    else:
        return ParameterValue(key, val)