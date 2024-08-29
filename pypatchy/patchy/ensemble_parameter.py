from __future__ import annotations
from dataclasses import dataclass, field
from typing import Union, Any, Mapping, Iterator

from .particle_adders import StageParticleAdder, RandParticleAdder, FromPolycubeAdder
from .pl.plparticleset import PLParticleSet, MultidentateConvertSettings

# from ..patchyio import get_writer # TODO: sort out this spaghtii

PARTICLE_TYPES_KEY = "particle_types"
MDT_CONVERT_KEY = "mdt_convert"
STAGES_KEY = "staging"


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
        return str(self.param_value)

    def __eq__(self, other: ParameterValue) -> bool:
        return self.param_name == other.param_name and self.value_name() == other.value_name()

    def str_verbose(self) -> str:
        """
        Returns: a string describing the parameter value
        """
        return f"{self.param_name}: {self.value_name()}"

    def __hash__(self) -> int:
        return hash((self.param_name, self.value_name(),))


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

    def str_verbose(self) -> str:
        return f"{self.param_name}: {self.value_name()}\n" + \
               "\n".join([v.str_verbose().replace("\n", "\n\t") for v in self.param_value.values()])

    def __iter__(self) -> Iterator[ParameterValue]:
        for v in self.param_value.values():
            if isinstance(v, ParamValueGroup):
                yield from v
            else:
                yield v

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
        PLParticleSet.__init__(self, particles.particles(), particles.get_src_map())
        self.set_name = name

    def value_name(self):
        return self.set_name

    def str_verbose(self) -> str:
        pset_str = f"Particle Set {self.set_name}"
        pset_str += f"\n\tNum. Particle Types: {self.num_particle_types()}"
        pset_str += f"\n\tNum. Colors: {len(self.patch_colors()) / 2}"


class MDTConvertParams(ParameterValue):
    convert_params_name: str

    def __init__(self, cvt_settings: MultidentateConvertSettings, convert_params_name: str = MDT_CONVERT_KEY):
        ParameterValue.__init__(self, MDT_CONVERT_KEY, cvt_settings)
        if not isinstance(cvt_settings, MultidentateConvertSettings):
            raise TypeError(f"Invalid multidentate convert settings object type {type(cvt_settings)}")
        self.convert_params_name = convert_params_name

    def value_name(self) -> str:
        return self.convert_params_name

    def str_verbose(self) -> str:
        cvt_params = f"Multidentate Convert Settings {self.convert_params_name}:\n" \
                     f"\tNum Teeth: {self.param_value.n_teeth}\n" \
                     f"\tDental Radius: {self.param_value.dental_radius}\n" \
                     f"\tTorsion: {self.param_value.torsion}\n" \
                     f"\tFollow Surface: {self.param_value.follow_surf}\n" \
                     f"\tEnergy Scale: {self.param_value.energy_scale_method}\n"
        # f"\tAlpha Scale: {self.param_value.alpha_scale_method}\n" # alpha scale is not used
        return cvt_params


# class StageInfo?

class StagedAssemblyParam(ParameterValue):
    """
    mostly just a grouped parameter system
    """
    param_value: dict[str, StageInfoParam]
    staging_type_name: str

    def __init__(self, staging_info: dict[str, dict], staging_val_name: str, staging_params_name: str = STAGES_KEY):
        ParameterValue.__init__(self, staging_params_name, {
            stage_name: StageInfoParam(stage_name, **stage_info) for stage_name, stage_info in staging_info.items()
        })
        # if isinstance(staging_info, dict):
        # else:
        #     if not isinstance(staging_info, list):
        # # TODO: MORE TYPE CHECKING
        # if not isinstance(staging_info, dict):
        #     raise TypeError("Incorrect type for stages info")
        # if not all(["t" in stage for stage in staging_info.values()]):
        #     raise TypeError("Missing start-time info for some stages")
        self.staging_type_name = staging_val_name

    def value_name(self) -> str:
        return self.staging_type_name

    def get_stages(self) -> dict[str, dict]:
        return self.param_value

    def stage(self, i: int) -> StageInfoParam:
        return list(self.param_value.values())[i]

    def str_verbose(self) -> str:
        staging_desc = f"Staging {self.staging_type_name}"
        for stage_name, stage_info in self.get_stages().items():
            staging_desc += f"Stage: {stage_name}"

        return staging_desc


# todo: dataclass?
# you should keep a const instance of this in a list of param vals but can clone them and edit
class StageInfoParam(Mapping):
    start_time: int
    stage_name: str
    add_method: Union[None, StageParticleAdder]
    info: dict[str, Any]  # TODO: more detail?
    allow_shortfall: bool

    def __init__(self, stage_name, **kwargs):
        self.stage_name = stage_name
        self.start_time = kwargs["t"] if "t" in kwargs else 0
        # if this stage adds particles
        if "add_method" in kwargs:
            if "density" not in kwargs:
                    raise TypeError("particle adder specified without density!")
            if isinstance(kwargs["add_method"], str):
                if kwargs["add_method"].upper() == "RANDOM":
                    self.add_method = RandParticleAdder(kwargs["density"])
                # backwards-compatiility with string addition
                elif "=" in kwargs["add_method"]:
                    self.add_method = FromPolycubeAdder(kwargs["add_method"].split("=")[1])
                else:
                    raise TypeError(f"Invalid 'add_method' provided: {kwargs['add_method']}")
            elif isinstance(kwargs["add_method"], dict):
                if "type" not in kwargs["add_method"]:
                    raise TypeError("No type specified for particle add method! Specify 'polycube', 'patchy', "
                                    "'random', 'fix', or. idk.")
                add_type = kwargs["add_method"]["type"]
                if add_type == 'random':
                    self.add_method = RandParticleAdder(kwargs["density"])
                elif add_type == "polycube":
                    self.add_method = FromPolycubeAdder(*kwargs["add_method"]["polycubes"], density=kwargs["density"])
                elif add_type == "patchy":
                    raise Exception("Not implemented yet")
            else:
                raise TypeError(f"Invalid 'add_method' provided: {kwargs['add_method']}")
        else:
            self.add_method = None

        if "allow_shortfall" in kwargs:
            self.allow_shortfall = kwargs["allow_shortfall"]
        else:
            self.allow_shortfall = False

        # add more params
        self.info = {
            key: value for key, value in kwargs.items() if key not in ("t", "add_method", "density")
        }

    def set_end_time(self, newval: int):
        self.info["steps"] = newval
    def get_end_time(self) -> int:
        return self.info["steps"]

    def get_start_time(self) -> int:
        return self.start_time

    def __getitem__(self, item: str):
        return self.info[item]

    def __len__(self) -> int:
        return len(self.info)

    def __iter__(self) -> Iterator:
        return iter(self.info)


def parameter_value(key: str, val: Union[dict, str, int, float, bool]) -> ParameterValue:
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
                raise Exception("Particle type parameters not currently supported!")
                # get_writer().set_directory(get_input_dir())
                # return ParticleSetParam(get_writer().read_particle_types(**data)) # TODO: TEST
            elif val["type"] == STAGES_KEY:
                if not "stages" in data:
                    raise TypeError(f"Missing key 'stages' in staging info param {data['name']}")
                return StagedAssemblyParam(data["stages"], param_name, key)
            else:
                raise Exception(f"Invalid object-parameter type {val['type']}")
        else:
            # if no type is specified this is a parameter group
            return ParamValueGroup(param_name=key, param_value={pkey: parameter_value(pkey, pval)
                                                                for pkey, pval in val.items()}, valname=param_name)
    else:
        return ParameterValue(key, val)
