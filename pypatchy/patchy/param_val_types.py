from __future__ import annotations

from dataclasses import field
from typing import Union, Mapping, Any, Iterator

from pypatchy.patchy.base_param_val import ParameterValue, InvalidParameterTypeError
from pypatchy.patchy.particle_adders import StageParticleAdder, RandParticleAdder, FromPolycubeAdder
from pypatchy.patchy.pl.plparticleset import PLParticleSet, MultidentateConvertSettings

PARTICLE_TYPES_KEY = "particle_types"
MDT_CONVERT_KEY = "mdt_convert"
STAGES_KEY = "staging"

# todo: dataclass?
# you should keep a const instance of this in a list of param vals but can clone them and edit
class ParticleSetParam(ParameterValue, PLParticleSet):
    set_name: str = field(init=False)

    def __init__(self, particles: PLParticleSet, name=PARTICLE_TYPES_KEY):
        ParameterValue.__init__(self, PARTICLE_TYPES_KEY, particles)
        PLParticleSet.__init__(self, particles.get_src_map(), intmat=particles.particles())
        self.set_name = name

    def value_name(self):
        return self.set_name

    def str_verbose(self) -> str:
        pset_str = f"Particle Set {self.set_name}"
        pset_str += f"\n\tNum. Particle Types: {self.num_particle_types()}"
        pset_str += f"\n\tNum. Colors: {len(self.patch_colors()) / 2}"
        return pset_str


class MDTConvertParams(ParameterValue):
    convert_params_name: str

    def __init__(self, cvt_settings: MultidentateConvertSettings, convert_params_name: str = MDT_CONVERT_KEY):
        ParameterValue.__init__(self, MDT_CONVERT_KEY, cvt_settings)
        if not isinstance(cvt_settings, MultidentateConvertSettings):
            raise InvalidParameterTypeError(f"Invalid multidentate convert settings object type {type(cvt_settings)}")
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


class StagedAssemblyParam(ParameterValue):
    """
    A parameter value that describes a staged assembly protocol
    mostly just a grouped parameter system
    """
    param_value: dict[str, StageInfoParam]
    staging_type_name: str

    def __init__(self,
                 staging_info: dict[str, Union[dict, StageInfoParam]],
                 staging_val_name: str,
                 staging_params_name: str = STAGES_KEY):
        ParameterValue.__init__(self,
                                param_name=staging_params_name,
                                param_value={
                                    stage_name: StageInfoParam(stage_name, **stage_info)
                                    for stage_name, stage_info in staging_info.items()
                                } if all([isinstance(v, dict) for v in staging_info.values()]) else staging_info)
        # if isinstance(staging_info, dict):
        # else:
        #     if not isinstance(staging_info, list):
        # TODO: MORE TYPE CHECKING
        # if not isinstance(staging_info, dict) or staging_info:
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

    def add_stage(self, stage: StageInfoParam):
        assert stage.stage_name in self.param_value
        self.param_value[stage.stage_name] = stage



class StageInfoParam(Mapping):
    """
    A simulation parameter that describes a single stage in a staged assembly protocol
    """
    start_time: int
    stage_name: str
    add_method: Union[None, StageParticleAdder]
    # input file params for stage
    info: dict[str, Any]  # TODO: more detail in type hint?
    allow_shortfall: bool

    def __init__(self, stage_name, **kwargs):
        self.stage_name = stage_name
        self.start_time = int(kwargs["t"]) if "t" in kwargs else 0
        # if this stage adds particles
        if "add_method" in kwargs:
            add_method = kwargs["add_method"]
            if isinstance(add_method, StageParticleAdder):
                self.add_method = add_method
            elif isinstance(add_method, str):
                raise TypeError("string-type add methods are no longer supported")
            else:
                if not isinstance(add_method, dict):
                    raise InvalidParameterTypeError("add_method", add_method)
                if "type" not in add_method:
                    self.add_method = RandParticleAdder(**kwargs["add_method"])
                    # change in behavior: if add method type is not specified, assume random
                #     raise TypeError("No type specified for particle add method! Specify 'polycube', 'patchy', "
                #                     "'random', 'fix', or. idk.")
                else:
                    add_type = add_method["type"]
                    del add_method["type"] # delete type key so we can call constructor w/ arg unpacking
                    if add_type == 'random':
                        self.add_method = RandParticleAdder(**kwargs["add_method"])
                    elif add_type == "polycube":
                        self.add_method = FromPolycubeAdder(**kwargs["add_method"])
                    elif add_type == "patchy":
                        raise NotImplementedError("Not implemented yet")
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
            key: value for key, value in kwargs.items() if key not in ("t", "add_method")
        }

    def set_end_time(self, newval: int):
        self.info["steps"] = newval

    def get_end_time(self) -> int:
        return self.info["steps"]

    def get_start_time(self) -> int:
        return self.start_time

    def __getitem__(self, item: str):
        """
        gets the value for an input file param specific to this stage
        :return: value for given stage info key
        """
        return self.info[item]

    def __len__(self) -> int:
        return len(self.info)

    def __iter__(self) -> Iterator:
        return iter(self.info)

    def __contains__(self, item: str) -> bool:
        return item in self.info