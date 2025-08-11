from __future__ import annotations

from typing import Any, Union

from ipy_oxdna.observable import Observable

from pypatchy.patchy.ensemble_parameter import EnsembleParameter, ParameterValue, ParamValueGroup, MDTConvertParams, \
    ParticleSetParam, StagedAssemblyParam, StageInfoParam
from pypatchy.patchy.particle_adders import StageParticleAdder, RandParticleAdder, FromPolycubeAdder, FromConfAdder
from pypatchy.patchy.pl.pljsonrw import PLJSONEncoder
from pypatchy.patchy.pl.plparticleset import MultidentateConvertSettings
from pypatchy.patchy.simulation_specification import ParamSet


class SimEnsembleJsonEncoder(PLJSONEncoder):
    """
    JSON encoder class for all stuff required by PatchySimulationEnsemble
    """

    def default(self, obj: Any) -> Union[dict, list, float, int, bool, str]:
        if type(obj) in [int, bool, str, float]:
            return obj
        elif type(obj) == dict:
            return {
                key: self.default([obj[key]]) for key in obj
            }
        elif type(obj) == list:
            return [
                self.default(val) for val in obj
            ]
        elif isinstance(obj, ParamSet):
            return {
                pv.param_name: self.default(pv)
                for pv in obj.param_vals
            }
        elif isinstance(obj, EnsembleParameter):
            return self.encode_ensemble_parameter(obj)
        elif isinstance(obj, ParameterValue):
            return self.encode_param_value(obj)
        elif isinstance(obj, Observable):
            return obj.export()
        # StageInfoParam should probably inherit from ParameterValue, but it doesn't right now
        # and i'm afraid to change that
        elif isinstance(obj, StageInfoParam):
            infodict = {
                "stage_name": obj.stage_name,
                **obj.info
            }
            if obj.start_time > 0:
                infodict["t"] = obj.start_time
            if obj.allow_shortfall:
                infodict["allow_shortfall"] = True
            if obj.add_method is not None:
                infodict["add_method"] = encode_add_method(obj.add_method)
            return infodict
        else:
            return super().default(obj)

    def encode_ensemble_parameter(self, obj: EnsembleParameter) -> list[str, list[Any]]:
        # important that dict keys match constructor args
        return [
            obj.param_key,
            [self.encode_param_value(v) for v in obj.param_value_set]
        ]

    def encode_param_value(self, obj: ParameterValue) -> dict:
        if isinstance(obj, ParamValueGroup):
            return {
                "param_name": obj.param_name,
                "valname": obj.value_name(),
                "param_value": {
                    pv.param_name: self.encode_param_value(pv) for pv in obj.param_value.values()
                }
            }
        # specific versions of parameter value group
        elif isinstance(obj, MDTConvertParams):
            mdt_dict = {
                "n_teeth": obj.param_value.n_teeth,
                "dental_radius": obj.param_value.dental_radius
            }
            # things with default value; don't waste space with defaults
            if not obj.param_value.torsion:
                mdt_dict["torsion"] = obj.param_value.torsion
            if not obj.param_value.follow_surf:
                mdt_dict["follow_surf"] = obj.param_value.follow_surf
            if obj.param_value.energy_scale_method != MultidentateConvertSettings.ENERGY_SCALE_LINEAR:
                if obj.param_value.energy_scale_method == MultidentateConvertSettings.ENERGY_SCALE_NONE:
                    mdt_dict["energy_scale_method"] = "none"
                elif obj.param_value.energy_scale_method == MultidentateConvertSettings.ENERGY_SCALE_LOG:
                    mdt_dict["energy_scale_method"] = "log"
                else:
                    # energy scale number is a scalar value
                    mdt_dict["energy_scale_method"] = obj.param_value.energy_scale_method
            if obj.param_value.alpha_scale_method != MultidentateConvertSettings.ALPHA_SCALE_NONE:
                # TODO: lipari-szabo method or something
                if obj.param_value.alpha_scale_method == -1:
                    raise Exception("lipari-szabo method not supported!")
                else:
                    # energy scale number is a scalar value
                    mdt_dict["alpha_scale_method"] = obj.param_value.alpha_scale_method
            return mdt_dict
        elif isinstance(obj, ParticleSetParam):
            return super().default(obj)
        elif isinstance(obj, StagedAssemblyParam):
            return [
                self.default(stage) for stage in obj.param_value.values()
            ]

        # default parameter value
        else:
            return obj.param_value

def encode_add_method(add_method: StageParticleAdder) -> dict:
    infodict = {}
    if isinstance(add_method, RandParticleAdder):
        infodict["particles"] = add_method.particles
        if not add_method.extern:
            infodict["extern"] = False
    elif isinstance(add_method, FromPolycubeAdder):
        infodict["polycubes"] = [
            {
                "n_copies": pc.n_copies,
                "patch_distance_multiplier": pc.patch_distance_multiplier,
                "polycube_file_path": {
                    'cube_types': pc.polycube_file_path.rule.toJSON(),
                    'cubes': [
                        cube.toJSON() for cube in pc.polycube_file_path.particles()
                    ]
                }
            } for pc in add_method.polycubes
        ]
    elif isinstance(add_method, FromConfAdder):
        raise NotImplemented("FromConfAdder not yet implemented!")
    else:
        raise TypeError(f"Invalid adder type {type(add_method)}")
    return infodict
