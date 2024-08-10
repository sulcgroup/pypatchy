import json
import numpy as np
from typing import Union, Any
from .plparticle import PLPatchyParticle
from .plpatch import PLPatch
from .plparticleset import PLParticleSet, PLSourceMap, PLMultidentateSourceMap, MultidentateConvertSettings


# TODO: fully test these! right now they are VERY MUCH stopgaps!
def encode_patch(patch: PLPatch) -> dict:
    return {
        "id": patch.type_id(),
        "strength": patch.strength(),
        "position": patch.position().tolist(),
        "a1": patch.a1().tolist(),
        "a2": patch.a2().tolist(),
        "color": patch.color()
    }


def decode_patch(patch_dict: dict[str, Any]) -> PLPatch:
    return PLPatch(**patch_dict)


def encode_particle_type(particle_type: PLPatchyParticle) -> dict:
    return {
        "type": particle_type.type_id(),
        "radius": particle_type.radius(),
        "patches": [patch.type_id() for patch in particle_type.patches()],
        "name": particle_type.name()
    }


def encode_particle_set(particle_set: PLParticleSet) -> dict:
    plset_dict = {
        "particle_types": [encode_particle_type(particle) for particle in particle_set.particles()],
        "patches": [encode_patch(patch) for patch in particle_set.patches()]
    }
    if particle_set.get_src_map():
        plset_dict["source_map"] = encode_src_map(particle_set.get_src_map())
    return plset_dict


def encode_src_map(source_map: PLSourceMap) -> dict:
    if isinstance(source_map, PLMultidentateSourceMap):
        return {
            "src_set": encode_particle_set(source_map.src_set()),
            "patch_map": source_map.patch_map(),
            "conversion_params": encode_mdt_settings(source_map.get_conversion_params())
        }


def encode_mdt_settings(mdt_settings: MultidentateConvertSettings) -> dict[str, Any]:
    settings_dict = {
        "n_teeth": mdt_settings.n_teeth,
        "dental_radius": mdt_settings.dental_radius,

    }
    if not mdt_settings.torsion:
        settings_dict["torsion"] = False
    if mdt_settings.follow_surf:
        settings_dict["follow_surf"] = True
    if mdt_settings.energy_scale_method != MultidentateConvertSettings.ENERGY_SCALE_LINEAR:
        settings_dict["energy_scale_method"] = mdt_settings.energy_scale_method
    if mdt_settings.alpha_scale_method != MultidentateConvertSettings.ALPHA_SCALE_NONE:
        settings_dict["alpha_scale_method"] = mdt_settings.alpha_scale_method
    return settings_dict


class PLJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Union[dict, list, float, int, bool, str]:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, PLPatch):
            encode_patch(obj)
        if isinstance(obj, PLPatchyParticle):
            return encode_particle_type(obj)
        if isinstance(obj, PLParticleSet):
            return encode_particle_set(obj)
        if isinstance(obj, PLSourceMap):
            return encode_src_map(obj)
        if isinstance(obj, MultidentateConvertSettings):
            return encode_mdt_settings(obj)
        return json.JSONEncoder.default(self, obj)

