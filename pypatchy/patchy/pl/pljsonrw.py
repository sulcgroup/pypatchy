import json
import numpy as np
from typing import Union, Any
from .plparticle import PLPatchyParticle
from .plpatch import PLPatch
from .plparticleset import PLParticleSet, PLSourceMap, PLMultidentateSourceMap, MultidentateConvertSettings

# TODO: fully test these! right now they are VERY MUCH stopgaps!

class PLJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, PLPatch):
            return {
                "type": obj.type_id(),
                "strength": obj.strength(),
                "position": obj.position().tolist(),
                "a1": obj.a1().tolist(),
                "a2": obj.a2().tolist(),
                "color": obj.color()
            }
        if isinstance(obj, PLPatchyParticle):
            return {
                "type_id": obj.type_id(),
                "position": obj.position().tolist(),
                "radius": obj.radius(),
                "v": obj.v.tolist(),
                "L": obj.L.tolist(),
                "a1": obj.a1.tolist(),
                "a3": obj.a3.tolist(),
                "patches": [self.default(patch) for patch in obj.patches()],
                "name": obj.name()
            }
        if isinstance(obj, PLParticleSet):
            return {
                "particles": [self.default(particle) for particle in obj.particles()],
                "source_map": self.default(obj.get_src_map().__dict__) if obj.get_src_map() else None
            }
        if isinstance(obj, PLSourceMap):
            if isinstance(obj, PLMultidentateSourceMap):
                return {
                    "src_set": self.default(obj.src_set()),
                    "patch_map": obj.patch_map(),
                    "conversion_params": self.default(obj.get_conversion_params())
                }
        if isinstance(obj, MultidentateConvertSettings):
            return {
                "n_teeth": obj.n_teeth,
                "dental_radius": obj.dental_radius,
                "torsion": obj.torsion,
                "follow_surf": obj.follow_surf,
                "energy_scale_method": obj.energy_scale_method,
                "alpha_scale_method": obj.alpha_scale_method
            }
        return json.JSONEncoder.default(self, obj)


class PLJSONDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj: dict) -> Union[PLPatch, PLPatchyParticle, PLParticleSet, MultidentateConvertSettings, dict]:
        if 'type' in obj and 'strength' in obj and 'position' in obj:
            return PLPatch(
                type_id=obj['type'],
                color=obj['color'],
                relposition=np.array(obj['position']),
                a1=np.array(obj['a1']),
                a2=np.array(obj['a2']),
                strength=obj['strength']
            )
        if 'type_id' in obj and 'index' in obj and 'position' in obj:
            patches = [self.object_hook(patch) for patch in obj['patches']]
            particle = PLPatchyParticle(
                patches=patches,
                type_id=obj['type_id'],
                index_=obj['index'],
                position=np.array(obj['position']),
                radius=obj['radius']
            )
            particle.v = np.array(obj['v'])
            particle.L = np.array(obj['L'])
            particle.a1 = np.array(obj['a1'])
            particle.a3 = np.array(obj['a3'])
            particle._name = obj['name']
            return particle
        if 'particles' in obj:
            particles = [self.object_hook(particle) for particle in obj['particles']]
            if obj['source_map']:
                source_map = self.object_hook(obj['source_map'])
                return PLParticleSet(particles=particles, source_map=source_map)
            else:
                return PLParticleSet(particles=particles)
        if 'src_set' in obj and 'patch_map' in obj:
            src_set = self.object_hook(obj['src_set'])
            patch_map = {int(k): set(v) for k, v in obj['patch_map'].items()}
            conversion_params = self.object_hook(obj['conversion_params'])
            return PLMultidentateSourceMap(src_set, src_mapping=patch_map,
                                           cvt_params=conversion_params)
        if 'n_teeth' in obj and 'dental_radius' in obj:
            return MultidentateConvertSettings(
                n_teeth=obj['n_teeth'],
                dental_radius=obj['dental_radius'],
                torsion=obj['torsion'],
                follow_surf=obj['follow_surf'],
                energy_scale_method=obj['energy_scale_method'],
                alpha_scale_method=obj['alpha_scale_method']
            )
        return obj

# Usage example
particle_set = PLParticleSet()  # Your PLParticleSet object
json_data = json.dumps(particle_set, cls=PLJSONEncoder)
print(json_data)

# Decoding example
decoded_particle_set = json.loads(json_data, cls=PLJSONDecoder)
print(decoded_particle_set)
