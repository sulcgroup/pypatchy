from __future__ import annotations

import copy
import math
from abc import ABC, abstractmethod
from dataclasses import field, dataclass
from typing import Union

import numpy as np
from scipy.spatial.transform import Rotation as R

from .plparticle import PLPatchyParticle
from .plpatch import PLPatch
from ...patchy_base_particle import BaseParticleSet
from ipy_oxdna.util import rotation_matrix


class PLSourceMap(ABC):
    __src_set: BaseParticleSet

    def __init__(self, src: PLParticleSet):
        self.__src_set = src

    def src_set(self) -> BaseParticleSet:
        return self.__src_set

    @abstractmethod
    def normalize(self):
        pass


class PLMultidentateSourceMap(PLSourceMap):
    # maps patch in source to patches in this set
    __patch_map: dict[int, set[int]]
    # reverse of __patch_map, maps patches in this set to patches in source
    __patch_src_map: dict[int, int]
    __conversion_params: MultidentateConvertSettings

    def __init__(self, src: PLParticleSet, src_mapping: dict[int, set[int]], cvt_params: MultidentateConvertSettings):
        super().__init__(src)
        self.__patch_map = src_mapping
        self.__patch_src_map = dict()
        self.__conversion_params = cvt_params
        for src_patch_idx, mdt_patches in self.__patch_map.items():
            for patch_id in mdt_patches:
                self.__patch_src_map[patch_id] = src_patch_idx

    def get_src_patch(self, patch: PLPatch) -> PLPatch:
        """
        Given a patch in this particle set, returns the corresponding unidentate patch
        in the source set
        """
        # unclear if this fail behavior is correct
        assert patch.get_id() in self.__patch_src_map, "Invalid patch provided as arg!"
        return self.src_set().patch(self.__patch_src_map[patch.get_id()])

    def patch_groups(self) -> list[set[int]]:
        return [set(patches) for patches in self.__patch_map.values()]

    def patch_map(self) -> dict[int, set[int]]:
        return self.__patch_map

    def map_patch(self, udt_id: int) -> set[int]:
        assert udt_id in self.__patch_map

        return self.__patch_map[udt_id]

    def normalize(self) -> PLMultidentateSourceMap:
        cvt_params = MultidentateConvertSettings(1,
                                                 0,
                                                 True,
                                                 True,
                                                 MultidentateConvertSettings.ENERGY_SCALE_NONE,
                                                 MultidentateConvertSettings.ALPHA_SCALE_NONE)
        return PLMultidentateSourceMap(self.src_set().normalize(),
                                       src_mapping=copy.deepcopy(self.patch_map()),
                                       cvt_params=cvt_params)

    def get_converstion_params(self) -> MultidentateConvertSettings:
        return copy.deepcopy(self.__conversion_params)


@dataclass
class MultidentateConvertSettings:

    ENERGY_SCALE_NONE = 0  # string key "none", no energy scaling
    ENERGY_SCALE_LINEAR = -1  # string key "linear" divide energy by num patches
    ENERGY_SCALE_LOG = -3  # string key "log", divide energy by ln n_teeth

    ALPHA_SCALE_NONE = 0  # do not scale the alpha
    ALHPA_SCALE_LS = 1  # use the lipari-szabo model to scale the alpha

    n_teeth: int = field()
    dental_radius: float = field()
    torsion: bool = field(default=True) # this will have no effect on non-torsional patches
    follow_surf: bool = field(default=False)
    # energy scale > 0 means "multiply the energy by the scale
    energy_scale_method: Union[int, float] = field(default=ENERGY_SCALE_NONE)  # TODO: flexable support for energy scaling
    alpha_scale_method: Union[int, float] = field(default=ALPHA_SCALE_NONE)

    def __post_init__(self):
        # handle strings
        # TODO: custom exceptions for invalid params
        if isinstance(self.energy_scale_method, str):
            if self.energy_scale_method.lower() == "none":
                self.energy_scale_method = MultidentateConvertSettings.ENERGY_SCALE_NONE
            elif self.energy_scale_method.lower() == "linear":
                self.energy_scale_method = MultidentateConvertSettings.ENERGY_SCALE_LINEAR
            elif self.energy_scale_method.lower() == "log":
                self.energy_scale_method = MultidentateConvertSettings.ENERGY_SCALE_LOG
            else:
                raise Exception(f"Invalid energy scale method {self.energy_scale_method}")

        if isinstance(self.alpha_scale_method, str):
            if self.alpha_scale_method.lower() == "none":
                self.alpha_scale_method = MultidentateConvertSettings.ALPHA_SCALE_NONE
            elif self.alpha_scale_method.lower() in ["ls", "lipari-szabo"]:
                raise Exception(f"lipari-szabo method not supported!")
            else:
                raise Exception(f"Invalid alpha scale method {self.alpha_scale_method}")


    def scale_energy(self, patch_energy: float) -> float:
        if self.energy_scale_method == self.ENERGY_SCALE_NONE:
            return patch_energy
        elif self.energy_scale_method == self.ENERGY_SCALE_LINEAR:
            return patch_energy / self.n_teeth
        elif self.energy_scale_method == self.ENERGY_SCALE_LOG:  # subhajit says this is wrong but I am leaving it as an option
            return patch_energy / np.log(self.n_teeth)  # ln n teeth, basically made up
        else:
            return self.energy_scale_method * patch_energy

    def scale_alpha(self, alpha: float):
        """
        """
        if self.alpha_scale_method == self.ALPHA_SCALE_NONE:
            return alpha
        elif self.alpha_scale_method == self.ALHPA_SCALE_LS:
            raise Exception("Subhajit hasn't derived this yet!")
        else:
            return self.energy_scale_method * alpha


class PLParticleSet(BaseParticleSet):
    """
    Particle set class for PL Particles
    The main reason this subclass was created was in order to store data for multidentate particle mapping
    But it can also store other mappings
    """

    __src_map: PLSourceMap

    # __udt_source: Union[None, BaseParticleSet]
    # # maps patch in source to patches in this set
    # __patch_map: Union[None, dict[int, set[int]]]
    # # reverse of __patch_map, maps patches in this set to patches in source
    # __patch_src_map: Union[None, dict[int, int]]

    def __init__(self,
                 particles: Union[None, list[PLPatchyParticle]] = None,
                 source_map: Union[PLSourceMap, None] = None
                 ):
        if particles is None:
            particles = []
        super().__init__(particles)
        self.__src_map = source_map

    def patch_colors(self):
        return set({patch.color() for patch in self.patches()})

    def get_src_map(self) -> PLSourceMap:
        return self.__src_map

    def get_src(self) -> PLParticleSet:
        return self.get_src_map().src_set()

    def has_udt_src(self) -> bool:
        return self.get_src() is not None and isinstance(self.get_src_map(), PLMultidentateSourceMap)

    def patch_groups(self, particle: Union[int, PLPatchyParticle, None] = None) -> list[set[int]]:
        """
        Returns the patches in this particle set, grouped by the
        """
        assert self.is_multidentate()
        if particle is None:
            return self.get_src_map().patch_groups()
        if isinstance(particle, int):
            particle = self.particle(particle)
        return list(filter(self.get_src_map().patch_groups(),
                           lambda patches: any([self.patch(patch_id) in particle for patch_id in patches])))

    def is_multidentate(self) -> bool:
        return self.has_udt_src() and self.get_src().num_patches() != self.num_patches()

    def particle(self, identifier: Union[int, str]) -> PLPatchyParticle:
        if isinstance(identifier, int):
            return BaseParticleSet.particle(self, identifier)
        else:
            for p in self.particles():
                if p.name() == identifier:
                    return p
            raise IndexError(f"No particle in this set with name {identifier}")

    def mdt_rep(self, udt_id: int) -> set[PLPatch]:
        assert self.has_udt_src()
        return {self.patch(i) for i in self.get_src_map().map_patch(udt_id)}

    def __contains__(self, item: Union[PLPatch, PLPatchyParticle]) -> bool:
        """
        Flexable method which can accept PLPatch or PLPatchyParticle objects
        """
        if isinstance(item, PLPatch):
            for patch in self.patches():
                # use patch ids
                if patch.get_id() == item.get_id():
                    # double check homeopathy (wrong word?)
                    assert patch.color() == item.color(), "Mismatch between patch colors"
                    assert patch.strength() == item.strength(), "Mismatch between patch sterengths"
                    # commenting out these assertions to deal w/ normalized multidentate particles
                    # assert (abs(patch.a1() - item.a1()) < 1e-6).all(), "Mismatch between patch a1 vectors"
                    # assert (abs(patch.a2() - item.a2()) < 1e-6).all(), "Mismatch between patch a3 vectors"
                    # assert (abs(patch.position() - item.position()) < 1e-6).all(), "Mismatch between patch position vectors"
                    return True
            return False
        elif isinstance(item, PLPatchyParticle):
            for particle in self.particles():
                if particle.get_type() == item.get_type():
                    assert item.num_patches() == particle.num_patches(), "Mismatch between particle type patch counts!"
                    return True
            return False
        else:
            raise TypeError(f"{str(item)} has invalid type {type(item)} for PLParticleSet::__contains__")

    def normalize(self) -> PLParticleSet:
        # the correct way to do this would be to make NormedParticleMap a SourceMap
        # and then have some sort of source-map-chaining
        # but i don't cherish THAT cost-benifit analysis
        if self.get_src_map() is not None:
            return PLParticleSet([p.normalize() for p in self.particles()],
                                 source_map=self.get_src_map().normalize())
        # if self.has_udt_src():
        #     return PLParticleSet([p.normalize() for p in self.particles()],
        #                          src=PLMultidentateSourceMap(self.get_src().src_set().normalize(),
        #                                                      src_mapping=copy.deepcopy(self.get_src().patch_map())))
        else:
            return PLParticleSet([p.normalize() for p in self.particles()])

    def to_multidentate(self,
                        mdt_params: MultidentateConvertSettings) -> PLParticleSet:
        """
        Converts a set of patchy particles to multidentate
        Returns:
            athe multidentate base particle set
        """
        new_particles: list[PLPatchyParticle] = [None for _ in self.particles()]
        patch_counter = 0
        new_patches = []
        id_map: dict[int, set[int]] = dict()
        # iter particles
        for i_particle, particle in enumerate(self):
            new_particle_patches = []
            # iter patches in particle
            for patch in particle.get_patches():
                teeth = [None for _ in range(mdt_params.n_teeth)]
                is_color_neg = patch.color() < 0
                # "normalize" color by making the lowest color 0
                if abs(
                    patch.color()) < 21:  assert not mdt_params.torsion, "Torsion cannot be on for same color binding " \
                                                                         "b/c IDK how that works and IDC enough to figure it out"
                # note that the converse is not true; we can have torsion w/o same color binding
                colornorm = abs(patch.color()) - 21
                id_map[patch.type_id()] = set()
                for tooth in range(mdt_params.n_teeth):

                    # grab patch position, a1, a2
                    position = np.copy(patch.position())
                    a1 = np.copy(patch.a1())
                    a2 = np.copy(patch.a2())
                    # if the particle type doesn't include an a2

                    # problem!!!!!
                    if a2 is None:
                        if mdt_params.torsion:
                            raise Exception("Cannot treat non-torsional particle set as torsional!")
                        else:
                            if mdt_params.dental_radius > 0:
                                raise Exception("Even for non-torsional particles, we need an a2 to align teeth "
                                                "unless teeth are superimposed (dental_radius = 0)")

                    # theta is the angle of the tooth within the patch
                    theta = tooth / mdt_params.n_teeth * 2 * math.pi

                    # assign colors
                    # torsional patches need to be assigned colors to
                    if mdt_params.torsion:
                        c = colornorm * mdt_params.n_teeth + tooth + 21
                        if is_color_neg:
                            # opposite-color patches have to be rotated opposite directions
                            # b/c mirroring
                            theta *= -1
                            # set color sign
                            c *= -1
                    else:
                        # non-torsional patches are VERY EASY because you just use the same color again
                        c = patch.color()
                        # theta doesn't need to be adjusted for parity because it's the sames

                    r = R.identity()
                    if mdt_params.dental_radius > 0:
                        if mdt_params.follow_surf:
                            # phi is the angle of the tooth from the center of the patch
                            psi = mdt_params.dental_radius / particle.radius()
                            psi_axis = np.cross(a1, a2)  # axis orthogonal to patch direction and orientation
                            # get rotation
                            r = R.from_matrix(rotation_matrix(psi_axis, psi))
                        else:
                            # move tooth position out of center
                            position += a2 * mdt_params.dental_radius
                        r = r * R.from_matrix(rotation_matrix(a1, theta))
                        position = r.apply(position)
                        a1 = r.apply(a1)
                        # using torsional multidentate patches is HIGHLY discouraged but
                        # this functionality is included for compatibility reasons
                        a2 = r.apply(a2)
                        teeth[tooth] = PLPatch(patch_counter,
                                               c,
                                               position,
                                               a1,
                                               a2,
                                               mdt_params.scale_energy(patch.strength()))
                    # compativility for multidentate patches with 0 radius - may be useful for DNA origami convert
                    else:
                        # simply use unidentat patch position and skip a2
                        teeth[tooth] = PLPatch(patch_counter,
                                               c,
                                               position,
                                               a1,
                                               strength=mdt_params.scale_energy(patch.strength()))

                    id_map[patch.type_id()].add(patch_counter)
                    patch_counter += 1
                # add all teeth
                new_particle_patches += teeth
            new_particles[i_particle] = PLPatchyParticle(type_id=particle.type_id(), index_=i_particle,
                                                         radius=particle.radius(),
                                                         particle_name=particle.name())
            new_particles[i_particle].set_patches(new_particle_patches)
            new_particles[i_particle] = new_particles[i_particle].normalize()
            new_patches += new_particle_patches
        particle_set = PLParticleSet(new_particles, PLMultidentateSourceMap(self, id_map, mdt_params))
        return particle_set
