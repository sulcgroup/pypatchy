from __future__ import annotations
from typing import Any

import numpy as np
from crisscross.core_functions.megastructures import Megastructure
from crisscross.core_functions.slats import Slat
from oxDNA_analysis_tools.UTILS.data_structures import Configuration
from pypatchy.patchy_base_particle import BaseParticleSet
from pypatchy.scene import Scene
from pypatchy.structure import Structure

from ..patchy_base_particle import PatchyBaseParticle, PatchyBaseParticleType, BasePatchType


# note: i'm like 60% sure this is totally different from Subhajit's PHB work

class PatchySlatType(PatchyBaseParticleType):
    # radius of slat in sim. units
    radius: float
    # length of slat in sim. units
    length: float
    # todo: defaults for radius + length?
    # unlike in most patchy particle classes patch list is ordered (by position)
    patches: list[SlatPatch]

    # todo: patches by helix

    # we can assume this is a 6hb - can number helices 1-6

    def __init__(self, uid: int, patches: list[SlatPatch]):
        super().__init__(uid, sorted(patches, key=lambda p: p.position))

    def name(self) -> str:
        # sure
        return f"6HB_Type{self.type_id()}"

    def radius(self, normal: np.ndarray = np.array([0, 1, 0])) -> float:
        """
        oh god we're using the normal
        """
        if np.allclose(normal, np.array([1, 0, 0])):
            return self.length / 2
        elif np.allclose(normal, np.array(0, 1, 0)):
            return self.radius
        else:
            raise Exception("jesus christ fuck no")

class PatchySlat(PatchyBaseParticle):
    # unit vector of central axis. axis is directional, see SlatPatch location repr
    a1: np.ndarray
    # unit vector for rotation of slat. unit vector represents direction of helix 0
    a2: np.ndarray

    type_particle: PatchySlatType

    def __init__(self,
                 uid: int,
                 type_particle: PatchySlatType,
                 position: np.ndarray,
                 a1: np.ndarray,
                 a2: np.ndarray):
        super().__init__(uid, type_particle.type_id(), position)
        self.type_particle = type_particle
        self.a1 = a1
        self.a2 = a2

    def rotation(self) -> Any:  # todo decide return type
        pass  # todo

    def patches(self):
        return self.type_particle.patches

    def patch(self, idx: int) -> SlatPatch:
        return self.type_particle.patch(idx)

    def num_patches(self) -> int:
        return self.type_particle.num_patches()

    def rotate(self, rotation: np.ndarray):
        # todo test rotmat valid
        self.a1 = rotation @ self.a1
        self.a2 = rotation @ self.a2


class SlatPatch(BasePatchType):
    """
    patch on a slat
    we can represent location as a cylindrical coordinate
    """

    # position of the patch along the length of the 6hb. direction of axis is assumed to be 2H -> 5H,
    # whatever that means
    position: float
    # helices are numbered 0-5 clockwise if you imagine the directional positioning line
    # along the helix pointing away from you
    helix_num: int

    # todo: overload BasePatchType._key_points or smth

    def __init__(self, uid: int, position: int, helix_num: int, color: int):
        super().__init__(uid, color)
        self.position = position
        self.helix_num = helix_num

    def colornum(self) -> int:
        return self.color()

    def can_bind(self, other: SlatPatch):
        return self.color() == other.color()  # hardcode slat colors to same-color binding

    def has_torsion(self):
        return False # todo: this shouldn't be an abstract method of BasePatchType


class SlatSet(BaseParticleSet):
    # anything?
    pass


# may want to extend Structure as well except that that's at terrible idea
class SlatsStructure(Scene, SlatSet):
    """
    wrapper for Megastructure class which implements Scene api
    eventually i will need a class for a patchy particle simulation of slats, this is not that
    """
    ms: Megastructure

    def __init__(self, ms: Megastructure):
        Scene.__init__(self)
        SlatSet.__init__(self)
        self.ms = ms


    def num_particle_types(self) -> int:
        return super(SlatsStructure, self).num_particle_types()

    def particle_types(self) -> SlatSet:
        return self

    def set_particle_types(self, ptypes: BaseParticleSet):
        raise Exception("Cannot set particle types in slats structure")

    def get_conf(self) -> Configuration:
        pass

    def particles_bound(self, p1: PatchyBaseParticle, p2: PatchyBaseParticle) -> bool:
        pass

    def patches_bound(self,
                      particle1: PatchyBaseParticle,
                      p1: BasePatchType,
                      particle2: PatchyBaseParticle,
                      p2: BasePatchType) -> bool:
        pass

