from __future__ import annotations

import itertools
import re
from copy import deepcopy
from pathlib import Path
from typing import Union

import numpy as np
from Bio.SVDSuperimposer import SVDSuperimposer
from oxDNA_analysis_tools.UTILS.data_structures import TopInfo, Configuration

from ..patchy_base_particle import BasePatchType, PatchyBaseParticleType, PatchyBaseParticle, \
    BaseParticleSet
from ..scene import Scene
from pypatchy.util import dist

# todo: bidict?
MGL_COLORS = [
    "blue",
    "green",
    "red",
    "darkblue",
    "darkgreen",
    "darkred"
]


class MGLParticle(PatchyBaseParticleType, PatchyBaseParticle):
    """
    The MGL specification doesn't really 'do' particle type
    The closest approximation is particle colors (see above) but even those aren't guarentees of anything
    """
    _position: np.ndarray
    _particle_color: str
    _radius: float

    def __init__(self,
                 position: np.ndarray,
                 radius: float,
                 color: str,
                 uid: int,
                 patches: list[BasePatchType]):
        PatchyBaseParticleType.__init__(self, uid, patches)
        PatchyBaseParticle.__init__(self, uid, uid, position)
        self._position = position
        self._particle_color = color
        self._radius = radius

    def get_type(self) -> int:
        return self.color() # shut up

    def color(self) -> str:
        return self._particle_color

    def name(self) -> str:
        return self.color()

    def color_idx(self) -> int:
        return MGL_COLORS.index(self.color())

    def radius(self, normal: np.ndarray = np.zeros(shape=(3,))) -> float:
        return self._radius  # assume a spherical particle

    def cms(self) -> np.ndarray:
        """
        Calculates the center of mass of the particle as defined as
        the average of the positions of the patches
        """
        return np.mean([patch.position() for patch in self.patches()], axis=0)

    def position(self):
        return self._position

    def set_position(self, new_position: np.ndarray):
        self._position = new_position

    def rotation(self) -> np.ndarray:
        """
        The concept of a particle rotation doesn't make sense for an mgl particle with
        no frame of reference
        """
        return np.identity(3)

    def rotate(self, rotation: np.ndarray):
        """
        "rotates" the particle in-place by applying a rotation matrix to the position
        of each patch on the particle
        """
        for p in self.patches():
            p.set_position(p.position() @ rotation)

    def patch_matrix(self) -> np.ndarray:
        return np.stack([p.position() for p in self.patches()])


class MGLPatch(BasePatchType):
    _width: float

    def __init__(self, uid: int, color: str, position: np.ndarray, width: float):
        super().__init__(uid, color)
        self._key_points = [position, ]
        self._width = width

    def width(self):
        return self._width

    def colornum(self):
        return MGL_COLORS.index(self.color())

    def position(self) -> np.ndarray:
        return self._key_points[0]

    def can_bind(self, other: MGLPatch):
        """
        MGL doesn't have a hard-and-fast can-bind behavior but for the purposes
        of this library I'm approximating it as "x" binds with "darkx" where x is any color
        """
        return self.color().endswith(other.color()) or other.color().endswith(self.color())

    def set_position(self, new_position: np.ndarray):
        assert new_position.shape == (3,)
        self._key_points[0] = new_position

    def has_torsion(self):
        return False # mgl patches are innately non-torsional


"""
A scene loaded from an MGL file
"""
class MGLScene(Scene):

    _box_size: np.ndarray

    def __init__(self, box_size: np.ndarray = np.zeros((3,))):
        super().__init__()
        self._box_size = box_size

    def box_size(self) -> np.ndarray:
        return self._box_size

    def box_valid(self) -> bool:
        return not (self.box_size() != 0).any()

    def add_particle(self, p: MGLParticle):
        self._particles.append(p)

    def type_level_map(self) -> dict[str, int]:
        """
        Produces a map of the level of each particle type (assumed to be defined uniquely by color)
        to the number of times that type occurs in this scene
        Important for exporting to patchy simulations
        """
        particle_map = {
            particle.color(): 0
            for particle in self.particle_types().particles()
        }
        for particle in self.particles():
            particle_map[particle.color()] += 1
        return particle_map

    def center(self) -> np.ndarray:
        """
        Returns the center of the scene as defined as the average of all the particle
        positions of the scene
        """
        return np.mean([p.position() for p in self.particles()])

    def recenter(self) -> MGLScene:
        new_scene = MGLScene()
        center = self.center()
        for p in self.particles():
            p = deepcopy(p)
            p.set_position(p.position() - center)
            new_scene.add_particle(p)
        return new_scene

    def patch_ids_unique(self) -> bool:
        patch_ids = list(itertools.chain.from_iterable(
            [[patch.get_id() for patch in particle.patches()] for particle in self.particles()]))
        if min(patch_ids) != 0 or max(patch_ids) != len(patch_ids) - 1:
            return False
        if len(np.unique(patch_ids)) != len(patch_ids):
            return False
        return True

    def particle_types(self) -> MGLParticleSet:
        """
        Returns a BaseParticleSet containing the particle types (as defined by color) in this scene
        WARNING: does not check if particles are actually identical - just checks particle colors!!!
        """
        particle_set = MGLParticleSet()
        colors = set()
        particle_type_counter = 0
        patch_type_counter = 0
        for particle in self._particles:
            if particle.color() not in colors:
                colors.add(particle.color())
                pcopy = deepcopy(particle)
                pcopy.set_id(particle_type_counter)
                particle_type_counter += 1
                for patch in pcopy.patches():
                    patch.set_id(patch_type_counter)
                    patch_type_counter += 1
                particle_set.add_particle(pcopy)

        return particle_set

    def num_particle_types(self) -> int:
        return len(self.particle_types().particles())

    def get_conf(self) -> Configuration:
        # TODO
        pass

    def particles_bound(self, p1: MGLParticle, p2: MGLParticle) -> bool:
        """
        Checks if two particles in this scene are bound
        """
        # iter patch pairs
        for p1patch, p2patch in zip(p1.patches(), p2.patches()):
            # check if patches are complimentary
            if self.patches_bound(p1, p1patch, p2, p2patch):
                return True
        return False

    def get_rot(self, p: Union[MGLParticle, int]) -> np.ndarray:
        """
        computes the rotation of an mgl particle based on its center and patches
        Returns:
            the matrix by which you would need to rotate the type particle of p to make it equal p
        """
        if isinstance(p, int):
            self.get_rot(self.particles()[p])
        else:
            sup = SVDSuperimposer()
            ptype = self.particle_types()[p.color()]
            # don't search orders
            m1 = np.concatenate([ptype.patch_matrix(), np.zeros((1, 3))])
            m2 = np.concatenate([p.patch_matrix(), np.zeros((1, 3))])
            sup.set(m1, m2)
            sup.run()
            assert sup.get_rms() < 1e-6
            rot, tran = sup.get_rotran()
            assert np.linalg.norm(tran) < 1e-3

            assert np.linalg.norm(m1 @ rot.T - m2) < 1e-3, "Bad rotation!"
            return rot.T


    def set_particle_types(self, ptypes: BaseParticleSet):
        """
        please don't
        ok to be more specific the concept of "particle type" doesn't apply super well to mgl
        so you can't really set the particle types k sorry
        """
        pass

    def patches_bound(self,
                      particle1: MGLParticle,
                      p1: MGLPatch,
                      particle2: MGLParticle,
                      p2: MGLPatch) -> bool:
        if p1.can_bind(p2):
            # check binding geometry
            # (warning: sus)
            patch1_pos = p1.position() @ particle1.rotation() + particle1.position()
            patch2_pos = p2.position() @ particle2.rotation() + particle2.position()
            # todo: verify that this behavior is correct!
            d = dist(patch1_pos, patch2_pos)
            # 4 * patch.width = 2 x patch radius x 2 patches
            bind_w = 2 * (2 * p1.width() + 2 * p2.width())
            if d <= bind_w:
                return True
        return False


class MGLParticleSet(BaseParticleSet):
    _colormap: dict[str, MGLParticle]

    def __init__(self, particles: Union[MGLParticleSet, list[MGLParticle], None]=None):
        self._colormap = {}
        super().__init__(particles)

    def add_particle(self, particle: MGLParticle):
        super().add_particle(particle)
        self._colormap[particle.color()] = particle

    def __getitem__(self, item: str) -> MGLParticle:
        return self._colormap[item]


def load_mgl(file_path: Path) -> MGLScene:
    assert file_path.is_file()
    with file_path.open("r") as f:
        # first line of mgl defines box
        first_line = f.readline()
        box_size = np.array([float(x) for x in re.findall(r'\d+\.\d+', first_line)])
        mgl = MGLScene(box_size)
        # each line after the first is a particle
        patch_idx = 0
        type_map: dict[str, MGLParticle] = {}
        for i, line in enumerate(f):
            # first part of an mgl line is the particle position, second is other info, third is patches
            particle_position, particle_data, patch_strs = re.split(r"[@M]", line)
            particle_position = np.array([float(x) for x in particle_position.split()])
            r, particle_color = particle_data.split()
            r = float(r)
            assert particle_color.startswith("C")
            particle_color = particle_color[particle_color.find("[") + 1:particle_color.rfind("]")]
            patch_strs = patch_strs.split()
            assert len(patch_strs) % 5 == 0
            patches = []
            # no delimeter between patches
            j = 0
            while j < len(patch_strs):
                # each patch is specied by 4 numbers and a string
                # the first three numbers specify the patch position, the fourth is the patch width,
                # the string is fifth and is the patch color
                patch_coords = np.array([float(x) for x in patch_strs[j:j + 3]])
                w = float(patch_strs[j + 3])
                patch_color = re.search(r"C\[(\w+)]", patch_strs[j + 4]).group(1)
                patches.append(MGLPatch(patch_idx, patch_color, patch_coords, w))
                patch_idx += 1
                j += 5

            p = MGLParticle(particle_position, r, particle_color, i, patches)
            mgl.add_particle(p)
        return mgl

