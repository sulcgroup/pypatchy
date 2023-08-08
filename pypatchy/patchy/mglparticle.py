from __future__ import annotations

import itertools
import re
from copy import deepcopy
from pathlib import Path

import numpy as np

from pypatchy.patchy_base_particle import BaseParticleSet, BasePatchType, PatchyBaseParticleType

# todo: bidict?
MGL_COLORS = [
    "blue",
    "green"
    "red",
    "darkblue",
    "darkgreen",
    "darkred"
]


class MGLParticle(PatchyBaseParticleType):
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
        super().__init__(uid, patches)
        self._position = position
        self._particle_color = color
        self._radius = radius

    def color(self) -> str:
        return self._particle_color

    def color_idx(self) -> int:
        return MGL_COLORS.index(self.color())

    def radius(self, normal: np.ndarray = np.zeros(shape=(3,))) -> float:
        return self._radius # assume a spherical particle

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


class MGLPatch(BasePatchType):
    _width: float

    def __init__(self, uid: int, color: str, position: np.ndarray, width: float):
        super().__init__(uid, color)
        self._key_points = [position,]
        self._width = width

    def width(self):
        return self._width

    def color_idx(self):
        return MGL_COLORS.index(self.color())

    def position(self) -> np.ndarray:
        return self._key_points[0]

    def can_bind(self, other: MGLPatch):
        """
        MGL doesn't have a hard-and-fast can-bind behavior but for the purposes
        of this library I'm approximating it as "x" binds with "darkx" where x is any color
        """
        return self.color().endswith(other.color()) or other.color().endswith(self.color())


class MGLScene:
    _box_size: np.ndarray
    _particles: list[MGLParticle]

    def __init__(self, box_size: np.ndarray=np.zeros((3,))):
        self._box_size = box_size
        self._particles = []

    def box_size(self) -> np.ndarray:
        return self._box_size

    def box_valid(self) -> bool:
        return not (self.box_size() != 0).any()

    def particles(self) -> list[MGLParticle]:
        """
        Returns a list of individual particles in this scene
        """
        return self._particles

    def add_particle(self, p: MGLParticle):
        self._particles.append(p)

    def num_particlees(self) -> int:
        return len(self._particles)

    def particle_set(self) -> BaseParticleSet:
        """
        Returns a BaseParticleSet containing the particle types (as defined by color) in this scene
        WARNING: does not check if particles are actually identical - just checks particle colors!!!
        """
        particle_set = BaseParticleSet()
        colors = set()
        for particle in self._particles:
            if particle.color() not in colors:
                colors.add(particle.color())
                particle_set.add_particle(deepcopy(particle))

        return particle_set

    def type_level_map(self) -> dict[str, int]:
        """
        Produces a map of the level of each particle type (assumed to be defined uniquely by color)
        to the number of times that type occurs in this scene
        Important for exporting to patchy simulations
        """
        particle_map = {
            particle.color(): 0
            for particle in self.particle_set().particles()
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

    def avg_pad_bind_distance(self):
        pass


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
                patch_coords = np.array([float(x) for x in patch_strs[j:j+3]])
                w = float(patch_strs[j+3])
                patch_color = re.search(r"C\[(\w+)]", patch_strs[j+4]).group(1)
                patches.append(MGLPatch(patch_idx, patch_color, patch_coords, w))
                patch_idx += 1
                j += 5

            p = MGLParticle(particle_position, r, particle_color, i, patches)
            mgl.add_particle(p)
        return mgl
