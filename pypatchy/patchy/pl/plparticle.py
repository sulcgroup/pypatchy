from __future__ import annotations

import copy

from typing import Union, Generator

import numpy as np

from .plpatch import PLPatch
from ...patchy_base_particle import PatchyBaseParticleType, PatchyBaseParticle
from ...util import random_unit_vector, selectColor

PATCHY_NULL_A1: np.ndarray = np.array([0, 0, 1])
PATCHY_NULL_A3: np.ndarray = np.array([1, 0, 0])


MGL_PARTICLE_COLORS = ['blue','green','red','black','yellow','cyan','magenta','orange','violet']
MGL_PATCH_COLORS = [
    'green','violet','pink','brown','orange','red','black']

class PLPatchyParticle(PatchyBaseParticleType, PatchyBaseParticle):
    # HATE making the particle type and the particle the same class but refactoring is objectively not the
    # best use of my time
    cm_pos: np.ndarray
    _radius: float
    v: np.ndarray
    L: np.ndarray
    a1: Union[None, np.ndarray]
    a3: Union[None, np.ndarray]
    _name: str

    all_letters = ['C', 'H', 'O', 'N', 'P', 'S', 'F', 'K', 'I', 'Y']

    def __init__(self, patches: list[PLPatch] = [],
                 particle_name: Union[str, None] = None,
                 type_id=0,
                 index_=0,
                 position=np.array([0., 0., 0.]),
                 radius=0.5):
        PatchyBaseParticleType.__init__(self, type_id, patches)
        PatchyBaseParticle.__init__(self, index_, type_id, position)

        self._radius = radius
        self.v = np.array([0., 0., 0.])
        self.L = np.array([0., 0., 0.])
        self.a1 = PATCHY_NULL_A1
        self.a3 = PATCHY_NULL_A3

        self._name = particle_name

    def name(self) -> str:
        if self._name is None:
            if self.get_uid():
                return f"particle_{self.get_uid()}"
            else:
                return f"particletype_{self.type_id()}"
        else:
            return self._name

    def radius(self, normal: np.ndarray = np.zeros(shape=(3,))) -> float:
        """
        Returns the radius of the particle
        Since the PLPatchyParticle class is assumed to be a sphere, the direction
        vector parameter is ignored
        """
        return self._radius

    def set_radius(self, radius):
        self._radius = radius

    def get_patches(self):
        return self._patches

    def set_patches(self, patches: list[PLPatch]):
        self._patches = patches

    def patches(self) -> list[PLPatch]:
        return self._patches

    def num_patches(self) -> int:
        return len(self._patches)

    def patch_by_id(self, patch_id: int) -> PLPatch:
        """
        warning: O(n) search so don't use too too often
        """
        patches = [p for p in self.patches() if p.type_id() == patch_id]
        assert len(patches) == 1
        return patches[0]

    def set_orientation(self, a1: np.ndarray, a3: np.ndarray):
        assert np.dot(a1, a3) < 1e-6
        self.a1 = a1
        self.a3 = a3

    def rotate(self, rot_matrix: np.ndarray):
        """
        Rotates particle in-place, applying rotation matrix to self.a1, self.a3, and all patches
        """
        assert rot_matrix.shape == (3, 3)
        assert abs(np.linalg.det(rot_matrix) - 1) < 1e-6
        self.a1 = rot_matrix @ self.a1
        self.a3 = rot_matrix @ self.a3

    def normalize(self) -> PLPatchyParticle:
        cpy = copy.deepcopy(self)
        # rotate particle by the transpose of the rotation matrix so that
        # the particle's rotation matrix = I
        cpy.rotate(self.rotmatrix().T)
        assert np.abs(cpy.rotmatrix() - np.identity(3)).sum() < 1e-6
        # rotate patches on cpy by original particle rotation to keep them in
        # same relative position
        for p in cpy.patches():
            p.rotate(self.rotmatrix())
        return cpy

    def matches(self, other: PLPatchyParticle) -> bool:
        if self.type_id() != other.type_id():
            return False
        for i, (p1, p2) in enumerate(zip(self.patches(), other.patches())):
            if p1.type_id() != p2.type_id():
                return False
            if (np.abs(p1.position() - p2.position()) > 1e-6).any():
                return False
        return True

    def set_random_orientation(self):
        self.a1 = random_unit_vector()
        x = random_unit_vector()
        # i had to replace this code Joakim or someone wrote because it's literally the "what not to do" solution
        # self.a1 = np.array(np.random.random(3))
        # self.a1 = self.a1 / np.sqrt(np.dot(self.a1, self.a1))
        # x = np.random.random(3)
        self.a3 = x - np.dot(self.a1, x) * self.a1
        self.a3 = self.a3 / np.sqrt(np.dot(self.a3, self.a3))
        if abs(np.dot(self.a1, self.a3)) > 1e-10:
            raise IOError("Could not generate random orientation?")

    def distance_from(self, particle: PLPatchyParticle, box_size: np.ndarray) -> float:
        d = self.cm_pos - particle.cm_pos
        d[0] -= np.rint(d[0] / float(box_size[0])) * box_size[0]
        d[1] -= np.rint(d[1] / float(box_size[1])) * box_size[1]
        d[2] -= np.rint(d[2] / float(box_size[2])) * box_size[2]
        return np.sqrt(np.dot(d, d))

    def add_patch(self, patch: PLPatch):
        if self._patches is None:
            self._patches = []
        self._patches.append(patch)

    def fill_patches(self, patch_array: list[PLPatch]):
        """
        Loads patch info from a provided list of patches, based on
        the patch IDs contained in self._patch_ids

        """
        if self._patch_ids is not None:
            self._patches = []
            # loop preloadad patch IDs
            for i, patch_id in enumerate(self._patch_ids):
                self._patches.append(copy.deepcopy(patch_array[patch_id]))

    def fill_configuration(self, ls: np.ndarray):
        self.cm_pos = np.array([float(x) for x in ls[0:3]])
        self.a1 = np.array([float(x) for x in ls[3:6]])
        self.a3 = np.array([float(x) for x in ls[6:9]])
        self.v = np.array([float(x) for x in ls[9:12]])
        self.L = np.array([float(x) for x in ls[12:15]])

    def save_conf_to_string(self) -> str:
        conf_string = '%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f' % (
            self.cm_pos[0], self.cm_pos[1], self.cm_pos[2], self.a1[0], self.a1[1], self.a1[2], self.a3[0], self.a3[1],
            self.a3[2], self.v[0], self.v[1], self.v[2], self.L[0], self.L[1], self.L[2])
        return conf_string + '\n'

    # def init_from_dps_file(self, ptype: int, patchids: list[int]):
    #     self._uid = ptype
    #     self._patch_ids = patchids

    def print_icosahedron(self, particle_color: str):
        return f"{np.array2string(self.cm_pos)[1:-1]} @ 0.5 C[{particle_color}] " \
               f"I {np.array2string(self.a1)[1:-1]} {np.array2string(self.a2)[1:-1]}"

    def get_a2(self) -> np.ndarray:
        return np.cross(self.a3, self.a1)

    a2: np.ndarray = property(get_a2)

    def export_to_mgl(self,
                      patch_width: float = 0.1,
                      patch_shrink_scale=0.) -> str:
        # TODO: patch shrink scale to avoid overlapping patches
        # sout = '%f %f %f @ %f C[%s] ' % (self.cm_pos[0],self.cm_pos[1],self.cm_pos[2],self._radius,particle_color)
        # sout = '%f %f %f @ %f C[%s] ' % (self.cm_pos[0], self.cm_pos[1], self.cm_pos[2], 0.4, particle_color)
        # write x,y,z of patch + color
        particle_color = selectColor(self.get_type(), fmt='arr')
        particle_color = ",".join([str(i) for i in particle_color])
        sout = f"{np.array2string(self.position())[1:-1]} @ {self.radius()} C[{particle_color}]\n"
        # mgl format for patches is hardcoded as KF-like
        # so instead we will use spheres
        for i, p in enumerate(self._patches):
            patch_color = selectColor(abs(p.color()), saturation=65 if p.color() > 0 else 25, fmt="arr")
            patch_color = ",".join([str(i) for i in patch_color])
            sout += f"{np.array2string(self.patch_position(p))[1:-1]} @ {patch_width} C[{patch_color}]\n"
        return sout

    def export_to_lorenzian_mgl(self,
                                patch_colors,
                                particle_color,
                                patch_width: float = 0.1,
                                patch_extension: float = 0.2) -> str:
        # sout = '%f %f %f @ %f C[%s] ' % (self.cm_pos[0],self.cm_pos[1],self.cm_pos[2],self._radius,particle_color)

        sout = '%f %f %f @ %f C[%s] ' % (self.cm_pos[0], self.cm_pos[1], self.cm_pos[2], 0.4, particle_color)
        if len(self._patches) > 0:
            sout = sout + 'M '
        for i, p in enumerate(self._patches):
            orientation = np.array([self.a1, self.a2, self.a3])
            # orientation = np.transpose(orientation)
            pos = np.dot(orientation,
                         p.position())  # np.array( np.dot(p._position , self.a1) , np.dot(p._position[1] , self.a2), np.dot( p._position[2] * self.a3
            pos *= (1.0 + patch_extension)
            # print 'Critical: ',p._type,patch_colors
            g = f"{np.array2string(pos)[1:-1]} {patch_width} C[{patch_colors[i]}]"
            # g = '%f %f %f %f C[%s] ' % (pos[0], pos[1], pos[2], patch_width, patch_colors[i])
            sout = sout + g
        return sout

    def patch_ids(self) -> Generator[int, None, None]:
        for patch in self.patches():
            yield patch.get_id()

    def export_to_xyz(self,
                      patch_width=0.4,
                      patch_extension=0.2) -> str:
        letter = PLPatchyParticle.all_letters[self.type_id()]
        sout = '%s %f %f %f ' % (letter, self.cm_pos[0], self.cm_pos[1], self.cm_pos[2])

        return sout

    def rotation(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.a1, self.a2, self.a3

    def rotmatrix(self) -> np.ndarray:
        return np.stack(self.rotation())

    def __contains__(self, item: PLPatch):
        return any([p == item for p in self.patches()])

    def patch_position(self, p: Union[int, PLPatch]) -> np.ndarray:
        """
        get patch position accounting for patch rotation & particle position
        """
        if isinstance(p, int):
            return self.patch_position(self.patch(p))
        return p.position() @ self.rotmatrix() + self.position()

    def patch_a1(self, p: Union[int, PLPatch]) -> np.ndarray:
        """
        get patch a1 vector in global space
        """
        if isinstance(p, int):
            return self.patch_position(self.patch(p))
        return p.a1() @ self.rotmatrix()

    def instantiate(self, uid: int) -> PLPatchyParticle:
        p = copy.deepcopy(self)
        p.set_uid(uid)
        return p
