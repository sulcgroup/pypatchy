from __future__ import annotations

import copy
import math
from abc import abstractmethod, ABC

from typing import Union, Generator

import numpy as np
from scipy.spatial.transform import Rotation as R

from .plpatch import PLPatch
from ...patchy_base_particle import PatchyBaseParticleType, PatchyBaseParticle, BaseParticleSet
from ...util import random_unit_vector, rotation_matrix


class PLPatchyParticle(PatchyBaseParticleType, PatchyBaseParticle):
    # HATE making the particle type and the particle the same class but refactoring is objectively not the
    # best use of my time
    cm_pos: np.ndarray
    unique_id: int
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

        self.unique_id = index_
        self._radius = radius
        self.v = np.array([0., 0., 0.])
        self.L = np.array([0., 0., 0.])
        self.a1 = np.array([0, 0, 1])
        self.a3 = np.array([1, 0, 0])

        self._name = particle_name

    def name(self) -> str:
        if self._name is None:
            if self.unique_id:
                return f"particle_{self.get_uid()}"
            else:
                return f"particletype_{self.type_id()}"
        else:
            return self._name

    def set_uid(self, new_uid: int):
        self.unique_id = new_uid

    def get_uid(self) -> int:
        return self.unique_id

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

    # def get_patch_position(self, patchid: int) -> np.ndarray:
    #     assert -1 < patchid < self.num_patches(), "Index out of bounds"
    #     p = self._patches[patchid]
    #     return p.position()[0] * self.a1 + p.position()[1] * self.a2 + p.position()[2] * self.a3
    #
    # def get_patch_orientation_a1(self, patchid) -> np.ndarray:
    #     p: PLPatch = self.patch(patchid)
    #     v = np.array(self.a1 * p.a1()[0] + self.a2 * p.a1()[1] + self.a3 * p.a1()[2])
    #     return v
    #
    # def get_patch_orientation_a2(self, patchid) -> np.ndarrayh:
    #     p: PLPatch = self.patch(patchid)
    #     if np.dot(p.a2(), p.a2()) < 0.9999:  # account for rounding errors
    #         print("PRIMARY MAX CRITICAL ERRROR MAX CRITICAL ERROR", p.a2(), np.dot(p.a2(), p.a2()))
    #     v = np.array(self.a1 * p.a2()[0] + self.a2 * p.a2()[1] + self.a3 * p.a2()[2])
    #     if np.dot(v, v) < 0.9999:  # account for rounding errors
    #         print("MAX CRITICAL ERROR MAX CRITICAL ERROR", v, np.dot(v, v))
    #     return v
    #
    # def set_patch_a2_orientation(self, patchid, new_a2):
    #     p: PLPatch = self.patch(patchid)
    #     coor = np.array([np.dot(new_a2, self.a1), np.dot(new_a2, self.a2), np.dot(new_a2, self.a3)])
    #     p.set_a2(coor / np.sqrt(np.dot(coor, coor)))

    # def aligned_with(self, p: PLPatchyParticle) -> Union[None, dict[int, int]]:
    #     # checks if paticles are aligned
    #     # print 'Verification of alignment ', self.cm_pos, p.cm_pos
    #     if len(self.patches()) != len(p.patches()):
    #         return None
    #     correspondence = {}
    #     for i, patchA in enumerate(self.patches()):
    #         positionA = self.get_patch_position(i)
    #         for j, patchB in enumerate(p.patches()):
    #             positionB = p.get_patch_position(j)
    #
    #             val = np.dot(positionA / norm(positionA), positionB / norm(positionB))
    #             if val > 1.0 - myepsilon:
    #                 if j in correspondence.values():
    #                     # print 'Error two patches would correspond to the same patch'
    #                     return None
    #                 else:
    #                     correspondence[i] = j
    #                     # print 'CHECKING patch positions, we have MATCH ' ,i,j, positionA, positionB, np.dot(positionA/l2norm(positionA),positionB/l2norm(positionB))
    #                     break
    #         if i not in correspondence.keys():
    #             # print 'Could not match patch ',i
    #             return None
    #
    #     # print 'Found perfect correspondence',correspondence
    #     return correspondence

    # def align_with(self, part2: PLPatchyParticle) -> tuple:
    #     all_pos = [x.position() for x in self.patches()]
    #     all2_pos = [x.position() for x in part2.patches()]
    #     print('Trying to align FROM:', all_pos, '\n TO: ', all2_pos)
    #     for pi, p in enumerate(self._patches):
    #         for qi, q in enumerate(self._patches):
    #             if qi != pi:
    #                 for li, l in enumerate(part2.patches()):
    #                     for fi, f in enumerate(part2.patches()):
    #                         if li != fi:
    #                             # print 'Aligning patch %d with %d; and %d with %d' % (pi,li,qi,fi)
    #                             v1 = p.position() / norm(p.position())
    #                             v2 = l.position() / norm(l.position())
    #                             b1 = q.position() / norm(q.position())
    #                             b2 = f.position() / norm(f.position())
    #                             v1 = np.matrix(v1).transpose()
    #                             b1 = np.matrix(b1).transpose()
    #                             B = v1 * v2 + b1 * b2
    #                             U, s, V = np.linalg.svd(B, full_matrices=True)
    #                             M = np.diag([1, 1, np.linalg.det(U) * np.linalg.det(V)])
    #                             R = U * M * V
    #                             rot = np.asarray(R)
    #                             test = copy.deepcopy(part2)
    #                             test.rotate(rot)
    #                             c = self.aligned_with(test)
    #                             if c is not None:
    #                                 # print 'Success! '
    #                                 xxx = [test.get_patch_position(i) for i in range(len(test.patches()))]
    #                                 # print 'Using rotation \n', rot
    #                                 # print 'After rotatoin patches change to ',xxx
    #                                 return c, rot
    #     # print 'MAXIMUM ERROR'
    #     raise IOError('Cannot align patches')

    # def align_with(self, part2, rot_matrix=None):
    #     # this method tries to align particle with particle p, so that their patches overlap
    #     if len(self._patches) != len(part2._patches):
    #         return False
    #
    #     all_pos = [x.position() for x in self.patches()]
    #     all2_pos = [x.position() for x in part2._patches]
    #     print('Trying to align FROM:', all_pos, '\n TO: ', all2_pos)
    #     rot = copy.deepcopy(rot_matrix)
    #     if rot_matrix != None:
    #         np.transpose(rot)
    #         test = copy.deepcopy(part2)
    #         test.rotate(rot)
    #         xxx = [test.get_patch_position(i) for i in range(len(test._patches))]
    #         print('Using rotation \n', rot_matrix)
    #         print('After rotatoin patches change to ', xxx)
    #         c = self.aligned_with(test)
    #         if c != None:
    #             return c, rot
    #         else:
    #             print('More detailed test of alignment failed')
    #             raise IOError('Could not align')
    #
    #     # TOHL JE BLBE!!
    #     all_pos = [x._position for x in self._patches]
    #     all2_pos = [x._position for x in part2._patches]
    #     print('Trying to align ', all_pos, ' to ', all2_pos)
    #     print('Of particles whose positions are', self.cm_pos, part2.cm_pos)
    #     for pi, p in enumerate(self._patches):
    #         for qi, q in enumerate(self._patches):
    #             if qi != pi:
    #                 for li, l in enumerate(part2._patches):
    #                     for fi, f in enumerate(part2._patches):
    #                         if li != fi:
    #                             print('Aligning patch %d with %d; and %d with %d' % (pi, li, qi, fi))
    #                             v1 = p._position / l2norm(p._position)
    #                             v2 = l._position / l2norm(l._position)
    #                             b1 = q._position / l2norm(q._position)
    #                             b2 = f._position / l2norm(f._position)
    #                             # print 'Positions are', v1,v2,b1,b2
    #                             # print 'NP.dot is ',np.dot(v1,v2)
    #                             theta = np.arccos(np.dot(v1, v2))
    #                             print('Theta is', theta)
    #                             if abs(theta) < myepsilon:
    #                                 r1 = np.eye(3)
    #                             else:
    #                                 u1 = np.cross(v1, v2)
    #                                 # print 'U1 is',u1
    #                                 if l2norm(u1) < myepsilon:  # this means u1 = -u2, we pick u1 as perpendicular to v1
    #                                     u1 = np.array([v1[1], -v1[0], 0])
    #                                     if (l2norm(u1) == 0):
    #                                         u1 = np.array([0, -v1[1], v1[2]])
    #                                 u1 = u1 / l2norm(u1)
    #                                 r1 = rotation_matrix(u1, theta)
    #                             v1 = np.dot(r1, v1)
    #                             b1 = np.dot(r1, b1)
    #                             # print r1
    #                             # print v1
    #                             print('Po natoceni', np.dot(v1, v2))
    #                             b1proj = b1 - np.dot(b1, v1)
    #                             b2proj = b2 - np.dot(b2, v1)
    #                             b1proj = b1proj / l2norm(b1proj)
    #                             b2proj = b2proj / l2norm(b2proj)
    #                             print('Dot and Theta2 is ', np.dot(b1proj, b2proj))
    #                             if np.dot(b1proj, b2proj) < 1 - myepsilon:
    #                                 theta2 = np.arccos(np.dot(b1proj, b2proj))
    #                                 u2 = v1
    #                                 u2 = u2 / l2norm(u2)
    #                                 r2 = rotation_matrix(u2, theta2)
    #                                 r2PI = rotation_matrix(u2, 2. * math.pi - theta2)
    #                             else:
    #                                 r2 = np.eye(3)
    #                                 r2PI = r2
    #                             v1 = np.dot(r2, v1)
    #                             b1old = b1
    #                             b1 = np.dot(r2, b1)
    #                             print('After final alignment', np.dot(v1, v2), np.dot(b1, b2))
    #
    #                             xxx = copy.deepcopy(part2)
    #                             rot = np.dot(r1, r2)
    #                             print("Trying rotation matrix ", rot)
    #                             np.transpose(rot)
    #                             xxx.rotate(rot)
    #                             print(xxx.export_to_mgl(
    #                                 ['blue', 'green', 'red', 'black', 'yellow', 'cyan', 'magenta', 'orange', 'violet'],
    #                                 'blue'))
    #
    #                             if np.dot(b1, b2) < 1 - myepsilon:
    #                                 b1 = b1old
    #                                 r2 = r2PI
    #                                 b1 = np.dot(r2, b1)
    #                                 if np.dot(b1, b2) < 1 - myepsilon:
    #                                     print('Alignment double failed', np.dot(b1, b2))
    #                                     continue
    #                                 else:
    #                                     print('Second PI alignment was successful')
    #                             test = copy.deepcopy(part2)
    #                             rot = np.dot(r1, r2)
    #                             print("Trying rotation matrix ", rot)
    #                             np.transpose(rot)
    #                             test.rotate(rot)
    #                             c = self.aligned_with(test)
    #                             if c != None:
    #                                 return c, rot
    #                             else:
    #                                 print('More detailed test of alignment failed')
    #
    #     # if we got all the way here
    #     return None, None

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

    # def save_type_to_string(self, extras={}) -> str:
    #     outs = 'particle_%d = { \n type = %d \n ' % (self.type_id(), self.type_id())
    #     outs = outs + 'patches = '
    #     for i, p in enumerate(self._patches):
    #         outs = outs + str(p.get_id())
    #         if i < len(self._patches) - 1:
    #             outs = outs + ','
    #     outs += "\n".join([f"{key} = {extras[key]}" for key in extras])
    #     outs = outs + ' \n } \n'
    #     return outs

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
                      patch_colors: list[str],
                      particle_color: str,
                      patch_width: float = 0.1,
                      patch_extension: float = 0.2) -> str:
        # sout = '%f %f %f @ %f C[%s] ' % (self.cm_pos[0],self.cm_pos[1],self.cm_pos[2],self._radius,particle_color)
        # sout = '%f %f %f @ %f C[%s] ' % (self.cm_pos[0], self.cm_pos[1], self.cm_pos[2], 0.4, particle_color)
        sout = f"{self.cm_pos[0]} {self.cm_pos[1]} {self.cm_pos[2]} @ {self.radius()} C[{particle_color}] "
        if len(self._patches) > 0:
            sout = sout + 'M '
        for i, p in enumerate(self._patches):
            pos = p.position()[0] * self.a1 + p.position()[1] * self.a2 + p.position()[2] * self.a3
            pos *= (1.0 + patch_extension)
            # print 'Critical: ',p._type,patch_colors
            # g = '%f %f %f %f C[%s] ' % (pos[0], pos[1], pos[2], patch_width, patch_colors[i])
            g = f"{np.array2string(pos)[1:-1]} {patch_width} {patch_colors[i]}"
            sout = sout + g
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

    def patch_position(self, p: Union[int, PLPatch]):
        if isinstance(p, int):
            return self.patch_position(self.patch(p))
        return p.position() @ self.rotmatrix() + self.position()


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

    def __init__(self, src: PLParticleSet, src_mapping: dict[int, set[int]]):
        super().__init__(src)
        self.__patch_map = src_mapping
        self.__patch_src_map = dict()
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
        return PLMultidentateSourceMap(self.src_set().normalize(), src_mapping=copy.deepcopy(self.patch_map()))


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

    def get_src_map(self) -> PLSourceMap:
        return self.__src_map

    def get_src(self) -> PLParticleSet:
        return self.get_src_map().src_set()

    def has_udt_src(self) -> bool:
        return self.get_src() is not None and isinstance(self.get_src(), PLMultidentateSourceMap)

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
        return {self.patch(i) for i in self.get_src().map_patch(udt_id)}

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
                        dental_radius: float,
                        num_teeth: int,
                        torsion: bool = True,
                        follow_surf: bool = False) -> PLParticleSet:
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
                teeth = [None for _ in range(num_teeth)]
                is_color_neg = patch.color() < 0
                # "normalize" color by making the lowest color 0
                if abs(patch.color()) < 21: assert not torsion, "Torsion cannot be on for same color binding " \
                                                                "b/c IDK how that works and IDC enough to figure it out"
                # note that the converse is not true; we can have torsion w/o same color binding
                colornorm = abs(patch.color()) - 21
                id_map[patch.get_id()] = set()
                for tooth in range(num_teeth):

                    # grab patch position, a1, a2
                    position = np.copy(patch.position())
                    a1 = np.copy(patch.a1())
                    a2 = np.copy(patch.a2())
                    # if the particle type doesn't include an a2

                    # problem!!!!!
                    if a2 is None:
                        if torsion:
                            raise Exception("Cannot treat non-torsional particle set as torsional!")
                        else:
                            if dental_radius > 0:
                                raise Exception(
                                    "Even for non-torsional particles, we need an a2 to align teeth unless teeth are superimposed (dental_radius = 0)")

                    # theta is the angle of the tooth within the patch
                    theta = tooth / num_teeth * 2 * math.pi

                    # assign colors
                    # torsional patches need to be assigned colors to
                    if torsion:
                        c = colornorm * num_teeth + tooth + 21
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
                    if dental_radius > 0:
                        if follow_surf:
                            # phi is the angle of the tooth from the center of the patch
                            psi = dental_radius / particle.radius()
                            psi_axis = np.cross(a1, a2)  # axis orthogonal to patch direction and orientation
                            # get rotation
                            r = R.from_matrix(rotation_matrix(psi_axis, psi))
                        else:
                            # move tooth position out of center
                            position += a2 * dental_radius
                        r = r * R.from_matrix(rotation_matrix(a1, theta))
                        position = r.apply(position)
                        a1 = r.apply(a1)
                        # using torsional multidentate patches is HIGHLY discouraged but
                        # this functionality is included for compatibility reasons
                        a2 = r.apply(a2)
                        teeth[tooth] = PLPatch(patch_counter, c, position, a1, a2, 1.0 / num_teeth)
                    # compativility for multidentate patches with 0 radius - may be useful for DNA origami convert
                    else:
                        # simply use unidentat patch position and skip a2
                        teeth[tooth] = PLPatch(patch_counter, c, position, a1, strength=1.0 / num_teeth)

                    id_map[patch.get_id()].add(patch_counter)
                    patch_counter += 1
                # add all teeth
                new_particle_patches += teeth
            new_particles[i_particle] = PLPatchyParticle(type_id=particle.type_id(), index_=i_particle,
                                                         radius=particle.radius())
            new_particles[i_particle].set_patches(new_particle_patches)
            new_patches += new_particle_patches
        particle_set = PLParticleSet(new_particles, PLMultidentateSourceMap(self, id_map))
        return particle_set
