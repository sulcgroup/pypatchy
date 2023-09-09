#!/usr/bin/env python
from __future__ import annotations
import itertools
import random
from pathlib import Path
from typing import Union, IO
# This file loads patchy particle file from topology and Configuration
import numpy as np
import copy

from numpy.linalg import norm

from pypatchy.patchy_base_particle import BasePatchType, PatchyBaseParticleType, BaseParticleSet

myepsilon = 0.00001

def load_patches(filename: Union[str, Path],
                 num_patches=0) -> list[PLPatchyParticle]:
    if isinstance(filename, str):
        filename = Path(filename)
    j = 0
    Np = 0
    patches = [PLPatchyParticle() for _ in range(num_patches)]

    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if len(line) > 1 and line[0] != '#':
                if 'patch_' and '{' in line:
                    strargs = []
                    k = j + 1
                    while '}' not in lines[k]:
                        strargs.append(lines[k].strip())
                        k = k + 1
                    patch = PLPatch()
                    # print 'Loaded patch',strargs
                    patch.init_from_string(strargs)
                    index = patch.type_id()
                    # flexable patch indexing
                    # probably not optimized speed wise but optimized for flexibility
                    if index >= len(patches):
                        patches += [None for _ in range(index - len(patches) + 1)]
                    patches[index] = patch
                    Np += 1
            j = j + 1

    if num_patches != 0 and Np != num_patches:
        raise IOError('Loaded %d patches, as opposed to the desired %d types ' % (Np, num_patches))
    return patches


def load_particles(filename: Union[str, Path],
                   patch_types: list[PLPatch],
                   num_particles=0) -> list[PLPatchyParticle]:
    particles: list[PLPatchyParticle] = [PLPatchyParticle() for _ in range(num_particles)]
    Np = 0
    with open(filename, 'r') as f:
        lines = f.readlines()
        j = 0
        for line in lines:
            line = line.strip()
            if len(line) > 1 and line[0] != '#':
                if 'particle_' and '{' in line:
                    strargs = []
                    k = j + 1
                    while '}' not in lines[k]:
                        strargs.append(lines[k].strip())
                        k = k + 1
                    particle = PLPatchyParticle()
                    # print 'Loaded particle ',strargs
                    particle.init_from_string(strargs)
                    particle.fill_patches(patch_types)
                    index: int = particle.type_id()
                    # flexable particle indexing
                    # probably not optimized speed wise but optimized for flexibility
                    if index >= len(particles):
                        particles += [None for _ in range(index - len(particles) + 1)]
                    particles[index] = copy.deepcopy(particle)
                    Np += 1
            j = j + 1
    return particles


def export_interaction_matrix(patches, filename="interactions.txt"):
    with open(filename, 'w') as f:
        f.writelines(
            [
                f"patchy_eps[{p1.type_id()}][{p2.type_id()}] = 1.0\n"
                for p1, p2 in itertools.combinations(patches, 2)
                if p1.color() == p2.color()
            ]
        )


class PLPatch(BasePatchType):
    _a1: np.ndarray
    _a2: np.ndarray
    _type: int
    _strength: float

    def __init__(self,
                 type_id: Union[None, int] = None,
                 color: Union[None, int] = None,
                 relposition: Union[None, np.ndarray] = None,
                 a1: Union[None, np.ndarray] = None,
                 a2: Union[None, np.ndarray] = None,
                 strength: float = 1.0):
        super().__init__(type_id, color)
        self._key_points = [
            relposition,
            a1,
            a2
        ]
        self._type = type_id
        self._strength = strength

    def type_id(self) -> int:
        return self._type

    def strength(self) -> float:
        return self._strength

    def position(self) -> np.ndarray:
        return self._key_points[0]

    def set_position(self, newposition: np.ndarray):
        self._key_points[0] = newposition

    def colornum(self) -> int:
        return self.color()

    def a1(self) -> np.ndarray:
        return self._key_points[1]

    def set_a1(self, new_a1: np.ndarray):
        self._key_points[1] = new_a1

    def a2(self) -> np.ndarray:
        return self._key_points[2]

    def set_a2(self, new_a2: np.ndarray):
        self._key_points[2] = new_a2

    def get_abs_position(self, r) -> np.ndarray:
        return r + self._position

    def save_to_string(self, extras={}) -> str:
        # print self._type,self._type,self._color,1.0,self._position,self._a1,self._a2

        outs = f'patch_{self.type_id()} = ' + '{\n ' \
                                              f'\tid = {self.type_id()}\n' \
                                              f'\tcolor = {self.color()}\n' \
                                              f'\tstrength = {self.strength()}\n' \
                                              f'\tposition = {np.array2string(self.position(), separator=",")[1:-1]}\n' \
                                              f'\ta1 = {np.array2string(self.a1(), separator=",")[1:-1]}\n'
        if self.a2() is not None:  # tolerate missing a2s
            outs += f'\ta2 = {np.array2string(self.a2(), separator=",")[1:-1]}\n'
        else:
            # make shit up
            outs += f'\ta2 = {np.array2string(np.array([0,0,0]), separator=",")[1:-1]}\n'
        outs += "\n".join([f"t\t{key} = {extras[key]}" for key in extras])
        outs += "\n}\n"
        return outs

    def init_from_dps_file(self, fname: str, line_number: int):
        handle = open(fname)
        line = handle.readlines()[line_number]
        positions = [float(x) for x in line.strip().split()]
        self._position = np.array(positions)

    def init_from_string(self, lines: list[str]):
        for line in lines:
            line = line.strip()
            if len(line) > 1 and line[0] != '#':
                if "id" in line:
                    val = line.split('=')[1]
                    try:
                        self._type = int(val)
                    except ValueError:
                        self._type = int(val.split('_')[1])
                if "color" in line:
                    vals = int(line.split('=')[1])
                    self._color = vals
                elif "a1" in line:
                    vals = line.split('=')[1]
                    x, y, z = [float(g) for g in vals.split(',')]
                    self.set_a1(np.array([x, y, z]))
                elif "a2" in line:
                    vals = line.split('=')[1]
                    x, y, z = [float(g) for g in vals.split(',')]
                    self.set_a2(np.array([x, y, z]))
                elif "position" in line:
                    vals = line.split('=')[1]
                    x, y, z = [float(g) for g in vals.split(',')]
                    self.set_position(np.array([x, y, z]))

    def can_bind(self, other: BasePatchType) -> bool:
        if abs(self.color()) > 20:
            return self.color() == -other.color()
        else:
            return self.color() == other.color()


class PLPatchyParticle(PatchyBaseParticleType):
    # HATE making the particle type and the particle the same class but refactoring is objectively not the
    # best use of my time
    cm_pos: np.ndarray
    unique_id: int
    _radius: float
    _patch_ids: Union[None, list[int]]
    v: np.ndarray
    L: np.ndarray
    a1: Union[None, np.ndarray]
    a3: Union[None, np.ndarray]

    all_letters = ['C', 'H', 'O', 'N', 'P', 'S', 'F', 'K', 'I', 'Y']

    def __init__(self, patches: list[PLPatch] = [], type_id=0, index_=0, position=np.array([0., 0., 0.]), radius=0.5):
        super().__init__(type_id, patches)
        self.cm_pos = position
        self.unique_id = index_
        self._radius = radius
        self._patch_ids = None
        self.v = np.array([0., 0., 0.])
        self.L = np.array([0., 0., 0.])
        self.a1 = None
        self.a3 = None

    def name(self) -> str:
        return f"particletype_{self.type_id()}"

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

    def patches(self):
        return self._patches

    def num_patches(self) -> int:
        return len(self._patches)

    def translate(self, translation_vector: np.ndarray):
        self.cm_pos += translation_vector

    def rotate(self, rot_matrix: np.ndarray):
        self.a1 = np.dot(rot_matrix, self.a1)
        self.a3 = np.dot(rot_matrix, self.a3)

    def set_random_orientation(self):
        self.a1 = np.array(np.random.random(3))
        self.a1 = self.a1 / np.sqrt(np.dot(self.a1, self.a1))
        x = np.random.random(3)
        self.a3 = x - np.dot(self.a1, x) * self.a1
        self.a3 = self.a3 / np.sqrt(np.dot(self.a3, self.a3))
        if abs(np.dot(self.a1, self.a3)) > 1e-10:
            raise IOError("Could not generate random orientation?")

    def get_patch_position(self, patchid: int) -> np.ndarray:
        assert -1 < patchid < self.num_patches(), "Index out of bounds"
        p = self._patches[patchid]
        return p.position()[0] * self.a1 + p.position()[1] * self.a2 + p.position()[2] * self.a3

    def get_patch_orientation_a1(self, patchid) -> np.ndarray:
        p: PLPatch = self.patch(patchid)
        v = np.array(self.a1 * p.a1()[0] + self.a2 * p.a1()[1] + self.a3 * p.a1()[2])
        return v

    def get_patch_orientation_a2(self, patchid) -> np.ndarrayh:
        p: PLPatch = self.patch(patchid)
        if np.dot(p.a2(), p.a2()) < 0.9999:  # account for rounding errors
            print("PRIMARY MAX CRITICAL ERRROR MAX CRITICAL ERROR", p.a2(), np.dot(p.a2(), p.a2()))
        v = np.array(self.a1 * p.a2()[0] + self.a2 * p.a2()[1] + self.a3 * p.a2()[2])
        if np.dot(v, v) < 0.9999:  # account for rounding errors
            print("MAX CRITICAL ERROR MAX CRITICAL ERROR", v, np.dot(v, v))
        return v

    def set_patch_a2_orientation(self, patchid, new_a2):
        p: PLPatch = self.patch(patchid)
        coor = np.array([np.dot(new_a2, self.a1), np.dot(new_a2, self.a2), np.dot(new_a2, self.a3)])
        p.set_a2(coor / np.sqrt(np.dot(coor, coor)))

    def aligned_with(self, p: PLPatchyParticle) -> Union[None, dict[int, int]]:
        # checks if paticles are aligned
        # print 'Verification of alignment ', self.cm_pos, p.cm_pos
        if len(self.patches()) != len(p.patches()):
            return None
        correspondence = {}
        for i, patchA in enumerate(self.patches()):
            positionA = self.get_patch_position(i)
            for j, patchB in enumerate(p.patches()):
                positionB = p.get_patch_position(j)

                val = np.dot(positionA / norm(positionA), positionB / norm(positionB))
                if val > 1.0 - myepsilon:
                    if j in correspondence.values():
                        # print 'Error two patches would correspond to the same patch'
                        return None
                    else:
                        correspondence[i] = j
                        # print 'CHECKING patch positions, we have MATCH ' ,i,j, positionA, positionB, np.dot(positionA/l2norm(positionA),positionB/l2norm(positionB))
                        break
            if i not in correspondence.keys():
                # print 'Could not match patch ',i
                return None

        # print 'Found perfect correspondence',correspondence
        return correspondence

    def align_with(self, part2: PLPatchyParticle) -> tuple:
        all_pos = [x.position() for x in self.patches()]
        all2_pos = [x.position() for x in part2.patches()]
        print('Trying to align FROM:', all_pos, '\n TO: ', all2_pos)
        for pi, p in enumerate(self._patches):
            for qi, q in enumerate(self._patches):
                if qi != pi:
                    for li, l in enumerate(part2.patches()):
                        for fi, f in enumerate(part2.patches()):
                            if li != fi:
                                # print 'Aligning patch %d with %d; and %d with %d' % (pi,li,qi,fi)
                                v1 = p.position() / norm(p.position())
                                v2 = l.position() / norm(l.position())
                                b1 = q.position() / norm(q.position())
                                b2 = f.position() / norm(f.position())
                                v1 = np.matrix(v1).transpose()
                                b1 = np.matrix(b1).transpose()
                                B = v1 * v2 + b1 * b2
                                U, s, V = np.linalg.svd(B, full_matrices=True)
                                M = np.diag([1, 1, np.linalg.det(U) * np.linalg.det(V)])
                                R = U * M * V
                                rot = np.asarray(R)
                                test = copy.deepcopy(part2)
                                test.rotate(rot)
                                c = self.aligned_with(test)
                                if c is not None:
                                    # print 'Success! '
                                    xxx = [test.get_patch_position(i) for i in range(len(test.patches()))]
                                    # print 'Using rotation \n', rot
                                    # print 'After rotatoin patches change to ',xxx
                                    return c, rot
        # print 'MAXIMUM ERROR'
        raise IOError('Cannot align patches')

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

    def save_type_to_string(self, extras={}) -> str:
        outs = 'particle_%d = { \n type = %d \n ' % (self.type_id(), self.type_id())
        outs = outs + 'patches = '
        for i, p in enumerate(self._patches):
            outs = outs + str(p.get_id())
            if i < len(self._patches) - 1:
                outs = outs + ','
        outs += "\n".join([f"{key} = {extras[key]}" for key in extras])
        outs = outs + ' \n } \n'
        return outs

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

    def init_from_string(self, lines: list[str]):
        for line in lines:
            line = line.strip()
            if len(line) > 1 and line[0] != '#':
                if "type" in line:
                    vals = int(line.split('=')[1])
                    self._type_id = vals
                if 'patches' in line:
                    vals = line.split('=')[1]
                    self._patch_ids = [int(g) for g in vals.split(',')]

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

    def export_to_lorenzian_patchy_str(self,
                                       ninstances: int,
                                       root: Path = Path("/")) -> str:
        """

        """

        patches_dat_filename = f"patches_{self.type_id()}.dat"
        particle_str = f"{ninstances} {self.num_patches()} {','.join([str(pid) for pid in self._patch_ids])} {patches_dat_filename}"
        patches_dat_filestr = "\n".join(
            [np.array2string(patch.position(), precision=4)[1:-1] for patch in self.patches()])
        with open(root / patches_dat_filename, 'w') as f:
            f.write(patches_dat_filestr)
        return particle_str

    def export_to_xyz(self,
                      patch_width=0.4,
                      patch_extension=0.2) -> str:
        letter = PLPatchyParticle.all_letters[self.type_id()]
        sout = '%s %f %f %f ' % (letter, self.cm_pos[0], self.cm_pos[1], self.cm_pos[2])

        return sout


# TODO: something with this class
class PLPSimulation:
    particles: list[PLPatchyParticle]
    _box_size: np.ndarray
    N: int
    _particle_types: BaseParticleSet
    _E_tot: float
    _E_pot: float
    _E_kin: float

    _particle_colors: list[str]
    _patch_colors: list[str]
    _complimentary_colors: dict[int, str]
    _colorbank: list[str]

    def __init__(self, seed=69):
        self._box_size = np.array([0., 0., 0.])
        self.N = 0
        self._patch_types = []
        self._particle_types = BaseParticleSet()
        self._E_tot = 0.
        self._E_pot = 0.
        self._E_kin = 0.



        self.particles = []
        random.seed(seed)

        self._colorbank = []

    def set_radius(self, radius):
        for p in self.particles:
            p.set_radius(radius)

    def generate_random_color(self, particle_id=-1) -> str:
        # random.seed(seed)
        if True or len(self._colorbank) < 1 or particle_id == -1:
            r = [random.random() for _ in range(3)]
            color = np.array2string(np.array(r), separator=",")[1:-1]
            # color = '%f,%f,%f' % (r[0], r[1], r[2])
            return color
        else:
            return self._colorbank[particle_id]

    def translate(self, translation_vector: np.ndarray):
        for p in self.particles:
            p.translate(translation_vector)

    def set_box_size(self, box: Union[np.ndarray, list]):
        self._box_size = np.array(box)

    def load_dps_input_file(self, input_file):
        handle = open(input_file)
        patchy_matrix_file = None
        for line in handle.readlines():
            line = line.strip()
            if (len(line) > 1 and line[0] != '#'):
                if 'DPS_interaction_matrix_file' in line:
                    val = line.split('=')[1].strip()
                    patchy_matrix_file = val

        if patchy_matrix_file is None:
            raise IOError('Could not find DPS_interaction_matrix_file key in the input file')

        mfile = open(patchy_matrix_file)
        interacting_patches = {}
        for line in mfile.readlines():
            if 'patchy_eps' in line:
                if float(line.strip().split('=')[1]) > 0:
                    vals = line.strip().split('=')[0].strip().split('eps')[1].replace('[', ' ').replace(']',
                                                                                                        ' ').split()
                    v1 = int(vals[0])
                    v2 = int(vals[1])
                    interacting_patches[v1] = v2
                    interacting_patches[v2] = v1

        self._colorbank = [self.generate_random_color() for x in range(max(interacting_patches.keys()))]

        self.interacting_patches = interacting_patches

    def load_dps_topology(self, topology_file):
        handle = open(topology_file)
        lines = handle.readlines()
        vals = lines[0].split()
        N = int(vals[0])
        Ntypes = int(vals[1])
        type_counts = []
        orientations = []
        patches = {}
        # for c in range(len(patches.keys())):
        particles = []  # [None for x in range(N)]

        for pid, line in enumerate(lines[1:]):
            if len(line.split()) >= 2:
                vals = line.split()
                count = int(vals[0])
                type_counts.append(count)
                pcount = int(vals[1])
                ptypes = vals[2]
                ptypes = [int(x) for x in ptypes.strip().split(',')]
                pgeometry = vals[3]
                patches = []
                index = 0
                for i, p in enumerate(ptypes):
                    patch = PLPatch()
                    patch.set_color(p)
                    patch.init_from_dps_file(pgeometry, i)
                    patches.append(patch)
                for j in range(count):
                    particle = PLPatchyParticle()
                    particle._type = pid
                    particle._patch_ids = ptypes
                    particle._patches = patches
                    particle.unique_id = index
                    index += 1
                    particles.append(particle)

        self.particles = particles

        # types = [int(x) for x in line[1].split()]
        # print 'Critical', line[1].split()
        # print types
        # print self._particle_types
        # print 'THERE ARE', len(self._particle_types), ' particle types '

        self.N = N
        if N != len(self.particles):
            raise IOError('Particle number mismatch while reading topology file')

    def load_input_file(self, input_file):
        patchy_file = None
        particle_file = None
        n_patches = 0
        n_particles = 0

        with open(input_file, 'r') as handle:
            for line in handle.readlines():
                line = line.strip()
                if len(line) > 1 and line[0] != '#':
                    if 'patchy_file' in line:
                        val = line.split('=')[1].strip()
                        patchy_file = val
                    elif 'particle_file' in line:
                        val = line.split('=')[1].strip()
                        particle_file = val
                    elif 'particle_types_N' in line:
                        val = line.split('=')[1].strip()
                        n_particles = int(val)
                    elif 'patch_types_N' in line:
                        val = line.split('=')[1].strip()
                        n_patches = int(val)

        assert patchy_file is not None, "No patch file specified"
        assert particle_file is not None, "No particle file specified"
        assert n_particles > 0, "Particle count not specified"
        assert n_patches > 0, "Patch count not specified"

        self._colorbank = [self.generate_random_color() for x in range(n_patches)]
        # print >> sys.stderr, "Loaded patchy_file: %s, particle_file: %s, N_part_types: %d, n_patch_types: %d " % (patchy_file,particle_file,self._N_particle_types, self._N_patch_types)
        # now process the patch types file:

        self._particle_types.add_patches(load_patches(patchy_file, n_patches))

        self._particle_types.add_particles(load_particles(particle_file, self._particle_types.patches(), n_particles))

        # print 'Critical', self._particle_types
        # print "LOADED", len(self._particle_types)

    def load_topology(self, topology_file):
        handle = open(topology_file, 'r')
        line = handle.readlines()
        vals = line[0].split()
        N = int(vals[0])
        Ntypes = int(vals[1])
        particle_types = [int(x) for x in line[1].split()]
        # print 'Critical', line[1].split()
        # print types
        # print self._particle_types
        # print 'THERE ARE', len(self._particle_types), ' particle types '
        self.particles = []
        for index, particle_type in enumerate(particle_types):
            p = copy.deepcopy(self._particle_types[particle_type])
            p.unique_id = index
            self.particles.append(p)

        handle.close()
        self.N = N
        if N != len(self.particles):
            raise IOError('Particle number mismatch while reading topology file')

    def save_topology(self, topology_file):
        handle = open(topology_file, 'w')
        handle.write('%d %d\n' % (len(self.particles), self._N_particle_types))
        outstr = ''
        for p in self.particles:
            outstr = outstr + str(p.type_id()) + ' '
        handle.write(outstr + '\n')
        handle.close()

    def save_patchy_types_file(self, ptypes_file: str):
        handle = open(ptypes_file, 'w')
        for p in self._patch_types:
            outs = p.save_to_string()
            handle.write(outs)
        handle.close()

    def save_particle_types_file(self, ptypes_file: str):
        handle = open(ptypes_file, 'w')
        for p in self._particle_types:
            outs = p.save_type_to_string()
            handle.write(outs)
        handle.close()

    def check_for_particle_overlap(self, particle, dist_cutoff=1.0):
        # print 'Adding ', particle.cm_pos
        for p in self.particles:
            dist = p.distance_from(particle, self._box_size)
            # print ' Looking at distance from ', p.cm_pos,dist
            if dist <= dist_cutoff:
                # print 'Unfortunately overlaps with ',p.cm_pos,dist
                return True
                # print 'Check is fine!'
        return False

    def save_configuration(self,
                           conf_name: str,
                           t=0.):
        handle = open(conf_name, 'w')
        handle.write('t = %f\nb = %f %f %f\nE = %f %f %f\n' % (
            t, self._box_size[0], self._box_size[1], self._box_size[2], self._E_pot, self._E_kin, self._E_tot))
        for p in self.particles:
            outs = p.save_conf_to_string()
            handle.write(outs)
        handle.close()

    def add_particles(self, particles, strict_check=True):
        # adds particles to the field, also initializes paricle types and patchy types based on these data
        # it overwrites any previosuly stored particle!!
        self.particles = copy.deepcopy(particles)
        self.N = len(particles)
        # now treat types:
        saved_types = {}
        for p in self.particles:
            if p.type_id() not in saved_types.keys():
                saved_types[p.type_id()] = copy.deepcopy(p)
        self._particle_types = []
        for i, key_id in enumerate(sorted(saved_types.keys())):
            if key_id != i and strict_check:
                raise IOError(
                    "Error while adding particles to the PLPSimulation class, indices of types are not correctly ordered")
            self._particle_types.append(copy.deepcopy(saved_types[key_id]))
        self._N_particle_types = len(self._particle_types)

        # now treat patches
        saved_patch_types = {}
        for p in self._particle_types:
            for patch in p.patches():
                if patch.get_id() not in saved_patch_types.keys():
                    saved_patch_types[patch.get_id()] = copy.deepcopy(patch)
        self._patch_types = []
        for i, key_id in enumerate(sorted(saved_patch_types.keys())):
            if key_id != i and strict_check:
                raise IOError(
                    "Error while adding patches to the PLPSimulation class, indices of types are not correctly ordered")
            self._patch_types.append(copy.deepcopy(saved_patch_types[key_id]))
        self._N_patch_types = len(self._patch_types)

    def insert_particle(self,
                        particle,
                        check_overlap=False):
        if check_overlap:
            if self.check_for_particle_overlap(particle):
                return False
        self.particles.append(particle)
        self.N += 1
        if particle.type_id() not in [x.type_id() for x in self._particle_types]:
            self._particle_types.append(copy.deepcopy(particle))
            self._N_particle_types += 1
        return True

    def load_configuration(self,
                           configuration_file: str,
                           conf_to_skip=0,
                           close_file=True):
        _conf = open(configuration_file, 'r')
        if conf_to_skip > 0:
            conf_lines = 3 + self.N
            for j in range(conf_lines * conf_to_skip):
                _conf.readline()

        self.read_next_configuration(_conf)

        if close_file:
            _conf.close()

        return _conf

    def load_from_files(self,
                        input_file: str,
                        topology_file: str,
                        config_file: str,
                        conf_to_skip=0):
        self.load_input_file(input_file)
        self.load_topology(topology_file)
        self.load_configuration(config_file, conf_to_skip)

    def load_from_dps_files(self,
                            input_file: str,
                            topology_file: str,
                            config_file: str,
                            conf_to_skip=0):
        self.load_dps_input_file(input_file)
        self.load_dps_topology(topology_file)
        self.load_configuration(config_file, conf_to_skip)

    def read_next_configuration(self, file_handle: IO) -> Union[bool, IO]:
        _conf = file_handle

        timeline = _conf.readline()
        time = 0.
        if len(timeline) == 0:
            return False
        else:
            time = float(timeline.split()[2])

        box = np.array([float(x) for x in _conf.readline().split()[2:]])
        [E_tot, E_pot, E_kin] = [float(x) for x in _conf.readline().split()[2:5]]

        self._box_size = box
        self._E_tot = E_tot
        self._E_pot = E_pot
        self._E_kin = E_kin

        for i in range(self.N):
            ls = _conf.readline().split()
            self.particles[i].fill_configuration(np.array(ls))

        return _conf

    def bring_in_box(self, all_positive=False):
        for p in self.particles:
            nx = np.rint(p.cm_pos[0] / float(self._box_size[0])) * self._box_size[0]
            ny = np.rint(p.cm_pos[1] / float(self._box_size[1])) * self._box_size[1]
            nz = np.rint(p.cm_pos[2] / float(self._box_size[2])) * self._box_size[2]
            # print np.array([nx,ny,nz])
            p.cm_pos -= np.array([nx, ny, nz])
            if all_positive:
                for i in range(3):
                    if p.cm_pos[i] < 0:
                        p.cm_pos[i] += self._box_size[i]

    def get_color(self, index: int):
        if abs(index) >= 20:
            index = abs(index) - 20
        # print index
        # print self._patch_colors
        # return self.generate_random_color()
        return self._patch_colors[index]
        #
        # if not ispatch:
        #     if index < 0 or index >= len(self._particle_colors):
        #         if index in self._complementary_colors.keys():
        #             return self._complementary_colors[index]
        #         else:
        #             return self.generate_random_color()
        #     else:
        #         return self._particle_colors[index]
        # else:
        #     if index < 0 or index >= len(self._patch_colors):
        #         if index in self._complementary_colors.keys():
        #             return self._complementary_colors[index]
        #         else:
        #             return self.generate_random_color(index)
        #     else:
        #         return self._patch_colors[index]

    def export_to_mgl(self,
                      filename: str,
                      regime: str = 'w',
                      icosahedron=True):
        out = open(filename, regime)
        sout = f".Box: {np.array2string(self._box_size, separator=',')[1:-1]}\n"
        # sout = ".Box:%f,%f,%f\n" % (self._box_size[0], self._box_size[1], self._box_size[2])
        for p in self.particles:
            patch_colors = [self.get_color(pat.color()) for pat in p.patches()]
            particle_color = self.get_color(p.type_id())
            sout = sout + p.export_to_mgl(patch_colors, particle_color) + '\n'
            if icosahedron:
                p.print_icosahedron(particle_color)

        out.write(sout)
        out.close()



    def export_to_lorenzian_mgl(self,
                                filename: str,
                                regime: str = 'w',
                                icosahedron: bool = True):
        out = open(filename, regime)
        sout = ".Box:%f,%f,%f\n" % (self._box_size[0], self._box_size[1], self._box_size[2])
        for p in self.particles:
            patch_colors = [self.get_color(pat.color()) for pat in p.patches()]
            particle_color = self.get_color(p.type_id())
            sout = sout + p.export_to_lorenzian_mgl(patch_colors, particle_color) + '\n'
            if icosahedron:
                p.print_icosahedron(particle_color)

        out.write(sout)
        out.close()

    def export_to_francesco_mgl(self,
                                filename: str,
                                regime: str = 'w',
                                icosahedron: bool = True):
        with open(filename, regime) as fout:
            sout = f".Box:{np.array2string(self._box_size, separator=',')}\n"
            # sout = ".Box:%f,%f,%f\n" % (self._box_size[0], self._box_size[1], self._box_size[2])
            for p in self.particles:
                patch_colors = [self.get_color(pat.color()) for pat in p.patches()]
                particle_type = p.type_id()
                patch_position_0 = p.cm_pos + p.get_patch_position(0)
                patch_position_1 = p.cm_pos + p.get_patch_position(1)
                line = '%d %f %f %f %f %f %f %f %f %f' % (
                    particle_type, p.cm_pos[0], p.cm_pos[1], p.cm_pos[2], patch_position_0[0], patch_position_0[1],
                    patch_position_0[2], patch_position_1[0], patch_position_1[1], patch_position_1[2])
                particle_color = self.get_color(p.type_id())
                sout = sout + line + '\n'  # p.export_to_lorenzian_mgl(patch_colors,particle_color) + '\n'
                if icosahedron:
                    p.print_icosahedron(particle_color)

            fout.write(sout)

    def export_to_xyz(self,
                      filename: str,
                      regime: str = 'w'):
        with open(filename, regime) as fout:

            sout = str(len(self.particles)) + '\n'
            sout += "Box:%f,%f,%f\n" % (self._box_size[0], self._box_size[1], self._box_size[2])
            for p in self.particles:
                sout = sout + p.export_to_xyz() + '\n'

            fout.write(sout)
