from __future__ import annotations

import copy
import itertools
import random
from typing import Union, Iterable

import numpy as np
from oxDNA_analysis_tools.UTILS.data_structures import Configuration

from pypatchy.patchy.pl.plpatchylib import PATCHY_CUTOFF
from pypatchy.patchy.pl.plparticle import PLPatchyParticle, PLParticleSet
from pypatchy.patchy.pl.plpatch import PLPatch
from pypatchy.patchy_base_particle import BaseParticleSet
from pypatchy.scene import Scene
from pypatchy.util import dist


class PLPSimulation(Scene):

    _box_size: np.ndarray
    _particle_types: PLParticleSet
    _E_tot: float
    _E_pot: float
    _E_kin: float

    _time: int

    def __init__(self, seed=69):
        super().__init__()
        self._box_size = np.array([0., 0., 0.])
        self._E_tot = 0.
        self._E_pot = 0.
        self._E_kin = 0.
        self._time = 0

        random.seed(seed)

        self._colorbank = []

    def set_radius(self, radius):
        for p in self._particles:
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
        for p in self._particles:
            p.translate(translation_vector)

    def box_size(self) -> np.ndarray:
        return self._box_size

    def set_box_size(self, box: Union[np.ndarray, list]):
        self._box_size = np.array(box)

    def check_for_particle_overlap(self, particle: PLPatchyParticle, dist_cutoff=1.0) -> bool:
        """
        Checks if a particle overlaps any other particles in the simulation.
        Returns:
            true if the particle overlaps another particle, false otherwise
        """
        # print 'Adding ', particle.cm_pos
        for p in self._particles:
            dist = self.dist_2particles(particle, p)
            # print ' Looking at distance from ', p.cm_pos,dist
            if dist <= dist_cutoff:
                # print 'Unfortunately overlaps with ',p.cm_pos,dist
                return True
                # print 'Check is fine!'
        return False

    def add_particles(self, particles: list[PLPatchyParticle], strict_check=True):
        # adds particles to the field, also initializes paricle types and patchy types based on these data
        # it overwrites any previosuly stored particle!!
        self._particles = copy.deepcopy(particles)
        self.N = len(particles)
        # now treat types:
        saved_types = {}
        for p in self.particles():
            if p.get_type() not in saved_types.keys():
                saved_types[p.get_type()] = copy.deepcopy(p)

    def insert_particle(self,
                        particle: PLPatchyParticle,
                        check_overlap=False):
        if check_overlap:
            if self.check_for_particle_overlap(particle):
                return False
        self._particles.append(particle)
        if particle.type_id() not in [x.type_id() for x in self._particle_types.particles()]:
            self._particle_types.add_particle(particle)
        return True

    def dist_2particles(self, p1: PLPatchyParticle, p2: PLPatchyParticle):
        """
        Calculate the wrapped distance between two points in a 3D box.

        :param p1: numpy array [x, y, z] for first point.
        :param p2: numpy array [x, y, z] for second point.
        :param box_size: numpy array [length, width, height] for the box dimensions.
        :return: wrapped distance between p1 and p2.
        """

        # Calculate the difference in each dimension
        delta = (p2.position() % self.box_size()) - (p1.position() % self.box_size())

        # Wrap the differences where necessary
        delta = np.where(np.abs(delta) > self.box_size() / 2, self.box_size() - np.abs(delta), delta)

        return np.linalg.norm(delta)

    def add_particle_rand_positions(self, particles: Iterable[PLPatchyParticle],
                                    nTries=1e3,
                                    overlap_min_dist: float=1.0):
        # loop particles
        for p in particles:
            t = 0
            # set max tries so it doesn't go forever
            while t < nTries:
                # random position
                new_pos = np.random.rand(3) * (self._box_size - overlap_min_dist) + (overlap_min_dist / 2)
                p.set_position(new_pos)
                if not self.check_for_particle_overlap(p, overlap_min_dist):
                    break
                t += 1
            if t == nTries:
                raise Exception(f"Could not find a position to place a particle! nTries={nTries}")
            # randomize orientation
            p.set_random_orientation()
            self.add_particle(p)

    def num_particle_types(self) -> int:
        return self.particle_types().num_particle_types()

    # def load_configuration(self,
    #                        configuration_file: str,
    #                        conf_to_skip=0,
    #                        close_file=True):
    #     _conf = open(configuration_file, 'r')
    #     if conf_to_skip > 0:
    #         conf_lines = 3 + self.N
    #         for j in range(conf_lines * conf_to_skip):
    #             _conf.readline()
    #
    #     self.read_next_configuration(_conf)
    #
    #     if close_file:
    #         _conf.close()
    #
    #     return _conf

    # def read_next_configuration(self, file_handle: IO) -> Union[bool, IO]:
    #     _conf = file_handle
    #
    #     timeline = _conf.readline()
    #     time = 0.
    #     if len(timeline) == 0:
    #         return False
    #     else:
    #         time = float(timeline.split()[2])
    #
    #     box = np.array([float(x) for x in _conf.readline().split()[2:]])
    #     [E_tot, E_pot, E_kin] = [float(x) for x in _conf.readline().split()[2:5]]
    #
    #     self._box_size = box
    #     self._E_tot = E_tot
    #     self._E_pot = E_pot
    #     self._E_kin = E_kin
    #
    #     for i in range(self.N):
    #         ls = _conf.readline().split()
    #         self._particles[i].fill_configuration(np.array(ls))
    #
    #     return _conf
    #
    # def bring_in_box(self, all_positive=False):
    #     for p in self._particles:
    #         nx = np.rint(p.cm_pos[0] / float(self._box_size[0])) * self._box_size[0]
    #         ny = np.rint(p.cm_pos[1] / float(self._box_size[1])) * self._box_size[1]
    #         nz = np.rint(p.cm_pos[2] / float(self._box_size[2])) * self._box_size[2]
    #         # print np.array([nx,ny,nz])
    #         p.cm_pos -= np.array([nx, ny, nz])
    #         if all_positive:
    #             for i in range(3):
    #                 if p.cm_pos[i] < 0:
    #                     p.cm_pos[i] += self._box_size[i]

    # def get_color(self, index: int):
    #     if abs(index) >= 20:
    #         index = abs(index) - 20
    #     # print index
    #     # print self._patch_colors
    #     # return self.generate_random_color()
    #     return self._patch_colors[index]
    #     #
    #     # if not ispatch:
    #     #     if index < 0 or index >= len(self._particle_colors):
    #     #         if index in self._complementary_colors.keys():
    #     #             return self._complementary_colors[index]
    #     #         else:
    #     #             return self.generate_random_color()
    #     #     else:
    #     #         return self._particle_colors[index]
    #     # else:
    #     #     if index < 0 or index >= len(self._patch_colors):
    #     #         if index in self._complementary_colors.keys():
    #     #             return self._complementary_colors[index]
    #     #         else:
    #     #             return self.generate_random_color(index)
    #     #     else:
    #     #         return self._patch_colors[index]

    # def export_to_mgl(self,
    #                   filename: str,
    #                   regime: str = 'w',
    #                   icosahedron=True):
    #     out = open(filename, regime)
    #     sout = f".Box: {np.array2string(self._box_size, separator=',')[1:-1]}\n"
    #     # sout = ".Box:%f,%f,%f\n" % (self._box_size[0], self._box_size[1], self._box_size[2])
    #     for p in self._particles:
    #         patch_colors = [self.get_color(pat.color()) for pat in p.patches()]
    #         particle_color = self.get_color(p.type_id())
    #         sout = sout + p.export_to_mgl(patch_colors, particle_color) + '\n'
    #         if icosahedron:
    #             p.print_icosahedron(particle_color)
    #
    #     out.write(sout)
    #     out.close()

    #
    # def export_to_lorenzian_mgl(self,
    #                             filename: str,
    #                             regime: str = 'w',
    #                             icosahedron: bool = True):
    #     out = open(filename, regime)
    #     sout = ".Box:%f,%f,%f\n" % (self._box_size[0], self._box_size[1], self._box_size[2])
    #     for p in self._particles:
    #         patch_colors = [self.get_color(pat.color()) for pat in p.patches()]
    #         particle_color = self.get_color(p.type_id())
    #         sout = sout + p.export_to_lorenzian_mgl(patch_colors, particle_color) + '\n'
    #         if icosahedron:
    #             p.print_icosahedron(particle_color)
    #
    #     out.write(sout)
    #     out.close()

    # def export_to_francesco_mgl(self,
    #                             filename: str,
    #                             regime: str = 'w',
    #                             icosahedron: bool = True):
    #     with open(filename, regime) as fout:
    #         sout = f".Box:{np.array2string(self._box_size, separator=',')}\n"
    #         # sout = ".Box:%f,%f,%f\n" % (self._box_size[0], self._box_size[1], self._box_size[2])
    #         for p in self._particles:
    #             patch_colors = [self.get_color(pat.color()) for pat in p.patches()]
    #             particle_type = p.type_id()
    #             patch_position_0 = p.cm_pos + p.get_patch_position(0)
    #             patch_position_1 = p.cm_pos + p.get_patch_position(1)
    #             line = '%d %f %f %f %f %f %f %f %f %f' % (
    #                 particle_type, p.cm_pos[0], p.cm_pos[1], p.cm_pos[2], patch_position_0[0], patch_position_0[1],
    #                 patch_position_0[2], patch_position_1[0], patch_position_1[1], patch_position_1[2])
    #             particle_color = self.get_color(p.type_id())
    #             sout = sout + line + '\n'  # p.export_to_lorenzian_mgl(patch_colors,particle_color) + '\n'
    #             if icosahedron:
    #                 p.print_icosahedron(particle_color)
    #
    #         fout.write(sout)

    # def export_to_xyz(self,
    #                   filename: str,
    #                   regime: str = 'w'):
    #     with open(filename, regime) as fout:
    #
    #         sout = str(len(self._particles)) + '\n'
    #         sout += "Box:%f,%f,%f\n" % (self._box_size[0], self._box_size[1], self._box_size[2])
    #         for p in self._particles:
    #             sout = sout + p.export_to_xyz() + '\n'
    #
    #         fout.write(sout)

    # def from_top_conf(self, top: TopInfo, conf: Configuration):
    #     rw = get_writer()
    #     rw.read_top(Path(top.path), self)
    #
    #     self._time = conf.time
    #     self._E_pot, self._E_kin, self._E_tot = conf.energy
    #     self._box_size = conf.box
    #
    #     for i, p in enumerate(self.particles()):
    #         p.a1 = conf.a1s[i, :]
    #         p.a3 = conf.a3s[i, :]
    #         p.set_position(conf.positions[i, :])
    # #
    # def to_top_conf(self, top_path: Path, conf_path: Path) -> tuple[TopInfo, Configuration]:
    #     rw = get_writer()
    #     top = TopInfo(str(top_path), self.num_particles())
    #     rw.write_top(top_path, self)
    #
    #     particle_positions = np.stack([p.position() for p in self.particles()])
    #     particle_a1s = np.stack([p.a1 for p in self.particles()])
    #     particle_a3s = np.stack([p.a3 for p in self.particles()])
    #
    #     conf = Configuration(self._time,
    #                          self._box_size,
    #                          np.array([self._E_pot, self._E_kin, self._E_tot]),
    #                          particle_positions,
    #                          particle_a1s,
    #                          particle_a3s)
    #
    #     return top, conf

    def particle_types(self) -> PLParticleSet:
        return self._particle_types

    def set_particle_types(self, ptypes: PLParticleSet):
        """
        this seems simple. it is not!
        """
        self._particle_types = ptypes

    def get_conf(self) -> Configuration:
        positions: np.ndarray = np.array([
            p.position() for p in self.particles()
        ])
        a1s: np.ndarray = np.array([
            p.a1 for p in self.particles()
        ])
        a3s: np.ndarray = np.array([
            p.a3 for p in self.particles()
        ])

        return Configuration(
            self._time,
            self.box_size(),
            np.array([self._E_pot, self._E_kin, self._E_tot]),
            positions,
            a1s,
            a3s
        )

    def set_time(self, t):
        self._time = t

    def patchy_interaction_strength(self, p1: PLPatchyParticle, p2: PLPatchyParticle) -> float:
        """
        Computes the energy of the patch-patch iteraction strength between two particles.
        does not consider hard-sphere repulsiob potential
        # TODO: FINISH WRITING
        """
        energy = 0
        for p1patch, p2patch in zip(p1.patches(), p2.patches()):
            # check if patches are complimentary
            if p1patch.can_bind(p2patch):
                # check binding geometry
                # (warning: sus)
                patch1_pos = p1.patch_position(p1patch)
                patch2_pos = p2.patch_position(p2patch)
                # todo: verify that this behavior is correct!
                d = dist(patch1_pos, patch2_pos)
                # 4 * patch.width = 2 x patch radius x 2 patches

    def particles_bound(self,
                        p1: Union[PLPatchyParticle, int],
                        p2: Union[PLPatchyParticle, int]) -> bool:
        """
        Checks if two particles in this scene are bound
        """
        if isinstance(p1, int):
            return self.particles_bound(self.get_particle(p1), self.get_particle(p2))
        # TODO: employ patch-patch energy computations
        # return self.patchy_interaction_strength(p1, p2) < -0.1
        # for now, assume any two complimentary patches within 0.1 units are bound
        for p1patch, p2patch in itertools.product(p1.patches(), p2.patches()):
            if self.patches_bound(p1, p1patch, p2, p2patch):
                return True

        return False

    def patches_bound(self,
                      particle1: PLPatchyParticle,
                      p1: PLPatch,
                      particle2: PLPatchyParticle,
                      p2: PLPatch) -> bool:
        # TODO: better calculationzs
        patch1_pos = particle1.patch_position(p1)
        patch2_pos = particle2.patch_position(p2)
        d = dist(patch1_pos, patch2_pos)
        if d <= PATCHY_CUTOFF:
            return True
        return False
