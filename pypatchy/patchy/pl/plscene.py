from __future__ import annotations

import copy
import itertools
import math
import random
from typing import Union, Iterable

import numpy as np
from oxDNA_analysis_tools.UTILS.RyeReader import inbox
from oxDNA_analysis_tools.UTILS.data_structures import Configuration
from pypatchy.cell_lists import CellLists

from .plparticle import PLPatchyParticle
from .plparticleset import PLParticleSet, MultidentateConvertSettings
from .plpatch import PLPatch
from .plpotential import PLPotential
from ...scene import Scene
from ...util import dist, pairwise, random_unit_vector

PATCHY_CUTOFF = 0.18


class PLPSimulation(Scene, CellLists):
    """
    Class for a patchy particle simulation.
    Note: in case the fact that this is written in Python doesn't make this abundantly clear,
    this class is NOT INTENDED FOR RUNNING MOLECULAR DYNAMICS SIMULATIONS!!!!
    Hence why as of this time I have not written methods for forces - only energies
    """
    _particle_types: PLParticleSet
    _temperature: float
    _E_tot: float
    _E_pot: float
    _E_kin: float

    _time: int

    potentials: list[PLPotential]

    def __init__(self, seed=69):
        super().__init__()
        CellLists.__init__(self)
        self._box_size = np.array([0., 0., 0.])
        self._E_tot = 0.
        self._E_pot = 0.
        self._E_kin = 0.
        self._time = 0
        self._temperature = 0

        random.seed(seed)

        self._colorbank = []
        self.potentials = []

    def add_potential(self, potential: PLPotential):
        self.potentials.append(potential)

    def inbox(self):
        """
        moves all particles to inside the box
        """
        assert (self._box_size > 0).all()
        self.set_conf(inbox(self.get_conf()))

    def set_temperature(self, t: float):
        self._temperature = t

    def T(self) -> float:
        return self._temperature

    def generate_random_color(self, particle_id=-1) -> str:
        # random.seed(seed)
        if True or len(self._colorbank) < 1 or particle_id == -1:
            r = [random.random() for _ in range(3)]
            color = np.array2string(np.array(r), separator=",")[1:-1]
            # color = '%f,%f,%f' % (r[0], r[1], r[2])
            return color
        else:
            return self._colorbank[particle_id]

    def get_potential_energy(self) -> float:
        """
        computes average potential energy of the simulation from scratch
        """
        e = 0.
        checked_interactions: set[tuple] = set()
        for p1 in self.particles():
            for p2 in self.interaction_particles(p1):
                pair = (p1.get_uid(), p2.get_uid()) if p1.get_uid() < p2.get_uid() else (p2.get_uid(), p2.get_uid())
                if pair not in checked_interactions:
                    e_int = self.interaction_energy(p1, p2)
                    e += e_int
                    checked_interactions.add(pair)
        return e / self.num_particles()

    def get_energy_full(self) -> float:
        """
        computes average potential energy by looping energy of every particle pair
        O(n^2), so v. computationally intensive at high particle counts
        """
        e = 0.
        for p1, p2 in itertools.combinations(self.particles(), 2):
            e_int = self.interaction_energy(p1, p2)
            e += e_int
        return e / self.num_particles()

    def translate(self, translation_vector: np.ndarray):
        for p in self._particles:
            p.translate(translation_vector)
        if self.particle_cells is not None:
            self.apportion_cells()

    def sort_particles_by_type(self):
        """
        sorts particles in ascending order by type, without altering the particle layout
        """

        p_type_bins: list[list[PLPatchyParticle]] = [[] for _ in range(self.particle_types().num_particle_types())]
        for p in self.particles():
            p_type_bins[p.get_type()].append(p)
        self._particles = []
        self.particle_cells = dict() # reset cells (TODO: optimize?)
        for idx, p in enumerate(itertools.chain.from_iterable(p_type_bins)):
            p.set_uid(idx)
            self.add_particle(p)
        assert all(p1.get_type() <= p2.get_type() for p1, p2 in pairwise(self.particles()))

    def check_for_particle_overlap(self, particle: PLPatchyParticle, boltzmann: bool = True) -> bool:
        """
        Checks if a particle overlaps any other particles in the simulation.
        Returns:
            true if the particle overlaps another particle, false otherwise
        """
        # print 'Adding ', particle.cm_pos
        # please god let this be inboxed
        assert not boltzmann or self.T() > 0
        for p2 in self.interaction_particles(particle.position()):
            e = self.interaction_energy(particle, p2)
            if boltzmann:  # enable stochastic checking
                # boltzmann factor = e^(-energy / kT)
                boltzmann_factor = math.exp(-e / self.T())
                # if boltzmann factor is less than rng 0:1.0
                if boltzmann_factor < random.random():
                    return True
            elif e > 0:  # TODO: better energy cutoff?
                return True

            # dist = self.dist_2particles(particle, p)
            # # print ' Looking at distance from ', p.cm_pos,dist
            # if dist <= dist_cutoff:
            #     # print 'Unfortunately overlaps with ',p.cm_pos,dist
            #     return True
            #     # print 'Check is fine!'
        return False

    def apportion_cells(self):
        """
        apportions cells
        """
        CellLists.apportion_cells(self)
        self.apportion_cell_particles(self.particles())

    def add_particles(self, particles: list[PLPatchyParticle], strict_check=True):
        # adds particles to the field, also initializes paricle types and patchy types based on these data
        # it overwrites any previosuly stored particle!!
        for p in copy.deepcopy(particles):
            self.add_particle(p)
        # now treat types:
        saved_types = {}
        for p in self.particles():
            if p.get_type() not in saved_types.keys():
                saved_types[p.get_type()] = copy.deepcopy(p)

    def add_particle(self, p: PLPatchyParticle):
        super().add_particle(p)
        cell = self.get_cell(p.position())
        cell.particles.append(p)
        self.particle_cells[p.get_uid()] = cell

    def insert_particle(self,
                        particle: PLPatchyParticle,
                        check_overlap=False):
        if check_overlap:
            if self.check_for_particle_overlap(particle):
                return False
        self.add_particle(particle)
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

    def add_particle_rand_positions(self,
                                    particles: Iterable[PLPatchyParticle],
                                    nTries=1e3):
        # loop particles
        for p in particles:
            t = 0
            # set max tries so it doesn't go forever
            while t < nTries:
                # random position
                new_pos = np.random.rand(3) * self._box_size
                p.set_position(new_pos)
                if not self.check_for_particle_overlap(p):
                    break
                t += 1
            if t == nTries:
                raise Exception(f"Could not find a position t(o place a particle! nTries={nTries}")
            # randomize orientation
            p.set_random_orientation()
            self.add_particle(p)
        # the following assertion is useful for catching energy issues at low particle counts
        # but it's VERY VERY SLOW
        # assert self.get_potential_energy() < 0

    def add_conf_clusters(self,
                          conf_clusters: Iterable[PLPSimulation], # are you insane
                          nTries=1e3):
        # loop clusters
        for cluster in conf_clusters:
            t = 0
            # set max tries so it doesn't go forever
            while t < nTries:
                # random position
                new_pos = np.random.rand(3) * self._box_size
                cluster.translate(new_pos)
                if not self.check_for_particle_overlap(cluster):
                    break
                t += 1
            if t == nTries:
                raise Exception(f"Could not find a position t(o place a particle! nTries={nTries}")
            # randomize orientation
            # compute random rotation matrix
            a1 = random_unit_vector()
            x = random_unit_vector()
            # i had to replace this code Joakim or someone wrote because it's literally the "what not to do" solution
            # self.a1 = np.array(np.random.random(3))
            # self.a1 = self.a1 / np.sqrt(np.dot(self.a1, self.a1))
            # x = np.random.random(3)
            a3 = x - np.dot(a1, x) * a1
            a3 = a3 / np.sqrt(np.dot(a3, a3))
            rot = np.stack([a1, a2, a3])

    # the following assertion is useful for catching energy issues at low particle counts
    # but it's VERY VERY SLOW
    # assert self.get_potential_energy() < 0

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

    def set_conf(self, conf: Configuration, auto_inbox: bool = True):
        """
        sets particle positions from conf
        """
        if auto_inbox:
            conf = inbox(conf)
        # set scene time
        self.set_time(conf.time)
        # set scene energy
        self._E_pot, self._E_kin, self._E_tot = conf.energy
        # set box size
        self._box_size = conf.box
        # check conf size
        assert conf.positions.shape[1] == self.num_particles(), f"Mismatch between num particles in scene" \
                                                                f" ({self.num_particles()}) and conf" \
                                                                f" ({conf.positions.shape[1]})"
        # set particle positions, a1s, a3s
        for i, p in enumerate(self.particles()):
            p.set_position(conf.positions[i, :])
            p.a1 = conf.a1s[i, :]
            p.a3 = conf.a3s[i, :]

    def set_time(self, t):
        self._time = t

    def interaction_energy(self, p1: PLPatchyParticle, p2: PLPatchyParticle) -> float:
        """
        computes the interaction potential between two particles
        """
        e = 0.
        for potential_function in self.potentials:
            # TODO: non-periodic or semi-periodic dimensions
            e += potential_function.energy(self.box_size(), p1, p2)
        return e

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
        if not p1.can_bind(p2):
            return False
        # TODO: better calculationzs
        patch1_pos = particle1.patch_position(p1)
        patch2_pos = particle2.patch_position(p2)
        d = dist(patch1_pos, patch2_pos)
        if d <= PATCHY_CUTOFF:
            return True
        return False

    def add(self, other: PLPSimulation, cubize_box: bool = True):
        # assert self.particle_types() == other.particle_types() # WARNING
        if cubize_box:
            box_max_dimension = max(np.concatenate([self.box_size(), other.box_size()]))
            self.set_box_size(np.array([box_max_dimension, box_max_dimension, box_max_dimension]))
        else:
            self.set_box_size(np.max([self.box_size(), other.box_size()]))
        # WARNING: structures along periodic boundries will be VERY messed up by this process!!!
        for p in other.particles():
            p_copy: PLPatchyParticle = copy.deepcopy(p)
            p_copy.set_uid(len(self.particles()))
            self.add_particle(p_copy)

    def to_multidentate(self, *args, **kwargs) -> PLPSimulation:
        # dental_radius: float,
        # num_teeth: int,
        # torsion: bool = True,
        # follow_surf: bool = False
        # ) -> PLPSimulation:
        """
        Returns a multidentate version of the scene
        """
        if len(args) == 0:
            mdt_convert = MultidentateConvertSettings(**kwargs)
        else:
            mdt_convert = args[0]
        # convert scene particle set to multidentate
        mdt_particle_set = self.particle_types().to_multidentate(mdt_convert)
        assert all([np.abs(p.rotmatrix() - np.identity(3)).sum() < 1e-6 for p in mdt_particle_set.particles()])
        mdt_scene = PLPSimulation()
        # assign scene particle types
        mdt_scene.set_particle_types(mdt_particle_set)
        # loop particles in scene
        for particle in self.particles():
            # clone type particle from particle set
            mdt_particle = copy.deepcopy(mdt_particle_set.particle(particle.get_type()))
            # assign position
            mdt_particle.set_position(particle.position())
            # assign rotation
            # this is a absolute mess because default rotation isn't identity matrix
            # for some reason

            # 3x3 rotation matrices
            original_type_rot: np.ndarray = self.particle_types().particle(particle.get_type()).rotmatrix()
            mdt_type_rot: np.ndarray = mdt_particle_set.particle(particle.get_type()).rotmatrix()
            mdt_particle_rot: np.ndarray = mdt_particle.rotmatrix()
            particle_rot: np.ndarray = particle.rotmatrix()

            # now construct a rotation matrix to rotate mdt_type_rot such that the rotation from mdt_type_rot to new_rot
            # is the same as the rotation from original_type_rot to particle_rot

            # here's what chatGPT says:
            # Calculate the inverse (or transpose, since it's a rotation matrix) of original_type_rot
            inverse_original_type_rot = np.linalg.inv(original_type_rot)

            # Calculate the relative rotation from original_type_rot to particle_rot
            relative_rot = np.dot(inverse_original_type_rot, particle_rot)

            # Apply the relative rotation to mdt_type_rot to get new_rot
            new_rot = np.dot(mdt_type_rot, relative_rot)

            # Pre-compute the target rotation for comparison
            target_rot = np.dot(particle_rot, original_type_rot)

            # Apply new_rot to mdt_type_rot to get the resulting rotation
            resulting_rot = np.dot(mdt_type_rot, new_rot)

            # Use np.allclose to compare the resulting_rot and target_rot matrices within a tolerance
            assert np.allclose(resulting_rot, target_rot, atol=1e-6), "The new rotation does not align as expected."

            mdt_particle.rotate(new_rot)

            # assign UID
            mdt_particle.set_uid(particle.get_id())
            # add particle to scene
            mdt_scene.add_particle(mdt_particle)
        mdt_scene.set_box_size(self.box_size())
        return mdt_scene
