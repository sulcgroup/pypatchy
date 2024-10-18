from __future__ import annotations

import copy
import itertools
import math
import random
from pathlib import Path
from typing import Union, Iterable, Generator

import networkx as nx
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
        self.set_conf(inbox(self.get_conf(), center=False))

    def is_inboxed(self) -> bool:
        return all([
            all((particle.position() >= 0) & (particle.position() < self.box_size()))
            for particle in self.particles()])

    def set_temperature(self, t: float):
        self._temperature = t

    def T(self) -> float:
        return self._temperature

    def time(self) -> int:
        return self._time

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
                # 2-particle dist should always be >= r1+r2 but there are cases where we want to test for insanely high energies
                if self.dist_2particles(p1, p2) <= p1.radius() + p2.radius():
                    return float('inf')
                else:
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

    def translate(self, translation_vector: np.ndarray, inbox: bool = True, reapporiton_cells: bool = True):
        for p in self._particles:
            p.translate(translation_vector)
        if inbox:
            self.inbox()
        if reapporiton_cells and self.particle_cells is not None:
            self.apportion_cells()

    def rotate(self, rot: np.ndarray, center: Union[None, np.ndarray] = None):
        assert np.allclose(rot.transpose() @ rot, np.identity(3))
        assert abs(1-np.linalg.det(rot) < 1e-4)
        if center is None:
            center = self.cms()
        for p in self._particles:
            # move to rotation reference pt
            p.translate(-center)
            # set position to position rotated by matrix
            p.set_position(p.position() @ rot)
            # rotate particle in-place
            p.rotate(rot.T)
            # return from reference pt
            p.translate(center)

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

    def clear_particles(self):
        """
        removes all particles
        """
        # clear particles list
        self._particles.clear()
        # clear cells
        self.apportion_cells()

    def add_particle(self, p: PLPatchyParticle):
        super().add_particle(p)
        cell = self.get_cell(p.position())
        cell.particles.append(p)
        self.particle_cells[p.get_uid()] = cell

    def insert_particle(self,
                        particle: PLPatchyParticle,
                        check_overlap=False):
        """
        ADDS
        """
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

    def num_particle_types(self) -> int:
        return self.particle_types().num_particle_types()

    def export_to_mgl(self,
                      filename: Union[Path, str],
                      patches_w: float,
                      regime: str = 'w'):
        if isinstance(filename, str):
            filename = Path(filename)
        with filename.open(regime) as out:
            out.write(f".Box: {np.array2string(self._box_size, separator=',')[1:-1]}\n")
            # sout = ".Box:%f,%f,%f\n" % (self._box_size[0], self._box_size[1], self._box_size[2])
            for p in self._particles:
                out.write(p.export_to_mgl(patch_width=patches_w, patch_shrink_scale=0.0))

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
            conf = inbox(conf, center=False)
        # set scene time
        self.set_time(conf.time)
        # set scene energy
        self._E_pot, self._E_kin, self._E_tot = conf.energy
        # set box size
        self._box_size = conf.box
        # check conf size
        assert conf.positions.shape[0] == self.num_particles(), f"Mismatch between num particles in scene" \
                                                                f" ({self.num_particles()}) and conf" \
                                                                f" ({conf.positions.shape[0]})"
        # set particle positions, a1s, a3s
        for i, p in enumerate(self.particles()):
            p.set_position(conf.positions[i, :])
            p.a1 = conf.a1s[i, :]
            p.a3 = conf.a3s[i, :]

    def alter_box_size(self, new_box_size: Union[np.ndarray, float], cluster_energy_threshold: float):
        """
        This is VERY different from set_box_size
        """
        # if we've passed a scaling factor instead of an actual
        if not isinstance(new_box_size, np.ndarray):
            new_box_size = self.box_size()  * new_box_size
        assert len(self.potentials) > 0, "Cannot resize box without first adding interaction potentials"
        # find clusters
        clusters = list(self.split_scene_by_clusters(cluster_energy_threshold, 2))
        # to find counts of free-floating particles, start w/ total counts and count down
        free_particles_types = self.particle_type_counts()
        # iter clusters
        for cluster in clusters:
            for type_id, count in cluster.particle_type_counts().items():
                # mark particles of type type_id as accounted for
                free_particles_types[type_id] -= count
                assert free_particles_types[type_id] >= 0, f"Miscount (?) of particle type {type_id}!"
        # clear particles
        self.clear_particles()
        # set new box size
        self.set_box_size(new_box_size)
        # re-add clusters
        # in this operation, we lose uids, which is HOPEFULLY OKAY??
        self.add_conf_clusters(clusters)
        # add free particles
        for type_id, count in free_particles_types:
            self.add_particle_rand_positions([
                self.particle_types().particle(type_id).instantiate(self.num_particles()+i)
                for i in range(count)
            ])

    def set_time(self, t):
        self._time = t

    def cms(self) -> np.ndarray:
        position_sum = np.zeros((3,))
        for particle in self.particles():
            assert self.is_inboxed(),\
                "Cannot get cms of non-inboxed conf"
            position_sum += particle.position()
        return position_sum / self.num_particles()

    def interaction_energy(self, p1: Union[PLPatchyParticle, int], p2: Union[PLPatchyParticle, int]) -> float:
        """
        computes the interaction potential between two particles
        """
        e = 0.
        if isinstance(p1, int):
            p1 = self.get_particle(p1)
        if isinstance(p2, int):
            p2 = self.get_particle(p2)
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

    def compute_scene_graph(self, bond_energy: float = 0) -> nx.Graph:
        """
        Constructs a nx graph containing all particles in the scene
        """
        # use graph not digraph
        G = nx.Graph()
        # iter particles
        for p1 in self.particles():
            for p2 in self.interaction_particles(p1):
                e = self.interaction_energy(p1, p2)
                if e < bond_energy:
                    G.add_edge(p1.get_uid(), p2.get_uid(), e=e)
        return G

    def compute_scene_clusters(self, bond_energy: float=0, min_cluster_size: int = 1) -> Generator[nx.Graph]:
        G = self.compute_scene_graph(bond_energy=bond_energy)
        for g in nx.connected_components(G):
            if len(g) >= min_cluster_size:
                yield G.subgraph(g)

    def split_scene_by_clusters(self,bond_energy: float=0,  min_cluster_size: int = 2) -> Generator[PLPSimulation]:
        for cluster in self.compute_scene_clusters(bond_energy=bond_energy, min_cluster_size=min_cluster_size):
            yield self.subscene(cluster)

    def subscene(self, particle_ids: Iterable[int]) -> PLPSimulation:
        scene = PLPSimulation()
        scene.set_time(self.time())
        scene.set_temperature(self.T())
        scene.set_particle_types(self.particle_types())
        # list particles in cluster
        scene_particles = [copy.deepcopy(self.get_particle(particle_id)) for particle_id in particle_ids]
        # do box size computatioins
        # set box size for cluster scene (add padding)
        # scene.set_box_size(self.box_size())
        scene.set_box_size(np.stack([p.position() for p in scene_particles]).max(axis=0) + np.array([0.1, 0.1, 0.1]))
        scene.compute_cell_size(n_particles=len(scene_particles))
        scene.apportion_cells()
        # add particles
        for p in scene_particles:
            scene.add_particle(p)
        # inbox scene
        scene.inbox()
        scene.translate(-np.stack([p.position() for p in scene.particles()]).min(axis=0))
        return scene

    def add_conf_clusters(self, clusters: list[PLPSimulation], nTries=1e3):
        """
        Adds other PL confs ("clusters") to this simulation, using a similar method to add_particle_rand_positions
        """
        # loop particles
        for cluster in clusters:
            # we need to cache positions because PLPSimulation (unlike PLPatchyParticle)
            # has no method to set absolute position

            # set cluster box size
            cluster.set_box_size(self.box_size())
            conf_start = cluster

            t = 0
            # set max tries so it doesn't go forever
            while t < nTries:
                cluster = copy.deepcopy(conf_start)

                # # randomly rotate cluster
                a1 = random_unit_vector()
                x = random_unit_vector()
                # self.a1 = np.array(np.random.random(3))
                # self.a1 = self.a1 / np.sqrt(np.dot(self.a1, self.a1))
                # x = np.random.random(3)
                a3 = x - np.dot(a1, x) * a1
                # normalize a3 vector
                a3 = a3 / np.sqrt(np.dot(a3, a3))
                if abs(np.dot(a1, a3)) > 1e-10:
                    raise Exception("Could not generate random orientation?")
                # please tell me this produces a random rotation matrix
                cluster.rotate(np.stack([a1, np.cross(a3, a1), a3]))

                # random position
                new_pos = np.random.rand(3) * self._box_size
                cluster.translate(new_pos, True, False)

                cluster.inbox()

                assert np.allclose(self.box_size(), cluster.box_size())
                if not self.check_conf_overlap(cluster):
                    self.add(cluster)
                    break
                t += 1
            if t == nTries:
                raise Exception(f"Could not find a position t(o place a cluster! nTries={nTries}")

    def check_conf_overlap(self, cluster: PLPSimulation) -> bool:
        """
        checks if the provided cluster would create high-energy interactions (vol excl overlap)
        uses this.potentials
        """
        # not using cells for these
        e = 0.
        for p1, p2 in itertools.product(self.particles(), cluster.particles()):
            for potential in self.potentials:
                e += potential.energy(self.box_size(), p1, p2)
            # should return false if overlap is present, true if energetically acceptable
        return np.exp(-e / self.T()) <= random.random()
