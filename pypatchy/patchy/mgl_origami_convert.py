import math
from copy import deepcopy
from pathlib import Path
from typing import Generator, Iterable, Union

import numpy as np
from oxDNA_analysis_tools.UTILS.RyeReader import Configuration
from Bio.SVDSuperimposer import SVDSuperimposer

from .dna_particle import DNAParticle
from .patchy_scripts import mgl_to_pl
from .pl.plparticle import PLPatchyParticle, PLPatch
from .pl.plscene import PLPSimulation
from ..dna_structure import DNABase, construct_strands, BASE_BASE, rc, POS_BASE, DNAStructure

from random import choice
import itertools

from ..patchy.mgl import MGLParticle, MGLPatch, MGLScene
from ..patchy_base_particle import PatchyBaseParticle
from ..scene import Scene
from ..util import dist, normalize, get_output_dir


# todo: MORE PARAMETERS
def patch_positions_to_bind(patch_positions1: Iterable[int],
                            patch_positions2: Iterable[int],
                            conf1: Configuration,
                            conf2: Configuration) -> list[tuple[int, int]]:
    # get patch positions on particle 2 as an array
    pps2 = np.array([conf2.positions[pid] for pid in patch_positions2])

    # get all possible permutations of patch positions on particle 1
    perms = np.array(list(itertools.permutations([conf1.positions[pid] for pid in patch_positions1])))

    # compute vector difference between the particle 2 patch positions and the permutation matrix
    diffs = pps2[np.newaxis, :, :] - perms  # vector difference
    distancesqrd = np.sum(np.multiply(diffs, diffs), axis=(2,))  # dot product, gives us distance squared
    # sum of squares
    sums = np.sum(distancesqrd, axis=(1,))

    bestmatch = np.argmin(sums)
    best_patch1_positions: list[int] = list(itertools.permutations(patch_positions1))[bestmatch]
    return zip(
        best_patch1_positions,
        patch_positions2)

    # pairs = [] # we store pairs of corresponding indicesreverse
    # pp_pos2 = deepcopy(patch_positions2) #to be expendable
    # # go over every patch in the list for 1st particle
    # for patch_id1 in patch_positions1:
    #     # calculate the distances to all patches of particle 2
    #     dists = [dist(conf1.positions[patch_id1] - conf2.positions[id]) for id in pp_pos2]
    #     # figure out the index of the min particle 75
    #     min_dist = min(dists)
    #     id_min = dists.index(min_dist)
    #     # eliminate that index from the positions to scan against
    #     pairs.append(
    #         (patch_id1, pp_pos2.pop(id_min))
    #     )
    # return pairs # the closest patches on the 2 particles provided


def pairwise(iterable):
    # because i'm stuck on python 3.8
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def patch_idxs_to_bind(patch_id_1: int,
                       patch_id_2: int,
                       dna1: DNAParticle,
                       dna2: DNAParticle) -> list[tuple[int, int]]:
    """
    Parameters:
        patch_id_1 (int): indexer for patch 1
        patch_id_2 (int): indexer for patch 2
        dna1 (DNAParticle): first dna particle to bind
        dna2 (DNAParticle): second dna particle to bind

    Return:
        a tuple of pairs of indexes in patch 1 and patch 2 where the indexes are of the strand indexes (i'm so sorry)

    """

    patch_positions1 = dna1.dna_patch_positions(patch_id_1)
    patch_positions2 = dna2.dna_patch_positions(patch_id_2)
    # get patch positions on particle 2 as an array

    # get all possible permutations of patch positions on particle 1
    perms = np.array(list(itertools.permutations([
        patch_positions1[i, :] for i in range(patch_positions1.shape[0])]
    )))

    # compute vector difference between the particle 2 patch positions and the permutation matrix
    diffs = patch_positions2[np.newaxis, :, :] - perms  # vector difference
    distancesqrd = np.sum(np.multiply(diffs, diffs), axis=(2,))  # dot product, gives us distance squared
    # sum of squares
    sums = np.sum(distancesqrd, axis=(1,))

    bestmatch = np.argmin(sums)
    best_patch1_idxs: list[int] = list(itertools.permutations(range(len(patch_positions1))))[bestmatch]
    return zip(
        best_patch1_idxs,
        range(patch_positions2.shape[0]))


class MGLOrigamiConverter:
    """
    This class facilitates the conversion of a patchy particle model (in MGL format) to
    a set of connected DNA origamis
    """
    color_sequences: dict[Union[int, str], str]
    bondcount: int
    particle_type_map: dict[str, DNAParticle]
    patchy_scene: PLPSimulation
    # mapping where keys are PL particle UIDs and values are DNA particles
    dna_particles: dict[int, DNAParticle]
    padding: float  # manually-entered extra spacing
    dist_ratio: float

    def __init__(self,
                 scene: Union[MGLScene, PLPSimulation],
                 particle_types: Union[Path, DNAParticle,
                                       dict[str, DNAParticle]],
                 # additional arguements (optional)
                 spacer_length: int = 16,
                 particle_delta: float = 1.2,
                 bond_length: float = 0.4,
                 cos_theta_max: float = 0.95,
                 sticky_length: int = 18,
                 padding: float = 1.0,
                 flexable_patch_distances: bool = False,
                 expected_num_edges: int = -1
                 ):
        """
        Initializes this converter using a few required params and a lot of optional fine-tuning params
        Parameters:
            scene: an MGLScene object representing the structure we're making with the particles
            particle_types: describes the DNA origamis that will be used for particles.
            can be a single origami or a dict relating mgl colors to origamis
            spacer_length: length of poly-T spacer between particle and sticky end
            particle_delta: distance beyond which we can assume particles do not interact
            bond_length: distance beyond which we can assume patches do not interact. TODO: replace with some sort of gaussian?
            cos_theta_max: maximim cos(theta) angle between patches (TODO: interact with patch width?)
            sticky_length: length of sticky end sequences
            padding: magic number to get a little bit of space
        """

        # set parameters
        self.spacer_length = spacer_length
        self.particle_delta = particle_delta
        self.bond_length = bond_length
        self.cos_theta_max = cos_theta_max
        self.sticky_length = sticky_length

        # prep variables to use when building our structure
        self.color_sequences = {}
        self.color_match_overrides = {}
        self.bondcount = 0
        self.padding = padding
        self.dist_ratio = None
        self.dna_particles = None

        # inputs
        if isinstance(scene, MGLScene):
            scene = mgl_to_pl(scene)
        self.patchy_scene = scene

        # if only one particle type
        if isinstance(particle_types, DNAParticle):
            self.particle_type_map = {
                particle.get_type(): particle_types for particle in self.patchy_scene.particle_types().particles()
            }

        else:  # forward-proof for heterogenous systems
            assert isinstance(particle_types, dict)
            self.particle_type_map = particle_types

        # optional parameters
        self.flexable_patch_distances = flexable_patch_distances
        self.expected_num_edges = expected_num_edges
        assert not flexable_patch_distances or expected_num_edges > 0

    def get_dna_origami(self, particle_type: Union[str, int, PatchyBaseParticle]) -> DNAParticle:
        if isinstance(particle_type, PatchyBaseParticle):
            return self.get_dna_origami(particle_type.get_type())
        else:
            assert particle_type in self.particle_type_map, f"Particle type {particle_type} not in type map!"
            return self.particle_type_map[particle_type]

    def get_full_conf(self) -> DNAStructure:
        print("merging the topologies")
        particles = self.get_particles()
        merged_conf = sum(particles[1:], start=particles[0])
        return merged_conf

    def get_dist_ratio(self) -> float:
        """
        gets the scaling value used to convert the particle scene to the oxDNA conf
        calculates value if it has not already been computed
        Returns:
            a conversion factor in units of oxDNA Units / Scene Units
        """

        if self.dist_ratio is None:
            dist_ratios = []
            dna_distance = (2 * self.spacer_length + self.sticky_length) * BASE_BASE

            # WARNING! this algorithm could become very intense very quickly for large scenes
            for p1, p2 in self.patchy_scene.iter_bound_particles():

                # compute distance between two particles in the scene (mgl or pl)
                particles_distance = dist(p1.position(), p2.position())
                # technically radius() isn't a member of BaseParticle
                # radii: float = p1.radius() + p2.radius()

                p1_dna = self.particle_type_map[p1.get_type()]
                p2_dna = self.particle_type_map[p2.get_type()]

                # compute DNA particle "radius" as determined by average distance between the patch and cms
                p1_dna_rad = p1_dna.center2patch_conf()
                p2_dna_rad = p2_dna.center2patch_conf()

                # distance between cmss of two dna particles
                # should be very close to constant across the system
                dna_distance_2p = p1_dna_rad + p2_dna_rad + dna_distance
                dist_ratio = dna_distance_2p / particles_distance
                dist_ratios.append(dist_ratio)
            dist_ratio = np.mean(dist_ratios)
            assert not np.isnan(dist_ratio)
            if np.std(dist_ratios) > 0.05 * dist_ratio:
                print(f"Warning: The standard deviation of individual particle distance ratios {np.std(dist_ratios)}"
                      f" is more than 5% of the value of the mean distance ratio {dist_ratio}.")
            self.dist_ratio = dist_ratio

        return self.dist_ratio


    # TODO: write better?
    def color_sequence(self, colorstr: str) -> str:
        # if this color isn't in our color bank
        if colorstr not in self.color_sequences:
            # ...but its matching color is
            if self.get_color_match(colorstr) in self.color_sequences:
                # use the reverse compliment of the matching color sequenece
                self.color_sequences[colorstr] = rc(self.color_sequence(self.get_color_match(colorstr)))
                print(f"Assigning color {colorstr} sequence {self.color_sequences[colorstr]}"
                      f" (reverse compliment of {self.get_color_match(colorstr)})")
            else:
                # if neither our color nor its match is in the color seqs, generate a new sequence
                # todo: smarter?
                self.color_sequences[colorstr] = "".join(
                    choice(["A", "T", "C", "G"]) for _ in range(self.sticky_length))
                print(f"Assigning color {colorstr} random sequence {self.color_sequences[colorstr]}")

        return self.color_sequences[colorstr]

    def assign_color_sequence(self, color: int, seq: str):
        """
        Assigns the color given by colorstr the specific sequence sequence specified
        Automatically assigns the corresponding color the reverse compliment
        """
        assert len(seq) == self.sticky_length, "Incompatible sequence length"
        self.color_sequences[color] = seq
        self.color_sequences[-color] = rc(seq)

    def get_color_match(self, color: Union[int, str]) -> Union[int, str]:
        if color in self.color_match_overrides:
            return self.color_match_overrides[color]
        else:
            if isinstance(color, str):
                if color.startswith("dark"):
                    return color[4:]
                else:
                    return f"dark{color}"
            else:
                if not isinstance(color, int):
                    raise TypeError(f"Invalid color type {type(color)}")
                else:
                    return -color

    def get_expected_pad_distance(self) -> float:
        return (self.sticky_length + 2.0 * self.spacer_length) * POS_BASE

    def set_color_match(self,
                        colorstr: str,
                        match: str,
                        reverse_match: bool = True):
        self.color_match_overrides[colorstr] = match
        if reverse_match:
            self.color_match_overrides[match] = colorstr

    def match_patches_to_strands(self,
                                 p: PLPatchyParticle,
                                 dna: DNAParticle) -> np.ndarray:
        """
        Computes a rotation of this DNA particle that makes the patches on that particle line up with the
        3' ends of patch strands.
        Parameters:
            patch_groups: groups of patches that should be kept as a block for the purposes of
        Returns:
            a tuple where the first element is a rotation matrix (3x3), and the second is a mapping of patch IDs
            to patch strand
            For all but unidentate patches, the length of the sets of strands will be 1
        """
        # construct list of patch groups (can be 1 patch / group)
        # load each patch on the mgl particle

        # for patch in p.patches():
            # scale patch with origami
            # skip magic padding when scaling patch local coords
            # patchy_patches.append(patch.position() / self.scale_factor(p))
        # assert len(patchy_patches) == len(self.patch_centerpoints())
        # generate matrix of origami patch matrices
        best_rms = np.Inf
        # test different patch arrangements in mgl vs. origami. use best option.
        # best_order should be ordering of dna patches which best matches particle patches
        # iterate possible n-length permutations of the patches on the DNA particle,
        # where n is the number of patches on the patchy particles

        # compute best ordering of patches
        best_rot, patch_group_map = self.align_patches(p, dna)

        for udt_id, strand_group_idx in patch_group_map.items():
            strand_group = dna.patch_strand_ids[strand_group_idx]
            patch_group = self.patchy_scene.particle_types().mdt_rep(udt_id)
            # compute strand mapping
            strand_map: dict[int, int] = dna.align_patch_strands(patch_group, strand_group, dna.scale_factor(p))
            dna.assign_patches_strands(strand_map)

        print(f"Superimposed DNA origami on patchy particle {p.get_id()} with RMS={best_rms}")
        print(f"RMS / circumfrance: {best_rms / (2 * math.pi * dna.center2patch_conf())}")
        return best_rot

    def link_patchy_particle(self, p: PLPatchyParticle, dna: DNAParticle):
        """
        Links a patchy particle to a DNA structure, rotating the structure so that the orientation of the 3' ends of
        the patch strands match the position of the patches on the particle
        Note: PLEASE do not link multiple DNAParticle instances to the same patchy particle or vice
        versa!!!

        Parameters:
            p (PatchyBaseParticle): a patchy particle to link to a DNA particle
            dna (DNAParticle): a dna particle to link with the patchy particle
        """
        # assert len(self.patch_strand_ids) >= p.num_patches() or p.num_patches() % len(self.patch_strand_ids) == 0, \
        assert len(dna.flat_strand_ids) >= p.num_patches(), \
            f"Not enough patches on DNA particle ({len(dna.patch_strand_ids)}) to link DNA particle to " \
            f"on patchy particle({p.num_patches()})!"
        # compute strand mapping
        if not dna.has_strand_map():
            rot = self.match_patches_to_strands(p, dna)

        # call self.transform BEFORE linking the particle!
        dna.transform(rot)
        # self.patch_strand_ids = [self.patch_strand_ids[i] for i in best_order[:-1]] # skip last position (centerpoint)
        dna.linked_particle = p
        self.dna_particles[p.get_id()] = dna

    def align_patches(self, p: PLPatchyParticle, dna: DNAParticle) -> tuple[np.ndarray, dict[int, int]]:
        """
        Aligns patch centerpoints with the average positions of multidentate patches
        Returns:
            the ordering of groups of dna strand IDs that best matches the patches on the provided particle
        """
        sup = SVDSuperimposer()
        # clone dna particle
        pl_patch_centers = {}  # start with centerpoint of the particle (assumed to be 0,0,0)
        # load each patch on the patchy particle
        for patch in p.patches():
            # if patch is multidentate, get src patch id as map key
            if self.patchy_scene.particle_types().is_multidentate():
                patch_type_id = self.patchy_scene.particle_types().get_src_patch(patch).get_id()
            else:
                # use patch id as map key, will end up with identity mappigng
                patch_type_id = patch.get_id()
            if patch_type_id not in pl_patch_centers:
                pl_patch_centers[patch_type_id] = list()
            # scale patch with origami
            pl_patch_centers[patch_type_id].append(patch.position() / dna.scale_factor(p))
        # remember patch id order
        pid_order = list(pl_patch_centers.keys())
        assert len(pid_order) <= len(dna.patch_strand_ids)
        # compute patch centers
        pl_centers = np.stack([np.mean(plocs, axis=0) for plocs in pl_patch_centers.values()])
        assert pl_centers.shape == (len(pid_order), 3)
        # generate matrix of origami patches + centerpoint
        dna_patch_cmss = dna.get_patch_cmss()
        best_rms = np.Inf
        best_order = None
        best_rot = None
        cms = dna.cms()
        # loop permutations of patche strand gorups
        for perm in itertools.permutations(range(len(dna.patch_strand_ids)), r=len(pid_order)):
            # configure svd superimposer
            m1 = np.concatenate([pl_centers, np.zeros(shape=(1, 3))], axis=0)
            m2 = np.concatenate([dna_patch_cmss[perm, :], cms[np.newaxis, :]], axis=0)
            if m1.shape != m2.shape:
                raise Exception(f"Mismatch between patch position matrix on pl particle ({m1.shape}) "
                                f"and on dna particle ({m2.shape})")
            sup.set(m1, m2)

            # FIRE
            sup.run()
            # if rms is better
            if sup.get_rms() < best_rms:
                # save order
                best_order = dict(zip(pid_order, perm))
                best_rot = sup.rot
                # save rms
                best_rms = sup.get_rms()
        return best_rot, best_order

    def position_particles(self) -> list[DNAParticle]:
        """
        Positions particles?
        IDK
        """
        self.dna_particles = dict()
        # get scene particles
        particles = self.patchy_scene.particles()
        placed_confs = []  # output prom_p
        pl = len(particles)
        for i, particle in enumerate(particles):
            # clone dna particle
            origami: DNAParticle = deepcopy(self.get_dna_origami(particle))
            print(f"{i + 1}/{pl}", end="\r")
            # link DNA particle to pl particle
            self.link_patchy_particle(particle, origami)
            linker_len = (self.spacer_length * 2 + self.sticky_length) * BASE_BASE,
            # scale_factor = origami.scale_factor(particle) / self.padding
            scale_factor = self.get_dist_ratio() * self.padding
            # magic numbers needed again for things not to clash
            # origami.box = self.mgl_scene.box_size() / scale_factor
            # scale factor tells us how to convert MGL distance units into our DNA model distances
            origami.transform(tran=particle.position() * scale_factor)

            # we finished the positioning
            placed_confs.append(origami)
        print()
        return placed_confs

    # def particle_pair_candidates(self) -> Generator[tuple[MGLParticle], None, None]:
    #     """
    #     Returns all possible pairs of particles,
    #     as defined by interaction range between centers of mass
    #     """
    #     handeled_candidates = set()
    #     for i, p1 in enumerate(self.patchy_scene.particles()):
    #         for j, p2 in enumerate(self.patchy_scene.particles()):
    #             # if the particles are different and the distance is less than the maximum interaction distance
    #             if i != j and dist(p1.cms(),
    #                                p2.cms()) <= self.particle_delta:
    #                 if (i, j) not in handeled_candidates and not (j, i) in handeled_candidates:  # prevent repeats
    #                     handeled_candidates.add((i, j))
    #                     assert (p1.type_id(), p2.type_id()) == (i, j)
    #                     yield p1, p2

    # def patches_to_bind(p1, p2, patch_delta, cos_theta_max):
    #     for q,patch_1 in enumerate(p1.patches):
    #         for z,patch_2 in enumerate(p2.patches):
    #             if dist(p1.cms + patch_1.pos - (p2.cms+patch_2.pos)) <= patch_delta and colors_pair(patch_1, patch_2):
    #                 #https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
    #                 patch1norm = patch_1.pos / np.linalg.norm(patch_1.pos)
    #                 patch2norm = patch_2.pos / np.linalg.norm(patch_2.pos)
    #                 costheta = abs(float(np.clip(np.dot(patch1norm, patch2norm), -1.0, 1.0)))
    #                  # confusing,ly, cos theta-max is the cosine of the maximum angle
    #                  # so we check if cos theta is LESS than cos theta-max
    #                 if costheta >= cos_theta_max:
    #                     yield (q, patch_1), (z, patch_2)

    # def patches_to_bind(self,
    #                     particle_1: MGLParticle,
    #                     particle_2: MGLParticle) -> Generator[tuple[tuple[int, MGLPatch],
    #                                                                 tuple[int, MGLPatch]],
    #                                                           None,
    #                                                           None]:
    #     """
    #     Returns:
    #         a generator which produces pairs of tuples, each of which consists of patch index and patch object
    #     """
    #     assert particle_1.type_id() != particle_2.type_id()
    #     # keep in mind: we can't use patch internal IDs here because it enumerates differently!
    #     # find all possible pairings between two patches on particle 1 and particle 2
    #     # filter patches that don't pair
    #     possible_bindings = list(self.patchy_scene.iter_binding_patches(particle_1, particle_2))
    #     # sort by distance, ascending order
    #
    #     def sort_by_distance(p):
    #         patch1, patch2 = p
    #         return dist(particle_1.position() + patch1.position(),
    #                     particle_2.position() + patch2.position())
    #
    #     possible_bindings.sort(key=sort_by_distance)
    #     # lists for patches that have been handled on particles 1 and 2
    #     handled_p1 = set()
    #     handled_p2 = set()
    #     # iterate through possible pairs of patches
    #     for patch_1, patch_2 in possible_bindings:
    #         # skip patches we've already handled
    #         if not handled_p1[q] and not handled_p2[z]:
    #             # if the two patches are within bonding distance (mgl units)
    #             patch1_position = particle_1.cms() + patch_1.position()
    #             patch2_position = particle_2.cms() + patch_1.position()
    #             # compute patch distance
    #             patches_distance = dist(patch1_position, patch2_position)
    #             if (patches_distance <= self.bond_length) or \
    #                     (self.flexable_patch_distances and (self.bondcount < self.expected_num_edges)):
    #                 # https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
    #                 # normalize patches
    #                 patch1norm = patch_1.position() / np.linalg.norm(patch_1.position())
    #                 patch2norm = patch_2.position() / np.linalg.norm(patch_2.position())
    #                 costheta = abs(float(np.clip(np.dot(patch1norm,
    #                                                     patch2norm), -1.0, 1.0)))
    #                 # confusing,ly, cos theta-max is the cosine of the maximum angle
    #                 # so we check if cos theta is LESS than cos theta-max
    #                 if costheta >= self.cos_theta_max:
    #                     handled_p1[q] = handled_p2[z] = True
    #                     yield (q, patch_1), (z, patch_2)
    #             else:  # can do this because list is sorted
    #                 break

    def bind_particles3p(self,
                         dna_particles: list[DNAParticle]):
        """
        Creates double-stranded linkers to bind particles together
        """

        # use pl function to iter bound particles
        for p1, p2 in self.patchy_scene.iter_bound_particles():
            p1_dna = self.get_dna_particle(p1)
            p2_dna = self.get_dna_particle(p2)
            # iter pairs of patches that connect p1 to p2
            for patch1, patch2 in self.patchy_scene.iter_binding_patches(p1, p2):
                self.bind_patches_3p(p1_dna,
                                      patch1,
                                      p2_dna,
                                      patch2)

        #OLD CODE

        # assert self.patchy_scene.patch_ids_unique()
        # please please do not call this method with dna particles that don't correspond
        # in order to the mgl particles

        # patch_occupancies = [False for _ in itertools.chain.from_iterable(
        #     [[patch.get_id() for patch in particle.patches()] for particle in self.patchy_scene.particles()])]
        #
        # # loop possible pairs of particles
        # # patch_occupancies = [[False for _ in p.patches()] for p in particles]
        #
        # patchpaircount = 0
        # # loop particles
        # for p1, p2 in self.patchy_scene.iter_bound_particles():
        #     # i and j are particle unique ids
        #     # actually UIDs not type IDs
        #     i = p1.type_id()
        #     j = p2.type_id()
        #     assert i != j, "Particle should not bind to self!"
        #     # grab DNAParticle objects
        #     p1_dna: DNAParticle = dna_particles[i]
        #     p2_dna: DNAParticle = dna_particles[j]
        #     # loop through the patch pairs on each particle that can bind
        #     for (q, patch1), (z, patch2) in self.patches_to_bind(p1, p2):
        #         assert -1 < patch1.get_id() < len(patch_occupancies), "Mismatch between patch ID on structure and scene patches!"
        #         assert -1 < patch2.get_id() < len(patch_occupancies), "Mismatch between patch ID on structure and scene patches!"
        #         assert patch1.get_id() != patch2.get_id()
        #         # if either patch is bound, stop!
        #         if not patch_occupancies[patch1.get_id()] and not patch_occupancies[patch2.get_id()]:
        #             patch_occupancies[patch1.get_id()] = patch_occupancies[patch2.get_id()] = True
        #
        #             # for the positions of each of the 2 patches that are about to bind to each other
        #             for patch_idx1, patch_idx2 in patch_idxs_to_bind(q,
        #                                                              z,
        #                                                              p1_dna,
        #                                                              p2_dna):
        #                 self.bind_patches_3p(p1_dna,
        #                                      q,
        #                                      patch_idx1,
        #                                      patch1,
        #                                      p2_dna,
        #                                      z,
        #                                      patch_idx2,
        #                                      patch2)
        #
        #             self.bondcount += 1
        #         else:
        #             print("Patch occupied!")
        # print(f"Created {self.bondcount} dna patch bonds")

    def bind_patches_3p(self,
                        particle1: DNAParticle,
                        patch1: PLPatch,
                        particle2: DNAParticle,
                        patch2: PLPatch):
        """
        Binds two patches together by adding sticky ends at the 3' ends.
        Parameters:
            particle1_dna (DNAParticle): DNAParticle object representing particle a
            dna_patch1_id (int): the index in DNAParticleType::patch_strand_ids of the patch to bind
            dna_patch1_strand (int): the index of the strand in the patch strands (NOT particle strands!)
            patch1 (MGLPatch): mgl patch object?
            particle2_dna (DNAParticle): DNAParticle object representing particle b
            dna_patch2_id (int): the index in DNAParticleType::patch_strand_ids of the patch to bind
            dna_patch2_strand (int): the index of the strand in the patch strands (NOT particle strands!)
            patch2 (MGLPatch): mgl patch object?
        """

        # find nucleotides which will be extended to form patches

        start_position1 = particle1.patch_3p(patch1).pos
        start_position2 = particle2.patch_3p(patch2).pos

        start_vector_1 = normalize(start_position2 - start_position1)

        patch_distance = dist(start_position1, start_position2)

        # check distances
        if patch_distance > 2 * self.get_expected_pad_distance():
            print(f"Distance between patches = {patch_distance}. "
                  f"Expected distance {self.get_expected_pad_distance()}")

        # retrieve sequences from map + add spacers
        patch1_seq = self.spacer_length * "T" + self.color_sequence(patch1.color())
        patch2_seq = self.spacer_length * "T" + self.color_sequence(patch2.color())

        strand1, strand2 = construct_strands(patch1_seq + self.spacer_length * "T",  # need to add fake spacer to make code no go boom
                                             start_position1 + start_vector_1 * BASE_BASE,
                                             start_vector_1,
                                             rbases=patch2_seq)
        strand1 = strand1[:len(patch1_seq)]  # shave off dummy spacer from previous line of code

        particle1.patch_strand(patch1).prepend(strand1[::-1])
        particle2.patch_strand(patch2).prepend(strand2[::-1])


    # def bind_patches_3p(self,
    #                     particle1_dna: DNAParticle,
    #                     dna_patch1_id: int,
    #                     dna_patch1_strand: int,
    #                     patch1: MGLPatch,
    #                     particle2_dna: DNAParticle,
    #                     dna_patch2_id: int,
    #                     dna_patch2_strand: int,
    #                     patch2: MGLPatch):
    #     """
    #     Binds two patches together by adding sticky ends at the 3' ends.
    #     Parameters:
    #         particle1_dna (DNAParticle): DNAParticle object representing particle a
    #         dna_patch1_id (int): the index in DNAParticleType::patch_strand_ids of the patch to bind
    #         dna_patch1_strand (int): the index of the strand in the patch strands (NOT particle strands!)
    #         patch1 (MGLPatch): mgl patch object?
    #         particle2_dna (DNAParticle): DNAParticle object representing particle b
    #         dna_patch2_id (int): the index in DNAParticleType::patch_strand_ids of the patch to bind
    #         dna_patch2_strand (int): the index of the strand in the patch strands (NOT particle strands!)
    #         patch2 (MGLPatch): mgl patch object?
    #     """
    #     # find nucleotides which will be extended to form patches
    #     patch1_nucleotide: DNABase = particle1_dna.patch_strand_3end(dna_patch1_id, dna_patch1_strand)
    #     patch2_nucleotide: DNABase = particle2_dna.patch_strand_3end(dna_patch2_id, dna_patch2_strand)
    #
    #     start_position1 = patch1_nucleotide.pos
    #     start_position2 = patch2_nucleotide.pos
    #
    #     start_vector_1 = normalize(start_position2 - start_position1)
    #
    #     patch_distance = dist(start_position1, start_position2)
    #
    #     # check distances
    #     if patch_distance > 2 * self.get_expected_pad_distance():
    #         print(f"Distance between patches = {patch_distance}. "
    #               f"Expected distance {self.get_expected_pad_distance()}")
    #
    #     # retrieve sequences from map + add spacers
    #     patch1_seq = self.spacer_length * "T" + self.color_sequence(patch1.color())
    #     patch2_seq = self.spacer_length * "T" + self.color_sequence(patch2.color())
    #
    #     strand1, strand2 = construct_strands(patch1_seq + self.spacer_length * "T",  # need to add fake spacer to make code no go boom
    #                                          start_position1 + start_vector_1 * BASE_BASE,
    #                                          start_vector_1,
    #                                          rbases=patch2_seq)
    #     strand1 = strand1[:len(patch1_seq)]  # shave off dummy spacer from previous line of code
    #
    #
    #     particle1_dna.patch_strand(dna_patch1_id, dna_patch1_strand).prepend(strand1[::-1])
    #     particle2_dna.patch_strand(dna_patch2_id, dna_patch2_strand).prepend(strand2[::-1])

    def get_particles(self) -> list[DNAParticle]:
        """
        Returns DNA structures that correspond to particles
        """
        if self.dna_particles is None:
            print("positioning particles")
            self.position_particles()
        return list(self.dna_particles.values())

    def get_dna_particle(self, pl: Union[int, PLPatchyParticle]) -> DNAParticle:
        """
        Gets the DNA particle instace (not type) for a PL particle instance
        """
        # position particles if missing
        self.get_particles()
        if isinstance(pl, PLPatchyParticle):
            return self.get_dna_particle(pl.get_id())
        else:
            assert isinstance(pl, int)
            assert pl in self.dna_particles
            return self.dna_particles[pl]

    def construct_types(self):
        """
        Constructs "type particles" for each DNA particle.
        Important so we can keep same color strands the same on types
        """
        pass

    def convert(self):
        """
        Converts a scene containing joined MGL particles to an oxDNA model consisting of
        DNA origamis joined by sticky end handles.
        """

        # sanitize inputs
        particles = self.get_particles()

        print("binding particles using 3p patches")
        self.bind_particles3p(particles)
        assert self.expected_num_edges == -1 or self.bondcount == self.expected_num_edges, \
            "Wrong number of bonds created!"

        print("Done!")

    def save_top_dat(self, write_top_path: Union[Path, str], write_conf_path: Union[Path, str]):
        if isinstance(write_top_path, str):
            write_top_path = Path(write_top_path)
        if not write_top_path.is_absolute():
            write_top_path = get_output_dir() / write_top_path

        if isinstance(write_conf_path, str):
            write_conf_path = Path(write_conf_path)
        if not write_conf_path.is_absolute():
            write_conf_path = get_output_dir() / write_conf_path

        assert write_top_path.parent.exists()
        assert write_conf_path.parent.exists()
        if write_top_path.parent != write_conf_path.parent:
            print("You're technically allowed to do this but I do wonder why")

        merged_conf = self.get_full_conf()

        merged_conf.export_top_conf(write_top_path, write_conf_path)
        print(f"Wrote topopogy file to {str(write_top_path)}")

        print(f"Wrote conf file to {str(write_conf_path)}")
    def save_oxview(self, write_oxview_path: Union[Path, str]):
        if isinstance(write_oxview_path, str):
            write_oxview_path = Path(write_oxview_path)
        if not write_oxview_path.is_absolute():
            write_oxview_path = get_output_dir() / write_oxview_path

        merged_conf = self.get_full_conf()
        merged_conf.export_oxview(write_oxview_path)
        print(f"Wrote OxView file to {str(write_oxview_path)}")