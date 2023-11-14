import math
from copy import deepcopy
from pathlib import Path
from typing import Generator, Iterable, Union

import numpy as np
from oxDNA_analysis_tools.UTILS.RyeReader import Configuration
from Bio.SVDSuperimposer import SVDSuperimposer

from .dna_particle import DNAParticle
from ..dna_structure import DNABase, construct_strands, BASE_BASE, rc, POS_BASE

from random import choice
import itertools

from ..patchy.mglparticle import MGLParticle, MGLPatch
from ..patchy_base_particle import Scene, PatchyBaseParticle

dist = lambda a, b: np.linalg.norm(a - b)
normalize = lambda v: v / np.linalg.norm(v)




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
    clusters: list[dict]
    color_sequences: dict[str, str]
    bondcount: int
    particle_type_map: dict[str, DNAParticle]
    mgl_scene: Scene

    def __init__(self,
                 mgl: Scene,
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
            mgl: an MGLScene object representing the structure we're making with the particles
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

        # inputs
        self.mgl_scene = mgl

        # if only one particle type
        if isinstance(particle_types, DNAParticle):
            self.particle_type_map = {
                particle.color(): particle_types for particle in self.mgl_scene.particle_types().particles()
            }

        else:  # forward-proof for heterogenous systems
            assert isinstance(particle_types, dict)
            self.particle_type_map = particle_types

        # compute padding automatically
        if padding == -1:
            mgl_pad = self.mgl_scene.avg_pad_bind_distance()

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

    def assign_color_sequence(self, colorstr: str, seq: str):
        """
        Assigns the color given by colorstr the specific sequence sequence specified
        Automatically assigns the corresponding color the reverse compliment
        """
        assert len(seq) == self.sticky_length, "Incompatible sequence length"
        self.color_sequences[colorstr] = seq
        self.color_sequences[self.get_color_match(colorstr)] = rc(seq)

    def get_color_match(self, colorstr: str) -> str:
        if colorstr in self.color_match_overrides:
            return self.color_match_overrides[colorstr]
        else:
            if colorstr.startswith("dark"):
                return colorstr[4:]
            else:
                return f"dark{colorstr}"

    def get_expected_pad_distance(self) -> float:
        return (self.sticky_length + 2.0 * self.spacer_length) * POS_BASE

    def patches_can_bind(self,
                         patch1: MGLPatch,
                         patch2: MGLPatch) -> bool:
        return self.get_color_match(patch1.color()) == patch2.color()

    def set_color_match(self,
                        colorstr: str,
                        match: str,
                        reverse_match: bool = True):
        self.color_match_overrides[colorstr] = match
        if reverse_match:
            self.color_match_overrides[match] = colorstr

    def position_particles(self) -> list[DNAParticle]:
        """
        Positions particles?
        IDK
        """
        # get scene particles
        particles = self.mgl_scene.particles()
        placed_confs = []  # output prom_p
        # everyone's most favorite aligning tool
        sup = SVDSuperimposer()
        pl = len(particles)
        for i, particle in enumerate(particles):
            # clone dna particle
            origami: DNAParticle = deepcopy(self.get_dna_origami(particle))
            print(f"{i + 1}/{pl}", end="\r")
            origami.link_patchy_particle(particle)

            scale_factor = origami.scale_factor(particle) / self.padding
            # magic numbers needed again for things not to clash
            # origami.box = self.mgl_scene.box_size() / scale_factor
            # scale factor tells us how to convert MGL distance units into our DNA model distances
            origami.transform(tran=particle.position() / scale_factor)

            # we finished the positioning
            placed_confs.append(origami)
        print()
        return placed_confs

    def particle_pair_candidates(self) -> Generator[tuple[MGLParticle], None, None]:
        """
        Returns all possible pairs of particles,
        as defined by interaction range between centers of mass
        """
        handeled_candidates = set()
        for i, p1 in enumerate(self.mgl_scene.particles()):
            for j, p2 in enumerate(self.mgl_scene.particles()):
                # if the particles are different and the distance is less than the maximum interaction distance
                if i != j and dist(p1.cms(),
                                   p2.cms()) <= self.particle_delta:
                    if (i, j) not in handeled_candidates and not (j, i) in handeled_candidates:  # prevent repeats
                        handeled_candidates.add((i, j))
                        assert (p1.type_id(), p2.type_id()) == (i, j)
                        yield p1, p2

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

    def patches_to_bind(self,
                        particle_1: MGLParticle,
                        particle_2: MGLParticle) -> Generator[tuple[tuple[int, MGLPatch],
                                                                    tuple[int, MGLPatch]],
                                                              None,
                                                              None]:
        """
        Returns:
            a generator which produces pairs of tuples, each of which consists of patch index and patch object
        """
        assert particle_1.type_id() != particle_2.type_id()
        # keep in mind: we can't use patch internal IDs here because it enumerates differently!
        # find all possible pairings between two patches on particle 1 and particle 2
        possible_bindings = list(itertools.product(enumerate(particle_1.patches()), enumerate(particle_2.patches())))
        # filter patches that don't pair
        possible_bindings = [(patch1, patch2) for patch1, patch2 in possible_bindings
                             if self.patches_can_bind(patch1[1], patch2[1])]

        # sort by distance, ascending order

        def sort_by_distance(p):
            patch1, patch2 = p
            return dist(particle_1.cms() + patch1[1].position(),
                        particle_2.cms() + patch2[1].position())

        possible_bindings.sort(key=sort_by_distance)
        # lists for patches that have been handled on particles 1 and 2
        handled_p1 = [False for _ in particle_1.patches()]
        handled_p2 = [False for _ in particle_2.patches()]
        # iterate through possible pairs of patches
        for (q, patch_1), (z, patch_2) in possible_bindings:
            # skip patches we've already handled
            if not handled_p1[q] and not handled_p2[z]:
                # if the two patches are within bonding distance (mgl units)
                patch1_position = particle_1.cms() + patch_1.position()
                patch2_position = particle_2.cms() + patch_1.position()
                # compute patch distance
                patches_distance = dist(patch1_position, patch2_position)
                if (patches_distance <= self.bond_length) or \
                        (self.flexable_patch_distances and (self.bondcount < self.expected_num_edges)):
                    # https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
                    # normalize patches
                    patch1norm = patch_1.position() / np.linalg.norm(patch_1.position())
                    patch2norm = patch_2.position() / np.linalg.norm(patch_2.position())
                    costheta = abs(float(np.clip(np.dot(patch1norm,
                                                        patch2norm), -1.0, 1.0)))
                    # confusing,ly, cos theta-max is the cosine of the maximum angle
                    # so we check if cos theta is LESS than cos theta-max
                    if costheta >= self.cos_theta_max:
                        handled_p1[q] = handled_p2[z] = True
                        yield (q, patch_1), (z, patch_2)
                else:  # can do this because list is sorted
                    break

    def bind_particles3p(self,
                         dna_particles: list[DNAParticle]):
        """
        Creates double-stranded linkers to bind particles together
        """
        assert self.mgl_scene.patch_ids_unique()
        # please please do not call this method with dna particles that don't correspond
        # in order to the mgl particles

        patch_occupancies = [False for _ in itertools.chain.from_iterable(
            [[patch.get_id() for patch in particle.patches()] for particle in self.mgl_scene.particles()])]

        # loop possible pairs of particles
        # patch_occupancies = [[False for _ in p.patches()] for p in particles]

        patchpaircount = 0
        # loop particles
        for p1, p2 in self.particle_pair_candidates():
            # i and j are particle unique ids
            # actually UIDs not type IDs
            i = p1.type_id()
            j = p2.type_id()
            assert i != j, "Particle should not bind to self!"
            # grab DNAParticle objects
            p1_dna: DNAParticle = dna_particles[i]
            p2_dna: DNAParticle = dna_particles[j]
            # loop through the patch pairs on each particle that can bind
            for (q, patch1), (z, patch2) in self.patches_to_bind(p1, p2):
                assert -1 < patch1.get_id() < len(patch_occupancies), "Mismatch between patch ID on structure and scene patches!"
                assert -1 < patch2.get_id() < len(patch_occupancies), "Mismatch between patch ID on structure and scene patches!"
                assert patch1.get_id() != patch2.get_id()
                # if either patch is bound, stop!
                if not patch_occupancies[patch1.get_id()] and not patch_occupancies[patch2.get_id()]:
                    patch_occupancies[patch1.get_id()] = patch_occupancies[patch2.get_id()] = True

                    # for the positions of each of the 2 patches that are about to bind to each other
                    for patch_idx1, patch_idx2 in patch_idxs_to_bind(q,
                                                                     z,
                                                                     p1_dna,
                                                                     p2_dna):
                        self.bind_patches_3p(p1_dna,
                                             q,
                                             patch_idx1,
                                             patch1,
                                             p2_dna,
                                             z,
                                             patch_idx2,
                                             patch2)

                    self.bondcount += 1
                else:
                    print("Patch occupied!")
        print(f"Created {self.bondcount} dna patch bonds")

    def bind_patches_3p(self,
                        particle1_dna: DNAParticle,
                        dna_patch1_id: int,
                        dna_patch1_strand: int,
                        patch1: MGLPatch,
                        particle2_dna: DNAParticle,
                        dna_patch2_id: int,
                        dna_patch2_strand: int,
                        patch2: MGLPatch):
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
        patch1_nucleotide: DNABase = particle1_dna.patch_strand_3end(dna_patch1_id, dna_patch1_strand)
        patch2_nucleotide: DNABase = particle2_dna.patch_strand_3end(dna_patch2_id, dna_patch2_strand)

        start_position1 = patch1_nucleotide.pos
        start_position2 = patch2_nucleotide.pos

        start_vector_1 = normalize(start_position2 - start_position1)

        patch_distance = dist(start_position1, start_position2)

        # check distances
        if patch_distance > 2 * self.get_expected_pad_distance():
            print(f"Distance between patches = {patch_distance}. "
                  f"Expected distance {self.get_expected_pad_distance()}")

        # retrieve sequences from map + add spacers
        patch1_seq = self.spacer_length * "T" + self.color_sequence(patch1.color())
        patch2_seq = self.spacer_length * "T" + self.color_sequence(patch2.color())

        strand1, strand2 = construct_strands(patch1_seq + self.spacer_length * "T", # need to add fake spacer to make code no go boom
                                             start_position1 + start_vector_1 * BASE_BASE,
                                             start_vector_1,
                                             rbases=patch2_seq)
        strand1 = strand1[:len(patch1_seq)] # shave off dummy spacer from previous line of code

        particle1_dna.patch_strand(dna_patch1_id, dna_patch1_strand).prepend(strand1)
        particle2_dna.patch_strand(dna_patch2_id, dna_patch2_strand).prepend(strand2)

    def convert(self,
                write_top_path: Union[Path, None] = None,
                write_conf_path: Union[Path, None] = None,
                write_oxview_path: Union[Path, None] = None):
        """
        Converts a scene containing joined MGL particles to an oxDNA model consisting of
        DNA origamis joined by sticky end handles.
        """

        print("positioning particles")
        particles = self.position_particles()

        print("binding particles using 3p patches")
        self.bind_particles3p(particles)
        assert self.expected_num_edges == -1 or self.bondcount == self.expected_num_edges, "Wrong number of bonds created!"

        print("merging the topologies")

        merged_conf = sum(particles[1:], start=particles[0])
        # spit out the topology
        if write_top_path:
            assert write_conf_path
            assert write_top_path.parent.exists()
            assert write_conf_path.parent.exists()
            if write_top_path.parent != write_conf_path.parent:
                print("You're technically allowed to do this but I do wonder why")
            merged_conf.export_top_conf(write_top_path, write_conf_path)
            print(f"Wrote topopogy file to {str(write_top_path)}")

            print(f"Wrote conf file to {str(write_conf_path)}")

        if write_oxview_path:
            merged_conf.export_oxview(write_oxview_path)
            print(f"Wrote OxView file to {str(write_oxview_path)}")
