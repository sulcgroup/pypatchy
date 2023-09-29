import math
from copy import deepcopy
from pathlib import Path
from typing import Generator, Any, Iterable, Union

import numpy as np
from oxDNA_analysis_tools.UTILS.RyeReader import Configuration, linear_read, get_traj_info
from Bio.SVDSuperimposer import SVDSuperimposer
from pypatchy import oxutil

from pypatchy.oxutil import generate_helix_coords, write_configuration_header, write_configuration, write_oxview, \
    POS_BASE, generate_spacer, assign_coords
from random import choice
import itertools

from pypatchy.patchy.mglparticle import MGLScene, MGLParticle, MGLPatch
from pypatchy.patchy_base_particle import Scene

dist = lambda a, b: np.linalg.norm(a - b)
normalize = lambda v: v / np.linalg.norm(v)


def rc(s):
    mp = {"A": "T", "T": "A", "C": "G", "G": "C"}
    return "".join([mp[c] for c in s][::-1])


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

class DNAParticleType:
    """
    A DNA origami that will be used to make a particle in the output particle thing
    i'm very tired rn
    """
    topology: oxutil.TopInfo
    conf: Configuration

    # list containing the id of the last base in each strand
    strand_delims: list[int]

    # list of list of base ids that are tips from which we can extend single strandedd overhangs
    # not updated as top is modified!
    patch_positions: list[list[int]]

    def __init__(self, top_file: Path, dat_file: Path, patch_positions: list[list[int]]):
        """
        Constructs an object to facilitate conversion of patchy particle to DNA origami
        in a multicomponent structure
        """
        # import the topology
        self.topology, base_2_strand = oxutil.read_top(top_file)
        self.strand_delims = []
        # compute strand deliminators
        for (i1, s1), (i2, s2) in pairwise(base_2_strand.items()):
            # if i1 is last element in strand1
            if s1 != s2:
                # put it on the list
                self.strand_delims.append(i1)

        # now import  the origami
        self.conf = next(linear_read(get_traj_info(str(dat_file)), self.topology))[0]

        self.patch_positions = patch_positions

    def get_patch_cmss(self) -> np.ndarray:
        """
        get the centers of mass of the patches
        """
        patch_cmss = []
        for patch in self.patch_positions:
            cms = np.zeros(3)
            for i in patch:
                cms += self.conf.positions[i]
            patch_cmss.append(
                cms / len(patch)
            )
        a = np.array(patch_cmss)
        assert a.shape == (len(self.patch_positions), 3)
        return a

    def conf_cms(self):
        """
        Computes the average of the positions of the nucleotides in this conf
        """
        return np.sum(self.conf.positions, 0) / len(self.conf.positions)

    def center2patch_conf(self):
        """
        Computes the average distance between the center of the structure and the
        positions of the patches
        """
        return np.mean(
            np.linalg.norm((self.get_patch_cmss() - self.conf_cms()), axis=1)
        )

    def modify_strand3p(self, strand_id: int, seq: str) -> Iterable[int]:
        """
        Modifies a topology to add a sequence to the 3' end
        Args:
            strand_id : the ID of the strand to modify
            seq : the seqence to append, in 5'->3' order

        Returns:
            the ids of the bases that have been added
        """
        # grab seq length
        connection_length = len(seq)
        strand_old_len = len(self.topology.strands[strand_id].bases)
        assert connection_length < strand_old_len, "No I will not explain."
        # # we know we work with the 3 end so it's the beginning + sticky_length we have to modify
        # # construct list of bases with new identities at beginning of topology (b/c oxDNA thinks in 3'->5' for some reason?)
        #
        # new_bases = [oxutil.Base(t, b.p3, b.p5) for t, b in zip(seq, self.topology.strands[strand_id].bases[:connection_length])]
        # # "shift" topology by appending the existing strand to our new bases
        # new_bases.extend([oxutil.Base(
        #     t,
        #     p3 + connection_length if p3 != -1 else new_bases[-1].p3, # link end of old strand to new strand
        #     p5 + connection_length if p5 != -1 else -1)
        #                   for t, p3, p5 in self.topology.strands[strand_id].bases])
        # # empty previous bases as we work with tuples and we can't just reasign
        # # so clear strand bases
        # self.topology.strands[strand_id].bases.clear()
        # # and add our new copy strand (with our new sequence at the (3') beginning)
        # self.topology.strands[strand_id].bases.extend(new_bases)
        # assert len(self.topology.strands[strand_id].bases) == strand_old_len + connection_length

        # get id of residue at strand head
        start_id = self.topology.strands[strand_id].bases[1].p3
        # make new strand
        strand = oxutil.make_strand(strand_id, start_id, seq + oxutil.get_seq(self.topology.strands[strand_id]))
        # assign to topology
        self.topology.strands[strand_id] = strand
        assert len(self.topology.strands[strand_id].bases) == strand_old_len + connection_length
        # update num bases
        self.topology.nbases += connection_length
        # update strand delims
        for i in range(strand_id, len(self.strand_delims)):
            self.strand_delims[i] += connection_length
            # skip the strand we are working on for shifting, because we have already done it
            if i > strand_id:
                # shift strand to the right
                self.topology.strands[i] = self.topology.strands[i] >> connection_length

        # update patch positions
        self.patch_positions = [[
            baseid if baseid < start_id else baseid + connection_length for baseid in patch]
            for patch in self.patch_positions
        ]

        # update conf
        # a1s
        self.conf.a1s = np.resize(self.conf.a1s, (self.topology.nbases, 3)) # resize array
        self.conf.a1s[start_id + connection_length:, :] = self.conf.a1s[start_id:-connection_length, :] # move data
        self.conf.a1s[start_id: start_id+connection_length, :] = np.nan # wipe cells

        # a3s
        self.conf.a3s = np.resize(self.conf.a3s, (self.topology.nbases, 3)) # resize array
        self.conf.a3s[start_id + connection_length:, :] = self.conf.a3s[start_id:-connection_length, :] # move data
        self.conf.a3s[start_id: start_id+connection_length, :] = np.nan # wipe cells

        # positions
        self.conf.positions = np.resize(self.conf.positions, (self.topology.nbases, 3)) # resize array
        self.conf.positions[start_id + connection_length:, :] = self.conf.positions[start_id:-connection_length, :] # move data
        self.conf.positions[start_id: start_id+connection_length, :] = np.nan # wipe cells

        return range(start_id, start_id + connection_length)


    def generate_3p_ids(self, sid, num_ids):
        strand_head_id = self.topology.strands[sid].bases[0].p5 - 1
        return range(strand_head_id, strand_head_id + num_ids)

    def scale_factor(self, p: MGLParticle) -> float:
        """
        Computes the particle type's scale factor
        The scale factor converts between units in MGL space and units in nucleotide-level-model
        space, and is thus very important for placing origamis relative to each other
        """
        # distance between origami patch and center
        center2patch_conf = self.center2patch_conf()
        # distance between an arbitrary mgl patch and center - approximates particle radius
        # TODO: why not just use the actual radius?
        center2patch = np.round(np.linalg.norm(p.patch(0).position()), 1)
        # compute the scale factor by dividing the distance between the patches on the mgl
        # by the distance between the patches on the origami, and applying a handy scale factor
        # hashtag ratio
        return center2patch / center2patch_conf

    def base2strand(self, bid: int) -> int:
        # loop start ids
        for i, end_id in enumerate(self.strand_delims):
            # if start id is less than the last base in this strand
            if bid <= end_id:
                return i
        return len(self.strand_delims)


def patch_idxs_to_bind(patch_id_1: int,
                       patch_id_2: int,
                       dna1: DNAParticleType,
                       dna2: DNAParticleType) -> list[tuple[int, int]]:
    patch_positions1 = dna1.patch_positions[patch_id_1]
    patch_positions2 = dna2.patch_positions[patch_id_2]
    conf1 = dna1.conf
    conf2 = dna2.conf
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
    best_patch1_idxs: list[int] = list(itertools.permutations(range(len(patch_positions1))))[bestmatch]
    return zip(
        best_patch1_idxs,
        range(len(patch_positions2)))

class MGLOrigamiConverter:
    """
    This class facilitates the conversion of a patchy particle model (in MGL format) to
    a set of connected DNA origamis
    """
    clusters: list[dict]
    color_sequences: dict[str, str]
    bondcount: int
    particle_structure: dict[str, DNAParticleType]
    mgl_scene: Scene

    def __init__(self,
                 mgl: Scene,
                 particle_types: Union[Path, DNAParticleType,
                                       dict[str, DNAParticleType]],
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
        self.clusters = [{} for _ in range(len(mgl._particles()))]
        self.bondcount = 0
        self.padding = padding

        # inputs
        self.mgl_scene = mgl

        # if only one particle type
        if isinstance(particle_types, DNAParticleType):
            self.particle_structure = {
                particle.color(): particle_types for particle in self.mgl_scene.particle_set().particles()
            }

        else:  # forward-proof for heterogenous systems
            assert isinstance(particle_types, dict)
            self.particle_structure = particle_types

        # compute padding automatically
        if padding == -1:
            mgl_pad = self.mgl_scene.avg_pad_bind_distance()

        # optional parameters
        self.flexable_patch_distances = flexable_patch_distances
        self.expected_num_edges = expected_num_edges
        assert not flexable_patch_distances or expected_num_edges > 0

    def get_dna_origami(self, particle_type: Union[str, MGLParticle]) -> DNAParticleType:
        if isinstance(particle_type, MGLParticle):
            return self.get_dna_origami(particle_type.color())
        else:
            return self.particle_structure[particle_type]

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

    def position_particles(self) -> list[DNAParticleType]:
        """
        Positions particles?
        IDK
        """
        # get scene particles
        particles = self.mgl_scene._particles()
        placed_confs = []  # output prom_p
        # everyone's most favorite aligning tool
        sup = SVDSuperimposer()
        pl = len(particles)
        for i, particle in enumerate(particles):
            # clone dna particle
            origami: DNAParticleType = deepcopy(self.get_dna_origami(particle))
            print(f"{i + 1}/{pl}", end="\r")
            mgl_patches = []  # start with centerpoint of the particle (assumed to be 0,0,0)
            # load each patch on the mgl particle
            for patch in particle.patches():
                # scale patch with origami
                # skip magic padding when scaling patch local coords
                mgl_patches.append(patch.position() / origami.scale_factor(particle))
            assert len(mgl_patches) == len(origami.patch_positions)
            mgl_patches.append(np.zeros((3,)))
            # generate matrix of origami patches + centerpoint
            m2 = np.concatenate([origami.get_patch_cmss(), origami.conf_cms()[np.newaxis, :]], axis=0)
            best_rms = np.Inf
            # test different patch arrangements in mgl vs. origami. use best option.
            best_order = None
            for perm in itertools.permutations(enumerate(mgl_patches)):
                new_order, m1 = np.array(perm).T
                if new_order[len(origami.patch_positions)] != len(
                        origami.patch_positions):  # definately not optimized lmao
                    continue
                sup.set(np.stack(m1), m2)
                sup.run()
                if sup.get_rms() < best_rms:
                    best_order = new_order
                    best_rms = sup.get_rms()
                    rot, tran = sup.get_rotran()

            # iter through patches in best_order, skipping zeroth element (centroid)
            for j, pidx in enumerate(best_order[:-1]):
                residue_ids: list[int] = origami.patch_positions[j]
                newval = particle.patch(pidx).add_key_point(residue_ids)
                assert newval == 1

            # origami.patch_positions = [origami.patch_positions[x] for x in best_order if x != len(origami.patch_positions)]
            # sup.set(m1, m2)
            # sup.run()
            # rot, trn = sup.get_rotran()
            # best_rms = sup.get_rms()
            print(f"Superimposed DNA origami on patchy particle with RMS={best_rms}")
            print(f"RMS / circumfrance: {best_rms / (2 * math.pi * origami.center2patch_conf())}")

            scale_factor = origami.scale_factor(particle) / self.padding
            # magic numbers needed again for things not to clash
            origami.conf.box = self.mgl_scene.box_size() / scale_factor
            # scale factor tells us how to convert MGL distance units into our DNA model distances
            origami.conf.positions = np.einsum('ij, ki -> kj',
                                               rot,
                                               origami.conf.positions) + (particle.position() / scale_factor)
            origami.conf.a1s = np.einsum('ij, ki -> kj', rot, origami.conf.a1s)
            origami.conf.a3s = np.einsum('ij, ki -> kj', rot, origami.conf.a3s)

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
        for i, p1 in enumerate(self.mgl_scene._particles()):
            for j, p2 in enumerate(self.mgl_scene._particles()):
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
                         dna_particles: list[DNAParticleType]):
        """
        Creates double-stranded linkers to bind particles together
        """
        assert self.mgl_scene.patch_ids_unique()
        # please please do not call this method with dna particles that don't correspond
        # in order to the mgl particles

        patch_occupancies = [False for _ in itertools.chain.from_iterable(
            [[patch.get_id() for patch in particle.patches()] for particle in self.mgl_scene._particles()])]

        # loop possible pairs of particles
        # patch_occupancies = [[False for _ in p.patches()] for p in particles]
        # there's probably an oxpy method to convert bp to oxdna distance units

        patchpaircount = 0
        # loop particles
        for p1, p2 in self.particle_pair_candidates():
            # actually UIDs not type IDs
            i = p1.type_id()
            j = p2.type_id()
            assert i != j
            p1_dna: DNAParticleType = dna_particles[i]
            p2_dna: DNAParticleType = dna_particles[j]
            # loop through the patch pairs on each particle that can bind
            for (q, patch1), (z, patch2) in self.patches_to_bind(p1, p2):
                assert -1 < patch1.get_id() < len(patch_occupancies)
                assert -1 < patch2.get_id() < len(patch_occupancies)
                assert patch1.get_id() != patch2.get_id()
                # if either patch is bound, stop!
                if not patch_occupancies[patch1.get_id()] and not patch_occupancies[patch2.get_id()]:
                    patch_occupancies[patch1.get_id()] = patch_occupancies[patch2.get_id()] = True

                    conf1, conf2 = p1_dna.conf, p2_dna.conf

                    # I've saved the residue indices in the mgl patches as key points, for some reason
                    # not limited to 3 elements
                    p1_patch_idxs = patch1.get_key_point(1)
                    p2_patch_idxs = patch2.get_key_point(1)
                    # for the positions of each of the 2 patches that are about to bind to each other
                    for patch_idx1, patch_idx2 in patch_idxs_to_bind(q,
                                                                        z,
                                                                        p1_dna, p2_dna):
                        patch_id1 = p1_dna.patch_positions[q][patch_idx1]
                        patch_id2 = p2_dna.patch_positions[z][patch_idx2]
                        self.bind_patches_3p(p1_dna,
                                             p2_dna,
                                             patch_id1,
                                             patch_id2,
                                             i,
                                             j,
                                             patch1,
                                             patch2,
                                             dna_particles)
                    self.bondcount += 1
                else:
                    print("Patch occupied!")
        print(f"Created {self.bondcount} dna patch bonds")

    def bind_patches_3p(self,
                        p1_dna: DNAParticleType,
                        p2_dna: DNAParticleType,
                        patch_id1: int,
                        patch_id2: int,
                        i: int,
                        j: int,
                        patch1: MGLPatch,
                        patch2: MGLPatch,
                        particles: list
                        ):
        conf1, conf2 = p1_dna.conf, p2_dna.conf
        top1, top2 = p1_dna.topology, p2_dna.topology

        # make sure both patch nucleotides are at the ends of their respective strands
        # TODO: make sure this is correct
        assert patch_id1 == 0 or p1_dna.base2strand(patch_id1) != p1_dna.base2strand(patch_id1 - 1)
        assert patch_id2 == 0 or p2_dna.base2strand(patch_id2) != p2_dna.base2strand(patch_id2 - 1)
        # get positions of our patch strand origins
        start_position1 = conf1.positions[patch_id1]
        start_position2 = conf2.positions[patch_id2]
        # check distances
        if dist(start_position1, start_position2) > 2 * self.get_expected_pad_distance():
            print(f"Distance between patches = {dist(start_position1, start_position2)}. "
                  f"Expected distance {self.get_expected_pad_distance()}")

        # generate conf (residue positions) for sticky-end double helix
        # trust me, do this first
        midpoint = (start_position1 + start_position2) / 2
        coords1, coords2 = generate_helix_coords(self.sticky_length,
                                                 start_pos=midpoint,
                                                 helix_direction=start_position2 - start_position1)

        # lets modify the topology somehow
        # 1st figure out the strand index
        sid1 = p1_dna.base2strand(patch_id1)
        sid2 = p2_dna.base2strand(patch_id2)

        # first_midpoint = (start_position1 + start_position2 - self.spacer_length * POS_BASE) / 3
        # second_midpoint = first_midpoint * 2
        # construct spacer sequence and coords
        # scale the spacers to be evenly spaced between the origin point and
        # the sticky-end helix (hopefully this will make relax nicer?)
        p1_ids = p1_dna.generate_3p_ids(sid1, self.spacer_length + self.sticky_length)
        p2_ids = p2_dna.generate_3p_ids(sid2, self.spacer_length + self.sticky_length)
        if self.spacer_length > 0:
            t_coords1 = generate_spacer(self.spacer_length,
                                        start_position=start_position1,
                                        end_position=coords1[0][0],
                                        stretch=False)
            t_coords2 = generate_spacer(self.spacer_length,
                                        start_position=start_position2,
                                        end_position=coords2[0][0],
                                        stretch=False)
            # spacer_ids1 = generate_3p_ids(patch_id1, self.spacer_length)
            # spacer_ids2 = generate_3p_ids(patch_id2, self.spacer_length)

            # first modify strand to add spacer
            # modify_strand3p(top1, sid1, "T" * self.spacer_length)
            # modify_strand3p(top2, sid2, "T" * self.spacer_length)

        # retrieve sequences from map
        sticky_seq1 = self.color_sequence(patch1.color())
        sticky_seq2 = self.color_sequence(patch2.color())

        # then modify strand to add sticky ends
        p1_dna.modify_strand3p(sid1, sticky_seq1 + "T" * self.spacer_length)
        p2_dna.modify_strand3p(sid2, sticky_seq2 + "T" * self.spacer_length)

        # generate 3prime ids for sticky end sequences
        # sticky_id1s = generate_3p_ids(patch_id1 + self.spacer_length, self.sticky_length)
        # sticky_id2s = generate_3p_ids(patch_id2 + self.spacer_length, self.sticky_length)

        # update conf for spacer positions

        # single-stranded segments are prepended to top, so first section is sticky, seccond is spacer
        if self.spacer_length > 0:
            assign_coords(conf1, p1_ids[:self.spacer_length], t_coords1)
            assign_coords(conf2, p2_ids[:self.spacer_length], t_coords2)
        # make sure overhangs comply with the helix
        assign_coords(conf1, p1_ids[self.spacer_length:], coords1)
        assign_coords(conf2, p2_ids[self.spacer_length:], coords2)

        # add bond helix nucleotides to cluster lists
        for x in p1_ids[self.spacer_length:]:
            self.clusters[i][x] = self.bondcount + len(particles)
        for x in p2_ids[self.spacer_length:]:
            self.clusters[j][x] = self.bondcount + len(particles)

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
        merged_tops = oxutil.merge_tops([p.topology for p in particles])
        # spit out the topology
        if write_top_path:
            assert write_conf_path
            assert write_top_path.parent.exists()
            assert write_conf_path.parent.exists()
            if write_top_path.parent != write_conf_path.parent:
                print("You're technically allowed to do this but I do wonder why")
            oxutil.write_top(
                merged_tops, str(write_top_path)
            )
            print(f"Wrote topopogy file to {str(write_top_path)}")

            print("printing confs")
            with write_conf_path.open("w") as file:
                write_configuration_header(file, particles[0].conf)
                cl = len(particles)
                for i, p in enumerate(particles):
                    print(f"{i + 1}/{cl}", end="\r")
                    write_configuration(file, p.conf)
            print(f"Wrote conf file to {str(write_top_path)}")

        if write_oxview_path:
            write_oxview([p.topology for p in particles],
                         [p.conf for p in particles], self.clusters, write_oxview_path)
            print(f"Wrote OxView file to {str(write_oxview_path)}")
