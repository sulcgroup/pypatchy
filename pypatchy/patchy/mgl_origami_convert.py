import math
from pathlib import Path
from typing import Generator, Any

import numpy as np
from oxDNA_analysis_tools.UTILS.RyeReader import *
from Bio.SVDSuperimposer import SVDSuperimposer
from pypatchy.oxutil import generate_helix_coords, write_configuration_header, write_configuration, write_oxview, \
    POS_BASE, generate_spacer, generate_3p_ids, assign_coords
from pypatchy.oxutil import Base, merge_tops, read_top, write_top
from random import choice
import itertools

from pypatchy.patchy.mglparticle import MGLScene, MGLParticle, MGLPatch

dist = lambda a, b: np.linalg.norm(a - b)
normalize = lambda v: v / np.linalg.norm(v)


def rc(s):
    mp = {"A": "T", "T": "A", "C": "G", "G": "C"}
    return "".join([mp[c] for c in s][::-1])



# todo: MORE PARAMETERS
def patch_positions_to_bind(patch_positions1,
                            patch_positions2,
                            conf1,
                            conf2):
    pps2 = np.array([conf2.positions[pid] for pid in patch_positions2])

    perms = np.array(list(itertools.permutations([conf1.positions[pid] for pid in patch_positions1])))

    diffs = pps2[np.newaxis, :, :] - perms  # vector difference
    distancesqrd = np.sum(np.multiply(diffs, diffs), axis=(2,))  # dot product, gives us distance squared
    sums = np.sum(distancesqrd, axis=(1,))

    bestmatch = np.argmin(sums)
    return zip(
        list(itertools.permutations(patch_positions1))[bestmatch],
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


def modify_strand3p(top, strand_id: int, seq: str):
    connection_length = len(seq)
    # we know we work with the 3 end so it's the beginning + sticky_length we have to modify
    new_bases = [Base(t, b.p3, b.p5) for t, b in zip(seq, top.strands[strand_id].bases[0:connection_length])]
    new_bases.extend(top.strands[strand_id].bases[connection_length:])
    # empty previous bases as we work with tuples and we can't just reasign
    top.strands[strand_id].bases.clear()
    top.strands[strand_id].bases.extend(new_bases)


class DNAParticleType:
    """
    A DNA origami that will be used to make a particle in the output particle thing
    i'm very tired rn
    """
    topology: Any
    conf: Any
    base2strand: dict
    patch_positions: list[list[int]]

    def __init__(self, top_file: Path, dat_file: Path, patch_positions: list[list[int]]):
        """
        Constructs an object to facilitate conversion of patchy particle to DNA origami
        in a multicomponent structure
        """
        # import the topology
        self.topology, self.base2strand = read_top(top_file)

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


class MGLOrigamiConverter:
    """
    This class facilitates the conversion of a patchy particle model (in MGL format) to
    a set of connected DNA origamis
    """
    clusters: list[dict]
    color_sequences: dict[str, str]
    bondcount: int
    particle_structure: dict[str, DNAParticleType]

    def __init__(self,
                 mgl: MGLScene,
                 particle_types: Union[DNAParticleType,
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
        self.clusters = [{} for _ in range(len(mgl.particles()))]
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
        self.color_sequences[self.get_color_match(colorstr)] = seq

    def get_color_match(self, colorstr: str) -> str:
        if colorstr in self.color_match_overrides:
            return self.color_match_overrides[colorstr]
        else:
            if colorstr.startswith("dark"):
                return colorstr[4:]
            else:
                return f"dark{colorstr}"

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
        particles = self.mgl_scene.particles()
        placed_confs = []  # output prom_p
        # everyone's most favorite aligning tool
        sup = SVDSuperimposer()
        pl = len(particles)
        for i, particle in enumerate(particles):
            origami = deepcopy(self.get_dna_origami(particle))
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

            for j, pidx in enumerate(best_order[:-1]):
                newval = particle.patch(pidx).add_key_point(origami.patch_positions[j])
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
        assert self.mgl_scene.patch_ids_unique()
        particles = self.mgl_scene.particles()
        # please please do not call this method with dna particles that don't correspond
        # in order to the mgl particles

        patch_occupancies = [False for _ in itertools.chain.from_iterable(
            [[patch.get_id() for patch in particle.patches()] for particle in self.mgl_scene.particles()])]

        # loop possible pairs of particles
        # patch_occupancies = [[False for _ in p.patches()] for p in particles]
        # there's probably an oxpy method to convert bp to oxdna distance units
        expected_pad_distance = (self.sticky_length + 2.0 * self.spacer_length) * POS_BASE
        patchpaircount = 0
        # loop particles
        for p1, p2 in self.particle_pair_candidates():
            i = p1.type_id()
            j = p2.type_id()
            assert i != j
            p1_dna = dna_particles[i]
            p2_dna = dna_particles[j]
            # loop through the patch pairs on each particle that can bind
            for (q, patch1), (z, patch2) in self.patches_to_bind(p1, p2):
                assert -1 < patch1.get_id() < len(patch_occupancies)
                assert -1 < patch2.get_id() < len(patch_occupancies)
                assert patch1.get_id() != patch2.get_id()
                # if either patch is bound, stop!
                if not patch_occupancies[patch1.get_id()] and not patch_occupancies[patch2.get_id()]:
                    patch_occupancies[patch1.get_id()] = patch_occupancies[patch2.get_id()] = True

                    conf1, conf2 = p1_dna.conf, p2_dna.conf
                    top1, top2 = dna_particles[i].topology, dna_particles[j].topology

                    p1_patch_idxs = patch1.get_key_point(1)
                    p2_patch_idxs = patch2.get_key_point(1)
                    # for the positions of each of the 2 patches that are about to bind to each other
                    for patch_id1, patch_id2 in patch_positions_to_bind(p1_patch_idxs,
                                                                        p2_patch_idxs,
                                                                        conf1, conf2):
                        # make sure both patch nucleotides are at the ends of their respective strands
                        # TODO: make sure this is correct
                        assert p1_dna.base2strand[patch_id1] != p1_dna.base2strand[patch_id1-1]
                        assert p2_dna.base2strand[patch_id2] != p1_dna.base2strand[patch_id2-1]
                        start_position1 = conf1.positions[patch_id1]
                        start_position2 = conf2.positions[patch_id2]
                        if dist(start_position1, start_position2) > 2 * expected_pad_distance:
                            print(f"Distance between patches = {dist(start_position1, start_position2)}. "
                                  f"Expected distance {expected_pad_distance}")

                        # generate conf for sticky-end double helix
                        # trust me, do this first
                        midpoint = (start_position1 + start_position2) / 2
                        coords1, coords2 = generate_helix_coords(self.sticky_length,
                                                                 start_pos=midpoint,
                                                                 helix_direction=start_position2 - start_position1,
                                                                 perp=conf1.a1s[patch_id1])

                        first_midpoint = (start_position1 + start_position2 - self.spacer_length * POS_BASE) / 3
                        second_midpoint = first_midpoint * 2
                        # construct spacer sequence and coords
                        # scale the spacers to be evenly spaced between the origin point and
                        # the sticky-end helix (hopefully this will make relax nicer?)
                        t_coords1 = generate_spacer(self.spacer_length,
                                                    start_position=start_position1,
                                                    end_position=coords1[0][0],
                                                    stretch=False)
                        t_coords2 = generate_spacer(self.spacer_length,
                                                    start_position=start_position2,
                                                    end_position=coords2[0][0],
                                                    stretch=False)
                        spacer_ids1 = generate_3p_ids(patch_id1, self.spacer_length)
                        spacer_ids2 = generate_3p_ids(patch_id2, self.spacer_length)

                        # lets modify the topology somehow
                        # 1st figure out the strand index
                        sid1 = p1_dna.base2strand[patch_id1]
                        sid2 = p2_dna.base2strand[patch_id2]

                        # first modify strand to add spacer
                        modify_strand3p(top1, sid1, "T" * self.spacer_length)
                        modify_strand3p(top2, sid2, "T" * self.spacer_length)

                        # retrieve sequences from map
                        sticky_seq1 = self.color_sequence(patch1.color())
                        sticky_seq2 = self.color_sequence(patch2.color())

                        # then modify strand to add sticky ends
                        modify_strand3p(top1, sid1, sticky_seq1)
                        modify_strand3p(top2, sid2, sticky_seq2)

                        # generate 3prime ids for sticky end sequences
                        sticky_id1s = generate_3p_ids(patch_id1 + self.spacer_length, self.sticky_length)
                        sticky_id2s = generate_3p_ids(patch_id2 + self.spacer_length, self.sticky_length)

                        # update conf for spacer positions
                        assign_coords(conf1, spacer_ids1, t_coords1)
                        assign_coords(conf2, spacer_ids2, t_coords2)
                        # make sure overhangs comply with the helix
                        assign_coords(conf1, sticky_id1s, coords1)
                        assign_coords(conf2, sticky_id2s, coords2)

                        # add bond helix nucleotides to cluster lists
                        for x in sticky_id1s:
                            self.clusters[i][x] = self.bondcount + len(particles)
                        for x in sticky_id2s:
                            self.clusters[j][x] = self.bondcount + len(particles)
                    self.bondcount += 1
                else:
                    print("Patch occupied!")
        print(f"Created {self.bondcount} dna patch bonds")

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
        merged_tops = merge_tops([p.topology for p in particles])
        # spit out the topology
        if write_top_path:
            assert write_conf_path
            assert write_top_path.parent.exists()
            assert write_conf_path.parent.exists()
            if write_top_path.parent != write_conf_path.parent:
                print("You're technically allowed to do this but I do wonder why")
            write_top(
                merged_tops, str(write_top_path)
            )

            print("printing confs")
            with write_conf_path.open("w") as file:
                write_configuration_header(file, particles[0].conf)
                cl = len(particles)
                for i, p in enumerate(particles):
                    print(f"{i + 1}/{cl}", end="\r")
                    write_configuration(file, p.conf)

        if write_oxview_path:
            write_oxview([p.topology for p in particles],
                         [p.conf for p in particles], self.clusters, write_oxview_path)
