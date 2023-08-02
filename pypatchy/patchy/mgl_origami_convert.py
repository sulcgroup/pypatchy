from pathlib import Path
from typing import Generator, Any

from oxDNA_analysis_tools.UTILS.RyeReader import *
from Bio.SVDSuperimposer import SVDSuperimposer
from pypatchy.oxutil import generate_helix_coords, write_configuration_header, write_configuration, write_oxview
from pypatchy.oxutil import Base, merge_tops, read_top, write_top
from random import choice
import itertools

from pypatchy.patchy.mglparticle import MGLScene, MGLParticle, MGLPatch

dist = lambda a, b: np.linalg.norm(a - b)
normalize = lambda v: v / np.linalg.norm(v)


def rc(s):
    mp = {"A": "T", "T": "A", "C": "G", "G": "C"}
    return "".join([mp[c] for c in s][::-1])


def get_color_match(colorstr: str):
    if colorstr.startswith("dark"):
        return colorstr[4:]
    else:
        return f"dark{colorstr}"


def assign_coords(conf, indices, coords):
    for cds, idx in zip(coords, indices):
        conf.positions[idx] = cds[0]
        conf.a1s[idx] = cds[1]
        conf.a3s[idx] = cds[2]

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


def modify_strand3p(top, strand_id, seq):
    connection_lenght = len(seq)
    # we know we work with the 3 end so it's the beginning + sticky_length we have to modify
    new_bases = [Base(t, b.p3, b.p5) for t, b in zip(seq, top.strands[strand_id].bases[0:connection_lenght])]
    new_bases.extend(top.strands[strand_id].bases[connection_lenght:])
    # empty previous bases as we work with tuples and we can't just reasign
    top.strands[strand_id].bases.clear()
    top.strands[strand_id].bases.extend(new_bases)

class DNAParticleType:
    topology: Any
    conf: Any
    base2strand: dict
    patch_positions: list[list[int]]

    def __init__(self, top_file: Path, dat_file: Path, patch_positions: list[list[int]]):
        # import the topology
        self.topology, self.base2strand = read_top(top_file)

        # now import  the origami
        self.conf = next(linear_read(get_traj_info(str(dat_file)), self.topology))[0]

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
        return np.array(patch_cmss)

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
            np.linalg.norm((self.get_patch_cmss() - self.conf_cms()), axis = 1)
        )

    def scale_factor(self, p: MGLParticle, magic: float) -> float:
        # distance between origami patch and center
        center2patch_conf = self.center2patch_conf()
        # distance between mgl patch and center
        center2patch = np.round(np.linalg.norm(p.patch(0).position()), 1)
        # hashtag ratio
        return center2patch / center2patch_conf / magic


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
                 magic_padding: float = 1.4
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
            magic_padding: magic number to get a little bit of space
        """

        # set parameters
        self.spacer_length = spacer_length
        self.particle_delta = particle_delta
        self.bond_length = bond_length
        self.cos_theta_max = cos_theta_max
        self.sticky_length = sticky_length

        # prep variables to use when building our structure
        self.color_sequences = {}
        self.clusters = [{} for _ in range(len(mgl.particles()))]
        self.bondcount = 0
        self.magic_padding = magic_padding

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
            if get_color_match(colorstr) in self.color_sequences:
                # use the reverse compliment of the matching color sequenece
                self.color_sequences[colorstr] = rc(self.color_sequence(get_color_match(colorstr)))
                print(f"Assigning color {colorstr} sequence {self.color_sequences[colorstr]}"
                      f" (reverse compliment of {get_color_match(colorstr)}")
            else:
                # if neither our color nor its match is in the color seqs, generate a new sequence
                # todo: smarter?
                self.color_sequences[colorstr] = "".join(
                    choice(["A", "T", "C", "G"]) for _ in range(self.sticky_length))
                print(f"Assigning color {colorstr} random sequence {self.color_sequences[colorstr]}")

        return self.color_sequences[colorstr]

    def position_particles(self):
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
            origami = self.get_dna_origami(particle)
            print(f"{i + 1}/{pl}", end="\r")
            m1 = []
            for patch in particle.patches():
                m1.append(patch.position() / origami.scale_factor(particle, self.magic_padding))
            m1 = np.array(m1)  # generate refference

            m2 = []
            # compile list of patch centers of mass
            for pcms in origami.patch_positions:
                m2.append([pcms[0], pcms[1], pcms[2]])
            m2 = np.array(m2)  # generate thing to rotateclusters

            sup.set(m1, m2)
            sup.run()
            rot, tran = sup.get_rotran()  # NOTE: our system has to be positioned at cms = (0,0,0)

            pconf = deepcopy(origami.conf)
            # magic numbers needed again for things not to clash
            pconf.box = self.mgl_scene.box_size() / origami.scale_factor(particle, self.magic_padding)
            pconf.positions = np.einsum('ij, ki -> kj', rot,
                                        pconf.positions) + particle.cms() / origami.scale_factor(particle, self.magic_padding)  # + tran
            pconf.a1s = np.einsum('ij, ki -> kj', rot, pconf.a1s)
            pconf.a3s = np.einsum('ij, ki -> kj', rot, pconf.a3s)

            # we finished the positioning
            placed_confs.append(pconf)
        print()
        return placed_confs

    def particle_pair_candidates(self) -> Generator[tuple[MGLParticle], None,None]:
        """
        Returns all possible pairs of particles,
        as defined by interaction range between centers of mass
        """
        handeled_candidates = set()
        for i, p1 in enumerate(self.mgl_scene.particles()):
            for j, p2 in enumerate(self.mgl_scene.particles()):
                if i != j and dist(p1.cms(),
                                   p2.cms()) <= self.particle_delta:
                    if (i, j) not in handeled_candidates and not (j, i) in handeled_candidates:  # prevent repeats
                        handeled_candidates.add((i, j))
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

        # keep in mind: we can't use patch internal IDs here because it enumerates differently!
        # find all possible pairings between two patches on particle 1 and particle 2
        possible_bindings = list(itertools.product(enumerate(particle_1.patches()), enumerate(particle_2.patches())))
        # filter patches that don't pair
        possible_bindings = [(patch1, patch2) for patch1, patch2 in possible_bindings if patch1[1].can_bind(patch2[1])]
        # sort by distance, ascending order
        possible_bindings.sort(key=lambda patch1, patch2: dist(particle_1.cms() + patch1[1].position(),
                                                               (particle_2.cms() + patch2[1].position())))
        # lists for patches that have been handled on particles 1 and 2
        handled_p1 = [False for _ in particle_1.patches()]
        handled_p2 = [False for _ in particle_2.patches()]
        # iterate through possible pairs of patches
        for (q, patch_1), (z, patch_2) in possible_bindings:
            # skip patches we've already handled
            if not handled_p1[q] and not handled_p2[z]:
                # if the two patches are within bonding distance
                if dist(particle_1.cms() + patch_1.position(),
                        (particle_2.cms() + patch_2.positon())) <= self.bond_length:
                    # https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
                    # normalize patches
                    patch1norm = patch_1.position() / np.linalg.norm(patch_1.position())
                    patch2norm = patch_2.position() / np.linalg.norm(patch_2.position())
                    costheta = abs(float(np.clip(np.dot(normalize(patch_1.position()),
                                                        normalize(patch_2.position())), -1.0, 1.0)))
                    # confusing,ly, cos theta-max is the cosine of the maximum angle
                    # so we check if cos theta is LESS than cos theta-max
                    if costheta >= self.cos_theta_max:
                        handled_p1[q] = handled_p2[z] = True
                        yield (q, patch_1), (z, patch_2)
                else:  # can do this because list is sorted
                    break

    def generate_3p_ids(self, patch_id):
        """
        Returns the 3-prime residue IDs of nucleotides in the sticky end starting at the provided residue IDs
        """
        return range(patch_id, patch_id + self.sticky_length)

    def bind_particles3p(self,
                         placed_confs,
                         topologies):
        particles = self.mgl_scene.particles()

        # loop possible pairs of particles
        patch_occupancies = [[False for _ in p.patches()] for p in particles]
        # loop particles
        for p1, p2 in self.particle_pair_candidates():
            i = p1.type_id()
            j = p2.type_id()
            # loop through the patch pairs on each particle that can bind
            for (q, patch1), (z, patch2) in self.patches_to_bind(p1, p2):
                if patch_occupancies[i][q] or patch_occupancies[j][z]:
                    continue
                conf1, conf2 = placed_confs[i], placed_confs[j]
                top1, top2 = topologies[i], topologies[j]
                p1_dna = self.get_dna_origami(p1)
                p2_dna = self.get_dna_origami(p2)
                # for the positions of each of the 2 patches that are about to bind to each other
                for patch_id1, patch_id2 in patch_positions_to_bind(p1_dna.patch_positions[q],
                                                                    p2_dna.patch_positions[z],
                                                                    conf1, conf2):
                    # retrieve sequences from map & add spacers
                    seq1 = self.color_sequence(patch1.color()) + "T" * self.spacer_length
                    seq2 = self.color_sequence(patch2.color()) + "T" * self.spacer_length
                    sticky_length = len(seq1)
                    # generate bonded helix between the particles
                    coords1, coords2 = generate_helix_coords(sticky_length,
                                                             start_pos=conf1.positions[patch_id1],
                                                             helix_direction=conf1.a3s[patch_id1],
                                                             perp=conf1.a1s[patch_id1])
                    id1s = self.generate_3p_ids(patch_id1)
                    id2s = self.generate_3p_ids(patch_id2)
                    # make sure overhangs comply with the helix
                    assign_coords(conf1, id1s, coords1)
                    assign_coords(conf2, id2s, coords2)
                    # lets modify the topology somehow
                    # 1st figure out the strand index
                    sid1 = self.get_dna_origami(p1).base2strand[patch_id1]
                    sid2 = self.get_dna_origami(p2).base2strand[patch_id2]

                    modify_strand3p(top1, sid1, seq1)
                    modify_strand3p(top2, sid2, seq2)
                    # add bond helix nucleotides to cluster lists
                    for x in id1s:
                        self.clusters[i][x] = self.bondcount + len(particles)
                    for x in id2s:
                        self.clusters[j][x] = self.bondcount + len(particles)
                    patch_occupancies[i][q] = patch_occupancies[j][z] = True
                    self.bondcount += 1
        print(f"Created {self.bondcount} bonds between particles")

    def convert(self,
                write_top_path: Union[Path, None] = None,
                write_conf_path: Union[Path, None] = None,
                write_oxview_path: Union[Path, None] = None):
        """
        Converts a scene containing joined MGL particles to an oxDNA model consisting of
        DNA origamis joined by sticky end handles.
        """

        print("generate topologies")
        topologies = [deepcopy(self.get_dna_origami(p)) for p in self.mgl_scene.particles()]

        print("positioning particles")
        placed_confs = self.position_particles()

        print("binding particles using 3p patches")
        self.bind_particles3p(placed_confs, topologies)

        print("merging the topologies")
        merged_tops = merge_tops(topologies)
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
                write_configuration_header(file, placed_confs[0])
                cl = len(placed_confs)
                for i, conf in enumerate(placed_confs):
                    print(f"{i + 1}/{cl}", end="\r")
                    write_configuration(file, conf)

        if write_oxview_path:
            write_oxview(topologies, placed_confs, self.clusters, write_top_path)

