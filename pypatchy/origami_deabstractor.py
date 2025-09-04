import copy
import os
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from random import choice
from typing import Union

from ipy_oxdna.structure_editor.dna_structure import DNAStructure, rc, construct_strands

from .patchy.dna_particle import DNAParticle
from .patchy.pl.plparticle import PLPatchyParticle
from .patchy_base_particle import PatchyBaseParticle
from .scene import Scene
from .util import get_output_dir


class OrigamiDeabstractor(ABC):  # TODO BETTER NAME
    scene: Scene
    check_strict_rc: bool = True
    # map where the keys are patchy particle type IDs and the values are DNA type particles
    particle_type_map: dict[int, DNAParticle]
    # "book" of sticky ends which have already been added to the scene, used
    # to find un-added stickies and add them later
    sticky_book: Union[set[tuple[int, int]], None] = set()
    # mapping where keys are PL particle UIDs and values are DNA particles
    dna_particles: dict[int, DNAParticle]
    sticky_length: Union[int, None]
    # color sequences, as strings
    color_sequences: dict[Union[int, str], str]
    bondcount: int
    scale_factor: Union[float, None] = None

    def __init__(self,
                 sticky_length: Union[int, None] = None,
                 spacer_length: Union[int, None] = None,
                 expected_num_edges: Union[int, None] = None):
        self.spacer_length = spacer_length
        self.dna_particles = dict()
        self.sticky_book = set()
        self.expected_num_edges = expected_num_edges
        self.bondcount = 0
        self.particle_type_map = dict()
        self.dna_particles = dict()
        self.sticky_length = sticky_length
        # prep variables to use when building our structure
        self.color_sequences = {}
        self.color_match_overrides = {}

    def set_track_conf_stickies(self, bNewVal):
        if bNewVal:
            self.sticky_book = set()
        else:
            self.sticky_book = None

    def is_track_conf_stickies(self) -> bool:
        return self.sticky_book is not None

    def get_dna_origami(self,
                        particle_type: Union[str, int, PatchyBaseParticle]) -> DNAParticle:
        if isinstance(particle_type, PatchyBaseParticle):
            return self.get_dna_origami(particle_type.get_type())
        else:
            assert particle_type in self.particle_type_map, f"Particle type {particle_type} not in type map!"
            return self.particle_type_map[particle_type]

    def get_full_conf(self, clusters=False) -> DNAStructure:
        """
        Combines all structures together into a single DNAStructure
        """
        print("merging the topologies")
        # return DNAStructure.merge_many(self.get_particles())
        particles = self.get_particles()
        if clusters:
            for p in particles:
                p.clear_clusters()
                for b in p.iter_bases():
                    p.assign_base_to_cluster(b.uid, 0)

        merged_conf = DNAStructure.merge_many(particles)
        return merged_conf

    def get_sticky_length(self) -> float:
        """
        This method is *not* a simple accessor for self.sticky_length!
        If sticky end length has been hard coded it will simply return self.sticky_length
        If sticky length is not hardcoded it will return the average sticky length
        this is important for computing spacing when positioning particles
        """
        if self.sticky_length is not None:
            return self.sticky_length
        else:
            assert len(
                self.color_sequences) > 0, "Can't dynamically calculate sticky end length without assigned sticky ends!"
            return sum([len(seq) for seq in self.color_sequences.values()]) / len(self.color_sequences)

    def color_sequence(self, colorstr: str) -> str:
        # if this color isn't in our color bank
        if colorstr not in self.color_sequences:
            # ...but its matching color is
            if self.get_color_match(colorstr) in self.color_sequences:
                # use the reverse compliment of the matching color sequenece
                self.color_sequences[colorstr] = rc(self.color_sequence(self.get_color_match(colorstr)))
                print(f"Assigning color {colorstr} sequence 5'->{self.color_sequences[colorstr]}->3'"
                      f" (reverse compliment of {self.get_color_match(colorstr)})")
            else:
                # if neither our color nor its match is in the color seqs, generate a new sequence
                # todo: smarter?
                assert self.sticky_length is not None, f"Auto-generation of sequences is off but you didn't provide a " \
                                                       f"color sequence for {colorstr}"
                self.color_sequences[colorstr] = "".join(
                    choice(["A", "T", "C", "G"]) for _ in range(self.sticky_length))
                print(f"Assigning color {colorstr} random sequence {self.color_sequences[colorstr]}")

        return self.color_sequences[colorstr]

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

    def ready_to_position(self) -> bool:
        return all([ptype.get_type() in self.particle_type_map
                    for ptype in self.scene.particle_types()])

    def position_particles(self):
        """
        Positions particles?
        IDK
        """
        assert self.ready_to_position()
        self.dna_particles = dict()
        # get scene particles
        particles: list[PLPatchyParticle] = self.scene.particles()
        pl = len(particles)
        for i, particle in enumerate(particles):
            # clone dna particle
            origami = self.get_dna_origami(particle).clone(copy_uuids=False)
            # origami: DNAParticle = deepcopy(self.get_dna_origami(particle))
            # origami: DNAParticle = self.get_dna_origami(particle).clone()
            print(f"{i}/{pl}", end="\r")
            assert origami.linked_particle.matches(particle)
            # align dna particle with patchy
            origami.instance_align(particle)
            scale_factor = self.get_scale_factor()
            # scale factor tells us how to convert MGL distance units into our DNA model distances
            # transform origami
            origami.transform(tran=particle.position() * scale_factor)

            # we finished the positioning
            self.dna_particles[particle.get_uid()] = origami
        print()

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
            return self.get_dna_particle(pl.get_uid())
        else:
            assert isinstance(pl, int)
            assert pl in self.dna_particles
            return self.dna_particles[pl]

    def dump_monomer(self, ptype: DNAParticle, fp: Path):
        assert ptype.has_linked(), "Cannot dump monomer for unlinked DNA particle"
        # if np.abs(ptype.linked_particle.rotation() - np.identity(3)).sum() > 1e-6:
        #     print("Warning: Particle has non-identity rotation! "
        #           "Patch a1s are going to be messed up! But you can technically proceed")
        cpy = copy.deepcopy(ptype)
        # clone to add strands
        # iter strand id map
        for patch_id in cpy.patch_strand_map:
            # particle types don't have sticky ends added until they're positioned so we need to add them
            # here before we export
            patch = cpy.linked_particle.patch_by_id(patch_id)
            seq = "T" * self.spacer_length + self.color_sequence(patch.color())
            # last-base a1 is a really bad method for doing strand vector, compute patch a1 instead
            strand1, _ = construct_strands(seq,
                                           start_pos=cpy.patch_strand(patch_id)[0].pos,
                                           helix_direction=patch.a1())

            cpy.patch_strand(patch_id).prepend(strand1[::-1])

        cpy.export_top_conf(fp / (cpy.linked_particle.name() + ".top"),
                            fp / (cpy.linked_particle.name() + ".dat"))

    def dump_monomers(self, fp: Union[None, Path, str] = None, only_present=False):
        """
        Saves all dna particle types in their own .top and .dat files
        """
        # handle inputs
        if fp is None:
            # default to drop files in output dir
            fp = get_output_dir()
        elif isinstance(fp, str):
            fp = Path(fp)
            if not fp.is_absolute():
                fp = get_output_dir() / fp
        # make output directory if it doesn't already exist
        if not fp.exists():
            os.makedirs(fp)
        # for each monomer
        for ptype in self.particle_type_map.values():
            # check that it actually exists in the scene, or that we've forced to dump all
            if not only_present or self.scene.particle_type_counts()[ptype.linked_particle.get_type()] > 0:
                # save monomer top and conf to file path
                self.dump_monomer(ptype, fp)

    @abstractmethod
    def convert(self, unbound_stickies: bool = True):
        pass

    def assign_particles(self,
                         dna: DNAParticle,
                         *args: Union[str, PLPatchyParticle, int]):
        """
        Assigns a dna particle to one or more patchy particle types
        """

        # if no particle was provided, assume we're assigning the same DNA particle to
        # all patchy particle types.
        if not len(args):
            patchy_types: list[PLPatchyParticle] = self.scene.particle_types().particles()
        else:
            patchy_types: list[PLPatchyParticle] = []
            for a in args:
                if isinstance(a, PLPatchyParticle):
                    patchy_types.append(a)
                else:
                    ptype = self.scene.particle_types().particle(a)
                    patchy_types.append(ptype)
        # for each particle type
        for patchy_type in patchy_types:
            # make a replica of the dna particle to work on
            dna_cpy = deepcopy(dna)
            # assign dna particle in to particle type
            self.particle_type_map[patchy_type.get_type()] = dna_cpy
            # link patches
            self.link_patchy_particle(patchy_type, dna_cpy)