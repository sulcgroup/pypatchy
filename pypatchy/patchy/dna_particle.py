import itertools
import math
from typing import Union

import numpy as np
from Bio.SVDSuperimposer import SVDSuperimposer

from ..patchy_base_particle import PatchyBaseParticle

from ..dna_structure import DNAStructure, DNABase, DNAStructureStrand
from .mglparticle import MGLParticle


class DNAParticle (DNAStructure):
    """
    A DNA origami that will be used to make a particle in the output particle thing
    i'm very tired rn
    """

    # list of strands which have 3' ends which are suitable for extension
    patch_strand_ids: list[list[int]]
    # i'm going back and forth on whether it's better to have this extend PatchyParticle or
    # include a patchy particle object as a member. currently going w/ the latter
    linked_particle: PatchyBaseParticle

    # list of list of base ids that are tips from which we can extend single strandedd overhangs
    # not updated as top is modified!
    # patch_positions: list[list[int]]

    def __init__(self,
                 structure: DNAStructure,
                 patches: list[list[int]],
                 patchy_particle: Union[None, PatchyBaseParticle] = None):
        """
        Constructs an object to facilitate conversion of patchy particle to DNA origami
        in a multicomponent structure
        """
        DNAStructure.__init__(self, structure.strands, structure.time, structure.box, structure.energy)
        assert all([all([-1 < sid < len(self.strands) for sid in patch]) for patch in patches]), \
            "Some patch strand IDs are invalid!"
        self.patch_strand_ids = patches
        self.linked_particle = patchy_particle
        # import the topology
        # self.topology, base_2_strand = oxutil.read_top(top_file)
        # self.strand_delims = []
        # # compute strand deliminators
        # for (i1, s1), (i2, s2) in pairwise(base_2_strand.items()):
        #     # if i1 is last element in strand1
        #     if s1 != s2:
        #         # put it on the list
        #         self.strand_delims.append(i1)
        #
        # # now import  the origami
        # self.conf = next(linear_read(get_traj_info(str(dat_file)), self.topology))[0]
        #
        # self.patch_positions = patch_positions

    def link_patchy_particle(self, p: PatchyBaseParticle):
        """
        Links a patchy particle to this DNA structure, rotating the structure so that the orientation of the 3' ends of
        the patch strands match the position of the patches on the particle
        Note: PLEASE do not link multiple DNAParticle instances to the same patchy particle or vice
        versa!!!
        """
        assert len(self.patch_strand_ids) == p.num_patches(), \
            f"Mismatch between number of patches on DNA particle ({len(self.patch_strand_ids)}) " \
            f"and on patchy particle({p.num_patches()})!"
        sup = SVDSuperimposer()
        patchy_patches: list[np.ndarray] = []
        # start with centerpoint of the particle (assumed to be 0,0,0)
        # load each patch on the mgl particle
        for patch in p.patches():
            # scale patch with origami
            # skip magic padding when scaling patch local coords
            patchy_patches.append(patch.position() / self.scale_factor(p))
        assert len(patchy_patches) == len(self.patch_centerpoints())
        patchy_patches.append(np.zeros((3,)))
        m2 = np.stack(patchy_patches)
        # generate matrix of origami patches + centerpoint
        dna_patches = [*self.get_patch_cmss(), self.cms()]
        best_rms = np.Inf
        # test different patch arrangements in mgl vs. origami. use best option.
        # best_order should be ordering of dna patches which best matches particle patches
        best_order = None
        for perm in itertools.permutations(enumerate(dna_patches)):
            new_order, m1 = zip(*perm)
            m1 = np.stack(m1)
            if new_order[len(self.patch_centerpoints())] != len(
                    self.patch_centerpoints()):  # definately not optimized lmao
                continue
            sup.set(np.stack(m1), m2)
            sup.run()
            if sup.get_rms() < best_rms:
                best_order = new_order
                best_rms = sup.get_rms()
                rot, tran = sup.get_rotran()
        print(f"Superimposed DNA origami on patchy particle {p.get_id()} with RMS={best_rms}")
        print(f"RMS / circumfrance: {best_rms / (2 * math.pi * self.center2patch_conf())}")        # call self.transform BEFORE linking the particle!
        self.transform(rot.T, tran)
        # apply reordering
        self.patch_strand_ids = [self.patch_strand_ids[i] for i in best_order[:-1]] # skip last position (centerpoint)
        self.linked_particle = p

    def linked_particle(self) -> PatchyBaseParticle:
        return self.linked_particle

    def has_linked(self) -> bool:
        return self.linked_particle is not None

    def get_patch_cmss(self) -> np.ndarray:
        """
        get the centers of mass of the patches
        """
        patch_cmss = []
        for patch in self.patch_strand_ids:
            cms = np.zeros(3)
            for strand_id in patch:
                cms += self.strands[strand_id][0].pos
            patch_cmss.append(
                cms / len(patch)
            )
        a = np.array(patch_cmss)
        assert a.shape == (len(self.patch_strand_ids), 3)
        return a

    def patch_centerpoints(self):
        return [
            self.dna_patch_positions(i).mean(axis=0) for i in range(len(self.patch_strand_ids))
        ]

    def patch_strand_3end(self, patch_idx: int, patch_strand_idx: int) -> DNABase:
        """
        Finds the DNA base object at the 3' end of a strand corresponding to a patch
        This method involves a lot of tricky indexing

        Parameters:
            patch_idx (int) the index in self.patch_strand_ids of the patch to find
            patch_strand_idx (int) the index in the list of patch strands of the strand to access
        """
        return self.patch_strand(patch_idx, patch_strand_idx)[0]  # return 3' residue

    def patch_strand(self, patch_idx: int, patch_strand_idx: int) -> DNAStructureStrand:
        """
        Returns strand object corresponding to patch and idx within patch

        Parameters:
            patch_idx (int) the index in self.patch_strand_ids of the patch to find
            patch_strand_idx (int) the index in the list of patch strands of the strand to access
        """
        return self.strands[self.patch_strand_id(patch_idx, patch_strand_idx)]

    def patch_strand_id(self, patch_idx: int, patch_strand_idx: int) -> int:
        """
        Returns strand id corresponding to patch and idx within patch

        Parameters:
            patch_idx (int) the index in self.patch_strand_ids of the patch to find
            patch_strand_idx (int) the index in the list of patch strands of the strand to access
        """
        assert -1 < patch_idx < len(self.patch_strand_ids), f"Invalid patch idx {patch_idx}"
        # get patch strands
        patch_strands: list[int] = self.patch_strand_ids[patch_idx]
        assert -1 < patch_strand_idx < len(patch_strands), f"Invalid patch strand index {patch_strand_idx}"
        # get strand ID
        strand_id = patch_strands[patch_strand_idx]
        assert -1 < strand_id < len(self.strands), f"Invaliud strand ID {strand_id}"
        return strand_id

    def dna_patch_positions(self, patch_idx: int) -> np.ndarray:
        """
        Parameters:
            patch_idx (int): index in patch_strand_ids of a patch

        Returns:
            an N x 3 array of coords of the 3' residue in each strand composing this patch
        """
        assert -1 < patch_idx < len(self.patch_strand_ids)
        return np.stack([self.strands[strand_idx][0].pos
                         for strand_idx in self.patch_strand_ids[patch_idx]], axis=0)

    def center2patch_conf(self):
        """
        Computes the average distance between the center of the structure and the
        positions of the patches
        """
        return np.mean(
            np.linalg.norm((self.get_patch_cmss() - self.cms()), axis=1)
        )

    # def modify_strand3p(self, strand_id: int, seq: str) -> Iterable[int]:
    #     """
    #     Modifies a topology to add a sequence to the 3' end
    #     Args:
    #         strand_id : the ID of the strand to modify
    #         seq : the seqence to append, in 5'->3' order
    #
    #     Returns:
    #         the ids of the bases that have been added
    #     """
    #     # # grab seq length
    #     # connection_length = len(seq)
    #     # strand_old_len = len(self.topology.strands[strand_id].bases)
    #     # assert connection_length < strand_old_len, "No I will not explain."
    #     # # # we know we work with the 3 end so it's the beginning + sticky_length we have to modify
    #     # # # construct list of bases with new identities at beginning of topology (b/c oxDNA thinks in 3'->5' for some reason?)
    #     # #
    #     # # new_bases = [oxutil.Base(t, b.p3, b.p5) for t, b in zip(seq, self.topology.strands[strand_id].bases[:connection_length])]
    #     # # # "shift" topology by appending the existing strand to our new bases
    #     # # new_bases.extend([oxutil.Base(
    #     # #     t,
    #     # #     p3 + connection_length if p3 != -1 else new_bases[-1].p3, # link end of old strand to new strand
    #     # #     p5 + connection_length if p5 != -1 else -1)
    #     # #                   for t, p3, p5 in self.topology.strands[strand_id].bases])
    #     # # # empty previous bases as we work with tuples and we can't just reasign
    #     # # # so clear strand bases
    #     # # self.topology.strands[strand_id].bases.clear()
    #     # # # and add our new copy strand (with our new sequence at the (3') beginning)
    #     # # self.topology.strands[strand_id].bases.extend(new_bases)
    #     # # assert len(self.topology.strands[strand_id].bases) == strand_old_len + connection_length
    #     #
    #     # # get id of residue at strand head
    #     # start_id = self.topology.strands[strand_id].bases[1].p3
    #     # # make new strand
    #     # strand = oxutil.make_strand(strand_id, start_id, seq + oxutil.get_seq(self.topology.strands[strand_id]))
    #     # # assign to topology
    #     # self.topology.strands[strand_id] = strand
    #     # assert len(self.topology.strands[strand_id].bases) == strand_old_len + connection_length
    #     # # update num bases
    #     # self.topology.nbases += connection_length
    #     # # update strand delims
    #     # for i in range(strand_id, len(self.strand_delims)):
    #     #     self.strand_delims[i] += connection_length
    #     #     # skip the strand we are working on for shifting, because we have already done it
    #     #     if i > strand_id:
    #     #         # shift strand to the right
    #     #         self.topology.strands[i] = self.topology.strands[i] >> connection_length
    #     #
    #     # # update patch positions
    #     # self.patch_positions = [[
    #     #     baseid if baseid < start_id else baseid + connection_length for baseid in patch]
    #     #     for patch in self.patch_positions
    #     # ]
    #     #
    #     # # update conf
    #     # # a1s
    #     # self.conf.a1s = np.resize(self.conf.a1s, (self.topology.nbases, 3))  # resize array
    #     # self.conf.a1s[start_id + connection_length:, :] = self.conf.a1s[start_id:-connection_length, :]  # move data
    #     # self.conf.a1s[start_id: start_id + connection_length, :] = np.nan  # wipe cells
    #     #
    #     # # a3s
    #     # self.conf.a3s = np.resize(self.conf.a3s, (self.topology.nbases, 3))  # resize array
    #     # self.conf.a3s[start_id + connection_length:, :] = self.conf.a3s[start_id:-connection_length, :]  # move data
    #     # self.conf.a3s[start_id: start_id + connection_length, :] = np.nan  # wipe cells
    #     #
    #     # # positions
    #     # self.conf.positions = np.resize(self.conf.positions, (self.topology.nbases, 3))  # resize array
    #     # self.conf.positions[start_id + connection_length:, :] = self.conf.positions[start_id:-connection_length,
    #     #                                                         :]  # move data
    #     # self.conf.positions[start_id: start_id + connection_length, :] = np.nan  # wipe cells
    #     #
    #     # return range(start_id, start_id + connection_length)

    def scale_factor(self, p: PatchyBaseParticle) -> float:
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

    def transform(self, rot: np.ndarray = np.identity(3), tran: np.ndarray = np.zeros(3)):
        DNAStructure.transform(self, rot, tran)
        if self.has_linked():
            self.linked_particle.rotate(rot)
            self.linked_particle.translate(tran)
