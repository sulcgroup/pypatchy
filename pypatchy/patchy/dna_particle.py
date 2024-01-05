from __future__ import annotations

import itertools
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Union

import numpy as np
from Bio.SVDSuperimposer import SVDSuperimposer

from .pl.plparticle import PLPatchyParticle
from .pl.plpatch import PLPatch
from ..patchy_base_particle import PatchyBaseParticle, BasePatchType

from ..dna_structure import DNAStructure, DNABase, DNAStructureStrand
from .mgl import MGLParticle
from ..util import is_sorted


class DNAParticle (DNAStructure):
    """
    A DNA origami that will be used to make a particle in the output particle thing
    i'm very tired rn
    """

    # list of strands which have 3' ends which are suitable for extension
    patch_strand_ids: list[list[int]]
    # mapping of patch ids on a patch particle to strand IDs
    patch_strand_map: dict[int, int]
    # i'm going back and forth on whether it's better to have this extend PatchyParticle or
    # include a patchy particle object as a member. currently going w/ the latter
    linked_particle: PLPatchyParticle

    flat_strand_ids: list[int] = property(lambda x: list(itertools.chain.from_iterable(x.patch_strand_ids)))

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
        self.patch_strand_map = None

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

    def linked_particle(self) -> PatchyBaseParticle:
        return self.linked_particle

    def has_linked(self) -> bool:
        return self.linked_particle is not None

    def has_strand_map(self) -> bool:
        """
        Returns True if the strands have already been mapped to patches
        This can be done manually or automatically by match_patches_to_strands
        """
        return self.patch_strand_map is not None

    def assign_patches_strands(self, strand_map: dict):
        if self.patch_strand_map is None:
            self.patch_strand_map = strand_map
        else:
            self.patch_strand_map.update(strand_map)

    def assign_patch_strand(self, patch: Union[PLPatch, int], strand_id: int):
        if self.patch_strand_map is None:
            self.patch_strand_map = {}
        if isinstance(patch, PLPatch):
            patch = patch.get_id()
        self.patch_strand_map[patch] = strand_id

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

    def patch_strand_id(self, patch: Union[int, PLPatch]) -> int:
        """
        Returns the DNA strand that corresponds to the provided patch
        """
        if not self.has_linked():
            raise Exception("Cannot get matching DNA strand from patch if no linked particle")
        if isinstance(patch, PLPatch):
            return self.patch_strand_id(patch.get_id())
        else:
            if not isinstance(patch, int):
                raise TypeError(f"Invalid patch identifier type {type(patch)}")
            if patch not in self.patch_strand_map:
                raise IndexError(f"No DNA strand corresponding to patch with ID {patch}")
            strand_id = self.patch_strand_map[patch]
            if strand_id > self.nstrands:
                raise IndexError(f"Patch strand ID {strand_id} is invalid")
            return strand_id

    def patch_strand(self, patch: Union[int, PLPatch]) -> DNAStructureStrand:
        return self.strands[self.patch_strand_id(patch)]

    def patch_3p(self, patch: Union[int, PLPatch]) -> DNABase:
        strand = self.patch_strand_id(patch)
        return self.strand_3p(strand)

    def patch_5p(self, patch: Union[int, PLPatch]) -> DNABase:
        strand = self.patch_strand_id(patch)
        return self.strand_5p(strand)

    def get_patch_strand_3p_positions(self) -> list[np.ndarray]:
        """
        Returns the positions of the 3' nucleotides of the patches, as matrices of coords
        Each patch coord group is returned as a Nx3 matrix, where N is the number of strands
        """
        patch_coordss: list[np.ndarray] = []
        for patch in self.patch_strand_ids:
            patch_coordss.append(np.stack([self.strands[strand_id][0].pos for strand_id in patch]))
        return patch_coordss

    def patch_centerpoints(self):
        return [
            self.dna_patch_positions(i).mean(axis=0) for i in range(len(self.patch_strand_ids))
        ]

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

    def scale_factor(self, p: Union[None, PatchyBaseParticle] = None) -> float:
        """
        Computes the particle type's scale factor
        The scale factor converts between units in pl patchy space and units in nucleotide-level-model
        space, and is thus very important for placing origamis relative to each other
        """
        if p is None:
            assert self.has_linked()
            p = self.linked_particle

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
        # if self.has_linked():
        #     self.linked_particle.rotate(rot)
        #     self.linked_particle.translate(tran)

    def instance_align(self, particle: PLPatchyParticle):
        """
        Aligns a dna particle which has already been linked to a patchy particle type
        to a specific patchy particle instance
        """
        assert particle.type_id() == self.linked_particle.get_type(), "Trying to instance align type no compatible!"
        assert abs(self.linked_particle.rotation() - np.identity(3)).sum() < 1e-6, "Trying to align instance where " \
                                                                                   "currently matches particle isn't " \
                                                                                   "identity rotation!"
        # set origami rotation to match particle instance rotation
        self.transform(rot=particle.rotmatrix())
        # link particle instance (replacing type particle)
        self.linked_particle = particle

    def check_align(self, tolerence: float = 0.1) -> bool:
        """
        checks if the dna particle is correctly aligned to patchy
        """
        if not self.has_linked():
            return False
        # scale factor
        sf = self.scale_factor()
        # distance from center to patches - approxiamtes particle radius
        center2patch_conf = self.center2patch_conf()
        for patch_id, strand_id in self.patch_strand_map.items():
            strand_3p = self.strand_3p(strand_id)
            patch = self.linked_particle.patch_by_id(patch_id)
            v1 = patch.position() @ self.linked_particle.rotmatrix() / sf
            v2 = strand_3p.pos - self.cms()

            v1 /= np.linalg.norm(v1)
            v2 /= np.linalg.norm(v2)
            s = np.cross(v1, v2)
            c = v1.dot(v2)
            a = np.arctan2(np.linalg.norm(s), c)
            if a / (2 * math.pi) > tolerence:
                return False
            # dist = strand_3p.pos - patch.position() / sf
            # dist_rel = np.linalg.norm(dist) / (2 * math.pi * center2patch_conf)
            # if dist_rel > tolerence:
            #     return False
        return True

    def get_strand_group_norms(self):
        a = np.stack([self.strand_group_norm(idx) for idx in range(len(self.patch_strand_ids))])
        assert a.shape == (len(self.patch_strand_ids), 3)
        return a

    def strand_group_norm(self, strand_group_idx: int):
        """
        computes the vector which is normal to the patch strand group
        algorithm was produced by chatgpt 3.5 because as I'm writing this openAI is being stupid about
        the lab card
        """

        # compute plane of strands
        patch = self.patch_strand_ids[strand_group_idx]
        # pos is a 3x1 vector not a 3x vector so concat not stack
        strand_3p_pts = np.stack([self.strands[strand_id][0].pos for strand_id in patch])
        centroid = self.dna_patch_positions(strand_group_idx).mean(axis=0)
        # subtract strand group center from strand positions
        strands_local = strand_3p_pts - centroid
        # i do not understand linalg well enough to follow this
        U, _, Vt = np.linalg.svd(strands_local - np.mean(strands_local, axis=1, keepdims=True))
        # normal vector produced by svd isn't nessicarily the correct one for the patch
        normal_vector = Vt[-1, :]

        dna_cms = self.cms()

        # Compute the signed distance from the plane to the additional point
        signed_distance = np.dot(normal_vector, dna_cms) + np.dot(centroid, normal_vector)

        # If the signed distance is positive, flip the sign of the normal vector
        if signed_distance > 0:
            normal_vector *= -1

        dna_radius = np.linalg.norm(centroid - dna_cms)

        normal_vector = normal_vector / np.linalg.norm(normal_vector) * dna_radius

        return normal_vector + centroid


@dataclass
class PatchyOriRelation:  # name?
    """
    Dataclass to describe the relation of a patchy particle to a DNA origami
    """
    patchy: PLPatchyParticle
    dna: DNAParticle
    rot: np.ndarray
    perm: dict[int, int]
    rms: float
