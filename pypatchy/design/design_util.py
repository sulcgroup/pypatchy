import copy
from typing import Union

import numpy as np

from .. import libpolycubes

from pypatchy.polycubeutil.polycubesRule import PolycubesRule, diridx, PolycubeRuleCubeType
from ..util import getRotations


def get_coords(rule: PolycubesRule, assemblyMode='seeded') -> np.array:
    """
    Uses libpolycubes to compute coordinate system produced by a rule
    """
    rule_str: str = str(rule)
    return libpolycubes.get_coords(rule_str, assemblyMode)


def all_overlap_rotation(ct1: PolycubeRuleCubeType, ct2: PolycubeRuleCubeType) -> Union[bool, np.ndarray]:
    ct2_diridxs = ct2.diridxs()  # don't recompute this unnessecarily
    # two cube types with different patch counts will never have an overlap rotation
    if ct1.num_patches() != ct2.num_patches():
        return False
    # loop all rotations
    for rot in getRotations():
        overlaps = True
        # check each patch in ct1
        for p in ct1.patches():
            # if this patch overlaps with a patch on ct2 when this rotation is applied
            # might be possible to optimize this with linear algebra
            d = np.matmul(rot, p.direction()).round()
            if diridx(d) not in ct2_diridxs:
                # set overlap flag
                overlaps = False
                # break
                break
        # if the program didn't find any non- overlapping patches
        if overlaps:
            return rot
    return False


def no_overlap_rotation(ct1: PolycubeRuleCubeType, ct2: PolycubeRuleCubeType) -> Union[bool, np.ndarray]:
    ct2_diridxs = ct2.diridxs()  # don't recompute this unnessecarily
    # loop all rotations
    for rot in getRotations():
        overlaps = False
        # check each patch in ct1
        for p in ct1.patches():
            # if this patch overlaps with a patch on ct2 when this rotation is applied
            # might be possible to optimize this with linear algebra
            d = np.matmul(rot, p.direction()).round()
            if diridx(d) in ct2_diridxs:
                # set overlap flag
                overlaps = True
                # break
                break
        # if the program didn't find any overlapping patches
        if not overlaps:
            return rot
    return False


def rotate_cube_type(ct1: PolycubeRuleCubeType, rotation: np.ndarray) -> PolycubeRuleCubeType:
    new_patches = [
        patch.rotate(rotation)
        for patch in ct1.patches()
    ]
    return PolycubeRuleCubeType(ct1.type_id(), new_patches, ct1.state_size(), ct1.effects(), ct1.name())