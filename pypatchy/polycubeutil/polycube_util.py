# utility functions for polyccubes that don't require a polycubes binary
# this module requires PolycubesStructure, PolycubeSStructureCube, and PolycubesRule,
# so don't import it fro those!
from __future__ import annotations

import copy
import itertools

import numpy as np

from .polycube_structure import PolycubeStructure
from .cube import PolycubesStructureCube
from .polycubesRule import PolycubesRule, PolycubeRuleCubeType, PolycubesPatch, rdir, \
    get_orientation, RULE_ORDER
from ..structure import FiniteLatticeStructure
from ..util import getRotations


def make_colors_consecutive(rule: PolycubesRule) -> PolycubesRule:
    """
    translation of Joakim Bohlin's `simplify` method from `utils.js`
    Simplifies a rule by making colors consecutive (so colors used in rule will be 1, 2, 3, etc.)

    :param rule: a polycubes rule
    :raise AssertionError: if the rule has patches with color value of zero
    :return: the rule, simplified
    :rtype: PolycubesRule
    """
    # construct set of only positive colors
    color_set = [c for c in rule.color_set() if c > 0]
    rule_simple = copy.deepcopy(rule)
    for patch in rule_simple.patches():
        # placeholder for patch color
        c = patch.color()
        assert c, "Zero-value colors are not supported"
        # offset by 1 to handle zero-indexed list
        assert abs(c) in color_set

        patch.set_color(color_set.index(abs(c)) + 1)
        # handle negative colors
        if c < 0:
            patch.set_color(patch.color() * -1)

    return rule_simple


def get_fully_addressable_rule(structure: FiniteLatticeStructure) -> PolycubeStructure:
    """
    Given a topology (as a FiniteLatticeStructure object), construct a polycubes structure where each location in the topology
    is a unique cube type, and each connection between two locations is mediated by a unique color.
    :param structure: the topology (as a FiniteLatticeStructure object)
    :return: A polycube of the topology with non-repeating cube types and colors
    :rtype: PolycubesStructure
    """
    # construct empty rule
    rule = PolycubesRule()
    # construct position map
    # PolycubeStructure obects are immutableish so we have to construct our return object last
    cubePosMap: dict[int, PolycubesStructureCube] = dict()

    # first pass: populate with empty cube types and patches
    patch_counter = 0
    for iCube in structure.vertices():
        cube_position = structure.cube(iCube)
        cube_type = PolycubeRuleCubeType(iCube, [])
        cubePosMap[iCube] = PolycubesStructureCube(iCube,
                                                   cube_position,
                                                   0,
                                                   cube_type)
        for e in structure.graph.edges(iCube):
            d = structure.graph.get_edge_data(*e)["dirIdx"]
            # set cube type and direction, leave orierntation and color blank
            cube_type.add_patch(PolycubesPatch(patch_counter,
                                               d, 0, 0))
            patch_counter += 1
        rule.add_particle(cube_type)

    # loop structure graph edges
    color_counter = 1
    # use undirected graph to iter edges to avoid duplicates
    for u, v in structure.graph.to_undirected().edges:
        # get cube types for both positions
        # cube instances should be unrotated since we initialized them w/o rotations
        uct = cubePosMap[u].get_cube_type()
        vct = cubePosMap[v].get_cube_type()
        # direciton idx of edge u->v
        ud: int = structure.graph.get_edge_data(u, v)["dirIdx"]

        # set patch orientations
        # compute default patch orientation for patch facing the direction ud
        patch_orientation = get_orientation(ud, 0)
        # set both patch orierntation vectors
        uct.patch(ud).set_align(patch_orientation)
        vct.patch(rdir(ud)).set_align(patch_orientation)

        # set patch colors
        uct.patch(ud).set_color(color_counter)
        vct.patch(rdir(ud)).set_color(-color_counter)
        color_counter += 1

    return PolycubeStructure(rule=rule, cubes=cubePosMap.values())


def coord_equal(a1: np.ndarray, a2: np.ndarray) -> bool:
    """
    Args:
        a1 : a N x 3 np array
        a2 : a N x 3 np array

    Return:
        false if the two coords are for structures with the same shape, true otherwise
    """
    if a1.shape[0] != a2.shape[0]:
        return False

    # hi this is Josh. I'd like very much to see a mathematical proof that the following order of operations
    # will produce equal numpy arrays if and only if the starting coordinate sets are equivalent

    # find center of masses of each shape
    com1: np.ndarray = (np.sum(a1, axis=0) / a1.shape[0]).round()
    com2: np.ndarray = (np.sum(a2, axis=0) / a2.shape[0]).round()

    a1 = a1 - com1[np.newaxis, :]
    a2 = a2 - com2[np.newaxis, :]

    for rot in getRotations():
        ra1 = rot @ a1
        for perm in itertools.permutations(range(a2.shape[0])):
            if np.array_equal(ra1[perm, :], a2):
                return True
    return False


def rotation_mapping_to_matrix(rotation_map: dict[int, int]) -> np.ndarray:
    """
    Written by chatGPT
    Convert a rotation map to a rotation matrix.
    :param rotation_map: a rotation map
    :return: a rotation matrix
    """

    # Create a rotation matrix from the rotation map
    rotation_matrix = np.zeros((3, 3))

    for i in range(3):
        # Find the corresponding vector from the mapping
        # Note: we only need to consider vectors with positive first component (1, 3, 5 in RULE_ORDER)
        # Because a rotation is completely determined by where it sends these three vectors
        vector = RULE_ORDER[rotation_map[2 * i + 1]]

        # Add the vector as a column in the rotation matrix
        rotation_matrix[:, i] = vector
    assert np.linalg.det(rotation_matrix) == 1

    # Calculate the quaternion using the function from the previous part
    return rotation_matrix
