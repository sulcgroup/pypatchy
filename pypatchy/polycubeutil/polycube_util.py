# utility functions for polyccubes that don't require a polycubes binary
import copy
import itertools

import numpy as np

from pypatchy.polycubeutil.polycube_structure import PolycubeStructure
from pypatchy.polycubeutil.cube import PolycubesStructureCube
from pypatchy.polycubeutil.polycubesRule import PolycubesRule, PolycubeRuleCubeType, PolycubesPatch, rdir, \
    get_orientation
from pypatchy.structure import FiniteLatticeStructure
from pypatchy.util import getRotations


def make_colors_consecutive(rule: PolycubesRule) -> PolycubesRule:
    """
    translation of Joakim Bohlin's `simplify` method from `utils.js`
    Simplifies a rule by making colors consecutive (so colors used in rule will be 1, 2, 3, etc.)
    Args:
        rule: a polycubes rule

    Returns:
        the rule, simplified
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
