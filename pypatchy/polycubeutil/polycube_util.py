# utility functions for polyccubes that don't require a polycubes binary
import copy
import itertools

import numpy as np

from pypatchy.polycubeutil.polycubesRule import PolycubesRule, RULE_ORDER, PolycubeRuleCubeType, PolycubesPatch, rdir, \
    get_orientation
from pypatchy.polycubeutil.structure import Structure, FiniteLatticeStructure, PolycubeStructure, PolycubesStructureCube
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
    # Find out which dimension has the fewest connectors

    # ok so this thing i'm about to comment out is joakim's algorithm. i like your code except it sucks so let me
    # write the code and maybe it will be really good

    # loop vertices in structure
    # for p_idx in range(structure.num_vertices()):
    #     p = structure.cube(p_idx)
    #     # loop directions
    #     for d in RULE_ORDER:
    #         # if vertex is connected to neigbor
    #         if structure.positions_connected(p, p + d):
    #             # loop dimensions?
    #             for i in range(3):
    #                 if np.array_equal(np.abs(d), dims[i]):
    #                     # increment
    #                     dimCount[i] += 1
    #                     break

    # compute dimension with..... fewest connections?
    # minDim = dims[dimCount.index(min(dimCount))]

    # loop edges in structure
    # this will technically loop all edges twice but for the purposes of computing the min dimension that's fine
    # for edge in structure.graph.edges:
    #     d = structure.graph.get_edge_data(*edge)["dirIdx"]
    #     dimension_idx = int(d / 2)  # rule order: dimension is idx / 2 rounded down
    #     dimCount[dimension_idx] += 1
    #
    # minDim = np.array([0, 0, 0])
    # minDim[dimCount.index(min(dimCount))] = 1

    # further problem. this code also sucks. not only is it badly written but i can tell without even running it
    # that it's very buggy
    # Initialise empty species
    # rule = []
    # cubePosMap = {}
    # for iCube, p in enumerate(coords):
    #     cubeType = []
    #     for i, d in enumerate(utils.getFaceRotations()):
    #         alignDir = d
    #         if not np.array_equal(utils.getRuleOrder()[i], minDim):
    #             alignDir = minDim
    #         cubeType.append({'color': 0, 'alignDir': alignDir})
    #     rule.append(cubeType)
    #     cubePosMap[posToString(p)] = iCube

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

    # # Set colors and alignment direcitons
    # colorCounter = 1
    # for iCube, p in enumerate(coords):
    #     found = False
    #     for iFace, d in enumerate(utils.getRuleOrder()):
    #         neigbourPos = p + d
    #         if bindingStr(p, neigbourPos) in connectors:
    #             found = True
    #             invDir = -d
    #             iFaceNeigh = list(np.array_equal(invDir, v) for v in utils.getRuleOrder()).index(True)
    #             iCubeNeigh = cubePosMap[posToString(neigbourPos)]
    #
    #             rule[iCube][iFace]['color'] = colorCounter
    #             rule[iCubeNeigh][iFaceNeigh]['color'] = -colorCounter
    #             rule[iCubeNeigh][iFaceNeigh]['alignDir'] = rule[iCube][iFace]['alignDir']
    #
    #             colorCounter += 1
    #     if not found:
    #         print("{} not found in connections".format(posToString(p)))
    #
    # rule = [[{
    #     'color': face['color'],
    #     'orientation': round(utils.getSignedAngle(
    #         utils.getFaceRotations()[i],
    #         face['alignDir'],
    #         utils.getRuleOrder()[i]
    #     ) * (2 / np.pi) + 4) % 4
    # } for i, face in enumerate(s)] for s in rule]

    # return rule

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
