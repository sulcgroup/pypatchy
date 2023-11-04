import copy

from ... import libpolycubes
from pypatchy.design.design_util import all_overlap_rotation, rotate_cube_type
from pypatchy.polycubeutil.polycube_util import get_fully_addressable_rule, coord_equal
from pypatchy.polycubeutil.polycubesRule import PolycubesRule
from pypatchy.structure import FiniteLatticeStructure, PolycubeStructure


def substitution_solve(s: FiniteLatticeStructure) -> PolycubesRule:
    """
    i'm trying to translate this from some of Joakim's code so bear with me...
    https://github.com/Akodiat/polycubes/blob/01e8f89a7a9fc6b34d990683b033ccdc8cd12041/js/utils.js#L706
    kinda
    i don't trust joakim's algorithm actually
    """
    # get fully addressable rule and structure
    fully_addr_polycube = get_fully_addressable_rule(s)
    # set of tried... idk? some kinda ordered pairs
    # aleady_tried: set[frozenset[int]] = set()

    updated_rule_set = True
    # execute steps until we're no longer able to make an update
    while updated_rule_set:
        updated_rule_set = False
        new_rule = try_simplify_rule(fully_addr_polycube.rule)
        # if the new rule is shorter than the current rule, we've updated it so keep thiniing
        updated_rule_set = new_rule.num_particle_types() != fully_addr_polycube.rule.num_particle_types()


def try_simplify_rule(structure: PolycubeStructure) -> PolycubesRule:
    # iter pairs of cube types in rule
    for ct1 in structure.rule.particles():
        for ct2 in structure.rule.particles()[:ct1.type_id()]:
            # don't try to combine cube types w/ different patch counts
            if ct1.get_type().num_patches() != ct2.get_type().num_patches():
                continue
            # filter combinations i've already tried
            # if frozenset({ct1.get_type().type_id(), ct2.get_type().type_id()}) in aleady_tried:
            #     continue
            new_rule = try_merge_cube_types(structure.rule, ct1, ct2)
            if new_rule.num_particle_types() != structure.rule.num_particle_types():
                if libpolycubes.isBoundedAndDeterministic(str(new_rule), 100, 'stochastic'):
                    coords = libpolycubes.getCoords(str(new_rule), 'stochastic', False, structure.num_vertices())
                    if coord_equal(coords, structure.matrix()):
                        return new_rule
    return structure.rule


def try_merge_cube_types(rule: PolycubesRule, ct1idx: int, ct2idx: int) -> PolycubesRule:
    colormap: dict[int, int] = dict()
    newRule = copy.deepcopy(rule)
    ct1 = newRule.particle(ct1idx)
    ct2 = newRule.particle(ct2idx)
    rot = all_overlap_rotation(ct1, ct2)
    if rot:
        ct2 = rotate_cube_type(ct2, rot)
        # iter patches in ct1
        for patch in ct1.patches():
            # map color from patch on ct2 to color on patch on ct1
            colormap[ct2.patch(patch.direction()).color()] = patch.color()
        for patch in newRule.patches():
            if patch.color() in colormap:
                patch.set_color(colormap[patch.color()])
            elif -patch.color() in colormap:
                patch.set_color(-colormap[-patch.color()])
        newRule.remove_cube_type(ct2)
        return newRule
    else:
        return rule
