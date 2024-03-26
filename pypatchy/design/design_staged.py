from ..polycubeutil.polycube_structure import PolycubeStructure
from ..polycubeutil.polycubesRule import PolycubesRule, RULE_ORDER


def design_staged(s: PolycubeStructure) -> PolycubesRule:
    # todo: checking
    surf_cubes = [ct for ct in s.particle_types() if ct.num_patches() < len(RULE_ORDER)]
    