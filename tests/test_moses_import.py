from dataclasses import dataclass
from typing import Optional

from pypatchy.polycubeutil.moses_import import moses_voxels_to_rule
from pypatchy.polycubeutil.polycubesRule import RULE_ORDER, diridx


@dataclass
class FakeBond:
    color: Optional[int]


@dataclass
class FakeVoxel:
    cargo: int
    cargo_coords: tuple
    bonds: dict
    id2: int = 0


def test_moses_voxels_to_rule_basic():
    """
    two voxels bonded along x (color 1 <-> -1), everything else unbonded,
    with distinct cargo/cargo_coords, to check patches, directions, and the
    parallel cargo list all line up correctly.
    """
    all_dirs = [(0.5, 0, 0), (-0.5, 0, 0), (0, 0.5, 0), (0, -0.5, 0), (0, 0, 0.5), (0, 0, -0.5)]

    voxel_a = FakeVoxel(
        cargo=1,
        cargo_coords=(0, 0, 0),
        bonds={**{d: FakeBond(None) for d in all_dirs}, (0.5, 0, 0): FakeBond(1)},
        id2=1,
    )
    voxel_b = FakeVoxel(
        cargo=2,
        cargo_coords=(0.3, 0, 0),
        bonds={**{d: FakeBond(None) for d in all_dirs}, (-0.5, 0, 0): FakeBond(-1)},
        id2=2,
    )

    rule, cargos, id2_map = moses_voxels_to_rule([voxel_a, voxel_b])

    assert rule.num_particle_types() == 2
    assert rule.num_colors() == 2  # color_set() counts +1 and -1 as distinct entries

    ct_a, ct_b = rule.particle(0), rule.particle(1)
    assert ct_a.num_patches() == 1
    assert ct_b.num_patches() == 1

    patch_a = ct_a.patches()[0]
    patch_b = ct_b.patches()[0]
    assert patch_a.color() == 1
    assert patch_b.color() == -1
    assert patch_a.dirIdx() == diridx(RULE_ORDER[1])  # +x
    assert patch_b.dirIdx() == diridx(RULE_ORDER[0])  # -x
    assert not patch_a.has_torsion()
    assert not patch_b.has_torsion()

    assert cargos[ct_a.type_id()] == (1, (0, 0, 0))
    assert cargos[ct_b.type_id()] == (2, (0.3, 0, 0))

    assert id2_map[1] == ct_a.type_id()
    assert id2_map[2] == ct_b.type_id()
