"""
Converts the output of a MOSES (github.com/ganglab-bnl/MOSES) run into a pypatchy
PolycubesRule.

MOSES voxels aren't imported directly as a class dependency -- this module only
duck-types against the attributes it needs (`id2`, `cargo`, `cargo_coords`, `bonds`,
and each bond's `color`), so it works against real `algorithm.lattice.Voxel.Voxel`
instances (e.g. `moses.mesovoxel.all_voxels()`) without pypatchy depending on the
MOSES package at all.

Two things intentionally don't carry over:
- Cargo is not representable on a PolycubesRule/PolycubeRuleCubeType (that layer is
  pure binding combinatorics), so it's returned as a side-channel list instead,
  indexed the same way as the rule's particle types.
- Patches are constructed without torsion (`orientation=None`). MOSES has no
  per-bond alignment concept at all -- it treats a voxel's relative orientation as
  fully pinned by its complete set of bonds (true for any voxel with >=2
  non-collinear structural bonds, i.e. essentially all of them in a periodic 3D
  crystal), rather than encoding rotational alignment per patch the way Polycubes
  does. So there's nothing to recover here, even in principle.
"""
from typing import Any, Iterable

import numpy as np

from .polycubesRule import PolycubesRule, PolycubeRuleCubeType, PolycubesPatch, diridx
from ..structure import GenericTypedStructure, TypedStructure, read_topology

# (cargo, cargo_coords) per cube type, indexed by PolycubeRuleCubeType.type_id()
CargoList = list[tuple[Any, tuple[float, float, float]]]


def moses_voxels_to_rule(voxels: Iterable) -> tuple[PolycubesRule, CargoList, dict[int, int]]:
    """
    Converts an iterable of MOSES voxel prototypes (e.g. `moses.mesovoxel.all_voxels()`)
    into a PolycubesRule, plus a parallel list of (cargo, cargo_coords) per cube type.

    Args:
        voxels: iterable of voxel-like objects, each with:
            - `.id2`: MOSES's unique-voxel-type id (used only to build `id2_map`)
            - `.cargo`: cargo type
            - `.cargo_coords`: 3-length cargo position/orientation marker
            - `.bonds`: dict mapping a (dx,dy,dz) vertex offset (magnitude 0.5) to a
              bond-like object with a `.color` attribute (signed int, or None if
              that face isn't bonded)
            Note: `Mesovoxel.all_voxels()` returns voxel ids (ints), not `Voxel`
            objects -- resolve them first, e.g.
            `[lattice.get_voxel(v) for v in moses.mesovoxel.all_voxels()]`.
    Returns:
        (rule, cargos, id2_map) where `cargos[rule.particle(i).type_id()]` is the
        (cargo, cargo_coords) pair for that cube type, and `id2_map` maps MOSES's
        `voxel.id2` to the corresponding `PolycubeRuleCubeType.type_id()` (needed to
        compute per-unit-cell stoichiometry via `moses_base_type_counts`).
    """
    rule = PolycubesRule()
    cargos: CargoList = []
    id2_map: dict[int, int] = {}

    for v in voxels:
        cube_type = PolycubeRuleCubeType(ct_id=0, patches=[])  # id assigned by add_particle
        for vertex, bond in v.bonds.items():
            if bond.color is None:
                continue
            dir_idx = diridx(np.round(np.array(vertex) * 2).astype(int))
            patch = PolycubesPatch(rule.num_patches(), int(bond.color), dir_idx)
            cube_type.add_patch(patch)
        rule.add_particle(cube_type)
        cargos.append((v.cargo, tuple(v.cargo_coords)))
        id2_map[v.id2] = cube_type.type_id()

    return rule, cargos, id2_map


def moses_base_type_counts(lattice, id2_map: dict[int, int], num_cube_types: int) -> list[int]:
    """
    Counts how many of each pypatchy cube type appear per unit cell, by counting
    `id2` occurrences across every "real" voxel in the lattice (`lattice.voxels`,
    which excludes MOSES's duplicated unit-cell-boundary voxels).

    Args:
        lattice: a MOSES Lattice (or duck-typed equivalent) with a `.voxels` list
            of voxel-like objects exposing `.id2`.
        id2_map: as returned by `moses_voxels_to_rule`.
        num_cube_types: `rule.num_particle_types()`.
    Returns:
        counts, where `counts[cube_type_id]` is that type's count per unit cell --
        directly usable as `base_type_counts` for an `AnnealingCrystalloydSimulations`
        (or other `VMMCBaseRunner`) instance.
    """
    counts = [0] * num_cube_types
    for v in lattice.voxels:
        counts[id2_map[v.id2]] += 1
    return counts


def moses_lattice_to_targets(lattice) -> tuple[list[TypedStructure], list[GenericTypedStructure]]:
    """
    Builds the (local, periodic) target structures needed by a VMMC-based
    crystalloyd simulation runner (`targets` / `periodic_targets`) directly from a
    MOSES lattice's own bond graph, typed by cargo (since it's the target's
    physical cargo arrangement being checked, not MOSES's internal, symmetry-driven
    voxel type multiplicity).

    Reuses `Structure.export_periodic_topology()`/`read_topology()` (rather than
    reimplementing local-vs-periodic edge classification) so the split matches
    exactly how the rest of pypatchy's design pipeline (e.g. `Polysat.target_structure`
    / `Polysat.local_target_structure`) defines it.

    Args:
        lattice: a MOSES Lattice (or duck-typed equivalent) with `.voxels` (real,
            de-duplicated voxels), `.is_unit_cell_voxel()`, and `.voxel_dict3`
            (unit-cell-duplicate-id -> real-id).
    Returns:
        (targets, periodic_targets) -- `targets` is the local (non-periodic)
        structure split into its connected components; `periodic_targets` gives,
        for each entry in `targets`, the full periodic structure it belongs to.
    """
    seen: set[tuple[int, int]] = set()
    bindings = []
    types: dict[int, int] = {}

    for v in lattice.voxels:
        types[v.id] = v.cargo
        for vertex, bond in v.bonds.items():
            du = diridx(np.round(np.array(vertex) * 2).astype(int))
            if (v.id, du) in seen:
                continue
            pv = bond.get_partner_voxel()
            pv_id = lattice.voxel_dict3[pv.id] if lattice.is_unit_cell_voxel(pv) else pv.id
            dv = diridx(np.round(-np.array(vertex) * 2).astype(int))
            bindings.append((v.id, du, pv_id, dv))
            seen.add((v.id, du))
            seen.add((pv_id, dv))

    full_structure = GenericTypedStructure(bindings=bindings, types=types)
    periodic_topo = full_structure.export_periodic_topology()
    periodic_full = read_topology(periodic_topo)

    local_topo = {"bindings": periodic_topo["bindings"]}
    if "nanoparticles" in periodic_topo:
        local_topo["nanoparticles"] = periodic_topo["nanoparticles"]
    local_structure = read_topology(local_topo)

    targets = list(local_structure.iter_components())
    periodic_targets = [periodic_full] * len(targets)
    return targets, periodic_targets