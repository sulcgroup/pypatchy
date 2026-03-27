"""
Conversion from Hash-CAD/crisscross Megastructure slats to PyPatchy raspberry patchy particles.

Geometry conventions:
- HANDLE_SPACING_NM (14 nm) is the physical distance between adjacent handle positions.
- SLAT_WIDTH_NM (10 nm) is the width of a six-helix bundle.
- Each slat is represented in a local coordinate frame:
    u-axis: along the slat's long axis (direction from first to last handle)
    w-axis: perpendicular to u in the xy-plane (the "width" direction)
    z-axis: the face normal (H2 face = +z, H5 face = -z)
- Patch relpositions use absolute nanometer distances.
- Repulsion sites are placed at each handle position along the u-axis with
  radius REPULSION_RADIUS_NM, approximating a cylindrical excluded volume.

Patch colors:
- ASSEMBLY_HANDLE with value N  -> color = +N
- ASSEMBLY_ANTIHANDLE with value N -> color = -N
"""
from __future__ import annotations

import numpy as np

from pypatchy.patchy.pl.plpatch import PLPatch
from pypatchy.patchy.pl.plparticle import PLPatchyParticle
from pypatchy.patchy.pl.plparticleset import PLParticleSet

HANDLE_SPACING_NM = 14.0   # nm between adjacent handle positions
SLAT_WIDTH_NM = 10.0       # nm width of a six-helix bundle
REPULSION_RADIUS_NM = 7.0  # radius of each repulsion sphere (half handle spacing)

ASSEMBLY_CATEGORIES = {'ASSEMBLY_HANDLE', 'ASSEMBLY_ANTIHANDLE'}


def _slat_local_frame(slat, grid_xd: float, grid_yd: float
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the local coordinate frame of a slat.

    Returns:
        center_nm: 2D center of the slat in nm (x, y)
        u_hat: unit vector along the slat's long axis in the xy-plane
        w_hat: unit vector perpendicular to u_hat in the xy-plane
    """
    coords_nm = np.array([
        [x * grid_xd * HANDLE_SPACING_NM, y * grid_yd * HANDLE_SPACING_NM]
        for y, x in slat.slat_position_to_coordinate.values()
    ])
    center_nm = coords_nm.mean(axis=0)

    direction = coords_nm[-1] - coords_nm[0]
    norm = np.linalg.norm(direction)
    u_hat = direction / norm if norm > 1e-9 else np.array([1.0, 0.0])
    w_hat = np.array([-u_hat[1], u_hat[0]])

    return center_nm, u_hat, w_hat


def slat_repulsion_sites(slat, grid_xd: float, grid_yd: float
                         ) -> list[tuple[np.ndarray, float]]:
    """
    Compute repulsion sites for a slat: one sphere per handle position,
    placed along the u-axis at z=0 in the slat's local frame.
    """
    center_nm, u_hat, _ = _slat_local_frame(slat, grid_xd, grid_yd)
    sites = []
    for y, x in slat.slat_position_to_coordinate.values():
        pos_nm = np.array([x * grid_xd * HANDLE_SPACING_NM, y * grid_yd * HANDLE_SPACING_NM])
        u_coord = float(np.dot(pos_nm - center_nm, u_hat))
        sites.append((np.array([u_coord, 0.0, 0.0]), REPULSION_RADIUS_NM))
    return sites


def slat_to_particle(slat, type_id: int, patch_id_offset: int,
                     grid_xd: float, grid_yd: float) -> PLPatchyParticle:
    """
    Convert a single Slat to a PLPatchyParticle.

    Only ASSEMBLY_HANDLE and ASSEMBLY_ANTIHANDLE handles become patches.
    H2 handles are placed at z = +SLAT_WIDTH_NM/2, H5 at z = -SLAT_WIDTH_NM/2.
    """
    center_nm, u_hat, w_hat = _slat_local_frame(slat, grid_xd, grid_yd)

    patches = []
    patch_local_id = 0

    for side_handles, z_offset in [(slat.H2_handles, +SLAT_WIDTH_NM / 2.0),
                                   (slat.H5_handles, -SLAT_WIDTH_NM / 2.0)]:
        for pos, handle in sorted(side_handles.items()):
            if handle.get('category') not in ASSEMBLY_CATEGORIES:
                continue
            y, x = slat.slat_position_to_coordinate[pos]
            pos_nm = np.array([x * grid_xd * HANDLE_SPACING_NM, y * grid_yd * HANDLE_SPACING_NM])
            delta = pos_nm - center_nm
            relpos = np.array([np.dot(delta, u_hat), np.dot(delta, w_hat), z_offset])
            a1 = relpos / np.linalg.norm(relpos)

            value = int(handle['value'])
            color = value if handle['category'] == 'ASSEMBLY_HANDLE' else -value

            patches.append(PLPatch(
                type_id=patch_id_offset + patch_local_id,
                color=color,
                relposition=relpos,
                a1=a1,
            ))
            patch_local_id += 1

    return PLPatchyParticle(
        patches=patches,
        particle_name=slat.ID,
        type_id=type_id,
    )


def megastructure_to_particle_set(mega) -> tuple[PLParticleSet, list[tuple[np.ndarray, float]]]:
    """
    Convert a Megastructure to a PLParticleSet and a matching list of repulsion sites.

    Each non-phantom slat becomes its own particle type. Patch IDs are assigned
    globally across all particle types.

    Repulsion sites are computed from the first slat and assumed to be the same
    geometry for all slats (i.e., all slats have the same number of handle positions).

    Returns:
        particle_set: PLParticleSet of all slat particle types
        repulsion_sites: list of (position_nm, radius_nm) tuples for RPatchyTopology
    """
    grid_xd = mega.grid_xd
    grid_yd = mega.grid_yd

    real_slats = {k: s for k, s in mega.slats.items() if s.phantom_parent is None}

    particle_types = []
    patch_id_offset = 0
    for type_id, slat in enumerate(real_slats.values()):
        particle = slat_to_particle(slat, type_id, patch_id_offset, grid_xd, grid_yd)
        patch_id_offset += particle.num_patches()
        particle_types.append(particle)

    first_slat = next(iter(real_slats.values()))
    repulsion_sites = slat_repulsion_sites(first_slat, grid_xd, grid_yd)

    return PLParticleSet(particle_types), repulsion_sites