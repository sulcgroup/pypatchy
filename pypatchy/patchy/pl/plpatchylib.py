# #!/usr/bin/env python
# from __future__ import annotations
# import itertools
# import math
# from pathlib import Path
# from typing import Union
# # This file loads patchy particle file from topology and Configuration
# import copy
#
# from pypatchy.patchy.pl.plparticle import PLPatchyParticle, PLParticleSet
# from pypatchy.patchy.pl.plpatch import PLPatch
#
# """
# Library for handling patchy-lock (PL) systems in pypatchy
# Where constants etc. exist they should be the same as / similar to the Patchy interactions
# in C++
# """
#
# myepsilon = 0.00001
#
# # cutoff patch-patch distance, after which interaction cannot occur
# PATCHY_CUTOFF = 0.18
# # copied from oxDNA defs.h
# # bounded arc-cosine?
# LRACOS = lambda x: 0 if x > 1 else math.pi if x < -1 else math.acos(x)
#
#
#
#
# # TODO: something with this class
from __future__ import annotations

import copy
from typing import Union

import numpy as np

from ..mgl import MGLParticle, MGLScene
from .plparticle import PLParticleSet, PLPatchyParticle
from .plpatch import PLPatch
from .plscene import PLPSimulation
from ...patchy_base_particle import BaseParticleSet
from ...polycubeutil.polycube_structure import PolycubeStructure
from ...polycubeutil.polycubesRule import PolycubesRule
from ...util import to_xyz, halfway_vector, normalize


def to_PL(particle_set: BaseParticleSet,
          num_teeth: int = 1,
          dental_radius: float = 0.) -> PLParticleSet:
    """
    I will freely admit I have no idea what "PL" means
    Whatever it means, this function converts an arbitrary particle set into
    PL particles and patches, applying multidentate where applicable.
    """
    # in the unlikely event this is already a PL particle set
    if isinstance(particle_set, PLParticleSet):
        if num_teeth == 1 and dental_radius == 0:
            return particle_set
        else:
            particles = particle_set
    elif isinstance(particle_set, PolycubesRule):
        particles = polycube_rule_to_PL(particle_set)
    elif all([isinstance(p, MGLParticle) for p in particle_set]):
        particles, _ = mgl_particles_to_pl(particle_set)
    else:
        raise Exception("Unrecognized particle types!!")

    # do multidentate convert
    if num_teeth > 1:
        particles = particles.to_multidentate(dental_radius, num_teeth)

    assert particles.num_particle_types() == particle_set.num_particle_types(), \
        f"Bad conversion! Number of particle types produced {particles.num_particle_types()} not equal to number of particles" \
        f" in input particle set {particle_set.num_particle_types()}!"

    return particles


def polycube_rule_to_PL(particle_set: PolycubesRule) -> PLParticleSet:
    particles: PLParticleSet = PLParticleSet()
    # iter particles
    for particle in particle_set:
        particle_patches = []
        # convert to pl patch
        for patch in particle.patches():
            relPosition = patch.position() / particle.radius()
            pl_color = patch.colornum() - 20 if patch.colornum() < 0 else patch.colornum() + 20
            assert patch.get_id() == particles.num_patches() + len(particle_patches)
            a1 = relPosition / np.linalg.norm(relPosition)
            # torsion
            if patch.has_torsion():
                plpatch = PLPatch(patch.get_id(),
                                  pl_color,
                                  relPosition,
                                  a1,
                                  patch.alignDir())
            # no torsion
            else:
                plpatch = PLPatch(patch.get_id(),
                                  pl_color,
                                  relPosition,
                                  a1)
            particle_patches.append(plpatch)

        # convert to pl particle
        # reuse type ids here, unfortunately
        particle = PLPatchyParticle(type_id=particle.type_id(), particle_name=particle.name(),
                                    index_=particle.type_id())
        particle.set_patches(particle_patches)

        particles.add_particle(particle)
    return particles


def PL_to_rule(particles: list[PLPatchyParticle], ) -> Union[None, PolycubesRule]:
    """
    Tries to convert the provided set of particles and patches to a Polycubes Rule.
    TODO: include dynamic effects!!!! somehow.
    Args:
        particles: a list of PLPatchyParticle objects
    Returns:
        the set of particles provided as a Polycubes rule if possible. none if not possible
    """
    return PolycubesRule(rule_json=[
        {
            "patches": [
                {
                    "dir": to_xyz(np.round((patch.position() / particle.radius()))),
                    "alignDir": to_xyz(np.round(patch.a2())),
                    "color": patch.color() - 20 if patch.color() > 0 else patch.color() + 20,
                    "state_var": 0,
                    "activation_var": 0
                }
                for patch in particle.patches()]
        }
        for particle in particles])


def polycube_to_pl(polycube: PolycubeStructure,
                   nteeth=1,
                   dental_radius=0,
                   pad_frac: float = 0.1) -> PLPSimulation:
    pl = PLPSimulation()
    # convert polycubes rule to multidentate patchy particles
    pl_types = to_PL(polycube.rule, nteeth, dental_radius)
    pl_types = pl_types.normalize()
    pl.set_particle_types(pl_types)
    mins = np.full(fill_value=np.inf, shape=3)
    maxs = np.full(fill_value=-np.inf, shape=3)
    # iter cubes in polycube
    for cube in polycube._particles:
        pl_type: PLPatchyParticle = pl_types.particle(cube.get_type())
        particle = PLPatchyParticle(copy.deepcopy(pl_type.patches()),
                                    type_id=pl_type.type_id(),
                                    index_=cube.get_id(),
                                    position=cube.position())
        # particle.a1 = POLYCUBE_NULL_A1
        # particle.a3 = POLYCUBE_NULL_A3\
        particle.rotate(cube.rotation().as_matrix())
        assert pl_type.matches(particle)
        pl.add_particle(particle)
        maxs = np.max([maxs, particle.position()], axis=0)
        mins = np.min([mins, particle.position()], axis=0)
    # compute box
    pad = (maxs - mins) * pad_frac + np.full(fill_value=1, shape=(3,))
    pl.translate(-mins + pad)
    pl.set_box_size(maxs - mins + 2 * pad)

    # verify
    for cube1, cube2 in polycube.iter_bound_particles():
        assert pl.particles_bound(cube1.get_id(), cube2.get_id())
        cube_bindngs_count = len(list(polycube.iter_binding_patches(cube1, cube2)))
        pl_bindings_count = len(
            list(pl.iter_binding_patches(pl.get_particle(cube1.get_id()), pl.get_particle(cube2.get_id()))))
        assert nteeth * cube_bindngs_count == pl_bindings_count

    return pl


def mgl_particles_to_pl(mgl_particles: BaseParticleSet,
                        ref_scene: Union[MGLScene, None] = None) -> tuple[PLParticleSet, dict[str, PLPatchyParticle]]:
    pset = PLParticleSet()
    patch_uid = 0
    patch_color_map: dict[str, int] = {}
    color_counter = 1
    particle_type_colormap: dict[str, PLPatchyParticle] = {}
    for ptypeidx, mgl_ptype in enumerate(mgl_particles):
        pl_patches = []
        for mgl_patch in mgl_ptype.patches():
            # patch width information is inevitably lost
            # TODO: mgl -> pl mdt taking into account width

            # work out mgl patch colors
            # if color isn't in the map
            if mgl_patch.color() not in patch_color_map:
                # if the color is "dark[SOMETHING]"
                if mgl_patch.color().startswith("dark"):
                    color_str = mgl_patch.color()[4:]
                    # if the nondark version is in the map
                    if color_str in patch_color_map:
                        # add the dark version
                        patch_color_map[mgl_patch.color()] = -patch_color_map[color_str]
                    else:
                        # add the non-dark and dark version
                        patch_color_map[color_str] = color_counter
                        patch_color_map[mgl_patch.color()] = -color_counter
                        color_counter += 1
                else:
                    # just add the color
                    patch_color_map[mgl_patch.color()] = color_counter
                    color_counter += 1

            patch_color = patch_color_map[mgl_patch.color()]

            p = PLPatch(patch_uid,
                        patch_color,
                        mgl_patch.position(),
                        mgl_patch.position() / np.linalg.norm(mgl_patch.position()))
            pl_patches.append(p)
            patch_uid += 1
        pl_ptype = PLPatchyParticle(pl_patches,
                                    type_id=ptypeidx)
        particle_type_colormap[mgl_ptype.color()] = pl_ptype
        pset.add_particle(pl_ptype)

    # if we've provided a reference scene, use it to position A2 vectors (so we can convert multidentate later)
    if ref_scene is not None:
        # cry a lot
        handled_patches: set[int] = set()
        for (particle1, particle2) in ref_scene.iter_bound_particles():
            for patch1, patch2 in ref_scene.iter_binding_patches(particle1, particle2):
                # if we have handled all patches, just stop
                if len(handled_patches) == pset.num_patches():
                    break
                # get particle 1 type and rotation
                ptype1 = ref_scene.particle_types()[particle1.color()]
                p1rot = ref_scene.get_rot(particle1)

                # get particle 2 type and rotation
                ptype2 = ref_scene.particle_types()[particle2.color()]
                p2rot = ref_scene.get_rot(particle2)

                # get patch 1 and 2 type IDs
                ppatchtype1 = ptype1.patch(patch1.position() @ p1rot.T).get_id()
                ppatchtype2 = ptype2.patch(patch2.position() @ p2rot.T).get_id()
                if patch1.get_id() not in handled_patches and patch2.get_id() not in handled_patches:
                    # theta = math.pi - angle_between(patch1.position() @ particle1.rotation(),
                    #                                 patch2.position() @ particle2.rotation())
                    midvector = halfway_vector(patch1.position(),
                                               patch2.position())
                    # compute patch oris
                    patch1ori = normalize(np.cross(
                        patch1.position(),
                        midvector)
                    )
                    patch2ori = -normalize(np.cross(
                        patch2.position(),
                        midvector)
                    )

                    assert np.linalg.norm(patch1ori - patch2ori) < 1e-7, "Patch orientation vectors not orthogonal!"

                    pset.patch(ppatchtype1).set_a2(patch1ori @ p1rot.T)
                    pset.patch(ppatchtype2).set_a2(patch2ori @ p2rot.T)
                    handled_patches.add(ppatchtype1)
                    handled_patches.add(ppatchtype2)

                elif patch1.get_id() in handled_patches:
                    patch1ori = pset.patch(ppatchtype1).a2() @ p1rot.T
                    pset.patch(ppatchtype1).set_a2(patch1ori @ p2rot.T)

                    handled_patches.add(ppatchtype1)
                elif patch2.get_id() in handled_patches:
                    patch2ori = pset.patch(ppatchtype2).a2() @ p2rot.T
                    pset.patch(ppatchtype2).set_a2(patch2ori @ p1rot.T)

                    handled_patches.add(ppatchtype2)
        assert pset.num_patches() == len(handled_patches)

    return pset, particle_type_colormap


def mgl_to_pl(mgl: MGLScene,
              pad_frac: float = 0.1) -> PLPSimulation:
    pl = PLPSimulation()
    pset, particle_type_colormap = mgl_particles_to_pl(mgl.particle_types(), mgl)
    pl.set_particle_types(pset)

    mins = np.full(fill_value=np.inf, shape=3)
    maxs = np.full(fill_value=-np.inf, shape=3)

    # convert scene
    for mgl_particle in mgl.particles():
        pl_type = particle_type_colormap[mgl_particle.color()]
        particle = PLPatchyParticle(copy.deepcopy(pl_type.patches()),
                                    type_id=pl_type.type_id(),
                                    index_=mgl_particle.get_id(),
                                    position=mgl_particle.position())
        particle.rotate(mgl_particle.rotation())
        pl.add_particle(particle)
        maxs = np.max([maxs, particle.position()], axis=0)
        mins = np.min([mins, particle.position()], axis=0)

    pl.set_particle_types(pl.particle_types().normalize())

    pad = (maxs - mins) * pad_frac + np.full(fill_value=1, shape=(3,))
    pl.translate(-mins + pad)
    pl.set_box_size(maxs - mins + 2 * pad)

    return pl


POLYCUBE_NULL_A1: np.ndarray = np.array([
    1,
    0,
    0
])
POLYCUBE_NULL_A3: np.ndarray = np.array([
    0,
    0,
    1
])
