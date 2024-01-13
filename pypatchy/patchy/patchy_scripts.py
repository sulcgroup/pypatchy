from __future__ import annotations

import copy
import math
import shutil
from pathlib import Path
from typing import Union

import numpy as np
from scipy.spatial.transform import Rotation as R

from .mgl import MGLScene, MGLPatch, MGLParticle
from .pl.plpatchylib import load_patches, load_particles, export_interaction_matrix
from .pl.plparticle import PLPatchyParticle, PLParticleSet
from .pl.plpatch import PLPatch
from .pl.plscene import PLPSimulation
from ..patchy_base_particle import BaseParticleSet
from ..polycubeutil.polycube_structure import PolycubeStructure
from ..polycubeutil.polycubesRule import PolycubesRule, PolycubeRuleCubeType
from ..util import rotation_matrix, get_input_dir, to_xyz, angle_between


def convert_multidentate(particles: PLParticleSet,
                         dental_radius: float,
                         num_teeth: int,
                         torsion: bool = True,
                         followSurf=False) -> PLParticleSet:
    """
    DEPRECATED. use object-oriented method instead
    """
    return particles.to_multidentate(dental_radius, num_teeth, torsion, followSurf)


def convert_udt_files_to_mdt(patches_file: Union[str, Path],
                             particles_file: Union[str, Path],
                             dental_radius: Union[float, str] = "0.5",
                             num_teeth: Union[int, str] = "4",
                             follow_surf: Union[str, bool] = "false"):
    # standardize args
    if isinstance(patches_file, str):
        patches_file = Path(patches_file)
    if isinstance(particles_file, str):
        particles_file = Path(particles_file)

    # if the user has given a bad file path
    if not patches_file.is_file():
        # try the input directory
        if (get_input_dir() / patches_file).is_file():
            patches_file = get_input_dir() / patches_file
        else:
            # throw a fit
            raise FileNotFoundError(f"File {patches_file} not found!")
    if not particles_file.is_file():
        # try uboyt directory
        if (get_input_dir() / particles_file).is_file():
            particles_file = get_input_dir() / particles_file
        else:
            # throw a fit
            raise FileNotFoundError(f"File {particles_file} not found!")

    dental_radius = float(dental_radius)
    num_teeth = int(num_teeth)
    follow_surf = follow_surf.lower() == "true"
    patches = load_patches(patches_file)
    particles = load_particles(particles_file, patches)
    new_particles = convert_multidentate(particles, dental_radius, num_teeth, follow_surf)

    new_particles_fn = f"{particles_file[:particles_file.rfind('.')]}_MDt.txt"
    new_patches_fn = f"{patches_file[:patches_file.rfind('.')]}_MDt.txt"
    with open(new_particles_fn, 'w') as f:
        for p in new_particles:
            f.write(p.save_type_to_string())

    with open(new_patches_fn, 'w') as f:
        for p in new_particles.patches():
            f.write(p.save_to_string())


def convert_flavian_to_lorenzian(patches_file: str,
                                 particles_file: str,
                                 top_file: str):
    assert Path(patches_file).is_file()
    assert Path(particles_file).is_file()
    assert Path(top_file).is_file()
    patches = load_patches(patches_file)
    particles = load_particles(particles_file, patches)

    # read num particles_
    with open(top_file, 'r') as f:
        nParticles, nParticleTypes = f.readline().split()
        particle_ids = list(map(lambda x: int(x), f.readline().split()))

    # back up file
    shutil.copyfile(top_file, top_file + ".bak")

    with open(top_file, 'w+') as f:
        f.write(f"{nParticles} {nParticleTypes}\n")
        f.writelines([particle.export_to_lorenzian_patchy_str(particle_ids.count(particle.type_id())) + "\n"
                      for particle in particles])

    export_interaction_matrix(patches)


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
        particles = convert_multidentate(particles, dental_radius, num_teeth)

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


# TODO: intgrate mapping of pl color to particle type into plparticle set the way udt sourcing is
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
        handled_patches: set[int] = {}
        for (particle1, particle2) in ref_scene.iter_bound_particles():
            for patch1, patch2 in ref_scene.iter_binding_patches(particle1, particle2):
                if patch1.get_id() not in handled_patches and patch2.get_id() in handled_patches:
                    # theta = math.pi - angle_between(patch1.position() @ particle1.rotation(),
                    #                                 patch2.position() @ particle2.rotation())
                    midvector

                elif patch1.get_id() not in handled_patches:
                    pass
                elif patch2.get_id() not in handled_patches:
                    pass

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

    pad = (maxs - mins) * pad_frac + np.full(fill_value=1, shape=(3,))
    pl.translate(-mins + pad)
    pl.set_box_size(maxs - mins + 2 * pad)

    return pl
