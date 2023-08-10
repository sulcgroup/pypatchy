from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Union

import numpy as np
from scipy.spatial.transform import Rotation as R

from pypatchy.patchy.plpatchy import PLPatchyParticle, PLPatch, load_patches, load_particles, export_interaction_matrix
from pypatchy.patchy_base_particle import BaseParticleSet
from pypatchy.util import rotation_matrix, get_input_dir


def convert_multidentate(particles: list[PLPatchyParticle],
                         dental_radius: float,
                         num_teeth: int,
                         followSurf=False) -> tuple[list[PLPatchyParticle], list[PLPatch]]:
    new_particles: list[PLPatchyParticle] = [None for _ in particles]
    patch_counter = 0
    new_patches = []
    for i_particle, particle in enumerate(particles):
        new_particle_patches = []
        for patch in particle.get_patches():
            teeth = [None for _ in range(num_teeth)]
            # "normalize" color by making the lowest color 0
            is_color_neg = patch.color() < 0
            colornorm = abs(patch.color()) - 21
            for tooth in range(num_teeth):
                # start with color
                c = colornorm * num_teeth + tooth + 21
                # grab patch position, a1, a2
                position = np.copy(patch.position())
                a1 = np.copy(patch.a1())
                a2 = np.copy(patch.a2())
                # theta is the angle of the tooth within the patch
                theta = tooth / num_teeth * 2 * math.pi
                if is_color_neg:
                    # opposite-color patches have to be rotated opposite directions
                    # b/c mirroring
                    theta *= -1
                    # set color sign
                    c *= -1
                r = R.identity()
                if followSurf:
                    # phi is the angle of the tooth from the center of the patch
                    psi = dental_radius / particle.radius()
                    psi_axis = np.cross(a1, a2)  # axis orthogonal to patch direction and orientation
                    # get rotation
                    r = R.from_matrix(rotation_matrix(psi_axis, psi))
                else:
                    # move tooth position out of center
                    position += a2 * dental_radius
                r = r * R.from_matrix(rotation_matrix(a1, theta))
                position = r.apply(position)
                a1 = r.apply(a1)
                # using torsional multidentate patches is HIGHLY discouraged but
                # this functionality is included for compatibility reasons
                a2 = r.apply(a2)
                teeth[tooth] = PLPatch(patch_counter, c, position, a1, a2, 1.0 / num_teeth)
                patch_counter += 1
            # add all teeth
            new_particle_patches += teeth
        new_particles[i_particle] = PLPatchyParticle(type_id=particle.type_id(), index_=i_particle,
                                                     radius=particle.radius())
        new_particles[i_particle].set_patches(new_particle_patches)
        new_patches += new_particle_patches
    return new_particles, new_patches


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
    [new_particles, new_patches] = convert_multidentate(particles, dental_radius, num_teeth, follow_surf)

    new_particles_fn = f"{particles_file[:particles_file.rfind('.')]}_MDt.txt"
    new_patches_fn = f"{patches_file[:patches_file.rfind('.')]}_MDt.txt"
    with open(new_particles_fn, 'w') as f:
        for p in new_particles:
            f.write(p.save_type_to_string())

    with open(new_patches_fn, 'w') as f:
        for p in new_patches:
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
          dental_radius: float = 0.) -> tuple[list[PLPatchyParticle], list[PLPatch]]:
    """
    I will freely admit I have no idea what "PL" means
    Whatever it means, this function converts an arbitrary particle set into
    PL particles and patches, applying multidentate where applicable.
    """
    particles: list[PLPatchyParticle] = []
    patches: list[PLPatch] = []
    # iter particles
    for particle in particle_set.particles():
        particle_patches = []
        # convert to pl patch
        for patch in particle.patches():
            relPosition = patch.position() / 2
            pl_color = patch.color() - 20 if patch.color() < 0 else patch.color() + 20
            assert patch.get_id() == len(patches) + len(particle_patches)
            particle_patches.append(PLPatch(patch.get_id(),
                                    pl_color,
                                    relPosition,
                                    patch.position(),
                                    patch.alignDir()))
        patches.extend(particle_patches)
        # convert to pl particle
        particle = PLPatchyParticle(type_id=particle.get_id(), index_=particle.get_id())
        particle.set_patches(particle_patches)

        particles.append(particle)

    # do multidentate convert
    if num_teeth > 1:
        particles, patches = convert_multidentate(particles, dental_radius, num_teeth)

    return particles, patches
