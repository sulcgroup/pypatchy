from __future__ import annotations

import shutil
from pathlib import Path
from typing import Union

import numpy as np

from .pl.plparticleset import PLParticleSet, MultidentateConvertSettings
from pypatchy.patchy.pl.patchyio import get_writer, FWriter, LWriter
from .pl.plpotential import PLFRPatchyPotential, PLFRExclVolPotential, PLLRExclVolPotential, PLLRPatchyPotential
from .pl.plscene import PLPSimulation
from ..util import get_input_dir


def convert_multidentate(particles: PLParticleSet,
                         dental_radius: float,
                         num_teeth: int,
                         torsion: bool = True,
                         followSurf=False) -> PLParticleSet:
    """
    DEPRECATED. use object-oriented method instead
    """
    return particles.to_multidentate(MultidentateConvertSettings(dental_radius, num_teeth, torsion, followSurf))


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
    get_writer("flavio").set_directory(patches_file.parent)
    particle_set = get_writer("flavio").read_particle_types(patches_file, particles_file)
    new_particles = convert_multidentate(particle_set, dental_radius, num_teeth, follow_surf)

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
    particle_set = get_writer("flavio").read_particle_types(patches_file, particles_file)
    topology: FWriter.PatchyTopology = get_writer("flavio").read_top(top_file)
    l_top: LWriter.PatchyTopology = LWriter.PatchyTopology(particle_set, topology.type_counts)
    get_writer("lorenzo").write_top(l_top, top_file)
    # # read num particles_
    # with open(top_file, 'r') as f:
    #     nParticles, nParticleTypes = f.readline().split()
    #     particle_ids = list(map(lambda x: int(x), f.readline().split()))
    #
    # # back up file
    # shutil.copyfile(top_file, top_file + ".bak")
    #
    # with open(top_file, 'w+') as f:
    #     f.write(f"{nParticles} {nParticleTypes}\n")
    #     f.writelines([particle.export_to_lorenzian_patchy_str(particle_ids.count(particle.type_id())) + "\n"
    #                   for particle in particles])
    #
    # export_interaction_matrix(patches)

# TODO: intgrate mapping of pl color to particle type into plparticle set the way udt sourcing is


def lorenzian_to_flavian(lorenzian_folder: Union[Path, str], flavian_folder: Union[Path, str],
                         top_name: str = "init.top", conf_name: str = "last_conf.dat"):
    """

    """
    if isinstance(lorenzian_folder, str):
        lorenzian_folder = Path(lorenzian_folder)
    if isinstance(flavian_folder, str):
        flavian_folder = Path(flavian_folder)
    assert flavian_folder != lorenzian_folder, "Must have different origin and destination folders"
    get_writer("lorenzo").set_directory(lorenzian_folder)
    topology: LWriter.LPatchyTopology = get_writer("lorenzo").read_top(top_name)
    get_writer("flavio").set_directory(flavian_folder)
    ftop = FWriter.FPatchyTopology(topology.particles())
    get_writer("flavio").write_top(ftop, top_name)
    get_writer("flavio").write_particles_patches(topology.particle_types, "particles.txt", "patches.txt")
    shutil.copyfile(lorenzian_folder / conf_name, flavian_folder / conf_name)


def int_mat_to_keywise(m: np.ndarray, particles: PLParticleSet) -> dict[tuple[int,int], float]:
    assert len(m.shape) == 2
    assert m.shape[0] == m.shape[1]
    int_map = dict()
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if m[(i, j)]:
                int_map[particles.patch(i).color(), particles.patch(j).color()] = m[i, j]
    return int_map



def add_standard_patchy_interaction(scene: PLPSimulation,
                                    sigma: float,
                                    interaction_matrix: Union[np.ndarray, None, dict[tuple[int,int]], float] = None):
    """
    quick function to add patchy interactions which mimic
     rovigatti/Interaction/PatchySwapInteraction
    """
    if interaction_matrix is None:
        interaction_matrix = PLLRPatchyPotential.make_interaction_matrix(scene.particle_types().patches())
    elif isinstance(interaction_matrix, np.ndarray):
        interaction_matrix = int_mat_to_keywise(interaction_matrix, scene.particle_types())
    scene.add_potential(PLLRExclVolPotential(
        rmax=2.01421
    ))
    # TODO: i'm like 99% sure we can ignore patchy interaction for this purpose
    patchy_potential = PLLRPatchyPotential(
        rmax=2.01421,  # cutoff for all interactions, computed assuming a particle w/ radius 0.5 and no spherical attraction
        interaction_matrix=interaction_matrix,
        sigma_ss=sigma
    )

    scene.add_potential(patchy_potential)

