import itertools
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import oxpy

import matplotlib.pyplot as plt
import pytest

from scipy.spatial.transform import Rotation as  R
from ipy_oxdna.oxdna_simulation import Simulation
from pypatchy.patchy.pl.patchyio import get_writer
from pypatchy.patchy.pl.plpatchylib import polycube_rule_to_PL
from pypatchy.patchy.pl.plpotential import PLFRExclVolPotential, PLFRTorsionalPatchyPotential, PLFRPatchyPotential
from pypatchy.patchy.pl.plscene import PLPSimulation
from pypatchy.polycubeutil.polycubesRule import PolycubesRule
from pypatchy.server_config import get_server_config

# just two particles

distances = np.linspace(0.95, 1.2, 25)
angles = np.linspace(0, np.pi / 2, 15)

rule = PolycubesRule(rule_str="1|||||_|-1||||")

def fr_notorsion_data(alpha=0.12) -> tuple[np.ndarray, np.ndarray]:
    # create empty scene
    scene = PLPSimulation()
    scene.set_box_size([3,3,3]) # sure
    scene.cell_size = 3.
    scene.apportion_cells()
    # set particle types to rule
    scene.set_particle_types(polycube_rule_to_PL(rule))
    # instantiate particles
    particle_1 = scene.particle_types().particle(0).instantiate(0)
    particle_1.set_rotation(np.identity(3))
    particle_2 = scene.particle_types().particle(1).instantiate(1)
    particle_2.set_rotation(np.identity(3))
    # particle 0 should be at 0,0,0
    scene.add_particle(particle_1)
    assert np.allclose(scene.get_particle(0).position(), np.array([0,0,0]))
    scene.add_particle(particle_2)
    particles_radius = 0.5

    two_r = 2 * particles_radius

    scene.add_potential(PLFRExclVolPotential(rmax=0.9988, # ????
                                             rstar=0.90530002117156982 * two_r,
                                             b=677.505671539  # from flavio's code
                                                 ))

    scene.add_potential(PLFRPatchyPotential(
        rmax=0.4 * 3 * two_r,
        alpha=alpha,
    ))

    energies_oxDNA = np.zeros(shape=(len(distances), ))
    # energies_oxDNA = np.zeros(shape=(len(distances), len(angles)))
    energies_pypatchy = np.zeros(shape=(len(distances), ))
    # energies_pypatchy = np.zeros(shape=(len(distances), len(angles)))
    with tempfile.TemporaryDirectory(delete=False) as td:
        os.chdir(td)

        w = get_writer("flavio")
        w.set_directory(td)
        for i, dist in enumerate(distances):
            # for j, angle in enumerate(angles):
            scene.get_particle(1).set_position(np.array([0,0, -dist]))
            scene.get_particle(1).set_rotation(R.from_rotvec(np.array([0, 0, math.pi])).as_matrix())

            # compite oxDNA energy
            w.write(scene)
            sim = Simulation(td, td)
            sim.build()
            sim.input["T"] = "25C"
            sim.input["PATCHY_alpha"] = alpha
            sim.input["particle_file"] = "particles.txt"
            sim.input["particle_types_N"] = rule.num_particle_types()
            sim.input["patchy_file"] = "patches.txt"
            sim.input["patch_types_N"] = rule.num_patches()
            sim.input["use_torsion"] = False
            sim.input["plugin_search_path"] = str(Path(get_server_config().oxdna_path) / "contrib" / "romano")
            sim.input["steps"] = 1
            sim.input["interaction_type"] = "PatchyShapeInteraction"
            sim.input["shape"] = "sphere"
            sim.input["backend"] = "CPU"
            with oxpy.Context():
                manager = oxpy.OxpyManager("input")
                energies_oxDNA[i] = manager.system_energy()
                del manager
            # compute pypatchy energy
            energies_pypatchy[i] = scene.get_potential_energy()

    return energies_oxDNA, energies_pypatchy

def fr_torsion_data(narrow_type=0, alpha=0.12) -> tuple[np.ndarray, np.ndarray]:
    # create empty scene
    scene = PLPSimulation()
    scene.set_box_size([3,3,3]) # sure
    scene.cell_size = 3.
    scene.apportion_cells()
    # set particle types to rule
    scene.set_particle_types(polycube_rule_to_PL(rule))
    # instantiate particles
    particle_1 = scene.particle_types().particle(0).instantiate(0)
    particle_1.set_rotation(np.identity(3))
    particle_2 = scene.particle_types().particle(1).instantiate(1)
    particle_2.set_rotation(np.identity(3))
    # particle 0 should be at 0,0,0
    scene.add_particle(particle_1)
    assert np.allclose(scene.get_particle(0).position(), np.array([0,0,0]))
    scene.add_particle(particle_2)
    particles_radius = 0.5

    two_r = 2 * particles_radius

    scene.add_potential(PLFRExclVolPotential(rmax=0.9988, # ????
                                             rstar=0.90530002117156982 * two_r,
                                             b=677.505671539  # from flavio's code
                                                 ))

    scene.add_potential(PLFRTorsionalPatchyPotential(
        rmax=0.4 * 3 * two_r,
        alpha=alpha,
        narrow_type=narrow_type
    ))

    energies_oxDNA = np.zeros(shape=(len(distances), len(angles)))
    energies_pypatchy = np.zeros(shape=(len(distances), len(angles)))
    with tempfile.TemporaryDirectory(delete=False) as td:
        os.chdir(td)

        w = get_writer("flavio")
        w.set_directory(td)
        for i, dist in enumerate(distances):
            for j, angle in enumerate(angles):
                scene.get_particle(1).set_position(np.array([0,0, -dist]))
                scene.get_particle(1).set_rotation(R.from_rotvec(np.array([angle, 0, math.pi])
                                                                 ).as_matrix())

                # compite oxDNA energy
                w.write(scene)
                sim = Simulation(td, td)
                sim.build()
                sim.input["T"] = "25C"
                sim.input["narrow_type"] = narrow_type
                sim.input["PATCHY_alpha"] = alpha
                sim.input["particle_file"] = "particles.txt"
                sim.input["particle_types_N"] = rule.num_particle_types()
                sim.input["patchy_file"] = "patches.txt"
                sim.input["patch_types_N"] = rule.num_patches()
                sim.input["use_torsion"] = True
                sim.input["plugin_search_path"] = str(Path(get_server_config().oxdna_path) / "contrib" / "romano")
                sim.input["steps"] = 1
                sim.input["interaction_type"] = "PatchyShapeInteraction"
                sim.input["shape"] = "sphere"
                sim.input["backend"] = "CPU"
                with oxpy.Context():
                    manager = oxpy.OxpyManager("input")
                    energies_oxDNA[i, j] = manager.system_energy()
                    del manager
                # compute pypatchy energy
                energies_pypatchy[i, j] = scene.get_potential_energy()

    return energies_oxDNA, energies_pypatchy

def FR_torsion_full_data(narrow_type=0, alpha=0.12) -> tuple[np.ndarray, np.ndarray]:
    # create empty scene
    scene = PLPSimulation()
    scene.set_box_size([3,3,3]) # sure
    scene.cell_size = 3.
    scene.apportion_cells()
    # set particle types to rule
    scene.set_particle_types(polycube_rule_to_PL(rule))
    # instantiate particles
    particle_1 = scene.particle_types().particle(0).instantiate(0)
    particle_1.set_rotation(np.identity(3))
    particle_2 = scene.particle_types().particle(1).instantiate(1)
    particle_2.set_rotation(np.identity(3))
    # particle 0 should be at 0,0,0
    scene.add_particle(particle_1)
    assert np.allclose(scene.get_particle(0).position(), np.array([0,0,0]))
    scene.add_particle(particle_2)
    particles_radius = 0.5

    two_r = 2 * particles_radius

    scene.add_potential(PLFRExclVolPotential(rmax=0.9988, # ????
                                             rstar=0.90530002117156982 * two_r,
                                             b=677.505671539  # from flavio's code
                                                 ))

    scene.add_potential(PLFRTorsionalPatchyPotential(
        rmax=0.4 * 3 * two_r,
        alpha=alpha,
        narrow_type=narrow_type
    ))

    energies_oxDNA = np.zeros(shape=(len(distances), len(angles), len(angles), len(angles)))
    energies_pypatchy = np.zeros(shape=(len(distances), len(angles), len(angles), len(angles)))
    with tempfile.TemporaryDirectory(delete=False) as td:
        os.chdir(td)

        w = get_writer("flavio")
        w.set_directory(td)
        for i, dist in enumerate(distances):
            for (j, theta), (k, phi), (n, psi) in itertools.product(enumerate(angles),
                                                                     enumerate(angles),
                                                                     enumerate(angles)):
                scene.get_particle(1).set_position(np.array([0,0, -dist]))
                scene.get_particle(1).set_rotation(R.from_rotvec(np.array([theta, phi, psi+math.pi])
                                                                 ).as_matrix())

                # compite oxDNA energy
                w.write(scene)
                sim = Simulation(td, td)
                sim.build()
                sim.input["T"] = "25C"
                sim.input["narrow_type"] = narrow_type
                sim.input["PATCHY_alpha"] = alpha
                sim.input["particle_file"] = "particles.txt"
                sim.input["particle_types_N"] = rule.num_particle_types()
                sim.input["patchy_file"] = "patches.txt"
                sim.input["patch_types_N"] = rule.num_patches()
                sim.input["use_torsion"] = True
                sim.input["plugin_search_path"] = str(Path(get_server_config().oxdna_path) / "contrib" / "romano")
                sim.input["steps"] = 1
                sim.input["interaction_type"] = "PatchyShapeInteraction"
                sim.input["shape"] = "sphere"
                sim.input["backend"] = "CPU"
                with oxpy.Context():
                    manager = oxpy.OxpyManager("input")
                    energies_oxDNA[i, j, k, n] = manager.system_energy()
                    del manager
                # compute pypatchy energy
                energies_pypatchy[i, j, k, n] = scene.get_potential_energy()

    return energies_oxDNA, energies_pypatchy

@pytest.mark.parametrize("alpha", [0.08, 0.012, 0.015])
def test_compare_energy_fr_notorsion(alpha):
    # Run the simulation to get distances and energies
    energies_oxDNA, energies_pypatchy = fr_notorsion_data(alpha)

    assert np.allclose(energies_oxDNA, energies_pypatchy)

    # Plot the results
    plt.figure(figsize=(8, 5))
    plt.plot(distances, energies_oxDNA, label='oxDNA Energy', linestyle='--', marker='o')
    plt.plot(distances, energies_pypatchy, label='pypatchy Energy', linestyle='-', marker='x')
    plt.xlabel('Distance')
    plt.ylabel('Energy')
    plt.title('Energy vs Distance')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

@pytest.mark.parametrize("narrow_type", [0, 1, 2, 3])
def test_compare_energy_fr_torsion(narrow_type):
    energies_oxDNA, energies_pypatchy = fr_torsion_data(narrow_type)

    # Grid
    D, A = np.meshgrid(distances, angles, indexing="ij")

    # Energy difference
    energy_diff = energies_oxDNA - energies_pypatchy

    # Find absolute min/max of both energy datasets
    global_min = min(energies_oxDNA.min(), energies_pypatchy.min())
    global_max = max(energies_oxDNA.max(), energies_pypatchy.max())

    assert np.allclose(energies_oxDNA, energies_pypatchy)

    # Plot energy difference
    plt.figure(figsize=(7, 6))
    c = plt.imshow(energy_diff.T, origin='lower', aspect='auto',
                   extent=[distances[0], distances[-1], angles[0], angles[-1]],
                   cmap='coolwarm', vmin=global_min - global_max, vmax=global_max - global_min)

    plt.title("Energy Difference (oxDNA - pypatchy)")
    plt.xlabel("Distance")
    plt.ylabel("Angle (rad)")
    plt.colorbar(c, label="Energy Difference (scaled to abs energy range)")
    plt.tight_layout()
    plt.show()

@pytest.mark.parametrize("narrow_type", [0, 1, 2, 3])

def test_torsion_energies_full(narrow_type):
    energies_oxDNA, energies_pypatchy = FR_torsion_full_data(narrow_type)

    assert np.allclose(energies_oxDNA, energies_pypatchy)


