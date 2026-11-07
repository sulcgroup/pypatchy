"""
Conservation-of-energy tests for oxDNA's patchy particle interactions, on both the CPU and CUDA
backends.

Both tests build the same two-particle bound patchy system and run it as an NVE (thermostat off)
MD trajectory using the DetailedPatchySwapInteraction plugin (oxDNA/contrib/rovigatti, format
"lorenzo") -- this is the patchy interaction that's actually compiled in this environment (unlike
RaspberryPatchyInteraction, whose "evans" plugin isn't part of this checkout). Total energy is
sampled throughout the run and checked for drift.

The CUDA test needs an `oxpy` build that was actually compiled with `-DCUDA=ON`. This repo's
default `oxDNA/build` is CPU-only, so the CUDA test auto-skips unless both a CUDA-capable `oxpy`
and a compiled `CUDADetailedPatchySwapInteraction.so` are found; see build_cuda_oxdna.sh for how
to produce them on a machine with the CUDA toolkit and a GPU.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oxpy_utils.oxdna_simulation import Simulation

from pypatchy.patchy.pl.patchyio import get_writer, LWriter
from pypatchy.patchy.pl.plparticleset import MultidentateConvertSettings
from pypatchy.patchy.pl.plpatchylib import polycube_rule_to_PL
from pypatchy.patchy.pl.plscene import PLPSimulation
from pypatchy.polycubeutil.polycubesRule import PolycubesRule

# two particle types, complementary on one face only -- same rule already exercised in
# test_sim_ensemble_json.py, so it's known to convert cleanly to a PL particle set
RULE_STR = "1|1|1|1|1|1_-1|||||"

OXDNA_ROOT = Path(__file__).resolve().parents[2] / "oxDNA"
# built in-place by contrib/rovigatti's CMakeLists (CMAKE_LIBRARY_OUTPUT_DIRECTORY = source dir),
# so this is the plugin dir for both the CPU-only default build and a `-DCUDA=ON` rebuild alike
PLUGIN_DIR = OXDNA_ROOT / "contrib" / "rovigatti"

# NVE run length: short enough to run in well under a second, long enough (with the 3x3x3 PBC box)
# for the pair to interact, separate, and re-encounter each other several times over
STEPS = 20000
DT = 1e-4
PRINT_ENERGY_EVERY = 100


def _bonded_pair_scene() -> PLPSimulation:
    """Two multidentate patchy particles, placed close enough to bind on the first step."""
    rule = PolycubesRule(rule_str=RULE_STR)
    scene = PLPSimulation()
    scene.set_box_size([3, 3, 3])
    scene.cell_size = 3.
    scene.apportion_cells()
    scene.set_particle_types(
        polycube_rule_to_PL(rule).to_multidentate(
            MultidentateConvertSettings(n_teeth=4, dental_radius=0.25)
        )
    )
    particle_1 = scene.particle_types().particle(0).instantiate(0)
    particle_1.set_rotation(np.identity(3))
    particle_2 = scene.particle_types().particle(1).instantiate(1)
    particle_2.set_rotation(np.identity(3))
    scene.add_particle(particle_1)
    assert np.allclose(scene.get_particle(0).position(), np.array([0, 0, 0]))
    scene.add_particle(particle_2)
    particle_2.set_position(np.array([.98, 0., 0.]))
    return scene


def _run_nve(sim_dir: Path, backend: str, backend_precision: str) -> pd.DataFrame:
    """
    Writes the bonded-pair scene, runs it as an NVE MD trajectory, and returns the energy_df
    (columns: time, U, K, Etot).
    """
    scene = _bonded_pair_scene()

    w = get_writer("lorenzo")
    w.set_directory(sim_dir)
    top = LWriter.LPatchyTopology(scene.particle_types(), scene.particle_type_counts())
    w.write(scene, topology="topology.top", conf_file="init.dat", scene_top=top)

    sim = Simulation(sim_dir, sim_dir)
    # NOTE: deliberately not using sim.build()/build_dat_top() here -- their file-detection glob
    # (find_conf_file: "first *.dat that isn't energy/trajectory/error_conf") can't tell init.dat
    # apart from the lorenzo writer's patches_N.dat sidecar files and may clobber one of them.
    # Since the writer already wrote exactly the files/names oxDNA needs, just point the input
    # file at them directly.
    sim.build_sim.build_sim_dir()
    sim.input.set_conf_file("init.dat")
    sim.input.set_top_file("topology.top")
    sim.input["seed"] = 42
    sim.input["T"] = "25C"
    sim.input["DPS_interaction_matrix_file"] = "interactions.txt"
    sim.input["DPS_sigma_ss"] = 0.082
    sim.input["DPS_normalise_patches"] = 0
    sim.input["max_density_multiplier"] = 10
    sim.input["interaction_type"] = "DetailedPatchySwapInteraction"
    sim.input["plugin_search_path"] = str(PLUGIN_DIR)
    sim.input["steps"] = STEPS
    sim.input["dt"] = DT
    sim.input["thermostat"] = "no"
    sim.input["print_conf_interval"] = STEPS
    sim.input["print_energy_every"] = PRINT_ENERGY_EVERY
    sim.input["use_edge"] = 0
    sim.input["backend"] = backend
    sim.input["backend_precision"] = backend_precision
    sim.build_sim.build_input()
    sim.sim_files.parse_current_files()

    sim.oxpy_run(subprocess=False, verbose=False, log=False)
    assert sim.oxpy_run.error_message is None, \
        f"oxDNA run failed:\n{sim.oxpy_run.error_message}"

    return sim.analysis.energy_df


def _assert_energy_conserved(df: pd.DataFrame, rel_tol: float = 1e-4):
    etot = df["Etot"].to_numpy()
    assert len(etot) > 10, "not enough energy samples to judge conservation"
    assert (df["U"].to_numpy() != 0).any(), \
        "potential energy was zero throughout -- the pair never interacted, so this isn't " \
        "actually exercising the interaction's force/energy consistency"

    rel_drift = (etot.max() - etot.min()) / abs(etot.mean())
    assert rel_drift < rel_tol, \
        f"total energy drifted by {rel_drift:.3e} (relative), exceeding tolerance {rel_tol:.3e}"


def test_energy_conservation_cpu(tmp_path):
    assert (PLUGIN_DIR / "DetailedPatchySwapInteraction.so").exists(), \
        f"DetailedPatchySwapInteraction.so not found in {PLUGIN_DIR} -- build the rovigatti " \
        f"contrib plugins first (`make rovigatti` from oxDNA/build)"

    df = _run_nve(tmp_path, backend="CPU", backend_precision="double")
    _assert_energy_conserved(df)


def _cuda_backend_reason_unavailable() -> str | None:
    """Returns a human-readable reason the CUDA test can't run, or None if it can."""
    if not (PLUGIN_DIR / "CUDADetailedPatchySwapInteraction.so").exists():
        return (
            f"CUDADetailedPatchySwapInteraction.so not found in {PLUGIN_DIR}. "
            "oxpy/rovigatti here were built without CUDA support -- see build_cuda_oxdna.sh."
        )

    import oxpy
    with oxpy.Context():
        inp = oxpy.InputFile()
        inp["sim_type"] = "MD"
        inp["backend"] = "CUDA"
        inp["backend_precision"] = "double"
        inp["steps"] = "1"
        try:
            oxpy.OxpyManager(inp)
        except oxpy.OxDNAError as e:
            if "is not allowed" in str(e):
                return (
                    "The CUDA backend was compiled without double precision "
                    "(rebuild oxDNA with -DCUDA_DOUBLE=ON): " + str(e)
                )
            if "not supported" in str(e):
                return (
                    "The currently-imported oxpy was not compiled with CUDA support "
                    "(backend='CUDA' is rejected). See build_cuda_oxdna.sh to rebuild it with "
                    "-DCUDA=ON on a machine with the CUDA toolkit and a GPU."
                )
            raise
    return None


def test_energy_conservation_cuda(tmp_path):
    reason = _cuda_backend_reason_unavailable()
    if reason is not None:
        pytest.skip(reason)

    df = _run_nve(tmp_path, backend="CUDA", backend_precision="double")
    _assert_energy_conserved(df)
