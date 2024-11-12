import shutil
import tempfile

import pytest

from pypatchy.patchy.simulation_ensemble import find_ensemble, build_ensemble, PatchySimulationEnsemble
from pypatchy.patchy.vis_lib import plot_energy
from pypatchy.server_config import load_server_settings
from pypatchy.util import cfg, get_input_dir, simulation_run_dir
from test_ensembles_basic import ensemble

from pypatchy.patchy.analysis_lib import LoadEnergies

from pathlib import Path
import pandas as pd

from pypatchy.patchy.pl.patchyio import get_writer
from pypatchy.patchy.pl.plscene import PLPSimulation
from pypatchy.patchy.simulation_ensemble import find_ensemble


@pytest.fixture
def temp_dir() -> str:
    # construct temporary directory to run

    with tempfile.TemporaryDirectory() as td:
        cfg.set('ANALYSIS', 'simulation_data_dir', str(td))
        yield td


def test_split_cluster_fr():
    """
    tests that we can split a flavio style conf by clusters
    """
    pass

@pytest.fixture
def tile4x4():
    """
    downloads 4x4 tile test data. this can take a bit so avoid unnessecary calls
    """
    return ensemble("4x4_tile", "2023-10-03", True)


def test_split_cluster_lr(temp_dir: str):
    """
    tests that we can split a lorenzo style conf by clusters
    """
    shutil.copy(Path(__file__).parent / "test_files" / "ensemble_specs" / "menger_cube_2024-04-16_metadata.json",
                get_input_dir())

    # copy menger cube particle types data
    (get_input_dir() / "particle_sets").mkdir(exist_ok=True)
    shutil.copytree(Path(__file__).parent / "test_files" / "particle_sets" / "menger_cube",
                    get_input_dir() / "particle_sets", dirs_exist_ok=True)

    e = find_ensemble("menger_cube", "2024-04-16")
    e.set_server_settings(load_server_settings("test_lr"))

    # copy test files to temp dir
    shutil.copytree(Path("/") / "home" / "josh" / "pypatchy_test" / "menger_cube_2024-04-16",
                    Path(temp_dir) / "menger_cube_2024-04-16",
                    dirs_exist_ok=True)

    e.info()

    for isim, sim in enumerate(e.get_simulation(T="0.06")):
        scene = e.get_scene(sim, "initial")
        ptypes = scene.particle_types()
        scene_dataframe = pd.DataFrame(
            columns=[f"particle_{n}_count" for n, _ in enumerate(ptypes)])  # row for each subscene
        clusters_dir = Path(temp_dir) / f"duplicate_{isim}_clusters"
        clusters_dir.mkdir(parents=True)
        cluster_scenes: list[PLPSimulation] = list(scene.split_scene_by_clusters(min_cluster_size=15,
                                                                                 bond_energy=0))
        cluster_scenes.sort(key=lambda x: x.num_particles(), reverse=True)
        for i, subscene in enumerate(cluster_scenes):
            writer = get_writer("flavio")
            writer.set_directory(clusters_dir / f"cluster{i}_flavio")
            writer.directory().mkdir()
            writer.write(subscene,
                         topology="cluster.top",
                         conf_file="cluster_init.conf",
                         particle_file="particles.txt",
                         patchy_file="patches.txt")
            row = {
                f"particle_{ptype}_count": count
                for ptype, count in subscene.particle_type_counts().items()
            }
            scene_dataframe = scene_dataframe.append(row, ignore_index=True)
        # add column to dataframe which is sum of other columns
        scene_dataframe["total_count"] = scene_dataframe.sum(axis=1)
        # sort dataframe by sum column (preserving index)
        scene_dataframe = scene_dataframe.sort_values(by="total_count", ascending=False)
        # scene_dataframe.to_csv(clusters_dir / "cluster_data.csv", index=False)


def test_analysis(temp_dir: str, tile4x4: PatchySimulationEnsemble):
    """
    """

    e = ensemble("4x4_tile", "2023-10-03", True)
    # CODE: download files from remote file server to path returned by `e.tld()`
    assert e.all_folders_exist()
    assert all([e.sim_stage_done(sim, e.sim_most_recent_stage(sim)) for sim in e.ensemble()])
    # more tests
    pass


def test_plot_energy(temp_dir: str):
    e = find_ensemble(name="menger_crystal_n5", date="2024-03-18")
    e.clear_pipeline()
    e.add_analysis_steps(
        LoadEnergies("load_energies", output_tstep=1)
    )
    e.lorenzian_to_flavian("~/lab_data/Simulations/")
    plot_energy(e, data_identifier=(("staging", "denovo"),), e_type="pe")
    # plot_energies(e, e.get_simulation(T=0.0675))

