"""
Tests basic stuff with loading ensembles
"""
import os
import shutil
import tempfile
from pathlib import Path
from typing import Union

import pytest

# check prereqs are ok
import test_prerequisites
from pypatchy.patchy.simulation_ensemble import find_ensemble, PatchySimulationEnsemble
from pypatchy.server_config import PatchyServerConfig, load_server_settings

from pypatchy.util import get_input_dir
from pypatchy.util import cfg
from ptutils import get_data


@pytest.fixture
def temp_dir() -> str:
    # construct temporary directory to run

    with tempfile.TemporaryDirectory() as td:
        cfg.set('ANALYSIS', 'simulation_data_dir', str(td))
        yield td



def ensemble(ensemble_name: str, date: Union[str, None] = None, dl: bool=False) -> PatchySimulationEnsemble:
    """
    function to copy a test ensemble to pypatchy input working directory
    Parameters:
        ensemble_name name of ensemble to grab
        date date of simulation ensemble to grab
        dl if true, will try to download from remote server
    """
    # copy ensemble files
    if date is None:
        shutil.copy(
            Path(__file__).parent / "test_files" / "ensemble_specs" / (
                    ensemble_name + ".json"),
            get_input_dir())
        e = find_ensemble(cfg=ensemble_name)
    else:
        shutil.copy(
            Path(__file__).parent / "test_files" / "ensemble_specs" / (
                    ensemble_name + "_" + date + "_metadata.json"),
            get_input_dir())
        e = find_ensemble(ensemble_name, date)
    if dl:
        get_data(e, str(e.tld()))
    e.info()
    return e


def test_example_basic_3DX(temp_dir: str):
    """
    test basic 3d cross
    """
    e = ensemble("example_basic")
    e.set_server_settings(load_server_settings("test_fr"))
    print("Performing setup")
    e.do_setup()
    print("Running test simulation")
    e.start_simulations()
    # del metadata file
    # is there no pathlib command for ths??
    os.remove(str(get_input_dir() / e.metadata_file))
    # todo: del analysis pipeline


def test_multidentate(temp_dir: str):
    e = ensemble("example_mdt")
    e.set_server_settings(load_server_settings("test_lr"))
    print("Performing setup")
    e.do_setup()
    print("Starting simulations")
    e.start_simulations()
    # del metadata file
    # is there no pathlib command for ths??
    os.remove(str(get_input_dir() / e.metadata_file))
    # todo: del analysis pipeline


def test_multiinit(temp_dir: str):
    # copy required polycube to scenes dir
    (get_input_dir() / "scenes").mkdir(exist_ok=True)
    shutil.copy(Path(__file__).parent / "test_files" / "particle_sets",
                get_input_dir() / "scenes")
    e = ensemble("example_multiinit")
    e.set_server_settings(load_server_settings("test_lr"))
    e.do_setup()
    e.start_simulations()


def test_staged(temp_dir: str):
    e = ensemble("example_staged")
    e.set_server_settings(load_server_settings("test_fr"))
    e.do_setup(stage="first_stage")
    e.start_simulations(stage="first_stage")
    e.do_setup(stage="second_stage")
    e.start_simulations(stage="second_stage")

def test_json_particles(temp_dir: str):
    pass

def test_scene_rw(temp_dir: str):
    """
    tests read/write of scene from an ensemble
    """
    fmt = "flavio"
    # extract and copy 3x3 wireframe cube data to temp directory
    test_data_date = ""
    e = ensemble("4x4_tile", "2023-10-03", True)
    assert e.all_folders_exist(), "Did not successfully copy test data!"
    pass
