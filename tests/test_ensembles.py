"""
Tests basic stuff with loading ensembles
"""
import os
import re
import shutil
import tempfile
from importlib import resources
from pathlib import Path

import pytest

# check prereqs are ok
import test_prerequisites
from pypatchy.patchy.simulation_ensemble import find_ensemble, PatchySimulationEnsemble
from pypatchy.server_config import PatchyServerConfig, load_server_settings

from pypatchy.util import get_input_dir
from pypatchy.util import cfg


@pytest.fixture
def temp_dir() -> str:
    # construct temporary directory to run

    with tempfile.TemporaryDirectory() as td:
        cfg.set('ANALYSIS', 'simulation_data_dir', str(td))
        yield td


def ensemble(ensemble_name: str):
    """
    function to copy a test ensemble to pypatchy input working directory
    """
    # copy ensemble files
    shutil.copy(
        Path(__file__).parent.parent / "spec_files" / "test_files" / "test_ensemble_specs" / (
                ensemble_name + ".json"),
        get_input_dir())
    return find_ensemble(cfg=ensemble_name)


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
    shutil.copy(Path(__file__).parent.parent / "spec_files" / "test_files" / "test_particle_sets",
                get_input_dir() / "scenes")
    e = ensemble("example_multiinit")
    e.set_server_settings(load_server_settings("test_lr"))
    e.do_setup()
    e.start_simulations()

def test_staged(temp_dir: str):
    e = ensemble("example_staged")
    e.do_setup(stage="first_stage")
    e.start_simulations(stage="first_stage")
    e.do_setup(stage="second_stage")
    e.start_simulations(stage="second_stage")

def test_analysis(temp_dir: str):
    pass