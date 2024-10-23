"""
Tests basic stuff with loading ensembles
"""
import re
import shutil
import tempfile
from importlib import resources
from pathlib import Path

import pytest

# check prereqs are ok
import test_prerequisites
from pypatchy.patchy.simulation_ensemble import find_ensemble, PatchySimulationEnsemble
from pypatchy.server_config import PatchyServerConfig

from pypatchy.util import get_input_dir
from pypatchy.util import cfg


@pytest.fixture
def temp_dir() -> str:
    # construct temporary directory to run

    with tempfile.TemporaryDirectory() as td:
        cfg.set(cfg['ANALYSIS']['simulation_data_dir'], td)
        yield td


def ensemble(ensemble_name: str):
    """
    function to copy a test ensemble to pypatchy input working directory
    """
    # copy ensemble files
    shutil.copy(
        Path(__file__).parent.parent / "spec_files" / "server_configs" / "test_files" / "test_ensemble_specs" / (
                ensemble_name + ".json"),
        get_input_dir())
    return find_ensemble(cfg=ensemble_name)


def test_example_basic_3DX(temp_dir: str):
    """
    test basic 3d cross
    """
    e = ensemble("example_basic")
    e.start_simulations()

