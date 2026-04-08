import tempfile

import pytest

from pypatchy.server_config import load_server_settings
from pypatchy.util import cfg
from test_ensembles_basic import ensemble


@pytest.fixture
def temp_dir() -> str:
    with tempfile.TemporaryDirectory() as td:
        cfg.set('ANALYSIS', 'simulation_data_dir', str(td))
        yield td


@pytest.mark.skip(reason="3x3_wireframe test data not yet available")
def test_clustering():
    fmt = "flavio"
    test_data_date = ""
    e = ensemble("3x3_wireframe")


def test_dilute(temp_dir: str):
    e = ensemble("test_dilute")
    e.set_server_settings(load_server_settings("test_fr"))
    e.do_setup(stage="first_stage")


def test_setup_from_polycubes(temp_dir: str):
    pass