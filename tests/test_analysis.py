import tempfile

import pytest

from pypatchy.util import cfg
from test_ensembles import ensemble


@pytest.fixture
def temp_dir() -> str:
    # construct temporary directory to run

    with tempfile.TemporaryDirectory() as td:
        cfg.set('ANALYSIS', 'simulation_data_dir', str(td))
        yield td


def test_analysis(temp_dir: str):
    """
    todo: provide raws data and analyze properly
    """
    e = ensemble("example_analysis")
    # CODE: download files from remote file server to path returned by `e.tld()`
    assert e.all_folders_exist()
    assert all([e.sim_stage_done(sim, e.sim_most_recent_stage(sim)) for sim in e.ensemble()])
    # more tests