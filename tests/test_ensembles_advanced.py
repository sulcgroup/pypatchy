from pypatchy.server_config import load_server_settings
from test_ensembles_basic import ensemble

def test_clustering():
    fmt = "flavio"
    # extract and copy 3x3 wireframe cube data to temp directory
    test_data_date = ""
    e = ensemble("3x3_wireframe")


def test_dilute(temp_dir: str):
    e = ensemble("test_dilute")
    e.set_server_settings(load_server_settings("test_fr"))
    e.do_setup(stage="first_stage")


def test_setup_from_polycubes(temp_dir: str):
    pass