from os.path import expanduser
from pathlib import Path

import pytest

def test_pypatchy_working_directory():
    assert Path(expanduser("~/.pypatchy")).is_dir(), \
        "No pypatchy working directory `~/.pypatchy` - try running configure.py"
    assert Path(expanduser("~/.pypatchy/input")).is_dir(), "`~/.pypatchy/input` directory does not exist"
    assert Path(expanduser("~/.pypatchy/output")).is_dir(), "`~/.pypatchy/output` directory does not exist"
    assert Path(expanduser("~/.pypatchy/settings.cfg")).is_file()

from pypatchy.server_config import load_server_settings

@pytest.mark.parametrize("spec_name", ["test_lr", "test_fr"])
def test_oxdna_install(spec_name):
    scfg = load_server_settings(spec_name)
    assert Path(scfg.oxdna_path).is_dir(), f"oxDNA install directory {scfg.oxdna_path} does not exist"
    assert (Path(scfg.oxdna_path) / "build" / "bin" / "oxDNA").is_file(),\
        f"oxDNA directory exists at {scfg.oxdna_path} but build/bin/oxDNA executable does not (oxDNA has not been built)"
    assert (Path(scfg.input_file_params["plugin_search_path"]) / f"{scfg.input_file_params['interaction_type']}.so").is_file(),\
        "Patchy interaction has not been compiled"