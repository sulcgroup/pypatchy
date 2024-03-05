"""
Tests basic stuff with loading ensembles
"""
import re
import shutil
from importlib import resources
from pathlib import Path

# check prereqs are ok
import test_prerequisites
from pypatchy.patchy.simulation_ensemble import find_ensemble, PatchySimulationEnsemble
from pypatchy.server_config import PatchyServerConfig

from pypatchy.util import get_input_dir

# copy ensemble spec test files to ~/.pypatchy/input
test_files_path = "spec_files/test_files/test_ensemble_specs"


def copy_ensemble_file(ensemble_name: str):
    """
    Finds an example ensemble spec in package resources pypatchy/patchy/spec_files/test_ensemble_specs
    and copies it to ~/.pypatchy/input

    """
    with resources.path("pypatchy", "spec_files") as resource_path:
        shutil.copy(resource_path / "server_configs" / "test_files" / "test_ensemble_specs" / (ensemble_name + ".json"),
                    get_input_dir())


def test_ensembles(settings: PatchyServerConfig):
    """
    tests all ensembles from specs in pypatchy/patchy/spec_files/test_ensembles
    """
    # run simulations in this process rather than submitting slurm jobs
    settings.is_slurm = False
    with resources.path("pypatchy", "spec_files") as resource_path:
        for fn in resource_path.glob("*"):
            # test ensembles begin with a number and "example", end with ".json
            if re.match(r"\d_example.*\.json", fn.stem):
                copy_ensemble_file(fn.stem)
                e = find_ensemble(fn.stem)
                e.set_server_settings(settings)
                test_ensemble(e)
                # delete copied file
                Path(get_input_dir() / fn.stem).unlink()
                # delete generated metadata file
                Path(get_input_dir() / e.metadata_file).unlink()


def test_ensemble(e: PatchySimulationEnsemble):
    """
    tests a patchy simulation ensemble
    """
    e.start_simulations()
    # delete generated files
    e.tld().rmdir()
