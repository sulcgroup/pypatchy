"""
Tests basic stuff with loading ensembles
"""
import shutil
from importlib import resources

# check prereqs are ok
import test_prerequisites

from pypatchy.util import get_input_dir

# copy ensemble spec test files to ~/.pypatchy/input
test_files_path = "spec_files/test_files/test_ensemble_specs"


def copy_ensemble_file(ensemble_name: str):
    with resources.path("pypatchy", "spec_files") as resource_path:
        shutil.copy(resource_path / "server_configs" / "test_files" / "test_ensemble_specs" / (ensemble_name + ".json"),
                    get_input_dir())

# i'm not precisely sure how to do these tests
