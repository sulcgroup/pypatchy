import tarfile
import tempfile
import urllib.request
from pathlib import Path

import pytest
from pypatchy.patchy.simulation_ensemble import PatchySimulationEnsemble
from pypatchy.util import cfg

RELEASE_BASE_URL = "https://github.com/sulcgroup/pypatchy-testdata/releases/download/test-data-v1"

# Map ensemble long names to their release asset filenames
RELEASE_ASSETS = {
    "4x4_tile_2023-10-03": "4x4_tile_2023-10-03.tar.gz",
}

def get_data(e: PatchySimulationEnsemble, copy_to: str):
    """
    Downloads test data for the given ensemble from the pypatchy GitHub release
    and extracts it to copy_to.
    """
    long_name = e.long_name()
    if long_name not in RELEASE_ASSETS:
        raise ValueError(
            f"No release asset registered for ensemble '{long_name}'. "
            f"Available: {list(RELEASE_ASSETS.keys())}"
        )

    asset_filename = RELEASE_ASSETS[long_name]
    url = f"{RELEASE_BASE_URL}/{asset_filename}"
    archive_path = Path(copy_to).parent / asset_filename

    print(f"Downloading {asset_filename} from GitHub releases...")
    urllib.request.urlretrieve(url, archive_path)

    print(f"Extracting to {copy_to}...")
    with tarfile.open(archive_path, "r:gz") as tar:
        # Extract into the parent of copy_to; the archive contains the ensemble dir
        tar.extractall(path=Path(copy_to).parent, filter="data")

    archive_path.unlink()
    print("Done.")


@pytest.fixture
def temp_dir() -> str:
    # construct temporary directory to run

    old_data_dir = cfg.get('ANALYSIS', 'simulation_data_dir')
    with tempfile.TemporaryDirectory() as td:
        cfg.set('ANALYSIS', 'simulation_data_dir', str(td))
        try:
            yield td
        finally:
            cfg.set('ANALYSIS', 'simulation_data_dir', old_data_dir)
