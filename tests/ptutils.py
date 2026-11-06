import importlib.util
import os
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import pytest
from pypatchy.patchy.pl.oxpy_energy import OxpyEnergyModel
from pypatchy.patchy.simulation_ensemble import PatchySimulationEnsemble
from pypatchy.util import cfg

# oxDNA checkout whose contrib plugins the energy tests load; override with the OXDNA_PATH environment variable
OXDNA_PATH = Path(os.environ.get("OXDNA_PATH", "/root/Github/oxDNA"))

# (writer format, contrib plugin dir, interaction) for each patchy model pypatchy supports
ENERGY_MODELS = {
    "flavio": ("romano", "PatchyShapeInteraction"),
    "lorenzo": ("rovigatti", "DetailedPatchySwapInteraction"),
    "raspberry": ("evans", "RaspberryPatchyInteraction"),
}


def plugin_available(patchy_format: str) -> bool:
    plugin_dir, interaction = ENERGY_MODELS[patchy_format]
    return importlib.util.find_spec("oxpy") is not None and \
        (OXDNA_PATH / "contrib" / plugin_dir / f"{interaction}.so").exists()


def requires_energy_model(patchy_format: str):
    """
    skip marker for tests that compute energies with the given model's oxDNA plugin
    """
    plugin_dir, interaction = ENERGY_MODELS[patchy_format]
    return pytest.mark.skipif(not plugin_available(patchy_format),
                              reason=f"needs oxpy and {interaction}.so in {OXDNA_PATH / 'contrib' / plugin_dir}")


def energy_model(patchy_format: str, **params) -> OxpyEnergyModel:
    """
    Builds an OxpyEnergyModel for one of pypatchy's patchy formats, with the same defaults pypatchy's
    default param set (spec_files/input_files/default.json) gives ensembles. Keyword args override them.
    """
    plugin_dir, interaction = ENERGY_MODELS[patchy_format]
    input_params = {
        "interaction_type": interaction,
        "plugin_search_path": str(OXDNA_PATH / "contrib" / plugin_dir),
        "T": 0.1,
    }
    if patchy_format == "flavio":
        input_params.update(PATCHY_alpha=0.12, PATCHY_radius=0.5, shape="sphere", use_torsion=1,
                            same_type_bonding=1, no_multipatch=1, interaction_tensor=0)
    elif patchy_format == "raspberry":
        input_params.update(PATCHY_alpha=0.12, narrow_type=0)
    elif patchy_format == "lorenzo":
        input_params.update(DPS_sigma_ss=0.082)
    input_params.update(params)
    return OxpyEnergyModel(patchy_format, input_params)

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
