"""
Tests all "server configs"
"""
from pathlib import Path
from pypatchy.server_config import list_server_configs, PatchyServerConfig

# params required for input files, which must be specified
reqd_input_file_params = [
    "backend",
    "backend_precision",
    "interaction_type",
    "plugin_search_path"
]


def test_server_configs():
    for config in list_server_configs():
        # check if oxDNA executable exists
        test_config(config)


def test_config(config: PatchyServerConfig):
    if not Path(config.oxdna_path).exists():
        raise Exception(f"Config {config.name} specifies {config.oxdna_path} which"
                        f"does not exist.")
    for paramname in reqd_input_file_params:
        if paramname not in config.input_file_params:
            raise Exception(f"Config {config.name} missing from `input_file_params`.")
    interaction_path = config.input_file_params["plugin_search_path"]
    interaction_name = config.input_file_params["interaction_type"]
    if not (Path(interaction_path) / f"{interaction_name}.so").exists():
        raise Exception(f"Config {config.name}: plugin specified "
                        f"{interaction_path} and {interaction_name}.so does not "
                        f"exist.")
    if config.input_file_params["backend"] not in ("CPU", "CUDA"):
        raise Exception(f"Invalid value for `backend` "
                        f"{config.input_file_params['backend']}")
    # TODO: test writer params?

