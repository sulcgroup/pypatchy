# create_config.py

# script created by chatGPT
# I hate this solution - if anyone comes up with a better one please lmk

import configparser
import os
import shutil
from pathlib import Path

import pkg_resources


def create_default_config():
    config = configparser.ConfigParser()

    # Add some default values
    config['ANALYSIS'] = {
        "simulation_data_dir": "/scratch/jrevan21/patchysimulations",
        "analysis_data_dir": "/scratch/jrevan21/analysis_space",
        "sample_every": 10,
        "cluster_file": "clusters.txt",
        "export_setting_file_name": "patchy_export_setup.json",
        "init_top_file_name": "init.top",
        "analysis_params_file_name": "analysis_params.json"
    }

    config["SETUP"] = {
        "server_config": "agave_classic"
    }

    # ... add other sections and values as needed ...

    home = Path.home()
    pypatchy_dir = home / '.pypatchy'
    config_file = pypatchy_dir / 'settings.cfg'

    # Create the configuration directory if it doesn't exist
    os.makedirs(pypatchy_dir, exist_ok=True)

    # Only write the configuration file if it doesn't exist
    if not os.path.exists(config_file):
        with open(config_file, 'w') as f:
            config.write(f)

    spec_files_source_dir = 'spec_files'
    spec_files_target_dir = pypatchy_dir / 'spec_files'

    assert pkg_resources.resource_exists('pypatchy', spec_files_source_dir)
    sub_dirs = pkg_resources.resource_listdir('pypatchy', spec_files_source_dir)
    for sub_dir in sub_dirs:
        files_in_sub_dir = pkg_resources.resource_listdir('pypatchy', spec_files_source_dir + '/' + sub_dir)
        for file in files_in_sub_dir:
            file_path_str = os.sep.join([spec_files_source_dir, sub_dir, file])
            source_file_path = pkg_resources.resource_filename('pypatchy', file_path_str)
            target_file_dir = spec_files_target_dir / sub_dir
            target_file_path = target_file_dir / file
            os.makedirs(target_file_dir, exist_ok=True)
            shutil.copyfile(source_file_path, target_file_path)

    os.makedirs(pypatchy_dir / "input", exist_ok=True)
    os.makedirs(pypatchy_dir / "output" / "logs", exist_ok=True)

if __name__ == '__main__':
    create_default_config()
