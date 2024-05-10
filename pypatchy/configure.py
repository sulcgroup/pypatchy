# create_config.py

# script created by chatGPT
# I hate this solution - if anyone comes up with a better one please lmk


import configparser
import os
import shutil
from pathlib import Path
from importlib import resources

def create_default_config():
    """
    Creates a default config file. Please modify the config file afterwards!
    """
    config = configparser.ConfigParser()

    # Add some default values
    config['ANALYSIS'] = {
        "simulation_data_dir": "/scratch/jrevan21/patchysimulations",
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
    if not config_file.exists():
        with open(config_file, 'w') as f:
            config.write(f)

    spec_files_source_dir = 'spec_files'
    spec_files_target_dir = pypatchy_dir / 'spec_files'

    # Access the package's resources using importlib.resources
    try:
        with resources.path('pypatchy', spec_files_source_dir) as path:
            sub_dirs = [d for d in os.listdir(path) if (path / d).is_dir()]
            for sub_dir in sub_dirs:
                if sub_dir != "test_files": # skip test_files
                    sub_dir_path = path / sub_dir
                    files_in_sub_dir = os.listdir(sub_dir_path)
                    for file in files_in_sub_dir:
                        source_file_path = sub_dir_path / file
                        target_file_dir = spec_files_target_dir / sub_dir
                        target_file_path = target_file_dir / file
                        os.makedirs(target_file_dir, exist_ok=True)
                        shutil.copyfile(source_file_path, target_file_path)
    except FileNotFoundError:
        print("Specified resource directory does not exist within the package.")

    os.makedirs(pypatchy_dir / "input" / "targets", exist_ok=True)
    os.makedirs(pypatchy_dir / "output" / "logs", exist_ok=True)

if __name__ == '__main__':
    create_default_config()
    # TODO: tests
