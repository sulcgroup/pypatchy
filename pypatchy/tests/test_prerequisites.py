"""
Tests that all prerequisites are installed and that all directory structures exist
"""
import os.path

# test oxpy and oat installation
try:
    import oxDNA_analysis_tools
    import oxpy
except ImportError as e:
    raise ImportError("oxDNA Analysis Tools (oat) python bindings are not installed!\n"
                    "See https://lorenzo-rovigatti.github.io/oxDNA/install.html#python-bindings for more info\n"
                    "Common issues here: oat or oxpy bindings installed in a different python env than pypatchy")

# test installation of ipy_oxdna
try:
    import ipy_oxdna
except ImportError as e:
    raise ImportError("ipy_oxDNA not installed! Install from https://github.com/mlsample/ipy_oxDNA")

# test directory structure
try:
    from pypatchy.util import *
    # input directory
    if not get_input_dir().exists():
        raise Exception("No directory for input spec files! Expected to find directory at "
                        f"{str(get_input_dir())}.")
    # logs directory
    if not get_log_dir().exists():
        raise Exception("No pypatchy logging directory! Expected to find directory at "
                        f"{str(get_input_dir())}.")
    # outputs
    if not get_output_dir().exists():
        raise Exception("No pypatchy output directory! Expected to find directory at "
                        f"{str(get_log_dir())}")
    # simulation run directory
    if not simulation_run_dir().exists():
        raise Exception(f"Specified simulation run directory {str(simulation_run_dir())} does not exist!")
    # TODO: are there more of these?

except ImportError as e:
    e.msg = e.msg + "\nTry running configure.py to automatically set up your pypatchy working" \
                    "directories"
    raise e
