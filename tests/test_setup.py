from os.path import expanduser
from pathlib import Path

import pytest
class TestPyPatchySetup:
    def test_pypatchy_working_directory(self):
        assert Path(expanduser("~/.pypatchy")).is_dir(), \
            "No pypatchy working directory `~/.pypatchy` - try running configure.py"
        assert Path(expanduser("~/.pypatchy/input")).is_dir(), "`~/.pypatchy/input` directory does not exist"
        assert Path(expanduser("~/.pypatchy/output")).is_dir(), "`~/.pypatchy/output` directory does not exist"
        assert Path(expanduser("~/.pypatchy/settings.cfg")).is_file()
