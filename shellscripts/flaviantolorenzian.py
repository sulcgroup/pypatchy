import os
import sys

from pypatchy.patchy.patchy_scripts import convert_flavian_to_lorenzian
from pypatchy.patchy.pl.patchyio import get_writer

# usage: flaviantolorenzian.py [patch file] [particle file] [top file]

if __name__ == "__main__":
    if len(sys.argv) > 2:
        get_writer("flavio").set_directory(os.getcwd())
        get_writer("lorenzo").set_directory(os.getcwd())
        convert_flavian_to_lorenzian(*sys.argv[1:])
