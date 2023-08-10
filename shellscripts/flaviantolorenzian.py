import sys

from pypatchy.patchy.patchy_scripts import convert_flavian_to_lorenzian

# usage: flaviantolorenzian.py [patch file] [particle file] [top file]

if __name__ == "__main__":
    if len(sys.argv) > 2:
        convert_flavian_to_lorenzian(*sys.argv[1:])
