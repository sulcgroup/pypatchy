import sys
from pypatchy.patchy.patchy_scripts import convert_udt_files_to_mdt

# usage: UDtoMDt [patch file] [particle file] [ [dental radius] [num teeth] [follow surface]]
if __name__ == "__main__":
    if len(sys.argv) > 2:
        convert_udt_files_to_mdt(*sys.argv[1:])