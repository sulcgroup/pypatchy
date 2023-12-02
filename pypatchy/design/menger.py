import sys
from copy import deepcopy

from pypatchy.polycubeutil.polycube_structure import PolycubeStructure


from pypatchy.structure import Structure
from pypatchy.util import get_output_dir

"""
Designs menger sponge topology / particle set
"""

def menger_crystal(n: int) -> PolycubeStructure:
    # base case: n=0, one cube
    if n == 0:

    else:
        # find smaller subunit
        ps = menger_crystal(n - 1)
        ps_vert = deepcopy(ps)
        ps_edge = deepcopy(ps)

def menger_cube(n: int) -> PolycubeStructure:
    # base case: n=0, one cube
    if n == 0:

    else:
        # find smaller subunit
        ps = menger_cube(n - 1)
        ps_vert = deepcopy(ps)
        ps_edge = deepcopy(ps)

if __name__ == "__main__":
    n = sys.argv[1]
    if len(sys.argv) > 2:
        write_path = get_output_dir() / sys.argv[2]
    else:
        write_path = get_output_dir() / f"menger_{n}.json"

