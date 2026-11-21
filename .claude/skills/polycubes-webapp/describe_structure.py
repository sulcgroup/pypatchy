#!/usr/bin/env python
"""
Describe a polycube structure the way the polycubes web app's environment panel does, but from
pypatchy: one entry per cube with its type, position, and each coloured patch in world
orientation (face, orientation, colour, allostery variables).

    python describe_structure.py structure.json            # readable table
    python describe_structure.py structure.json --json     # same data as JSON (compare with
                                                           #   render.js --env-json)
    python describe_structure.py structure.json --types    # cube types / the rule string

Accepts anything load_polycube reads (a {"cube_types", "cubes"} file). A path without a directory
is looked up under ~/.pypatchy/input (load_polycube's rule); use ./name.json for the cwd.
Run it in the pypatchy conda env.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from pypatchy.polycubeutil.polycube_structure import load_polycube
from pypatchy.polycubeutil.polycubesRule import RULE_ORDER

FACE_NAMES = ("left", "right", "bottom", "top", "back", "front")


def cube_entries(structure) -> list[dict]:
    """
    One dict per cube, sorted by position (the web app's cube names are random, so position is
    the key to match on). Patches are in world orientation: cube.patches() has already applied the
    cube's rotation, which is what the web panel shows too.
    """
    entries = []
    for cube in structure.particles():
        patches = []
        for patch in cube.patches():
            if patch.colornum() == 0:
                continue
            patches.append({
                "face": FACE_NAMES[patch.dirIdx()],
                "ori": patch.get_align_rot_num(),   # quarter turns about the face normal (0-3)
                "color": int(patch.colornum()),
                "stateVar": patch.state_var() if hasattr(patch, "state_var") else None,
                "actVar": patch.activation_var() if hasattr(patch, "activation_var") else None,
            })
        entries.append({
            "uid": cube.get_uid(),
            "type": f"CT{cube.get_cube_type().type_id()}",
            "position": tuple(int(round(x)) for x in cube.position()),
            "patches": sorted(patches, key=lambda p: FACE_NAMES.index(p["face"])),
        })
    return sorted(entries, key=lambda e: e["position"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("structure")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    parser.add_argument("--types", action="store_true", help="show cube types and the rule string instead")
    args = parser.parse_args()

    path = Path(args.structure)
    structure = load_polycube(path.resolve() if path.exists() else path)

    if args.types:
        print(f"rule string: {structure.rule}")
        counts = Counter(c.get_cube_type().type_id() for c in structure.particles())
        for ct in structure.rule:
            print(f"  CT{ct.type_id()}: {counts.get(ct.type_id(), 0)} cube(s)")
        return

    entries = cube_entries(structure)
    if args.json:
        print(json.dumps(entries, indent=1))
        return
    print(f"{len(entries)} cubes, {len(list(structure.rule))} cube types")
    for e in entries:
        print(f"cube {e['uid']:>3}  {e['type']}  at {e['position']}")
        for p in e["patches"]:
            print(f"     {p['face']:<6} ori {p['ori']}  colour {p['color']:>3}")


if __name__ == "__main__":
    main()
