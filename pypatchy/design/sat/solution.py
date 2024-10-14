from __future__ import annotations

import json
import logging

from pypatchy.design.sat.polycube_sat_problem import PolycubeSATProblem
from pypatchy.structure import *
from pypatchy.util import get_output_dir

from pypatchy.design.solve_utils import compute_coordinates, to_xyz, rot_idx_to_quaternion, quaternion_inverse
import re


class SATSolution:
    # list of variables that are true
    raw: frozenset[int]

    num_species: int
    num_colors: int
    # var C(s,p,c) = patch p on species s has color c
    C_vars: list[str]
    # var O(s,p,o) = patch p on species s has orientation o
    O_vars: list[str]

    rule: PolycubesRule

    # spacial map is a list of tuples where the first value is a particle species, the second is a rotation
    spacial_map: list[tuple[int, int]]
    nanoparticle_map: dict

    coordinate_map: Union[dict[int, np.array], None]

    def __init__(self,
                 sat_problem: PolycubeSATProblem,
                 sat_results: frozenset[int],
                 target_structure: Structure
                 ):
        """
        Constructor.
        Args:
            solver: an instance of Polysat, used to contextualize SAT solver data
            sat_results: a frozenset object containing variables (as numbers) in the solution that are True
        """
        self.raw = sat_results
        # save solver parameters
        self.num_species = sat_problem.nS
        self.num_colors = int(sat_problem.nC / 2 - 1)

        # members for color and orientation variables
        self.C_vars: list[str] = []
        self.O_vars: list[str] = []
        # loop through variables
        colorMap = {}
        self.rule = PolycubesRule(nS=sat_problem.nS)
        # we have to grab the color data and then process it, because otherwise we will
        # not know the color mapping soon enough (i'm so tired)
        p_color_data = []
        # we don't have to grab the patch orientation data and then process it but
        # it makes my life easier
        p_orientation_data = []
        colorCounter = 1
        # species identity, rotation tuple for each position in nL
        self.spatial_map = [(-1, -1) for _ in range(sat_problem.nL)]
        self.nanoparticle_map = {}
        assert all([
            any([
                sat_problem.P(l, s, r) in sat_results
                for s, r in itertools.product(range(sat_problem.nS), range(sat_problem.nR))])
            for l in range(sat_problem.nL)])
        for vname, vnum in sat_problem.list_vars():
            # if variable is True in solution
            if vnum in sat_results:
                # check if this variable is a color match spec
                m = re.match(r"B\((\d*),\s*(\d*)\)", vname)
                if m:
                    c1, c2 = m.groups()
                    assert (c1 not in colorMap or c2 not in colorMap)
                    if int(c1) < 2 or int(c2) < 2:
                        colorMap[c1] = 0
                        colorMap[c2] = 0
                    else:
                        colorMap[c1] = colorCounter
                        colorMap[c2] = -colorCounter
                        colorCounter += 1
                    continue

                # check if this variable is a patch color specifier
                m = re.match(r"C\((\d*),\s*(\d*),\s*(\d*)\)", vname)
                if m:
                    # save patch color data - have to apply after read pass so B(c1,c2) var data
                    # can be used to construct a rule
                    p_color_data.append(m.groups())
                    self.C_vars.append(vname)
                    continue

                # check if this variable is a position specifier
                m = re.match(r"P\((\d*),\s*(\d*),\s*(\d*)\)", vname)
                if m:
                    location, species, rotation = m.groups()
                    species = int(species)
                    self.spatial_map[int(location)] = (species, int(rotation))
                    continue

                # check if this variable is a patch orientation specifier
                if sat_problem.torsionalPatches:
                    m = re.match(r"O\((\d*),\s*(\d*),\s*(\d*)\)", vname)
                    if m:
                        p_orientation_data.append(m.groups())
                        self.O_vars.append(vname)
                        continue

                # check for nanoparticles
                if sat_problem.nNPT > 0:
                    # if there's 1 nanoparticle type
                    if sat_problem.nNPT == 1:
                        m = re.match(r"N\((\d*)\)", vname)
                        if m:
                            self.nanoparticle_map[int(m.groups()[0])] = 1
                            continue
                    # if there's more than 1 nanoparticle type
                    else:
                        m = re.match(r"N\((\d*),\s*(\d*)\)", vname)
                        if m:
                            species, nptype = m.groups()
                            self.nanoparticle_map[int(species)] = int(nptype)
                            continue

        assert all(species != -1 and rotation != -1 for species, rotation in self.spatial_map)

        # apply patch color data
        for s, p, c in p_color_data:
            particle_type_idx = int(s)
            patch_direction = RULE_ORDER[int(p)]
            if not self.rule.particle(particle_type_idx).has_patch(patch_direction):
                self.rule.add_particle_patch(particle_type_idx, PolycubesPatch(
                    uid=None,  # idx will be assigned in add method
                    color=colorMap[c],
                    direction=patch_direction,
                    orientation=get_orientation(int(p), 0)
                ))

        # if applicable, apply patch orientation data
        if len(p_orientation_data) > 0:
            for s, p, o in p_orientation_data:  # Patch on species l has orientation o
                # print("Patch {} on species {} has orientation {}".format(p, s, o))
                species_idx = int(s)
                direction_idx = int(p)
                rotation = int(o)
                self.rule.particle(species_idx).get_patch_by_diridx(direction_idx).set_align_rot(rotation)

        # construct map of location indexes to coordinates in 3-space....
        if not target_structure.is_multifarious() and not target_structure.is_crystal():
            self.coord_map = compute_coordinates(target_structure.bindings_list)
        else:
            self.coord_map = None

    def decRuleNew(self):
        return str(self.rule)

    def decRuleOld(self):
        return "_".join("|".join(
            f"{ct.get_patch_by_diridx(idx).color()}:{ct.get_patch_by_diridx(idx).get_align_rot_num()}"
            for idx, _ in enumerate(RULE_ORDER)
        ) for ct in self.rule.particles())

    def hasNanoparticles(self):
        return len(self.nanoparticle_map) > 0

    def numNanoparticleTypes(self):
        return np.unique(self.nanoparticle_map.values()).size - 1

    def printToLog(self, logger: logging.Logger=logging.root):
        logger.info("-----------------")
        logger.info(f"\tNum Species: {self.num_species}")
        logger.info(f"\tNum Colors: {self.num_colors}")
        logger.info(f"\tRule: {self.decRuleNew()}")
        logger.info(f"\tSpatial Map: {self.spatial_map}")
        if self.hasNanoparticles():
            logger.info(f"\tNum Nanoparticle Types: {self.numNanoparticleTypes()}")
            for nptype in range(self.numNanoparticleTypes()):
                logger.info(f"\t\tType {nptype + 1}: {[i for i, ct in enumerate(self.rule.particles()) if self.nanoparticle_map[i] == nptype + 1]}")

        logger.info("-----------------")

    def type_counts(self) -> list[int]:
        tyoe_counts = [0 for _ in range(self.num_species)]
        for species, _ in self.spatial_map:
            tyoe_counts[species] += 1
        return tyoe_counts

    def exportScene(self, modelname="scene"):
        data = {
            'cube_types': self.rule.toJSON(),
            'cubes': [
                self.cubeToJSON(i) for i in range(len(self.spatial_map))
            ]
        }
        if modelname.startswith("/"):
            p = Path(modelname)
        elif modelname.startswith("~"):
            p = Path.expanduser(modelname)
        elif modelname.endswith(".json"):
            p = get_output_dir() / "SAT" / modelname
        else:
            p = get_output_dir() / "SAT" / f"{modelname}_{self.num_species}S_{self.num_colors}C.json"
        try:
            with p.open("w+") as f:
                json.dump(data, f, indent=4)
        except FileNotFoundError as e:
            print(e.strerror)

    def cubeToJSON(self, i: int) -> dict:
        """
        Exports spacial data on a cube
        Args:
            i: the location index (in self.spacial_map) of the cube to export

        Returns:
            a dict of data for the spacial location of a cube
        """
        ctid, ctrot = self.spatial_map[i]

        return {
            "position": to_xyz(self.coord_map[i]),
            "rotation": {
                k: float(v) for k, v in zip(
                    ["w", "x", "y", "z"],
                    quaternion_inverse(rot_idx_to_quaternion(ctrot)))
                         },
            "state": [True, *([False] * 12)],  # ignore for now TODO come back to
            "type": ctid
        }

    def has_coord_map(self):
        return self.coord_map is not None
