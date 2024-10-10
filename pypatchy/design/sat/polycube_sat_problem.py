import itertools
from collections.abc import Generator
from pathlib import Path
from typing import Union, IO

import numpy as np

from pypatchy.design.sat.sat_problem import SATProblem, SATProblemPart, SATClause, exactly_one
from pypatchy.design.solve_utils import patchRotToVec, getFlatFaceRot, getIndexOf, patchVecToRot
from pypatchy.polycubeutil.polycubesRule import RULE_ORDER
from pypatchy.structure import Structure
from pypatchy.util import enumerateRotations


class PolycubeSATProblem(SATProblem):
    """
    represents a SAT problem relating on a polycube on a cubic lattice
    """

    # ------------------------- SAT Problem... hyperparameters? ------------------------------
    # number of locations
    nL: int
    # number of patches allowed per species
    nP: int
    # number of orientations allowed for a patch
    nO: int
    # number of nanoparticle types
    nNPT: int
    # number of rotations
    nR: int = property(lambda self: len(self.rotations))

    # list of all possible rotations (dependant on dimensionality, for now it's fixed) TODO: better
    rotations: dict[int: dict[int, int]]
    torsionalPatches: bool

    # list of "basic" SAT clauses
    basic_sat_parts: list[SATProblemPart]

    def get_basic_sat_clauses(self) -> Generator[SATClause, None, None]:
        for problem_part in self.basic_sat_parts:
            yield from problem_part
    basic_sat_clauses: property(get_basic_sat_clauses)

    def __init__(self,
                 num_species: int,
                 num_colors: int,
                 num_dimensions: int,
                 num_lattice_locations: int,
                 num_nanoparticle_types: int,
                 torsion: bool):
        super().__init__()
        # set up design parameters, but not the structure
        self.nL = num_lattice_locations
        self.nP = 6  #: Number of patches on a single particle
        self.torsionalPatches = torsion
        if self.torsionalPatches:
            self.nO = 4  #: Number of possible orientations for a patch, N,S,W,E
        self.nS = num_species
        self.nC = num_colors
        self.nD = num_dimensions
        self.nNPT = num_nanoparticle_types
        self.BCO_varlen = 0  # the number of clauses that determine B and C

        self.rotations = enumerateRotations()


    def get_solution_var_names(self, solution: set[int]) -> list[str]:
        return [vname for vname, vnum in self.variables.items() if vnum in solution]

    def add_constraints_all_particles(self):
        """
        Adds SAT clauses specifying that the solution must use all particle types
        for each species, we must have at least one position where the species is present (at any rotation)
        """
        self.add_problem_part(SATProblemPart("^S", ["l", "s", "r"]))
        for s in range(self.nS):
            self.sat_problem_parts["^S"] .clauses.append([self.P(l, s, r)
                                                         for l in range(self.nL)
                                                         for r in range(self.nR)])

    def add_constraints_all_patches(self):
        """
        adds SAT clauses specifying that the solution must use all patch types
        Returns:

        """
        self.add_problem_part(SATProblemPart("^C", ["s", "p", "c"]))
        for c in range(self.nC):
            self.sat_problem_parts["^C"].clauses.append([self.C(s, p, c)
                                                         for s in range(self.nS)
                                                         for p in range(self.nP)])

    def add_constraints_all_patches_except(self,
                                           forbidden: list[int],
                                           nonRequired: list[int] = [1]):
        """
        Adds constraints which require the solution to include all patches, w/ exceptions
        Colors in forbidden cannot be used in the solution at all, colors in nonRequired can be used but don't have to be
        """
        self.add_problem_part(SATProblemPart("^C", ["s", "p", "c"]))
        # loop colors
        for c in range(self.nC):
            # skip patches that aren't in either "forbidden" or "nonrequired"
            if c not in forbidden and c not in nonRequired:
                self.sat_problem_parts["^C"].clauses.append([self.C(s, p, c) for s in range(self.nS) for p in range(self.nP)])
            # Do not use forbidden color
            # for all patches p, species s, nForbidden of our list of forbidden colors
            for p, s, nForbidden in itertools.product(range(self.nP), range(self.nS), forbidden):
                self.sat_problem_parts["^C"].clauses.append(
                    [-self.C(s, p, nForbidden)]
                )

    def add_constraints_fixed_blank_orientation(self):
        """
        hardcodes blank patch orientations to 0
        otherwise we'd have a lot of redundant sat solutions to wade through
        """
        self.add_problem_part(SATProblemPart("|P", ["s", "p"]))
        for (s, p) in itertools.product(range(self.nS), range(self.nP)):
            self.sat_problem_parts["|P"].clauses.append((
                -self.C(s, p, 1),  # Either patch p on species s isn't empty
                self.O(s, p, 0)  # Or the patch is oriented up
            ))

    def add_constraints_no_self_complementarity(self, above_color=0):
        """
        forbids colors from being complimentary to themselves
        """
        self.add_problem_part(SATProblemPart("^CC", ["c"]))
        for c in range(above_color, self.nC):
            self.sat_problem_parts["^CC"].clauses.append([-self.B(c, c)])

    def fix_particle_colors(self, ptype: int, sid: int, cid: int):
        """
        requires patch ptype on species s to have color cid
        this function is never used and is inherited from JB, I have preserved it on
        the grounds that it may someday be useful
        """
        self.add_problem_part(SATProblemPart("SC", ["p", "s", "c"]))
        self.sat_problem_parts["SC"].clauses.append([self.C(sid, ptype, cid)])

    def fix_empties(self, top: Structure):
        """
        forces empty-binding colors to be 1
        """
        self.sat_problem_parts["E"] = SATProblemPart("E", ["l", "p"])
        for particle, patch in top.get_empties():
            self.sat_problem_parts["E"].clauses.append(self.fix_slot_colors(particle, patch, 1))
            # print("Particle {} patch {} should be empty".format(particle, patch))

    def fix_slot_colors(self, loc: int, p: int, cid: int) -> SATClause:
        """
        Forces patch p at locaion l to have color cid
        useful for fixing empties, may have other applications in more bespoke SAT designs
        """
        return [self.F(loc, p, cid)]

    def fix_all_color_interactions(self):
        """

        """
        self.add_problem_part(SATProblemPart("I", ["c{1}", "c{2}"]))
        for c in range(2, self.nC - 1, 2):
            self.sat_problem_parts["I"].clauses.append(self.fix_color_interaction(c, c + 1))

    def fix_color_interaction(self, c1: int, c2: int) -> SATClause:
        """
        hardcodes color c1 to interact with color c2
        """
        return [self.B(c1, c2)]

    def generate_BCO(self):
        """
        constructs all nessecary variables for vars B, C, and O
        """
        # make sure B, C and O vars are first:
        # var B(c1, c2) = color c1 is compatible with C2
        for (c1, c2) in itertools.product(range(self.nC), range(self.nC)):
            self.B(c1, c2)
        # var C = patch p on species s has color c
        for s, p, c in itertools.product(range(self.nS), range(self.nP), range(self.nC)):
            self.C(s, p, c)
        # var O = patch p on species s has color c
        if self.torsionalPatches:
            for s, p, o in itertools.product(range(self.nS), range(self.nP), range(self.nO)):
                self.O(s, p, o)

    def gen_legal_color_bindings(self):
        """
        Adds clause i (see https://doi.org/10.1021/acsnano.2c09677) - Legal color bindings:
        "Each color has exactly one color that it binds to"
        forall c1 exactly one c2 s.t. B(c1, c2)
        Returns: a list of CNF clauses that together represent clause i
        """
        constraints = []
        # loop colors, skipping color 0
        self.add_problem_part(SATProblemPart("B", ["c{1}", "c{2}"]))
        for c1 in range(self.nC):
            # each color should bind w/ exactly one other color
            constraints.extend(exactly_one([self.B(c1, c2) for c2 in range(self.nC)]))

        self.sat_problem_parts["B"].clauses = constraints

    def gen_legal_species_coloring(self):
        """
        Adds CNF clauses for Clause ii in the manuscript
        Legal species patch coloring (unnecesay, implied by "Legal species coloring in positions" and "Legal position
                ^^ (NOTE: this is a comment from Joakim's code; I have no idea what he means by it)
        patch coloring"):
        "Each patch on every species has exactly one color"
        forall s, forall p, exactly one c p.t. C(s, p, c)
        Returns: a list of CNF clauses for Clause ii
        """

        self.add_problem_part(SATProblemPart("C", ["s", "p", "c{k}", "c{l}"]))

        constraints = []
        for (s, p) in itertools.product(range(self.nS), range(self.nP)):
            constraints.extend(exactly_one([self.C(s, p, c) for c in range(self.nC)]))
        self.sat_problem_parts["C"].clauses = constraints

    def gen_legal_species_orientation(self):
        """
        Adds clause iix (viii?)
        Legal species patch orientation
        "Each patch on every species has exactly one orientation"
        forall s, forall p, exactly one o p.t. O(s, p, o)
        Returns: a list of SAT clauses representing clause iix (viii?)
        """
        self.add_problem_part(SATProblemPart("O", ["s", "p", "o"]))
        constraints = []
        if self.torsionalPatches:
            for s, p in itertools.product(range(self.nS), range(self.nP)):
                constraints.extend(exactly_one([self.O(s, p, o) for o in range(self.nO)]))
        self.sat_problem_parts["O"].clauses = constraints

    def gen_legal_patch_coloring(self):
        """
        Adds clause... I don't think this one is described in the paper?
        It may be an implicit result of clauses ii and v
        Legal position patch coloring:
        "Every position patch has exactly one color"
        for all l, p exactly one c st. F(l, p, c)
        Returns: a list of CNF clauses that prevent any position patch from having more than one color
        """
        self.add_problem_part(SATProblemPart("F", ["l", "p", "c"]))
        constraints = []
        for (l, p) in itertools.product(range(self.nL), range(self.nP)):
            constraints.extend(exactly_one([self.F(l, p, c) for c in range(self.nC)]))
        self.sat_problem_parts["F"].clauses = constraints

    def gen_legal_position_patch_orientation(self):
        """
        Adds clause... I don't think this one is explicitly described in the paper?
        It may be an implicit result of clauses iix (viii) and x
        Legal position patch orientation:
        "Every position patch has exactly one orientation"
        for all l, p exactly one o st. A(l, p, o)
        Returns: a list of CNF clauses that prevent any position patch from having more than one orientation
        """

        self.add_problem_part(SATProblemPart("A", ["l", "p", "o"]))

        constraints = []
        if self.torsionalPatches:
            for l, p in itertools.product(range(self.nP), range(self.nP)):
                constraints.extend(exactly_one([self.A(l, p, o) for o in range(self.nO)]))
        self.sat_problem_parts["A"].clauses = constraints

    def gen_forms_desired_structure(self, top: Structure):
        """
        adds sat clauses to design the desired structure

        Implements Clauses iv and ix in the paper
        Clause iv states that every two patches that bind to each other must have
        complimentary colors
        Clause ix states that every two patches that bind to each other must have
        compatible orientations
        Returns: a list of SAT clauses expressing Clauses iv and ix in the paper
        """

        # Forms desired crystal:
        # "Specified binds have compatible colors"
        # forall (l1, p1) binding with (l2, p2) from crystal spec:
        # forall c1, c2: F(l1, p1, c1) and F(l2, p2, c2) => B(c1, c2)
        self.add_problem_part(SATProblemPart("BF",
                                                      ["l{i}",
                                                       "p{i}",
                                                       "c{i}",
                                                       "l{j}",
                                                       "p{j}",
                                                       "c{j}"]))

        # - Forms desired crystal:
        # "Specified binds have compatible orientations"
        # 	forall (l1, p1) binding with (l2, p2) from crystal spec:
        # 		forall o1, o2: A(l1, p1, o1) and A(l2, p2, o2) => D(c1, c2)
        self.add_problem_part(SATProblemPart("DA",
                                                      ["l{i}",
                                                       "p{i}",
                                                       "o{i}",
                                                       "l{j}",
                                                       "p{j}",
                                                       "o{j}"]))

        for l1, p1, l2, p2 in top.bindings_list:
            # color matches
            for c1, c2 in itertools.product(range(self.nC), range(self.nC)):
                self.sat_problem_parts["BF"].clauses.append((-self.F(l1, p1, c1),
                                                             -self.F(l2, p2, c2),
                                                             self.B(c1, c2)))
            # species matches
            for o1, o2 in itertools.product(range(self.nO), range(self.nO)):
                self.sat_problem_parts["DA"].clauses.append((-self.A(l1, p1, o1),
                                                             -self.A(l2, p2, o2),
                                                             self.D(p1, o1, p2, o2)))

    def gen_hard_code_orientations(self):
        """
        Clause ix?
        Returns: A list of CNF clauses expressing Clause ix (i think?) from the paper

        """
        self.add_problem_part(SATProblemPart("D",
                                                      ["l{i}",
                                                       "p{i}",
                                                       "c{i}",
                                                       "l{j}",
                                                       "p{j}",
                                                       "c{j}"]))
        constraints: list[SATClause] = []
        # Hard-code patch orientations to bind only if they point in the same direction
        if self.torsionalPatches:
            for p1, p2 in itertools.combinations(range(self.nP), 2):
                for o1, o2 in itertools.product(range(self.nO), range(self.nO)):
                    v1 = patchRotToVec(p1, o1)
                    v2 = patchRotToVec(p2, o2)
                    # Do they point in the same global direction?
                    # And do the patches face each other?
                    if np.array_equal(v1, v2) and p2 % 2 == 0 and p2 + 1 == p1:
                        constraints.append([self.D(p1, o1, p2, o2)])
                        # print("patch {}, orientation {} binds with patch {}, orientation {}".format(p1, o1, p2, o2))
                    else:
                        constraints.append([-self.D(p1, o1, p2, o2)])
        self.sat_problem_parts["D"].clauses = constraints

    def gen_legal_species_placement(self):
        """
        Adds CNF clauses corresponding to SAT clause iii from the paper
        Returns: a list of CNF clauses expressing SAT clause iii

        """
        # - Legal species placement in positions:
        # "Every position has exactly one species placed there with exactly one rotation"
        #   forall l: exactly one s and r p.t. P(l, s, r)
        self.add_problem_part(SATProblemPart("P",
                                                     ["l",
                                                      "s{i}",
                                                      "r{i}",
                                                      "s{j}",
                                                      "r{j}"]))
        constraints: list[SATClause] = []
        for l in range(self.nL):
            # for all possible species, rotation pairs
            srs = itertools.product(range(self.nS), range(self.nR))
            location_vars = [self.P(l, s, r) for s, r in srs]
            constraints.extend(exactly_one(location_vars))
        self.sat_problem_parts["P"].clauses = constraints

    def gen_legal_species_position_coloring(self):
        """
        Adds clause v from the paper
        Clause v states that all patches in the lattice are colored correspondingly to
        their occupying species s, rotated by r
        Returns: A list of CNF clauses expressing SAT clause v

        """
        # - Legal species coloring in positions:
        # "Given a place, species and its rotation, the patch colors on the position and (rotated) species must be the same"
        #   for all l, s, r:
        #       P(l, s, r) => (forall p, c: F(l, p, c) <=> C(s, rotation(p, r), c))

        self.add_problem_part( SATProblemPart("rotC", ["l", "s", "r", "p", "c"]))
        constraints = []
        # for each position l in the lattice
        for (l, s, r, p, c) in itertools.product(range(self.nL),
                                                 range(self.nS),
                                                 range(self.nR),
                                                 range(self.nP),
                                                 range(self.nC)):
            p_rot = self.rotation(p, r)  # Patch after rotation
            # Species 's' rotated by 'r' gets color 'c' moved from patch 'p' to 'p_rot':
            constraints.append((
                -self.P(l, s, r),  # EITHER no species 's' at position 'l' with rot 'r'
                -self.F(l, p, c),  # OR no patch 'p' at position 'l' with color 'c'
                self.C(s, p_rot, c)  # OR patch 'p_rot' on species 's' DOES have the color 'c'
            ))
            constraints.append((
                -self.P(l, s, r),  # EITHER no species 's' at position 'l' with rot 'r'
                self.F(l, p, c),  # OR there is a patch 'p' at position 'l' with color 'c'
                -self.C(s, p_rot, c)  # OR there is no patch 'p_rot' on species 's' with the color 'c'
            ))
        self.sat_problem_parts["rotC"].clauses = constraints

    def gen_legal_species_patch_orientation(self):
        """
        Adds clause x from the paper
        Clause x states that each patch in the lattice must have an orientation corresponding to the
        patch on the species occupying that position, accounting for rotation
        Returns: a list of CNF clauses expressing clause x from the paper

        """
        # - Legal species patch orientation in positions:
        # "Given a place, species and its rotation, the patch orientations on the position and (rotated) species must be correct"

        self.add_problem_part(SATProblemPart("rotO", ["l", "s", "r", "p", "o"]))

        constraints = []
        if self.torsionalPatches:
            for (l, s, r, p, o) in itertools.product(range(self.nL),
                                                     range(self.nS),
                                                     range(self.nR),
                                                     range(self.nP),
                                                     range(self.nO)):
                p_rot = self.rotation(p, r)  # Patch after rotation
                o_rot = self.orientation(p, r, o)  # Patch orientation after rotation
                # Species 's' rotated by 'r' gets orientation 'o' of patch 'p' changed to 'o_rot' at the new path 'p_rot':
                # print("Species {} rotated by {}: patch {}-->{}, orientation {}-->{}".format(s, r, p, p_rot, o, o_rot))
                constraints.append((
                    -self.P(l, s, r),  # EITHER no species 's' at position 'l' with rot 'r'
                    -self.A(l, p, o),  # OR no patch 'p' at position 'l' with orientation 'o'
                    self.O(s, p_rot, o_rot)
                    # OR patch 'p_rot' on species 's' has the orientation 'o_rot'
                ))
                constraints.append((
                    -self.P(l, s, r),  # EITHER no species 's' at position 'l' with rot 'r'
                    self.A(l, p, o),  # OR there is a patch 'p' at position 'l' with orientation 'o'
                    -self.O(s, p_rot, o_rot)
                    # OR there is no patch 'p_rot' on species 's' with the orientation 'o_rot'
                ))
        self.sat_problem_parts["rotO"].clauses = constraints

    def gen_lock_patch_orientation(self):
        """
        I have no idea what this does
        It appears to only apply to 2D shapes
        Returns: a list of CNF clauses of some sort, if 2D, else an empty list

        """
        self.add_problem_part(SATProblemPart("lock", ["s", "p", "o"]))
        constraints = []
        if self.nD == 2:
            # Lock patch orientation if 2D
            for (s, p) in itertools.product(range(self.nS), range(self.nP)):
                # v = np.array([0, 1, 0])
                # o = utils.patchVecToRot(p, v)
                if p > 3:
                    # Patch p on species s is empty
                    constraints.append([self.C(s, p, 1)])
                o = getFlatFaceRot()[p]
                # Patch p has orientation 'o'
                constraints.append([self.O(s, p, o)])
        self.sat_problem_parts["lock"].clauses = constraints

        # OPTIONAL:
        # assign colors to all slots, if there are enough of them - MUCH FASTER
        # assert self.nS * self.nP == self.nC
        # c = 0
        # for s in range(self.nS):
        #    for p in range(self.nP):
        #        constraints.append([self.C(s, p, c)])
        #        c += 1
        # assert c == self.nC

        # symmetry breaking a little bit....
        # constraints.append([self.F(0, 0, 0)])
        # constraints.append([self.P(0, 0, 0)])

    def gen_nanoparticle_singleparticle(self, np_locations: list[int]):
        """
        generates nanoparticle clauses for a system with only one nanoparticle type
        Returns: a list of SAT clauses
        """
        constraints = []
        self.add_problem_part(SATProblemPart("NPs",
                                                       ["l", "s", "r"]))
        # for each location
        for l in range(self.nL):
            # if location l has a nanoparticle
            hasParticle = l in np_locations  # eval now to save time
            constraints.extend(itertools.chain.from_iterable(
                [
                    [
                        [self.N_single(s), -self.P(l, s, r)]
                        if hasParticle else
                        [-self.N_single(s), -self.P(l, s, r)]
                        for s in range(self.nS)
                    ]
                    for r in range(self.nR)]
            ))
        self.sat_problem_parts["NPs"].clauses = constraints

    def gen_nanoparticle_multiparticle(self, np_locations: dict[int, int]):
        """
        generates nanoparticle clauses for a system with multiple nanoparticle types
        Returns: a list of SAT clauses

        """
        assert self.nNPT > 1, "You're trying to add multi-type nanoparticle clauses for a system with only" \
                              " one nanoparticle type! Please reformat your setup json"

        self.add_problem_part(SATProblemPart("N", ["s", "t"]))
        # generate disjointitiy clause - each species s has exactly one nanoparticle type
        for s in range(self.nS):
            self.sat_problem_parts["N"].clauses.extend(exactly_one([
                self.N_multi(s, t) for t in range(self.nNPT)
            ]))

        self.add_problem_part(SATProblemPart("NPs",
                                                       ["l", "s", "r"]))

        # occupancy clause - each position must have its correct nanoparticle type
        for l in range(self.nL):
            # if location l has a nanoparticle
            nptype = np_locations[l]  # eval now to save time
            self.sat_problem_parts["NPs"].clauses.extend(itertools.chain.from_iterable(
                [
                    [
                        [self.N_multi(s, nptype), -self.P(l, s, r)]
                        for s in range(self.nS)
                    ]
                    for r in range(self.nR)]
            ))

    def gen_no_self_interact(self):
        """
        generates SAT problem part which states that colors cannot self-interact
        """
        constraints = []
        # for each species
        self.add_problem_part(SATProblemPart("~B", ["s", "c{1}", "c{2}", "p"]))
        for s in range(self.nS):
            # for each pair of colors
            for c1, c2 in itertools.combinations(range(self.nC), 2):
                if c1 == c2:
                    continue  # colors should never be self-interacting so we can safely ignore
                # either the colors do not interact
                var_no_interact = -self.B(c1, c2)
                for p in range(self.nP):
                    # this clause was written by chatGPT so I'm only like 70% sure I trust it
                    # The clause says: either var_no_interact is True (colors do not interact)
                    # or at least one patch is not c1 or at least one patch is not c2 for the species
                    constraints.append([var_no_interact, -self.C(s, p, c1), -self.C(s, p, c2)])

        self.sat_problem_parts["~B"].clauses = constraints

    def load_BC_constraints_from_text_sol(self, sol_file: Path, append=True) -> list[SATClause]:
        """
        I have never used this method and don't know that anyone has
         loads solution from written output (such as B(1,3), one clause per line) in myinput handle
         """
        with sol_file.open("r") as myinput:
            lines = [line.strip() for line in myinput.readlines()]
            new_constraints = []
            for vname in lines:
                if 'B' in vname or 'C' in vname:
                    new_constraints.append([self.variables[vname]])
                    # print(vname)
            if append:
                self.add_problem_part(SATProblemPart(sol_file.stem, []))
                # print 'Addding',new_constraints, 'to', self.basic_sat_clauses
                self.sat_problem_parts[sol_file.stem].clauses = new_constraints
                # print self.basic_sat_clauses
            return new_constraints

    # SAT variable methods
    def rotation(self, p: int, r: int) -> int:
        """ patch that p rotates to under rotation r """
        assert 0 <= p < self.nP
        assert 0 <= r < self.nR
        # assert all(len(set(rotations[r].keys())) == nP for r in rotations)
        # assert all(len(set(rotations[r].values())) == nP for r in rotations)
        assert len(self.rotations) == self.nR
        assert r in self.rotations
        assert p in self.rotations[r]
        return self.rotations[r][p]

    def B(self, c1: int, c2: int) -> int:
        """ color c1 binds with c2 """
        if c2 < c1:
            c1, c2 = c2, c1
        assert 0 <= c1 <= c2 < self.nC
        return self.variable("B", c1, c2)

    def D(self, p1: int, o1: int, p2: int, o2: int) -> int:
        """ patch p1, orientation o1 binds with patch p2, orientation o2 """
        if p2 < p1:
            o1, o2 = o2, o1
            p1, p2 = p2, p1
        assert 0 <= p1 <= p2 < self.nP
        assert 0 <= o1 < self.nO
        assert 0 <= o2 < self.nO
        return self.variable("D", p1, o1, p2, o2)

    def F(self, l: int, p: int, c: int) -> int:
        """ patch p at position l has color c """
        assert 0 <= l < self.nL
        assert 0 <= p < self.nP
        assert 0 <= c < self.nC
        return self.variable("F", l, p, c)

    def A(self, l: int, p: int, o: int) -> int:
        """ patch p at position l has orientation o """
        assert 0 <= l < self.nL
        assert 0 <= p < self.nP
        assert 0 <= o < self.nO
        return self.variable("A", l, p, o)

    def C(self, s: int, p: int, c: int) -> int:
        """ patch p on species s has color c """
        assert 0 <= s < self.nS
        assert 0 <= p < self.nP
        assert 0 <= c < self.nC
        return self.variable("C", s, p, c)

    def O(self, s: int, p: int, o: int) -> int:
        """ patch p on species s has orientation o """
        assert 0 <= s < self.nS
        assert 0 <= p < self.nP
        assert 0 <= o < self.nO
        return self.variable("O", s, p, o)
        # return self.variables.setdefault(f'O({s},{p},{o})', len(self.variables) + 1)

    def P(self, l: int, s: int, r: int) -> int:
        """ position l is occupied by species s with rotation r """
        assert 0 <= l < self.nL
        assert 0 <= s < self.nS
        assert 0 <= r < self.nR
        return self.variable("P", l, s, r)

    def orientation(self, p: int, r: int, o: int) -> int:
        """ new orientation for patch p with initial orientation o after getting rotated by r """
        assert 0 <= p < self.nP
        assert 0 <= r < self.nR
        assert 0 <= o < self.nO
        assert len(self.rotations) == self.nR
        assert r in self.rotations
        assert p in self.rotations[r]

        # Calculate patch that p rotates to:
        p_rot = self.rotations[r][p]
        # Calculate vector corresponding to o
        v = patchRotToVec(p, o)
        # Which p has the same vector as o?
        p_temp = getIndexOf(v, RULE_ORDER)
        assert (p != p_temp)
        # How would that p get rotated?
        p_temp_rot = self.rotations[r][p_temp]
        assert (p_rot != p_temp_rot)
        # And what vector does that correspond to?
        v_rot = RULE_ORDER[p_temp_rot]
        # And what orientation value does that give us?
        return patchVecToRot(p_rot, v_rot)

    ## NANOPARTICLE SAT VARIABLES ##
    def N_single(self, s: int) -> int:
        """species s has a nanoparticle
        Args:
            s: species

        Returns:
            index of variable N(s)
        """
        assert -1 < s < self.nS
        return self.variable("N", s)

    def N_multi(self, s: int, t: int) -> int:
        """
        species s has a nanoparticle of type t
        Args:
            s: species
            t: nanoparticle type

        Returns:
        """
        assert -1 < s < self.nS
        assert -1 < t < self.nNPT
        return self.variable("N", s, t)

    def check_bindings(self, bindings: dict):
        pids = set([x[0] for x in bindings.keys()] + [x[0] for x in bindings.values()])
        nL = 1 + max(pids)
        sids = [x[1] for x in bindings.keys()] + [x[1] for x in bindings.values()]
        nP = 1 + max(sids)

        assert all(len([
            (l, p) for p in range(self.nP) if (l, p) in bindings.keys() or (l, p) in bindings.values()
        ]) <= 6 for l in pids)

        # these asserts should be unnessecary but I'm leaving them anyway
        if self.nL is None:
            self.nL = nL
        elif self.nL != nL:
            raise IOError("Bindings text has different number of positions %d than imposed %d " % (self.nL, nL))

        if self.nP is None:
            self.nP = nP
        elif self.nP < nP:
            raise IOError("Bindings text has larger number of patches %d than imposed %d " % (self.nP, nP))

        return True