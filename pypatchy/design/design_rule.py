"""
Polycube SAT specification adapted from Lukas chrystal one.

"Atoms" are "movable and rotatable", have 6 slots
"Positions" are fixed in the crystal, have 6 slots and bind according to spec
The problem has 2 parts:
A. find color bindings and colorings of position slots where each patch neightbors according to crystal model have
    colors that bind
B. find colorings of atoms s.t. all crystal positions are identical to (some) species rotation. The species definitions
    must not allow for bad 2-binds

indexes:
- colors:   1...c...#c (variable number)
- atoms:    1...s...#s (variable number)
- slots:    0...p...5=#p-1 (bindings places on atoms - 0,1,2 on one side, 3,4,5 on the other)
- position: 1...l...16=#l (number of positions in the crystal)
- rotation: 1...r...6=#r possible rotations of an species
- condition: 1...d...#d conditions to avoid bad crystal
- qualification: 0..#q (0 for good crystal, one more for each bad one)

(boolean) variables:
- B(c1, c2): color c1 binds with c2 (n=#c*#c)
- F(l, p, c): patch p at position l has color c (n=#l*#p*#c)
- C(s, p, c): patch p on species s has color c (n=#s*#p*#c)
- P(l, s, r): position l is occupied by species s with rotation r (n=#l*#s*#r)

encoding functions:
- rotation(p, r) = patch that p rotates to under rotation r

See: https://arxiv.org/pdf/2207.06954v1.pdf
"""
import datetime
import itertools
import os
import re
import time
from enum import Enum
from pathlib import Path
from typing import Union, IO

from ..polycubeutil.polycubesRule import PolycubesRule, PolycubesPatch, get_orientation
from ..polycubeutil.polycubesRule import RULE_ORDER

from .solve_utils import patchRotToVec, getIndexOf, patchVecToRot, getFlatFaceRot, calcEmptyFromTop, \
    countParticlesAndBindings
import numpy as np
from pysat.formula import CNF
from pysat.solvers import Glucose4

from threading import Timer

from .sat_problem import SATProblem, SATClause, interrupt
from .solution import SATSolution

from .solve_params import *
import libpolycubes
import libtlm
from libtlm import TLMParameters, TLMPolycube

RELSAT_EXE = 'relsat'


# exit/error/response codes

# solut
class SolverResponse(Enum):
    SOLN_UNSAT = 0
    SOLN_UNDEFINED = 1
    SOLN_TIMEOUT = 2
    SOLN_ERR = 3


# Polycube SAT Solver
class Polysat(SATProblem):
    # number of patches allowed per species
    nP: int
    # number of orientations allowed for a patch
    nO: int
    # whether patches are torsional
    torsionalPatches: bool

    # number of nanoparticle types
    nNPT: int
    # map of locations of nanoparticles (a list of locations if we only have one np type, a dict otherwise
    np_locations: Union[dict[int, int], list[int]]

    additional_sat_clauses: list[SATClause]
    BCO_varlen: int
    # allo_clauses: Union[None, list[SATClause]]
    nanoparticle_clauses: Union[None, list[SATClause]]
    input_params: SolveParams

    # read nS, nC, etc. properties from self.input_params
    # nS = Number of distinct cube types for the solver
    nS: int = property(lambda self: self.input_params.nS)
    # nC = number of color variables, a function of input_params.nC but not indentical
    nC: int = property(lambda self: (self.input_params.nC * 2) + 1)
    nD: int = property(lambda self: self.input_params.nDim)

    n_TLM_steps: int
    tlm_record_interval: int

    def __init__(self,
                 params: SolveParams):
        super().__init__(params.get_logger(), params.solver_timeout)

        # save solve parameter set
        self.input_params = params

        # self.allostery_constraints = params.allo_limits

        # nL = number of locations
        self.nL, _ = countParticlesAndBindings(params.topology.bindings_list)
        self.internal_bindings = copy.deepcopy(params.topology.bindings_list)

        self.torsionalPatches = params.torsion
        if self.torsionalPatches:
            self.nO = 4  #: Number of possible orientations for a patch, N,S,W,E

        self.empty = calcEmptyFromTop(params.topology.bindings_list)
        self.set_crystal_topology(params.topology.bindings_list)

        if params.has_nanoparticles() and isinstance(params.nanoparticles, dict):
            # if the nanoparticle system has multiple nanoparticle types, we need to
            # add our dummy nanoparticle type and calcuate the number of nanoparticle types
            self.nNPT = max(params.nanoparticles.values())
            self.np_locations = {
                l: params.nanoparticles[l] if l in params.nanoparticles else self.nNPT
                for l in range(self.nL)
            }
            self.nNPT += 1  # add additional nanoparticle type for "no nanoparticle"
        elif isinstance(params.nanoparticles, list) and len(params.nanoparticles):
            self.np_locations = params.nanoparticles
            self.nNPT = 1
        else:
            self.nNPT = 0

        self.additional_sat_clauses = []  # some additional conditions
        self.BCO_varlen = 0  # the number of clauses that determine B and C
        # self.allo_clauses = None  # clauses to handle allostery
        self.nanoparticle_clauses = None

        # self.nA = 0 if self.allostery_complexity_constraints['type'] == 'none' else \
        #     (self.allostery_complexity_constraints['max_n_patches_with_allostery']
        #      if 'max_n_patches_with_allostery' in self.allostery_complexity_constraints else self.nP)

    def init(self, strict_counts=True):
        self.generate_constraints()

        # if problem has nanoparticles
        if self.nNPT > 0:
            # if the nanoparticle data is provided as a list, that means single np type
            if isinstance(self.np_locations, list):
                self.nanoparticle_clauses = self.gen_nanoparticle_singleparticle()
            else:
                self.nanoparticle_clauses = self.gen_nanoparticle_multiparticle()

        # Solution must use all particles
        if strict_counts:
            self.add_constraints_all_particles()

        # Solution must use all patches, except color 0 which should not bind
        self.add_constraints_all_patches_except([0], [1])

        # A color cannot bind to itself
        self.add_constraints_no_self_complementarity()

        # Make sure color 0 binds to 1 and nothing else
        # self.fix_color_interaction(0, 1)

        # Fix interaction matrix, to avoid redundant solution
        for c in range(2, self.nC - 1, 2):
            self.fix_color_interaction(c, c + 1)

        if self.nD == 3 and self.torsionalPatches:
            self.add_constraints_fixed_blank_orientation()

        # if self.allostery_constraints.allostery_type() is None:
        #     self.fix_empties()

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
        # return self.variables.setdefault(f'B({c1},{c2})',
        #                                  len(self.variables) + 1)

    def D(self, p1: int, o1: int, p2: int, o2: int) -> int:
        """ patch p1, orientation o1 binds with patch p2, orientation o2 """
        if p2 < p1:
            o1, o2 = o2, o1
            p1, p2 = p2, p1
        assert 0 <= p1 <= p2 < self.nP
        assert 0 <= o1 < self.nO
        assert 0 <= o2 < self.nO
        return self.variable("D", p1, o1, p2, o2)
        # return self.variables.setdefault(f'D({p1},{o1},{p2},{o2})',
        #                                  len(self.variables) + 1)

    def F(self, l: int, p: int, c: int) -> int:
        """ patch p at position l has color c """
        assert 0 <= l < self.nL
        assert 0 <= p < self.nP
        assert 0 <= c < self.nC
        return self.variable("F", l, p, c)
        # return self.variables.setdefault(f'F({l},{p},{c})',
        #                                  len(self.variables) + 1)

    def A(self, l: int, p: int, o: int) -> int:
        """ patch p at position l has orientation o """
        assert 0 <= l < self.nL
        assert 0 <= p < self.nP
        assert 0 <= o < self.nO
        return self.variable("A", l, p, o)
        # return self.variables.setdefault(f'A({l},{p},{o})', len(self.variables) + 1)

    def C(self, s: int, p: int, c: int) -> int:
        """ patch p on species s has color c """
        assert 0 <= s < self.nS
        assert 0 <= p < self.nP
        assert 0 <= c < self.nC
        return self.variable("C", s, p, c)
        # return self.variables.setdefault(f'C({s},{p},{c})',
        #                                  len(self.variables) + 1)

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
        # return self.variables.setdefault(f'P({l},{s},{r})', len(self.variables) + 1)

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
        # return self.variables.setdefault(f'N{s},{t}', len(self.variables) + 1)

    def S(self, s: int, p: int, v: int) -> int:
        """
        patch p on species s has variable v as a state variable
        Args:
            s:

        Returns:

        """
        assert -1 < s < self.nS
        assert -1 < p < self.nP
        return self.variable("S", s, p, v)

    def V(self, s: int, p: int, v: int) -> int:
        """
        patch p on species s has variable v as an activation variable
        Args:
            s:
            p:
            i:

        Returns:

        """
        return self.variable("V", s, p, v)

    def T(self, l: int, v: int) -> int:
        """
        value of state variable v for cube at position l
        Args:
            l:
            i:

        Returns:

        """
        return self.variable("T", l, v)

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

    def is_bound_patch(self, p: int, l: int) -> int:
        """
        Args:
            p: a patch index [0-6)
            l: a location index [0-self.nL)

        Returns: true if patch p at location l is involved in binding, false otherwise

        """
        return (p, l) in self.bindings or (p, l) in self.bindings.values()

    def all_bindings(self) -> list:
        return list(self.bindings.keys()) + list(self.bindings.values())

    def check_settings(self):
        # assert len(self.bindings) == (self.nL * self.nP) / 2.0
        assert len(set(self.bindings.values())) == len(self.bindings)
        # assert len(set(self.bindings) | set(self.bindings.values())) == self.nL * self.nP
        assert min([s for s, _ in self.bindings] + [s for _, s in self.bindings] +
                   [s for s, _ in self.bindings.values()] + [s for _, s in self.bindings.values()]) == 0
        assert max([s for s, _ in self.bindings] + [s for s, _ in self.bindings.values()]) == self.nL - 1
        assert max([s for _, s in self.bindings] + [s for _, s in self.bindings.values()]) == self.nP - 1
        for p in range(self.nP):
            for r in range(self.nR):
                self.rotation(p, r)

    def generate_constraints(self) -> list[SATClause]:
        """
        Adds all constraints from the original paper (clauses i - x)
        Returns: a list of CNF clauses expressing SAT clauses i - x in the original paper, plus
        a few more that are implicit in the paper)

        """
        self.generate_BCO()
        self.basic_sat_clauses = []
        self.BCO_varlen = len(self.variables)
        constraints = []
        # clause i
        constraints.extend(self.gen_legal_color_bindings())
        # clause ii
        constraints.extend(self.gen_legal_species_coloring())
        # ADD CRYSTAL and COLORS:
        # clause ???
        constraints.extend(self.gen_legal_patch_coloring())
        # clause ???
        constraints.extend(self.gen_legal_position_patch_orientation())
        # clauses iv & ix
        constraints.extend(self.gen_forms_desired_crystal())
        # clause ix?
        constraints.extend(self.gen_hard_code_orientations())
        # clause iii
        constraints.extend(self.gen_legal_species_placement())
        # clause iix, or viii if you know how to use roman numerals correctly
        constraints.extend(self.gen_legal_species_orientation())
        # clause v
        constraints.extend(self.gen_legal_species_position_coloring())
        # clause x
        constraints.extend(self.gen_legal_species_patch_orientation())
        # 2D-specific mystery clauses
        constraints.extend(self.gen_lock_patch_orientation())
        # apply
        self.basic_sat_clauses.extend(constraints)
        return constraints

    def generate_BCO(self):
        # make sure B, C and O vars are first:
        for c1 in range(self.nC):
            for c2 in range(self.nC):
                self.B(c1, c2)
        for s in range(self.nS):
            for p in range(self.nP):
                for c in range(self.nC):
                    self.C(s, p, c)
        if self.torsionalPatches:
            for s in range(self.nS):
                for p in range(self.nP):
                    for o in range(self.nO):
                        self.O(s, p, o)

    def gen_legal_color_bindings(self) -> list[SATClause]:
        """
        Adds clause i (see https://doi.org/10.1021/acsnano.2c09677) - Legal color bindings:
        "Each color has exactly one color that it binds to"
        forall c1 exactly one c2 s.t. B(c1, c2)
        Returns: a list of CNF clauses that together represent clause i
        """
        constraints = []
        # BASIC THINGS:
        #
        for c1 in range(self.nC):
            constraints.extend(self._exactly_one([self.B(c1, c2) for c2 in range(self.nC)]))

        return constraints

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

        constraints = []
        for s in range(self.nS):
            for p in range(self.nP):
                constraints.extend(self._exactly_one([self.C(s, p, c) for c in range(self.nC)]))
        return constraints

    def gen_legal_species_orientation(self):
        """
        Adds clause iix (viii?)
        Legal species patch orientation
        "Each patch on every species has exactly one orientation"
        forall s, forall p, exactly one o p.t. O(s, p, o)
        Returns: a list of SAT clauses representing clause iix (viii?)
        """
        constraints = []
        if self.torsionalPatches:
            for s in range(self.nS):
                for p in range(self.nP):
                    constraints.extend(self._exactly_one([self.O(s, p, o) for o in range(self.nO)]))
        return constraints

    def gen_legal_patch_coloring(self):
        """
        Adds clause... I don't think this one is described in the paper?
        It may be an implicit result of clauses ii and v
        Legal position patch coloring:
        "Every position patch has exactly one color"
        for all l, p exactly one c st. F(l, p, c)
        Returns: a list of CNF clauses that prevent any position patch from having more than one color
        """

        constraints = []
        for l in range(self.nL):
            for p in range(self.nP):
                constraints.extend(self._exactly_one([self.F(l, p, c) for c in range(self.nC)]))
        return constraints

    def gen_legal_position_patch_orientation(self):
        """
        Adds clause... I don't think this one is explicitly described in the paper?
        It may be an implicit result of clauses iix (viii) and x
        Legal position patch orientation:
        "Every position patch has exactly one orientation"
        for all l, p exactly one o st. A(l, p, o)
        Returns: a list of CNF clauses that prevent any position patch from having more than one orientation
        """

        constraints = []
        if self.torsionalPatches:
            for l in range(self.nL):
                for p in range(self.nP):
                    constraints.extend(self._exactly_one([self.A(l, p, o) for o in range(self.nO)]))
        return constraints

    def gen_forms_desired_crystal(self):
        """
        Implements Clauses iv and ix in the paper
        Clause iv states that every two patches that bind to each other must have
        complimentary colors
        Clause ix states that every two patches that bind to each other must have
        compatible orientations
        Returns: a list of SAT clauses expressing Clauses iv and ix in the paper
        """
        constraints = []

        # Forms desired crystal:
        # "Specified binds have compatible colors"
        # forall (l1, p1) binding with (l2, p2) from crystal spec:
        # forall c1, c2: F(l1, p1, c1) and F(l2, p2, c2) => B(c1, c2)
        for (l1, p1), (l2, p2) in self.bindings.items():
            for c1 in range(self.nC):
                for c2 in range(self.nC):
                    constraints.append((-self.F(l1, p1, c1), -self.F(l2, p2, c2), self.B(c1, c2)))

        # - Forms desired crystal:
        # "Specified binds have compatible orientations"
        # 	forall (l1, p1) binding with (l2, p2) from crystal spec:
        # 		forall o1, o2: A(l1, p1, o1) and A(l2, p2, o2) => D(c1, c2)
        if self.torsionalPatches:
            for (l1, p1), (l2, p2) in self.bindings.items():
                for o1 in range(self.nO):
                    for o2 in range(self.nO):
                        constraints.append((-self.A(l1, p1, o1), -self.A(l2, p2, o2), self.D(p1, o1, p2, o2)))
        return constraints

    def gen_hard_code_orientations(self):
        """
        Clause ix?
        Returns: A list of CNF clauses expressing Clause ix (i think?) from the paper

        """
        constraints = []
        # Hard-code patch orientations to bind only if they point in the same direction
        if self.torsionalPatches:
            for p1 in range(self.nP):
                for p2 in range(self.nP):
                    if p2 >= p1:
                        break
                    for o1 in range(self.nO):
                        for o2 in range(self.nO):
                            v1 = patchRotToVec(p1, o1)
                            v2 = patchRotToVec(p2, o2)
                            # Do they point in the same global direction?
                            # And do the patches face each other?
                            if np.array_equal(v1, v2) and p2 % 2 == 0 and p2 + 1 == p1:
                                constraints.append([self.D(p1, o1, p2, o2)])
                                # print("patch {}, orientation {} binds with patch {}, orientation {}".format(p1, o1, p2, o2))
                            else:
                                constraints.append([-self.D(p1, o1, p2, o2)])
        return constraints

    def gen_legal_species_placement(self) -> list[tuple[int, ...]]:
        """
        Adds CNF clauses corresponding to SAT clause iii from the paper
        Returns: a list of CNF clauses expressing SAT clause iii

        """
        # - Legal species placement in positions:
        # "Every position has exactly one species placed there with exactly one rotation"
        #   forall l: exactly one s and r p.t. P(l, s, r)
        constraints = []
        for l in range(self.nL):
            # for all possible species, rotation pairs
            srs = itertools.product(range(self.nS), range(self.nR))
            location_vars = [self.P(l, s, r) for s, r in srs]
            constraints.extend(self._exactly_one(location_vars))
        return constraints

    def gen_legal_species_position_coloring(self) -> list[tuple[int, int, int]]:
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
        constraints = []
        # for each position l in the lattice
        for l in range(self.nL):
            # for each species s
            for s in range(self.nS):
                # for all rotation r
                for r in range(self.nR):
                    # for each patch p
                    for p in range(self.nP):
                        # for each color c
                        for c in range(self.nC):
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
        return constraints

    def gen_legal_species_patch_orientation(self) -> list[tuple[int, int, int]]:
        """
        Adds clause x from the paper
        Clause x states that each patch in the lattice must have an orientation corresponding to the
        patch on the species occupying that position, accounting for rotation
        Returns: a list of CNF clauses expressing clause x from the paper

        """
        # - Legal species patch orientation in positions:
        # "Given a place, species and its rotation, the patch orientations on the position and (rotated) species must be correct"
        #   for all l, s, r:
        #       P(l, s, r) => (forall p, c: F(l, p, c) <=> C(s, rotation(p, r), c))
        constraints = []
        if self.torsionalPatches:
            for l in range(self.nL):
                for s in range(self.nS):
                    for r in range(self.nR):
                        for p in range(self.nP):
                            for o in range(self.nO):
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
        return constraints

    def gen_lock_patch_orientation(self):
        """
        I have no idea what this does
        It appears to only apply to 2D shapes
        Returns: a list of CNF clauses of some sort, if 2D, else an empty list

        """
        constraints = []
        if self.nD == 2:
            # Lock patch orientation if 2D
            for s in range(self.nS):
                for p in range(self.nP):
                    # v = np.array([0, 1, 0])
                    # o = utils.patchVecToRot(p, v)
                    if p > 3:
                        # Patch p on species s is empty
                        constraints.append([self.C(s, p, 1)])
                    o = getFlatFaceRot()[p]
                    # Patch p has orientation 'o'
                    constraints.append([self.O(s, p, o)])
        return constraints

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

    def gen_nanoparticle_singleparticle(self) -> list[SATClause]:
        """
        generates nanoparticle clauses for a system with only one nanoparticle type
        Returns: a list of SAT clauses
        """
        constraints = []
        # for each location
        for l in range(self.nL):
            # if location l has a nanoparticle
            hasParticle = l in self.np_locations  # eval now to save time
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
        return constraints

    def gen_nanoparticle_multiparticle(self) -> list[SATClause]:
        """
        generates nanoparticle clauses for a system with multiple nanoparticle types
        Returns: a list of SAT clauses

        """
        constraints = []
        # generate disjointitiy clause - each species s has exactly one nanoparticle type
        for s in range(self.nS):
            constraints.extend(self._exactly_one([
                self.N_multi(s, t) for t in range(self.nNPT)
            ]))

        # occupancy clause - each position must have its correct nanoparticle type
        for l in range(self.nL):
            # if location l has a nanoparticle
            nptype = self.np_locations[l]  # eval now to save time
            constraints.extend(itertools.chain.from_iterable(
                [
                    [
                        [self.N_multi(s, nptype), -self.P(l, s, r)]
                        for s in range(self.nS)
                    ]
                    for r in range(self.nR)]
            ))
        return constraints

    def gen_no_self_interact(self) -> list[SATClause]:
        constraints = []
        # for each species
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

        return constraints

    def fix_empties(self):
        for particle, patch in self.empty:
            self.fix_slot_colors(particle, patch, 1)
            # print("Particle {} patch {} should be empty".format(particle, patch))

    def output_cnf(self, out: Union[IO, None] = None) -> str:
        """ Outputs a CNF formula """
        num_vars = max(self.variables.values())
        num_constraints = len(self.basic_sat_clauses)
        # if self.allo_clauses is not None:
        #     num_constraints += len(self.allo_clauses)
        if self.nanoparticle_clauses is not None:
            num_constraints += len(self.nanoparticle_clauses)
        outstr = "p cnf %s %s\n" % (num_vars, num_constraints)
        # add basic clauses
        for c in self.basic_sat_clauses:
            outstr += ' '.join([str(v) for v in c]) + ' 0\n'

        # add nanoparticle clauses
        if self.nanoparticle_clauses is not None:
            for c in self.nanoparticle_clauses:
                outstr += ' '.join([str(v) for v in c]) + ' 0\n'
        # if self.allo_clauses is not None:
        #     for c in self.allo_clauses:
        #         outstr += ' '.join([str(v) for v in c]) + ' 0\n'
        if out is not None:
            out.write(outstr)
        return outstr

    #
    # def load_solution_from_lines(self, lines, maxvariable=None):
    #     """ loads solution from sat solution output in s string"""
    #     if len(lines) > 1:
    #         assert lines[0].strip() == 'SAT'
    #         satline = lines[1].strip().split()
    #     else:
    #         satline = lines[0].strip().split()
    #
    #     #line = myinput.readline().strip()
    #     #assert line == 'SAT'
    #     sols = [int(v) for v in satline]
    #     assert sols[-1] == 0
    #     sols = sols[:-1]
    #     assert len(sols) <= len(self.variables)
    #
    #     return [vname for vname, vnum in self.variables.items() if vnum in sols]

    def add_constraints_from_vnames(self, vnames: list[str]):
        constraints = []
        for vname in vnames:
            if vname not in self.variables:
                raise IOError("Trying to add variables that have not been defined, "
                              "probably incompatible problem formulation?")
            constraints.append(self.variables[vname])
        self.basic_sat_clauses.append(constraints)

    def convert_solution(self, sols):
        assert len(sols) <= len(self.variables)
        out = ""

        for vname, vnum in sorted(self.variables.items()):
            if vnum > len(sols):
                break
            if sols[vnum - 1] > 0:
                out += vname + '\n'
        return out

    def save_named_solution(self, solution, output, B=True, C=True, P=False):
        """saves text values of system constraints , such as B(2,3) etc"""
        handle = open(output, 'w')
        for vname, vnum in sorted(self.variables.items()):
            if vnum > len(solution):
                break
            if solution[vnum - 1] > 0:
                if 'B' in vname and B:
                    handle.write('%s\n' % (vname))
                elif 'C' in vname and C:
                    handle.write('%s\n' % (vname))
                elif 'P' in vname and P:
                    handle.write('%s\n' % (vname))
        handle.close()

    def load_constraints_from_sol(self, sol_file, append=False):
        """ loads solution from minisat output in myinput handle,
        adds it to self.additional_sat_clauses constraints """
        myinput = open(sol_file)
        line = myinput.readline().strip()
        assert line == 'SAT'
        sols = [int(v) for v in myinput.readline().strip().split()]
        assert sols[-1] == 0
        sols = sols[:-1]
        assert len(sols) <= len(self.variables)
        new_constraints = []
        for vname, vnum in sorted(self.variables.items()):
            if vnum > len(sols):
                break
            if sols[vnum - 1] > 0:
                new_constraints.append(self.variables[vname])
                # print(vname)
        if append:
            self.additional_sat_clauses.extend(new_constraints)
        return new_constraints

    def load_constraints_from_text_sol(self, sol_file, append=True):
        """ loads solution from written output (such as B(1,3),
         one clause per line) in myinput handle """
        myinput = open(sol_file)
        lines = [line.strip() for line in myinput.readlines()]
        new_constraints = []
        for vname in lines:
            new_constraints.append([self.variables[vname]])
            # print(vname)
        if append:
            # print 'Addding',new_constraints, 'to', self.basic_sat_clauses
            self.basic_sat_clauses.extend(new_constraints)
            # print self.basic_sat_clauses
        return new_constraints

    def load_BC_constraints_from_text_sol(self, sol_file, append=True):
        """ loads solution from written output (such as B(1,3), one clause per line) in myinput handle """
        myinput = open(sol_file)
        lines = [line.strip() for line in myinput.readlines()]
        new_constraints = []
        for vname in lines:
            if 'B' in vname or 'C' in vname:
                new_constraints.append([self.variables[vname]])
                # print(vname)
        if append:
            # print 'Addding',new_constraints, 'to', self.basic_sat_clauses
            self.basic_sat_clauses.extend(new_constraints)
            # print self.basic_sat_clauses
        return new_constraints

    def fill_constraints(self):
        self.generate_constraints()

    def add_constraints_all_particles(self):
        """
        Adds SAT clauses specifying that the solution must use all particle types

        """
        for s in range(self.nS):
            self.basic_sat_clauses.append([self.P(l, s, r) for l in range(self.nL) for r in range(self.nR)])

    def add_constraints_all_patches(self):
        """
        adds SAT clauses specifying that the solution must use all patch types
        Returns:

        """
        for c in range(self.nC):
            self.basic_sat_clauses.append([self.C(s, p, c) for s in range(self.nS) for p in range(self.nP)])

    def add_constraints_all_patches_except(self, forbidden: list[int], nonRequired: list[int] = [1]):
        for c in range(self.nC):
            if c not in forbidden and c not in nonRequired:
                self.basic_sat_clauses.append([self.C(s, p, c) for s in range(self.nS) for p in range(self.nP)])
            # Do not use forbidden color
            for p in range(self.nP):
                for s in range(self.nS):
                    for nForbidden in forbidden:
                        self.basic_sat_clauses.append(
                            [-self.C(s, p, nForbidden)]
                        )

    def add_constraints_fixed_blank_orientation(self):
        for p in range(self.nP):
            for s in range(self.nS):
                self.basic_sat_clauses.append((
                    -self.C(s, p, 1),  # Either patch p on species s isn't empty
                    self.O(s, p, 0)  # Or the patch is oriented up
                ))

    def add_constraints_no_self_complementarity(self, above_color=0):
        for c in range(above_color, self.nC):
            self.basic_sat_clauses.append([-self.B(c, c)])

    def fix_particle_colors(self, ptype, sid, cid):
        self.basic_sat_clauses.append([self.C(ptype, sid, cid)])

    def fix_slot_colors(self, ptype, sid, cid):
        self.basic_sat_clauses.append([self.F(ptype, sid, cid)])

    def fix_color_interaction(self, c1, c2):
        self.basic_sat_clauses.append([self.B(c1, c2)])

    def tlm_params(self,
                   rule: PolycubesRule,
                   temperature: float,
                   density: float,
                   type_counts: list[int],
                   interaction_matrix: dict[tuple[int,int],float]) -> TLMParameters:
        return TLMParameters(
            self.input_params.torsion,
            True,  # deplete types, reconsider hardcoding?
            temperature,
            density,
            str(rule),
            type_counts,
            self.n_TLM_steps,
            interaction_matrix,
            time.ctime(), # random seed
            self.tlm_record_interval
        )

    def forbidSolution(self, solution: SATSolution):
        """
        Forbid a specific solution, which while valid from the SAT perspective has
        been found to be invalid (unbounded, nondeterministic, cursed, etc.) by external tool
        Args:
            solution: a SATSolution object that has been found to be invalid and shouldn't be allowed in future solves

        Returns:

        """
        forbidden = []
        # forbid the solution by banning any solution where every variable in the Forbidden Solution is True
        for var in solution.C_vars + solution.O_vars:
            forbidden.append(-self.variables[var])
        # for vname in solution.split('\n'):
        #     if 'C' in vname or 'O' in vname:
        #         forbidden.append(-self.variables[vname])
        self.basic_sat_clauses.append(forbidden)

    def run_sat_solve(self, nSolutions: int, solver_timeout: Union[int, None]) -> Union[
            SolverResponse,
            list[SATSolution]]:
        if nSolutions == 1:  # Use minisat for single solutions
            result = self.run_glucose()
        else:  # use relsat for larger problems
            result = self.run_relsat(nSolutions=nSolutions, timeout=solver_timeout)

        if isinstance(result, SATSolution):
            result = [result]  # list form, to be consistant
        return result

    def run_glucose(self) -> Union[SolverResponse, SATSolution]:
        """
        Uses Glucose to solve the SAT problem
        Returns either a SATSolution object or a SolverResponse code specifying what didn't work
        """
        formula = CNF(from_string=self.output_cnf())
        tstart = datetime.datetime.now()
        self.logger.info(f"Starting solve with Glucose4, timeout {self.solver_timeout}")
        with Glucose4(bootstrap_with=formula.clauses) as m:
            # if the solver has a timeout specified
            if self.solver_timeout:
                timer = Timer(self.solver_timeout, interrupt, [m])
                timer.start()
                solved = m.solve_limited(expect_interrupt=True)
                timer.cancel()
            else:
                solved = m.solve()
            if solved:
                self.logger.info("Solved!")
                # pysat returns solution as a list of variables
                # which are "positive" if they're true and "negative" if they're
                # false.
                model = m.get_model()
                # we can pass the model directly to the SATSolution constructor because it will
                # just check for positive variables to be present
                return SATSolution(self, self.input_params, frozenset(model))
            else:
                # if the solve solver timed out
                if self.solver_timeout and (datetime.datetime.now() - tstart).seconds > self.solver_timeout:
                    return SolverResponse.SOLN_TIMEOUT
                # if the solver failed but didn't time out, conclude that the problem
                # is not satisfiable
                else:
                    return SolverResponse.SOLN_UNSAT

    def run_relsat(self, nSolutions: int, timeout: Union[int, None]) -> Union[SolverResponse, list[SATSolution]]:
        """
        Uses relsat to solve the SAT problem
        Args:
            nSolutions: the number of solutions to produce
            timeout: the length relsat should run before it gives up

        Returns:
            Either "TIMEOUT" or a tuple where the first element is the number of solutions
            produced and the second is a list of SATSolution objects
        """
        # construct filename for temporary file for SAT problem / solution
        tempfilename = '/tmp/temp_for_relsat.%s.cls' % (os.getpid())
        tempout = tempfilename + '.sol'
        # dump cnf to our new SAT file
        with open(tempfilename, 'w') as temp:
            self.output_cnf(temp)

        # here we execute
        # select lines from relsat output that don't contain the letter "c"
        # relsat output flags< lines that aren't SAT solutions by starting them with "c".
        # IDK why. "c" = comment?
        command = f"{RELSAT_EXE} -# {nSolutions}"
        if timeout is not None:
            command += f" -t {timeout}"
        else:
            command += " -t n"
        command += f" {tempfilename} | grep -v c > {tempout}"
        self.logger.info(f"Executing command ${command}")
        os.system(command)

        # open output file
        with open(tempout) as fout:
            out = fout.readlines()
            result = out[-1].strip()

            # if the problem was not satisfiable
            if result == 'UNSAT':
                return SolverResponse.SOLN_UNSAT
            # if a solution was found
            elif result == 'SAT':
                all_solutions = []
                # loop line in file
                for line in out:
                    if 'Solution' in line:
                        # process solution line formatting
                        myvars = line.strip().split(':')[1].strip()
                        varnames = frozenset(int(v) for v in myvars.split())
                        all_solutions.append(SATSolution(self, varnames))

                return all_solutions
            # if the process timed out
            elif result == 'TIME LIMIT EXPIRED':
                return SolverResponse.SOLN_TIMEOUT
            # if something else somehow happened
            else:
                self.logger.info(result)
                raise IOError("Found something else")

    def readSolution(self, sol: str) -> PolycubesRule:
        colorCounter = 1
        # need to map B variable indeces to colors
        colorMap: dict[int, int] = {}

        # initialize blank rule
        rule = PolycubesRule(nS=self.nS)
        # color pairing variables (B) that are true
        B_vars = re.findall(r'B\((\d+),(\d+)\)', sol)
        # iterate values for B
        for c1, c2 in B_vars:  # color c1 binds with c2
            # print("Color {} binds with {}".format(c1, c2))
            assert (c1 not in colorMap or c2 not in colorMap)
            # map colors
            if int(c1) < 2 or int(c2) < 2:
                colorMap[c1] = 0
                colorMap[c2] = 0
            else:
                colorMap[c1] = colorCounter
                colorMap[c2] = -colorCounter
                colorCounter += 1

        # patch color (C) variable matches
        C_matches = re.findall(r'C\((\d+),(\d+),(\d+)\)', sol)
        for s, p, c in C_matches:  # Patch p on species s has color c
            patch_direction = RULE_ORDER[p]
            if not rule.particle(p).has_patch(patch_direction):
                rule.add_particle_patch(s, PolycubesPatch(uid=None,  # idx will be assigned in add method
                                                          color=colorMap[c],
                                                          direction=patch_direction,
                                                          orientation=get_orientation(p, 0)
                                                          )
                                        )
        oMatches = re.findall(r'O\((\d+),(\d+),(\d+)\)', sol)
        if len(oMatches) > 0:
            for s, p, o in oMatches:  # Patch on species l has orientation o
                # print("Patch {} on species {} has orientation {}".format(p, s, o))
                rule.particle(s).get_patch_by_diridx(p).set_align_rot(o)

        return rule

    def readSolutionFromPath(self, path: Union[str, Path]) -> PolycubesRule:
        with open(path) as f:
            sol = f.read()

        return self.readSolution(sol)

    def get_solution_var_names(self, solution: set[int]) -> list[str]:
        return [vname for vname, vnum in self.variables.items() if vnum in solution]

    def find_solution(self):
        """
        big method! go go go!
        """
        nTries = 0
        good_soln = None

        while nTries < self.input_params.maxAltTries and not good_soln:
            # run SAT
            result = self.run_sat_solve(self.input_params.nSolutions,
                                        self.input_params.solver_timeout)

            # check response
            if result == SolverResponse.SOLN_TIMEOUT:
                self.logger.info("SAT solver timed out!")
                break  # assume more iterations will not help (???)
            elif result == SolverResponse.SOLN_UNSAT:
                self.logger.info('Sorry, no solution')
                break  # assume more iterations will not help
            elif len(result) > 0:  # found solutions!
                self.logger.info(f"Found {len(result)} possible solutions:")
                # for now let's use Joakim's rule formulation
                self.logger.info("\n".join("\t" + soln.decRuleOld() for soln in result))
                # skip Polycubes call if the target is a crystal

                good_soln = self.find_good_solution(result)
                if good_soln is None:
                    for soln in result:
                        self.forbidSolution(soln)
                nTries += 1
            else:
                self.logger.error("Undefined problem of some sort!!!")
                break
        return good_soln

    def find_good_solution(self, sat_solutions: list[SATSolution]) -> Union[SATSolution, None]:
        """
        ok this is complecated because what makes a solution "good" depends on what kind of structure we're
        trying to design here
        """
        for soln in sat_solutions:
            if self.input_params.crystal:
                if self.test_crystal(soln):
                    return soln
            elif self.input_params.is_multifarious():
                if self.test_multifarious_finite_size(soln):
                    return soln
            elif self.input_params.has_nanoparticles():
                # if the solve target has nanoparticles, you can still use polycubes
                # but warn the user that this may create false positives
                if self.test_type_specific_finite_size(soln):
                    return soln
            else:
                # loop each solution in the results
                if self.test_finite_size(soln):
                    return soln
        return None

    def test_crystal(self, sat_solution: SATSolution) -> bool:
        """
        this becomes. tricky.


        """
        T = 0.5

        polycube_results = libtlm.runSimulations(self.tlm_params(
            self.torsionalPatches, # torsion
            True, # Type depletion
            T,  # temperature
            0.1,  # density
            sat_solution.decRuleOld(),
            [200, 200],  # type counts
            int(5e6),  # steps
            1000  # data point interval

        ))


    def test_multifarious_finite_size(self, sat_solution: SATSolution) -> bool:
        """
        tricky
        """
        polycube_results = libtlm.runSimulations(self.tlm_params(

        ))

    def test_type_specific_finite_size(self, sat_solution: SATSolution) -> bool:
        """

        """
        pass

    def test_finite_size(self, sat_solution: SATSolution) -> bool:
        """
        this is old code but it should still work
        """
        self.logger.info(f"Testing rule {sat_solution.decRuleOld()}")
        # use "classic" polycubes to check if the rule is bounded and determinstic
        if libpolycubes.isBoundedAndDeterministic(sat_solution.decRuleOld(), isHexString=False):
            self.logger.info(f"{sat_solution.decRuleOld()} works!! We can stop now!")
            return True
        else:
            self.logger.info(f"{sat_solution.decRuleOld()} found to be unbounded and/or nondeterministic.")
            return False
