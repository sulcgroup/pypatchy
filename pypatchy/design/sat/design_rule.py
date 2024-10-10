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
import copy
import datetime
import itertools
import os
import re
from enum import Enum
from pathlib import Path
from threading import Timer

import libpolycubes
import libtlm
from pysat.formula import CNF
from pysat.solvers import Glucose4

from pypatchy.design.crystal_finder import find_crystal_temperature, DoesNotCrystalizeException
from pypatchy.design.sat.solution import SATSolution
from pypatchy.design.solve_params import *
from pypatchy.design.solve_utils import toPolycube
from pypatchy.polycubeutil.polycube_structure import PolycubeStructure
from pypatchy.polycubeutil.polycubesRule import PolycubesRule, PolycubesPatch, get_orientation
from pypatchy.polycubeutil.polycubesRule import RULE_ORDER
from .polycube_sat_problem import PolycubeSATProblem
from .sat_problem import SATClause, interrupt, SATProblemPart, exactly_one

RELSAT_EXE = 'relsat'


# exit/error/response codes

# solut
class SolverResponse(Enum):
    SOLN_UNSAT = 0
    SOLN_UNDEFINED = 1
    SOLN_TIMEOUT = 2
    SOLN_ERR = 3


class Polysat:

    # ----------------------------------- basic stuff ---------------------------------------------
    # bindings
    # bindings: dict[tuple[int, int]: tuple[int, int]]  # ngl I have no idea why they're like this
    # internal_bindings: list[tuple[int, int, int, int]]

    target_structure: Structure

    # logger for SAT solver
    logger: logging.Logger
    # timeout for SAT solver
    solver_timeout: Union[int, None]

    # whether patches are torsional
    torsionalPatches: bool

    # map of locations of nanoparticles (a list of locations if we only have one np type, a dict otherwise
    np_locations: Union[dict[int, int], list[int]]

    additional_sat_clauses: list[SATClause]
    BCO_varlen: int
    # allo_clauses: Union[None, list[SATClause]]
    nanoparticle_clauses: Union[None, list[SATClause]]
    input_params: SolveParams

    problem: PolycubeSATProblem
    bindings = property(lambda self: self.target_structure.bindings_list)

    def __init__(self,
                 params: SolveParams):
        self.logger = params.get_logger()
        self.solver_timeout = params.solver_timeout
        self.target_structure = Structure(bindings=params.bindings)

        # save solve parameter set
        self.input_params = params


        if params.has_nanoparticles() and isinstance(params.nanoparticles, dict):
            # if the nanoparticle system has multiple nanoparticle types, we need to
            # add our dummy nanoparticle type and calcuate the number of nanoparticle types
            nNPT = max(params.nanoparticles.values())
            self.np_locations = {
                l: params.nanoparticles[l] if l in params.nanoparticles else self.nNPT
                for l in self.target_structure.vertices()
            }
            nNPT += 1  # add additional nanoparticle type for "no nanoparticle"
        elif isinstance(params.nanoparticles, list) and len(params.nanoparticles):
            self.np_locations = params.nanoparticles
            nNPT = 1
        else:
            nNPT = 0

        # nL = number of locations
        # self.internal_bindings = copy.deepcopy(params.topology.bindings_list)
        self.problem = PolycubeSATProblem(
            params.nS,
            (params.nC + 1) * 2,
            params.nDim,
            params.topology.num_vertices(),
            nNPT,
            params.torsion
        )

        # self.empty = calcEmptyFromTop(params.topology.bindings_list)
        # self.set_crystal_topology(params.topology.bindings_list)

    def init(self, strict_counts=True):
        # generate basic constraints
        self.generate_constraints()

        # if problem has nanoparticles
        if self.problem.nNPT > 0:
            # if the nanoparticle data is provided as a list, that means single np type
            if isinstance(self.np_locations, list):
                self.nanoparticle_clauses = self.problem.gen_nanoparticle_singleparticle(self.np_locations)
            else:
                self.nanoparticle_clauses = self.problem.gen_nanoparticle_multiparticle(self.np_locations)

        # Solution must use all particles
        if strict_counts:
            self.problem.add_constraints_all_particles()

        # Solution must use all patches, except color 0 which should not bind
        # It is not necessary for a solution to have empty patches (e.g. if we
        # actually want an unbounded solution, so we don't require color 1)
        # in other words, color 0 is forbidden and color 1 is allowed but not required
        # todo: revisit
        self.problem.add_constraints_all_patches_except([0], [1])

        # A color cannot bind to itself
        # this is probably redundant w/ fix_color_interaction
        self.problem.add_constraints_no_self_complementarity()

        # Make sure color 0 binds to 1 and nothing else
        # self.fix_color_interaction(0, 1)

        # Fix interaction matrix, to avoid redundant solution
        # in other words, c2 should bind with c3, c4 with c5, etc.
        # if we don't do this then when we forbid a solution it will just find an alternative
        # arrangement of color vars

        # if we have torsional patches,
        if self.problem.nD == 3 and self.input_params.torsion:
            self.problem.add_constraints_fixed_blank_orientation()

        # force faces in topology which do not have bonds to not have patches
        self.problem.fix_empties(self.target_structure)

    def check_settings(self):
        # assert len(self.bindings) == (self.nL * self.nP) / 2.0
        # assert len(set(self.bindings.values())) == len(self.bindings)
        # assert len(set(self.bindings) | set(self.bindings.values())) == self.nL * self.nP
        # assert min([s for s, _ in self.bindings] + [s for _, s in self.bindings] +
        #            [s for s, _ in self.bindings.values()] + [s for _, s in self.bindings.values()]) == 0
        # assert max([s for s, _ in self.bindings] + [s for s, _ in self.bindings.values()]) == self.problem.nL - 1
        # assert max([s for _, s in self.bindings] + [s for _, s in self.bindings.values()]) == self.problem.nP - 1
        # for p, r in itertools.product(range(self.problem.nP), range(self.problem.nR)):
        #     self.problem.rotation(p, r)
        # todo: update
        pass

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



    def generate_constraints(self):
        """
        Adds all constraints from the original paper (clauses i - x)
        Returns: a list of CNF clauses expressing SAT clauses i - x in the original paper, plus
        a few more that are implicit in the paper)

        """
        self.problem.generate_BCO()
        self.BCO_varlen = len(self.problem.variables)
        constraints = []
        # clause i
        self.problem.gen_legal_color_bindings()
        # clause ii
        self.problem.gen_legal_species_coloring()
        # ADD CRYSTAL and COLORS:
        # clause ???
        self.problem.gen_legal_patch_coloring()
        # clause ???
        self.problem.gen_legal_position_patch_orientation()
        # clauses iv & ix
        self.problem.gen_forms_desired_structure(self.target_structure)
        # clause ix?
        self.problem.gen_hard_code_orientations()
        # clause iii
        self.problem.gen_legal_species_placement()
        # clause iix, or viii if you know how to use roman numerals correctly
        self.problem.gen_legal_species_orientation()
        # clause v
        self.problem.gen_legal_species_position_coloring()
        # clause x
        self.problem.gen_legal_species_patch_orientation()
        # 2D-specific mystery clauses
        self.problem.gen_lock_patch_orientation()
        # apply

    def save_named_solution(self, solution, output, B=True, C=True, P=False):
        """saves text values of system constraints , such as B(2,3) etc"""
        handle = open(output, 'w')
        for vname, vnum in sorted(self.problem.variables.items()):
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
        formula = CNF(from_string=self.problem.output_cnf())
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
            self.problem.output_cnf(temp)

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
                        all_solutions.append(SATSolution(self,varnames))

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
                        self.problem.forbidSolution(soln)
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
            elif self.input_params.is_multifarious:
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

    def polycube_satisfies(self, soln: list[int], crystal: PolycubeStructure) -> bool:
        """
        sets up and executes a sat-solve to test if the polycube object given matches the given SAT
        problem
        """
        # skip polycubes with more locations than the crystal unit cell
        if crystal.num_particles() > self.problem.nL:
            crystal_test_problem = copy.deepcopy(self.problem)
            part_pstar = SATProblemPart("P*", ["l*", "l"])
            crystal_test_problem.add_problem_part(part_pstar)
            pstar = lambda lstar, l: crystal_test_problem.variable("P*", lstar, l)
            # for each location  lin the test crystal
            for l in range(crystal_test_problem.nL):
                # exactly one location lStar maps to locatioon l
                part_pstar.clauses.extend(
                    exactly_one([
                                pstar(lstar, l)
                                 for lstar in range(crystal.num_particles())]))
            part_sstar = SATProblemPart("S*", ["l", "l*"])
            crystal_test_problem.add_problem_part(part_sstar)
            # for lstar -> l
            for (lstar, cube), l in itertools.product(enumerate(crystal.particles()), range(crystal_test_problem.nL)):
                # either lstar -/> l or the species at l is the same as the one at lstar
                part_sstar.clauses.extend([
                    -pstar(lstar, l)
                    *[crystal_test_problem.P(l, cube.type_id(), r) for r in crystal_test_problem.nR]
                ])

            # useful thing that will help us later
            Estar = lambda l1, l2, lstar_1, lstar_2: crystal_test_problem.variable("E*",
                                                  *[l1, l2 if l1 < l2 else l2, l1],
                                                  *[lstar_1, lstar_2 if lstar_1 < lstar_2 else lstar_2, lstar_1])

            # loop each pair of bound locations in the unit cell
            # we can actually ignore binding directions here

            part_graph_homomorph = SATProblemPart("G", ["l{i}", "l{j}", "lstar{k}", "lstar{l}"])
            # for each pair of linked positions in the unit cell:
            for (l1, _), (l2, _) in self.target_structure.bindings_list:
                # some l1, lstar1, l2, lstar2 exists where P*(l1, lstar1) and p*(l2, lstar2)
                # or vice versa
                # unfortuantely this becomes Hard
                # best approach may actually be to define this variable E*(l1, l2, lstar_1, lstar_2
                part_graph_homomorph.clauses.append([
                    Estar(l1, l2, lstar_1, lstar_2)
                    for (lstar_1, _), (lstar_2, _) in crystal.bindings_list
                ])

                # now let's constrain E*
                # E*(l1, l2, l*1, l*2) <-> P*(l1, lstar1) ^ P*(l2, lstar2)
                # i had chatGPT decompose E*(l1, l2, l*1, l*2) <-> P*(l1, lstar1) ^ P*(l2, lstar2) into CNF
                # let's pray it's actually correct, chatGPT's record on math problems is not great
                # in cnf this is (¬E∗(l1,l2,l∗1,l∗2)∨P∗(l1,l∗1))∧(¬E∗(l1,l2,l∗1,l∗2)∨P∗(l2,l∗2))∧(¬P∗(l1,l∗1)∨¬P∗(l2,l∗2)∨E∗(l1,l2,l∗1,l∗2))
                for (lstar_1, _), (lstar_2, _) in crystal.bindings_list:
                    # (¬E∗(l1,l2,l∗1,l∗2) ∨ P∗(l1,l∗1))
                    part_graph_homomorph.clauses.append([
                        -Estar(l1, l2, lstar_1, lstar_2),
                        pstar(lstar_1, l1)
                    ])
                    # (¬E∗(l1,l2,l∗1,l∗2)∨P∗(l2,l∗2))
                    part_graph_homomorph.clauses.append([
                        -Estar(l1, l2, lstar_1, lstar_2),
                        pstar(lstar_2, l2)
                    ])
                    # (¬P∗(l1,l∗1) ∨ ¬P∗(l2,l∗2) ∨ E∗(l1,l2,l∗1,l∗2))
                    part_graph_homomorph.clauses.append([
                        Estar(l1, l2, lstar_1, lstar_2),
                        -pstar(lstar_1, l1),
                        -pstar(lstar_2, l2)
                    ])
            # we have constructed SAT problem
        else:
            # polycube w/ less particles than the unit cell is nessecarily not a crystal
            return False

        # try to find solution
        formula = CNF(from_string=crystal_test_problem.output_cnf())
        tstart = datetime.datetime.now()
        self.logger.info(f"Starting crystal structure SAT test with Glucose4, timeout {self.solver_timeout}")
        with Glucose4(bootstrap_with=formula.clauses) as m:
            # if the solver has a timeout specified
            if self.solver_timeout:
                timer = Timer(self.solver_timeout, interrupt, [m])
                timer.start()
                solved = m.solve_limited(assumptions=soln, expect_interrupt=True)
                timer.cancel()
            else:
                solved = m.solve(assumptions=soln)
            if solved:
                self.logger.info("SAT test passed!")
                # we don't actually need to know anything about the SAT problem, only that it was solved
            else:
                # if the solve solver timed out
                if self.solver_timeout and (datetime.datetime.now() - tstart).seconds > self.solver_timeout:
                    raise Exception("SolverTimeoutException")
                # if the solver failed but didn't time out, conclude that the problem
                # is not satisfiable
                else:
                    return False

        # clear Pstar variables
        for lstar, l in itertools.product(range(crystal.num_particles()), range(self.nL)):
            del self.variables[f"Pstar({lstar, l})"]

    def test_crystal_structure(self, crystal_structure: PolycubeStructure,
                               unit_cell_structure: PolycubeStructure) -> bool:
        """
        tests if the crystal matches the provided polycube unit cell
        """
        if crystal_structure.rule != unit_cell_structure.rule:
            self.logger.warning("Trying to compare structures that don't share the same particle set!")
            return False
        # try to overlay unit cell on crystal structure
        # todo?

    def test_crystal(self, sat_solution: SATSolution) -> bool:
        """
        this becomes. tricky.
        this method will only work for non-multifarious crystals


        """

        solution_vars = [p if p in sat_solution.raw else -p for p, _ in enumerate(self.problem.variables) ]

        try:
            polycube_data, T = find_crystal_temperature(sat_solution.decRuleOld(),
                                                        sat_solution.type_counts(),
                                                        self.input_params.tlm_params)
            self.logger.info(f"Rule {str(sat_solution.rule)} crystallizes at T={T}, testing particle types...")
            # hardcode sat solution into solution (i promise this makes sense probably dude trust me bro)

            # only use final timestep polycubes
            final_records: list[libtlm.TLMHistoricalRecord] = [records[-1] for records in polycube_data]
            # convert polycube data to objects
            polycube_data: list[list[PolycubeStructure]] = [[toPolycube(sat_solution.rule, pc) for pc in record] for record in final_records]
            for polycube in itertools.chain.from_iterable(polycube_data):
                # of the polycube has formed a crystal other than the desired one, return false
                # use SAT-solver approach
                if not self.polycube_satisfies(solution_vars, polycube):
                    return False

        except DoesNotCrystalizeException as e:
            self.logger.info(str(e))
            self.logger.info(f"Forbidding solution {str(e)}")
            return False


    def test_multifarious_finite_size(self, sat_solution: SATSolution) -> bool:
        """
        tricky
        """


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
