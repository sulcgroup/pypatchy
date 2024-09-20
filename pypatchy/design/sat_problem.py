import copy
import itertools
import logging
from abc import ABC, abstractmethod
import re
from typing import Union, IO, Iterable

from pypatchy.util import enumerateRotations

SATClause = Union[tuple[int, ...], list[int, ...]]


class SATProblem(ABC):
    # number of locations
    nL: int
    # number of rotations
    nR: int
    # bindings
    bindings: dict[tuple[int, int]: tuple[int, int]]  # ngl I have no idea why they're like this
    internal_bindings: list[tuple[int, int, int, int]]

    # map of SAT variable names to SAT variable numbers
    variables: dict[str, int]

    # list of all possible rotations (dependant on dimensionality, for now it's fixed)
    rotations: dict[int: dict[int, int]]

    # list of "basic" SAT clauses
    basic_sat_clauses: list[SATClause]

    # logger for SAT solver
    logger: logging.Logger
    # timeout for SAT solver
    solver_timeout: Union[int, None]

    def __init__(self, logger: logging.Logger, timeout: int):
        self.logger = logger
        self.solver_timeout = timeout

        self.rotations = enumerateRotations()
        self.basic_sat_clauses = []  # string of s basic sat clause
        self.nR = len(self.rotations)
        self.variables = {}
        self.nP = 6  #: Number of patches on a single particle

    @abstractmethod
    def init(self):
        pass

    @abstractmethod
    def generate_constraints(self) -> list[SATClause]:
        pass

    def attach_logger(self, logger: logging.Logger):
        self.logger = logger

    def list_vars(self) -> list[tuple[str, int]]:
        """
        Returns: a list of tuples where each tuple is (name of variable, variable number)

        """
        return list(self.variables.items())

    def variable(self, name: str, *args: int) -> int:
        """
        Returns a variable X(a, b, c,...) where X is a string and a,b,c... are ints
        to an int value so it can be used in the SAT problem

        If the variable does not already exist, it is assigned a new int value based on the length
        of self.variables

        Best practice: This function should not be used as-is but should be used in functions
        in subclasses of type `def X(a,b,...)`
        Args:
            name: the name of the variable
            *args: a tuple of int characteristics (word?) for the variable)

        Returns:
            an int value to correspond to the variable in a CNF expression of the SAT problem
        """
        str_key = f"{name}({','.join([str(a) for a in args])})"
        return self.variables.setdefault(str_key, len(self.variables) + 1)

    def num_variables(self):
        return len(self.variables)

    def num_clauses(self):
        return len(self.basic_sat_clauses)

    def get_solution(self, sat_results: frozenset) -> list[tuple[str, int]]:
        """
        Given a solution represented as a set of variable numbers (ints), returns a list of tuple representations
        of the variables that were true in that solution
        Args:
            sat_results: a set of ints that are variables that are true in a solution to the sat problem

        Returns: a list of tuple representations of the variables that were true in that solution

        """
        return [(vname, vnum) for vname, vnum in self.list_vars() if vnum in sat_results]

    def get_solution_vars(self, sat_results: frozenset) -> dict[str, list[tuple[int, tuple[int, ...]]]]:
        """
        Returns the solution to this SAT problem organized by variable.
        Each variable name is a dict key, and the value is a tuple where the first element is the variable
        index, and the second is a tuple of variable attributes
        Args:
            sat_results: a frozenset of ints that are variables in the solution

        Returns:
            the solution as a dict of variables
        """
        solution_vars = dict()
        for vname, vnum in self.get_solution(sat_results):
            m = re.match(r"(\w)\(([\d\s,]*)\)", vname)
            assert m is not None
            if m.group(1) not in solution_vars:
                solution_vars[m.group(1)] = []
            numbers = tuple(map(int, re.findall(r"\d+", m.group(2))))
            solution_vars[m.group(1)].append((vnum, numbers))
        return solution_vars

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

    def set_crystal_topology(self, bindings: Iterable[tuple[int, int, int, int]]):
        """
        Accepts an array of integer tuples bindings, of format [particle_id1,patch1,particle_id_2,patch2], where particle_id1 uses patch1 to bind to particle_id2 on patch2
        Each interacting pair is only to be listed once
        """
        self.bindings = {
            (int(p1), int(s1)): (int(p2), int(s2))
            for (p1, s1, p2, s2) in bindings
        }
        self.check_bindings(self.bindings)

    def _exactly_one(self, vs: list[int]) -> list[SATClause]:
        """
        returns a list of constraints implementing "exacly one of vs is true"
        This is accomplished by first adding a constraint requiring that one value in the list be true,
        and then adding constraints requiring that for any arbirtrary pair of variables
        in the list, one variable in the pair be false
        """
        # assert all(v > 0 for v in vs)
        # assert len(vs) > 1
        # # add a constratint requiring that any of the variables be true
        # constraints = [tuple(sorted(vs))]
        # # outer loop for variables
        # for v1 in sorted(vs):
        #     # inner loop for variables
        #     for v2 in sorted(vs):
        #         # don't double-count variables
        #         if v2 >= v1:
        #             break
        #         # add constraints specifying that one of these two be false
        #         constraints.append((-v1, -v2))
        # assert len(set(constraints)) == (len(vs) * (len(vs) - 1)) / 2 + 1
        # return constraints
        return [tuple(sorted(vs)), *self._at_most_one(*vs)]


    def _at_most_one(self, *args: int) -> list[SATClause]:
        assert all(v > 0 for v in args)
        assert len(args) > 1
        constraints = []
        # outer loop for variables
        for v1 in sorted(args):
            # inner loop for variables
            for v2 in sorted(args):
                # don't double-count variables
                if v2 >= v1:
                    break
                # add constraints specifying that one of these two be false
                constraints.append((-v1, -v2))
        assert len(set(constraints)) == (len(args) * (len(args) - 1)) / 2
        return constraints


    def _n_or_fewer(self, vs, n):
        # don't call this function with more than then number of vars, or with less than 1
        assert 0 < n < len(vs)
        if n == 1:
            # first constraints in exactly_one is [v1 or v2 or v3... or vn]
            return self._exactly_one(vs)[1:]
        else:
            constraints = []
            combos = itertools.combinations(vs, n + 1)
            for c in combos:
                constraints.append((
                    -v for v in c
                ))

    @abstractmethod
    def output_cnf(self, out: Union[IO, None] = None) -> str:
        """ Outputs a CNF formula """
        pass

    def dump_cnf_to_file(self, fname):
        parameters = self.output_cnf()
        with open(fname, 'w') as outf:
            outf.write(parameters)


def interrupt(s):
    print("Timeout. Interrupting solve...")
    s.interrupt()

