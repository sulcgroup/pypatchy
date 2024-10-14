from __future__ import annotations
from typing import TYPE_CHECKING
import itertools
import re
from typing import Union, IO

if TYPE_CHECKING:
    from pypatchy.design.sat.solution import SATSolution

SATClause = Union[tuple[int, ...], list[int, ...]]


def exactly_one(vs: list[int]) -> list[SATClause]:
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
    return [tuple(sorted(vs)), *at_most_one(*vs)]


def n_or_fewer(vs: list[int], n: int):
    # don't call this function with more than then number of vars, or with less than 1
    assert 0 < n < len(vs)
    if n == 1:
        # first constraints in exactly_one is [v1 or v2 or v3... or vn]
        return exactly_one(vs)[1:]
    else:
        constraints = []
        combos = itertools.combinations(vs, n + 1)
        for c in combos:
            constraints.append((
                -v for v in c
            ))

def at_most_one(*args: int) -> list[SATClause]:
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


class SATProblemPart:
    """
    we define a "problem part" as a boolean statement involving variables, e.g.
    (F(li,pi,ci) ∧ (Flj,pj,cj) ⇒ B(ci,cj)
    which are NOT in CNF. (e.gs. in Table 2 of https://pubs.acs.org/doi/10.1021/acsnano.2c09677), where
    the author confusingly refers to problem parts as "SAT Clauses". For our purposes a SAT Clause
    is a boolean formula using only 'v' and '~' (e.g. a v b v ~c), which forms part of a CNF
    statement https://en.wikipedia.org/wiki/Conjunctive_normal_form
    a problem part can and usually does consist of a large number of SAT clauses, but
    a SAT clause can only correspond to one problem part
    """

    problem_part_name: str
    clauses: list[SATClause]

    def __init__(self, name: str, vars: list[str]):
        self.problem_part_name = name
        self.clauses = []

    def __iter__(self):
        yield from self.clauses


class RepeatPartException(BaseException):
    pass


class SATProblem:
    """
    purpose of this class: store information about a sat problem, seperating it from
    methods about executing the problem or specifics about clause meaning

    """
    # for defn. of "problem  part" see the SATProblem class doc
    sat_problem_parts: dict[str, SATProblemPart]
    variables: dict[str, int]

    def __init__(self):
        self.sat_problem_parts = dict()
        self.variables = dict()

    def get_problem_part(self, name: str) -> SATProblemPart:
        return self.sat_problem_parts[name]

    def add_problem_part(self, part: SATProblemPart, replace_if_exist: bool=False):
        if part.problem_part_name in self.sat_problem_parts and not replace_if_exist:
            raise RepeatPartException(f"The SAT problem already has a part with name {part.problem_part_name}")
        self.sat_problem_parts[part.problem_part_name] = part

    def clear_problem_part(self, name: str):
        del self.sat_problem_parts[name]

    def __iter__(self):
        yield from itertools.chain.from_iterable(self.sat_problem_parts.values())

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

    def num_variables(self) -> int:
        return len(self.variables)

    def num_clauses(self) -> int:
        return sum([len(part.clauses) for part in self.sat_problem_parts.values()])

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
        n = 1
        while f"forbidden{n}" in self.sat_problem_parts:
            n += 1
        problem_part = SATProblemPart(f"forbidden{n}", [])
        self.add_problem_part(problem_part)
        problem_part.clauses = forbidden

    def output_cnf(self, out: Union[IO, None] = None) -> str:
        """ Outputs a CNF formula """
        num_vars = max(self.variables.values())
        # if self.allo_clauses is not None:
        #     num_constraints += len(self.allo_clauses)
        nclauses = 0
        # add basic clauses
        outstr = str()
        for c in self:
            nclauses += 1
            outstr += ' '.join([str(v) for v in c]) + ' 0\n'
        outstr = "p cnf %s %s\n" % (num_vars, nclauses) + outstr

        if out is not None:
            out.write(outstr)
        return outstr

    def describe_brief(self) -> str:
        return f"SAT Problem with {self.num_variables()} vars and {self.num_clauses()} clauses."

    def dump_cnf_to_file(self, fname):
        parameters = self.output_cnf()
        with open(fname, 'w') as outf:
            outf.write(parameters)


def interrupt(s):
    print("Timeout. Interrupting solve...")
    s.interrupt()
