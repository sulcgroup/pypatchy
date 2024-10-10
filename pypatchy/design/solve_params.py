import copy
import itertools
import logging
from dataclasses import dataclass, field
from typing import Union

from pypatchy.structure import Structure


# class AllosteryLimits:
#     def __init__(self):
#         self.allow_allostery = True
#
#     def isAllowed(self):
#         return True
#
#     def allostery_type(self):
#         if self.allow_allostery:
#             return None


# NO_ALLOSTERY = AllosteryLimits()
# NO_ALLOSTERY.allow_allostery = False


@dataclass
class CrystallizationTestParams:
    """
    dataclass that holds wide-scale info about how to test a crystal, for the sat solver
    """
    n_replicas: int = field(default=24)
    n_unit_cells: int = field(default=400)  # can apply even to finite size assemblies
    # simulation step count - I strongly advise against leaving this as default!
    n_steps: int = field(default=int(5e6))
    # historical record interval
    record_interval: int = field(default=1000)
    density: float = field(default=0.1)

    # min, max, interval temperatures
    Tmin: float = field(default=0)
    Tmax: float = field(default=1)
    Tinterval: float = field(default=0.001)

    torsion: bool = field(default=True)

    def get_total_type_counts(self, cube_type_counts: list[int]):
        return [n * self.n_unit_cells for n in cube_type_counts]

    @property
    def t_interval_valid(self):
        return self.Tmax - self.Tmin > self.Tinterval

class SolveParams:
    """
    A class that holds the parameters for a SAT solver execution
    It include number of species, number of colors, etc.
    """
    # structure name
    name: str
    # structure topology
    bindings: list[tuple[int, int, int, int]]
    #
    extraConnections: list[tuple[int, int, int, int]]
    # number of species to solve for
    nS: int
    # number of colors to solve for
    nC: int
    # forbid self interact?
    forbid_self_interact: bool
    # positions of nanoparticles
    nanoparticles: Union[dict[int, int], list[int]]
    # number of dimensions (currently locked to 3)
    nDim: int
    # if the structure we're trying to solve for is a crystal
    crystal: bool
    # maximum alternative attempts to solve problem
    maxAltTries: int
    # number of solutions to find (should always be 1?)
    nSolutions: int
    # number of seconds to run the SAT solver for before giving up
    solver_timeout: int
    structure_ids: Union[dict[int, int], None]

    tlm_params: CrystallizationTestParams


    def __init__(self, name, **kwargs):
        self.name = name
        top = copy.deepcopy(kwargs["bindings"])
        self.bindings = copy.deepcopy(kwargs["bindings"])
        if 'extraConnections' in kwargs:
            self.extraConnections = kwargs['extraConnections']
        else:
            self.extraConnections = []

        self.nS = kwargs["nSpecies"] if "nSpecies" in kwargs else None
        self.nC = kwargs["nColors"] if "nColors" in kwargs else None
        # Use 1 solution at a time(glucose) unless otherwise specified
        # TODO: option "dynamic" to choose nSolutions depending on topology size
        self.nSolutions = kwargs["nSolutions"] if "nSolutions" in kwargs else 1
        self.nDim = kwargs["nDim"] if "nDim" in kwargs else 3
        self.torsion = kwargs["torsion"] if "torsion" in kwargs else True
        self.maxAltTries = kwargs["maxAltTries"] if "maxAltTries" in kwargs else 32
        self.nanoparticles = {int(k): kwargs["nanoparticles"][k] for k in kwargs["nanoparticles"]}\
            if "nanoparticles" in kwargs else {}
        # fix nanoparticles if num np types == 1
        if len(self.nanoparticles) > 0 and len(set(self.nanoparticles.values())) == 1:
            self.nanoparticles = list(self.nanoparticles.values())
        self.crystal = bool(kwargs["crystal"]) if "crystal" in kwargs else 'extraConnections' in kwargs
        self.solver_timeout = kwargs["solve_timeout"] if "solve_timeout" in kwargs else 21600
        self.forbid_self_interact = bool(kwargs["forbid_self_interact"]) if "forbid_self_interact" in kwargs else False
        # default timeout: 6 hrs, which is probably both too long and not long enough
        # structure IDS - used for multifarious design
        # maybe unused
        self.structure_ids = None

        # if we're testing a crystal, initialize parameters to pass to tlm
        if self.crystal:
            if "tlm_params" in kwargs:
                self.tlm_params = CrystallizationTestParams(torsion=self.torsion, **kwargs["tlm_params"])
            else:
                # default tlm params, whatever
                # warn?
                self.tlm_params = CrystallizationTestParams(torsion=self.torsion)

    topology: Structure = property(lambda self: Structure(bindings=self.bindings + self.extraConnections))
    is_multifarious: bool = property(lambda self: self.structure_ids is not None)

    def get_locations(self) -> set[int]:
        # can ignore crystal bindings
        return set(itertools.chain.from_iterable([(x[0], x[2]) for x in self.bindings]))

    def num_locations(self) -> int:
        return len(self.get_locations())

    def has_nanoparticles(self):
        return len(self.nanoparticles) > 0

    def brief_descr(self):
        return f"{self.nC},{self.nS}"


    def get_logger(self):
        logger_name = self.get_logger_name()
        assert logger_name in logging.Logger.manager.loggerDict
        return logging.getLogger(logger_name)

    def get_logger_name(self):
        return f"{self.name}_{self.nS}S_{self.nC}C"

    # depreacted: use self.structure_ids instead
    # def is_multifarious(self):
    #     return Structure(bindings=self.bindings).is_multifarious()

    # def __add__(self, other: SolveParams) -> SolveParams:
    #     """
    #     NOT COMMUTITIVE
    #     copies most stuff of self but adds bindings from other
    #     """
    #     combined: SolveParams = copy.deepcopy(self)
    #     combined.name += "_" + other.name
    #     location_map: dict[int, int] = dict()
    #
    #     if not combined.is_multifarious:
    #         combined.structure_ids = {l: 0 for l in combined.get_locations()}
    #
    #     def lmap (i: int) -> int:
    #         if i not in location_map:
    #             iNewVal = 0
    #             # find a number which is not in the set of existing location indices
    #             while iNewVal in combined.get_locations().union(location_map.values()):
    #                 iNewVal += 1
    #             location_map[i] = iNewVal
    #         return location_map[i]
    #
    #     combined.bindings += [
    #         (lmap(i), di, lmap(j), j)
    #         for i, di, j, dj in other.bindings
    #     ]
    #     combined.extraConnections += [
    #         (lmap(i), di, lmap(j), j)
    #         for i, di, j, dj in other.extraConnections
    #     ]
    #
    #     # update multifarious location map
    #     combined.structure_ids.update({l: 0 for l in location_map.values()})
    #
    #     # and i think this is. good?
    #     return combined


def construct(name: str, min_nC: int, max_nC: int, min_nS: int, max_nS: int, max_diff: Union[int, None],
              **kwargs) -> list[SolveParams]:
    """
    what
    """
    paramsets = [SolveParams(
        name,
        nSpecies=nS,
        nColors=nC,
        **kwargs)
                 for nC, nS in itertools.product(range(min_nC, max_nC), range(min_nS, max_nS))
                 if max_diff is None or abs(nS - nC) <= max_diff]
    paramsets.sort(key=lambda p: p.nS + p.nC)
    return paramsets
