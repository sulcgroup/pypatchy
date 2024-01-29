import copy
import itertools
import logging
from typing import Union

from pypatchy.structure import Structure


class AllosteryLimits:
    def __init__(self):
        self.allow_allostery = True

    def isAllowed(self):
        return True

    def allostery_type(self):
        if self.allow_allostery:
            return None


NO_ALLOSTERY = AllosteryLimits()
NO_ALLOSTERY.allow_allostery = False


class SolveParams:
    """
    A class that holds the parameters for a SAT solver execution
    """
    topology: Structure
    bindings: list[tuple[int, int, int, int]]
    nS: int
    nC: int
    forbid_self_interact: bool

    def __init__(self, name, **kwargs):
        self.name = name
        top = copy.deepcopy(kwargs["bindings"])
        self.bindings = copy.deepcopy(kwargs["bindings"])
        if 'extraConnections' in kwargs:
            top += kwargs['extraConnections']
        self.topology = Structure(bindings=top)
        self.nS = kwargs["nSpecies"] if "nSpecies" in kwargs else None
        self.nC = kwargs["nColors"] if "nColors" in kwargs else None
        # Use 1 solution at a time(glucose) unless otherwise specified
        # TODO: option "dynamic" to choose nSolutions depending on topology size
        self.nSolutions = kwargs["nSolutions"] if "nSolutions" in kwargs else 1
        self.nDim = kwargs["nDim"] if "nDim" in kwargs else 3
        self.torsion = kwargs["torsion"] if "torsion" in kwargs else True
        self.maxAltTries = kwargs["maxAltTries"] if "maxAltTries" in kwargs else 32
        self.nanoparticles = kwargs["nanoparticles"] if "nanoparticles" in kwargs else {}
        self.crystal = bool(kwargs["crystal"]) if "crystal" in kwargs else 'extraConnections' in kwargs
        self.allo_limits = kwargs["allo_limits"] if "allo_limits" in kwargs else NO_ALLOSTERY
        self.solver_timeout = kwargs["solve_timeout"] if "solve_timeout" in kwargs else 21600
        self.forbid_self_interact = bool(kwargs["forbid_self_interact"]) if "forbid_self_interact" in kwargs else False
        # default timeout: 6 hrs, which is probably both too long and not long enough

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

    def is_multifarious(self):
        return Structure(bindings=self.bindings).is_multifarious()


def construct(name: str, min_nC: int, max_nC: int, min_nS: int, max_nS: int, max_diff: Union[int, None],
              **kwargs) -> list[SolveParams]:
    paramsets = [SolveParams(
        name,
        nSpecies=nS,
        nColors=nC,
        **kwargs)
                 for nC, nS in itertools.product(range(min_nC, max_nC), range(min_nS, max_nS))
                 if max_diff is None or abs(nS - nC) <= max_diff]
    paramsets.sort(key=lambda p: p.nS + p.nC)
    return paramsets
