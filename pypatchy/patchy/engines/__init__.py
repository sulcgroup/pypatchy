"""
Simulation engines: the programs a PatchySimulationEnsemble can build, run and read simulations with.
A server config picks one with its "engine" field (default "oxdna").
"""
from .base import SimulationEngine
from .hoomd_engine import HOOMDEngine
from .oxdna import OxDNAEngine

ENGINES: dict[str, type[SimulationEngine]] = {
    OxDNAEngine.name: OxDNAEngine,
    HOOMDEngine.name: HOOMDEngine,
}


def get_engine(name: str) -> SimulationEngine:
    """
    :param name: an engine name, as given in a server config's "engine" field
    :return: the engine with that name
    """
    if name not in ENGINES:
        raise ValueError(f"No simulation engine called {name!r}. Options are: {', '.join(ENGINES)}")
    return ENGINES[name]()
