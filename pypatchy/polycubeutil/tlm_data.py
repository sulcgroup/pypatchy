# classes with idential APIs to libtlm pybind classes
# as such do NOT use "from tlm_data import ...", use "import tlm_data" for namespace clarity
# placing this in pypatchy.polycubeutil so it's usable without libtlm installed

from __future__ import annotations

import numpy as np

from pypatchy.util import from_xyz


class TLMHistoryRecord:
    """
    python copy for libtlm.TLMHistoryRecord
    """
    # ---- functions mimicking libtlm.TLMHistoryRecord API ---------------------------
    def stepCount(self) -> int:
        return self.__steps

    def numPolycubes(self) -> int:
        return len(self.__polycubes)

    def numCubeInstances(self) -> int:
        return self.__num_cubes_insts

    def energy(self) -> float:
        return self.__energy

    # ------ internal storage data
    __steps: int
    __polycubes: list[TLMPolycube]
    __num_cubes_insts: int #save this as an int
    __energy: float

    def __init__(self, **kwargs):
        """
        constructor from kwargs or more likely from a json dict
        we are not in favor of flexability here
        """
        assert "step_number" in kwargs and "polycubes" in kwargs and "bond_energy" in kwargs
        self.__steps = kwargs["step_number"]
        self.__polycubes = [TLMPolycube(**pcdata) for pcdata in kwargs["polycubes"]]
        self.__num_cubes_insts = sum([len(pc.getCubes()) for pc in self.__polycubes])
        self.__energy = kwargs["bond_energy"]

    def getPolycubes(self) -> list[TLMPolycube]:
        return self.__polycubes

class TLMPolycube:
    """
    python copy for libtlm.TLMPolycube
    """
    # ----------- libtlm.TLMPolycube API functions
    def getID(self) -> int:
        return self.__uid

    def numCubes(self) -> int:
        return len(self.__cubes)

    def getCubes(self) -> list[TLMCubeData]:
        return self.__cubes

    # ---- interalal storage vars
    __uid: int
    __cubes: list[TLMCubeData]

    def __init__(self, uid, cubes):
        self.__uid = uid
        self.__cubes = [TLMCubeData(**cube) for cube in cubes]

class TLMCubeData:
    """
    python copy for libtlm.TLMCubeData
    """
    def getUID(self) -> int:
        return self.__uid

    def getTypeID(self) -> int:
        return self.__type_id

    def getPosition(self) -> np.ndarray:
        return self.__position

    def getRotation(self) -> np.ndarray: # quaternion *sigh*
        return self.__rotation

    def getState(self) -> np.ndarray:
        return self.__state

    # ---- internal storage vars
    __uid: int
    __type_id: int
    __position: np.ndarray
    __rotation: np.ndarray
    __state: np.ndarray

    def __init__(self, **kwargs):
        # name
        self.__uid = int(kwargs["name"][4:])
        self.__type_id = kwargs["type"]
        self.__position = from_xyz(kwargs["position"])
        self.__state = np.array(kwargs["state"])
        cr = kwargs["rotation"]
        # quaternion order in np is xyzw, i think?
        self.__rotation = np.array((
                        cr['x'],
                        cr['y'],
                        cr['z'],
                        cr['w']
                    ))
