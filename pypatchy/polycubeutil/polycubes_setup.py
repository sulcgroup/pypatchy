### TODO: PyBind11 integration!!
from pathlib import Path

from pypatchy.polycubeutil.polycubesRule import PolycubesRule


class PolycubesSetup:
    """
    currently a wrapper for a polycubes rule
    """
    __rule: PolycubesRule

    def __init__(self, r: PolycubesRule):
        self.__rule = r

    def rule(self) -> PolycubesRule:
        return self.__rule

def setup_from_json(data: dict) -> PolycubesSetup:
    if "rule" in data:
        rule = PolycubesRule(rule_str=data["rule"])
    else:
        rule = PolycubesRule(rule_json=data["cube_types"])

def setup_from_inputfile(f: Path) -> PolycubesSetup:
    """
    constructs setup info object from a oxdna-style input file
    """
    pass