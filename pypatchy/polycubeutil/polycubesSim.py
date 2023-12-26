import json
from pathlib import Path
from typing import Union

from pypatchy.polycubeutil.polycube_structure import PolycubeStructure
from pypatchy.polycubeutil.polycubesRule import PolycubesRule
from pypatchy.polycubeutil.polycubes_setup import setup_from_json, PolycubesSetup
from pypatchy.util import get_output_dir, get_input_dir

# TODO: read json tag names from tlm c++

class PolycubesSimStep:
    """
    A step in a (TLM?) polycubes simulation
    """
    structures: list[PolycubeStructure]
    step_num: int

    def __init__(self, step_num: int, pcs: list[PolycubeStructure]):
        self.step_num = step_num
        self.structures = pcs

def load_sim_step(data: dict, ctxt: PolycubesSetup) -> PolycubesSimStep:
    step_num = data["step_number"]
    structures: list[PolycubeStructure] = []
    for pc_data in data["polycubes"]:
        structures.append(PolycubeStructure(rule=ctxt.rule(), structure=pc_data))
    return PolycubesSimStep(step_num, structures)


class PolycubesSim:
    """
    Represents a polycubes simulation.
    Can import from TLM or (TODO) the web app
    """
    cube_types: PolycubesRule # all cube types used in this simulation
    steps: dict[int, PolycubesSimStep]
    # todo: staging info?

def load_sim_data(file: Union[Path, str]) -> PolycubesSim:
    if isinstance(file, str):
        return load_sim_data(get_input_dir() / file)
    else:
        with file.open("r") as f:
            data = json.load(f)
        setup = setup_from_json(data["setup"])
        records = [PolycubesSimStep(setup, record) for record in json["history"]]
        return PolycubesSim(setup, records)

def load_tlm_data(file_name: str) -> PolycubesSim:
    return load_sim_data(get_output_dir() / "tlm" / file_name)