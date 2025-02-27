import shutil
from pathlib import Path

import pytest

from pypatchy.polycubeutil.polycube_structure import load_polycube, PolycubeStructure
from pypatchy.polycubeutil.polycubesRule import PolycubesRule


def test_read_string_rule():
    """
    test that we can load rule from a string
    """
    # using the menger cube rule
    rule_string = "1:2||1:2||1:2|_-1:3|2|9||13:3|_-2||3||5|_-3|4:1||-10:2||_-4||-6:2||-7|_-5:1|6:2|-11|||_7|8:1||||_-8||-8||-8|_-20:3||-21:1||16|_20|22:1||10||_-22:2||23:3|||15:3_-23|24|||11:1|_-24:2||19|||14:3_-19|18|-9|||_-18:1||17:3|||12:2_-17|21|||-13:1|_-16|-14:1||||_-15|-15:1||||_-12|-12:3||||"
    rule = PolycubesRule(rule_str=rule_string)
    assert rule.num_particle_types() == 19
    assert rule.num_colors() == 48
    # TODO: more tests?

@pytest.fixture
def ducky():
    """
    test that we can load a polycube from a json file
    """
    polycube_file = Path(__file__).parent / "test_files" / "cube_structures" / "duck_polycube.json"
    polycube = load_polycube(polycube_file)
    assert polycube.rule == PolycubesRule(rule_str="1:3|3:3||3|3|3_|||3:1|-2:2|_-1||||2:1|_-3:1|||||")
    assert polycube.num_vertices() == 8
    assert polycube.num_connections() == 7
    # todo: more tests?
    return polycube

def test_polycube_json(ducky: PolycubeStructure):
    pass # just run the fixture

def test_make_rule_continuous():
    """
    tests code which splices out unused colors
    """
    rule_expanded = "1:3|3:3||3|3|3_|||3:1|-5:2|_-1||||5:1|_-3:1|||||"
    rule = PolycubesRule(rule_str=rule_expanded)
    rule.make_colors_continuous()
    assert rule.num_particle_types() == 4
    assert rule.num_colors() == 3*2