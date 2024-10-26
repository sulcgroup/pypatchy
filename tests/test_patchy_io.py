"""
This file contains code for testing the read/write capabilities of pypatchy
specifically with regard to
"""
from pathlib import Path

import pytest

from pypatchy.patchy.pl.patchyio import get_writer

@pytest.fixture
def cross_fr():
    """
    tests flavio-format writer
    """
    writer = get_writer("flavio")
    writer.set_directory(Path(__file__) / "patchy_scenes" / "3dcross_scene_fr")
    scene = writer.read_scene("topology.top",
                              "last_conf.dat",
                              writer.read_particle_types(patchy_file="patches.txt",
                                                         particle_file="particles.txt"))
    return scene

@pytest.fixture
def cross_lr():
    writer = get_writer("lorenzo")
    writer.set_directory(Path(__file__) / "patchy_scenes" / "3dcross_scene_lr")
    scene = writer.read_scene("topology.top", "last_conf.dat", writer.read_particle_types("topology.top", "interactions.txt"))
    return scene

@pytest.fixture
def cross_mgl():
    pass # TODO

@pytest.fixture
def cross_sr():
    """tests subhajit-format writer"""
    writer = get_writer("subhajit")
    # TODO: subhajit test 


def test_fwriter_read(cross_fr):
    pass

def test_lwriter_read(cross_lr):
    """
    tests lorenzo-format writer
    """

