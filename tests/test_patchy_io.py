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
    writer.set_directory(Path(__file__).parent / "test_files" / "patchy_scenes" / "3dcross_fr")
    scene = writer.read_scene("topology.top",
                              "last_conf.dat",
                              writer.read_particle_types(patchy_file="patches.txt",
                                                         particle_file="particles.txt"))
    return scene


@pytest.fixture
def cross_lr():
    writer = get_writer("lorenzo")
    writer.set_directory(Path(__file__).parent / "test_files" / "patchy_scenes" / "3dcross_lr")
    scene = writer.read_scene("topology.top", "last_conf.dat",
                              writer.read_particle_types("topology.top", "interactions.txt"))
    return scene


@pytest.fixture
def cross_mgl():
    pass  # TODO


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
    pass


def test_read_loro_test():
    """
    tests the files included by micha with his ppview program
    https://github.com/zoombya/ppview/tree/main/public/LORO_patch_example
    """
    writer = get_writer("lorenzo")
    writer.set_directory(Path(__file__).parent / "test_files" / "patchy_scenes" / "loro_eg")
    scene = writer.read_scene("LORO.lorenzo.topology.top",
                              "last_conf.dat",
                              writer.read_particle_types("LORO.lorenzo.topology.top",
                                                         "LORO.interaction_matrix.txt"))
    scene.to_multidentate(n_teeth=3, dental_radius=0.05)
    pass
