"""
This file contains code for testing the read/write capabilities of pypatchy
specifically with regard to
"""
import tempfile
from pathlib import Path

import pytest

from pypatchy.patchy.pl.patchyio import get_writer
from pypatchy.patchy.pl.plparticleset import MultidentateConvertSettings
from pypatchy.patchy.pl.plpatchylib import polycube_to_pl
from pypatchy.patchy.pl.plpotential import PLFRPatchyPotential
from pypatchy.patchy.pl.plscene import PLPSimulation
from pypatchy.polycubeutil.polycube_structure import PolycubeStructure

from test_polycube_lib import ducky

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


def test_convert_duck(ducky: PolycubeStructure) -> PLPSimulation:
    scene: PLPSimulation = polycube_to_pl(ducky)
    scene.add_potential(
        PLFRPatchyPotential(
            rmax=0.4 * 3,
            alpha=0.12
        )
    )
    with tempfile.TemporaryDirectory() as td:
        writer = get_writer("flavio")
        writer.set_directory(td)
        writer.write(scene,
                     topology="topology.top",
                     conf_file="last_conf.dat",
                     particle_file="particles.txt",
                     patchy_file="patches.txt")
        print(td)
        return scene

# todo: put this in a different file
@pytest.fixture
def multidentate_patchy_ducky(ducky: PolycubeStructure) -> PLPSimulation:
    scene = polycube_to_pl(ducky, MultidentateConvertSettings(n_teeth=4, dental_radius=0.25))
    scene.add_potential(
        PLFRPatchyPotential(
            rmax=0.4 * 3,
            alpha=0.12
        )
    )
    return scene

def test_mdt_ducky(multidentate_patchy_ducky: PLPSimulation):
    # assert len(multidentate_patchy_ducky.compute_scene_graph().edges) == 14
    # text export (FR patchy)
    with tempfile.TemporaryDirectory() as td:
        writer = get_writer("flavio")
        writer.set_directory(td)
        writer.write(multidentate_patchy_ducky,
                     topology="topology.top",
                     conf_file="last_conf.dat",
                     particle_file="particles.txt",
                     patchy_file="patches.txt")
        print(td)
        pass