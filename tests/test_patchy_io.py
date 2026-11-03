"""
This file contains code for testing the read/write capabilities of pypatchy
specifically with regard to
"""
import tempfile
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from pypatchy.patchy.pl.patchyio import get_writer, read_mgl
from pypatchy.patchy.pl.plparticleset import MultidentateConvertSettings
from pypatchy.patchy.pl.plpatchylib import polycube_to_pl
from pypatchy.patchy.pl.plpotential import PLFRPatchyPotential, PLLRPatchyPotential
from pypatchy.patchy.pl.plscene import PLPSimulation
from pypatchy.polycubeutil.polycube_structure import PolycubeStructure

from test_polycube_lib import ducky

@pytest.fixture
def temp_dir() -> str:
    # construct temporary directory to run

    with tempfile.TemporaryDirectory() as td:
        yield td

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
                              writer.read_particle_types(topology="topology.top",
                                                         DPS_interaction_matrix_file="interactions.txt"))
    return scene


@pytest.fixture
def cross_mgl():
    pass  # TODO


# @pytest.fixture
def test_cross_sr(temp_dir, cross_fr):
    """tests subhajit-format writer"""
    writer = get_writer("subhajit")
    writer.set_directory(temp_dir)
    writer.write(
        cross_fr,
        topology="topology.top",
        conf_file="last_conf.dat",
    )
    pass


def test_cross_raspberry(temp_dir, cross_fr):
    """tests subhajit-format writer"""
    writer = get_writer("raspberry")
    writer.set_directory(temp_dir)
    writer.write(
        cross_fr,
        topology="topology.top",
        conf_file="last_conf.dat",
    )
    pass


def test_raspberry_round_trip_default_sites(temp_dir, cross_fr):
    """with no custom repulsion sites, every type gets the default cube shell and it survives a round trip"""
    writer = get_writer("raspberry")
    writer.set_directory(temp_dir)
    writer.write(cross_fr, topology="topology.top", conf_file="last_conf.dat")

    top = writer.read_top("topology.top")
    defaults = writer.RPatchyTopology.default_repulsion_sites()
    assert len(top.particle_repulsion_sites) == len(defaults)
    for (pos, r), (dpos, dr) in zip(top.particle_repulsion_sites, defaults):
        assert np.allclose(pos, dpos) and r == pytest.approx(dr)
    for ptype in cross_fr.particle_types():
        assert top.type_rsite_lists[ptype.type_id()] == list(range(len(defaults)))
    assert top.num_particles() == cross_fr.num_particles()


def test_raspberry_per_type_repulsion_sites(temp_dir, cross_fr):
    """different particle types can use different repulsion sites, including none at all"""
    writer = get_writer("raspberry")
    writer.set_directory(temp_dir)
    type_ids = [ptype.type_id() for ptype in cross_fr.particle_types()]
    assert len(type_ids) >= 2, "test needs at least two particle types"

    sites = [(np.array([0., 0., 0.]), 0.5),
             (np.array([0.3, 0., 0.]), 0.2),
             (np.array([-0.3, 0., 0.]), 0.2)]
    rsite_lists = {tid: [0] for tid in type_ids}
    rsite_lists[type_ids[0]] = [1, 2]  # dumbbell-ish shape
    rsite_lists[type_ids[1]] = []  # no steric repulsion at all
    scene_top = writer.RPatchyTopology(cross_fr.particle_types(),
                                       cross_fr.particle_type_counts(),
                                       repulsion_sites=sites,
                                       type_rsite_lists=rsite_lists)
    writer.write(cross_fr, topology="topology.top", conf_file="last_conf.dat", scene_top=scene_top)

    # the type with no sites must omit the field entirely - oxDNA would read "-1" as a site index
    with open(Path(temp_dir) / "topology.top") as f:
        ic_lines = {int(line.split()[1]): line.split() for line in f if line.startswith("iC")}
    assert len(ic_lines[type_ids[1]]) == 4
    assert ic_lines[type_ids[0]][4] == "1,2"

    top = writer.read_top("topology.top")
    assert top.type_rsite_lists == rsite_lists
    for (pos, r), (spos, sr) in zip(top.particle_repulsion_sites, sites):
        assert np.allclose(pos, spos) and r == pytest.approx(sr)
    assert len(top.type_repulsion_sites(type_ids[0])) == 2
    assert top.type_repulsion_sites(type_ids[1]) == []


def test_raspberry_read_optional_fields(temp_dir):
    """the reader accepts every iC form the oxDNA parser accepts"""
    with open(Path(temp_dir) / "topology.top", "w") as f:
        f.write("3 3\n"
                "iP 0 1.0 21 0,0,0.5 0,0,1\n"
                "iP 1 1.0 -21 0,0,-0.5 0,0,-1\n"
                "iR 0.25,0.25,0.25 0.25\n"
                "iR -0.25,-0.25,-0.25 0.25\n"
                "iC 0 1 0 0,1\n"  # standard
                "iC 1 1 1 1 2\n"  # trailing state size
                "iC 2 1 -1\n")  # no patches, no repulsion sites
    writer = get_writer("raspberry")
    writer.set_directory(temp_dir)
    top = writer.read_top("topology.top")
    assert top.type_rsite_lists == {0: [0, 1], 1: [1], 2: []}
    assert top.particle_types.particle(2).num_patches() == 0
    assert top.particle_types.particle(1).num_patches() == 1


def test_raspberry_invalid_repulsion_sites(cross_fr):
    writer = get_writer("raspberry")
    type_ids = [ptype.type_id() for ptype in cross_fr.particle_types()]
    with pytest.raises(ValueError, match="only 1 sites"):
        writer.RPatchyTopology(cross_fr.particle_types(), cross_fr.particle_type_counts(),
                               repulsion_sites=[((0, 0, 0), 0.5)],
                               type_rsite_lists={tid: [1] for tid in type_ids})
    with pytest.raises(ValueError, match="No repulsion site list"):
        writer.RPatchyTopology(cross_fr.particle_types(), cross_fr.particle_type_counts(),
                               repulsion_sites=[((0, 0, 0), 0.5)],
                               type_rsite_lists={})
    with pytest.raises(ValueError, match="non-positive radius"):
        writer.RPatchyTopology(cross_fr.particle_types(), cross_fr.particle_type_counts(),
                               repulsion_sites=[((0, 0, 0), 0.)])


def test_fwriter_read(cross_fr):
    pass

@pytest.fixture
def icos_mgl() -> PLPSimulation:
    """
    loads a single icosahedron from an mgl file
    """
    p = Path(__file__).parent / "test_files" / "patchy_scenes" / "icos_capsid" / "icosahedron.mgl"
    # second arg is patch color interactions, decimals  are interation strengths
    scene = read_mgl(p,
                     {
                         ("blue", "darkblue"): 1.,
                         ("green", "darkgreen"): 1.,
                         ("red", "darkred"): 1.
                     }
                     )

    # sigma ss is largely just the number that made this work for me
    scene.add_potential(PLLRPatchyPotential(
        rmax=1.2,
        interaction_matrix=scene.particle_types().interaction_matrix(),
        sigma_ss=0.7
    ))

    # scene.export_to_mgl("icos.mgl", patches_w=0.2)
    assert scene.particle_types().num_particle_types() == 3
    G = scene.compute_scene_graph()
    assert len(G.edges) == 30
    return scene

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
                              writer.read_particle_types(topology="LORO.lorenzo.topology.top",
                                                         DPS_interaction_matrix_file="LORO.interaction_matrix.txt"))
    scene.to_multidentate(MultidentateConvertSettings(n_teeth=3,
                                                      dental_radius=0.05,
                                                      torsion=False
                                                      ))
    pass


def test_convert_duck(ducky: PolycubeStructure):
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

# todo: put this in a different file
@pytest.fixture
def multidentate_patchy_ducky(ducky: PolycubeStructure) -> PLPSimulation:
    scene = polycube_to_pl(ducky, MultidentateConvertSettings(n_teeth=4, dental_radius=0.5))
    scene.add_potential(
        PLFRPatchyPotential(
            rmax=0.4 * 3,
            alpha=0.12
        )
    )
    return scene

def test_mdt_ducky(multidentate_patchy_ducky: PLPSimulation):
    assert len(multidentate_patchy_ducky.compute_scene_graph().edges) == 7 # i counted and it is 7
    assert len(multidentate_patchy_ducky.scene_multigraph().edges) == 7 * 4
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

def test_chain(ducky: PolycubeStructure):
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
        G_old = scene.compute_scene_graph()
        scene = writer.read_scene(
            "topology.top",
            "last_conf.dat",
            writer.read_particle_types(
            particle_file="particles.txt",
            patchy_file="patches.txt"))
        scene.add_potential(
            PLFRPatchyPotential(
                rmax=0.4 * 3,
                alpha=0.12
            )
        )
        G_new = scene.compute_scene_graph()
        assert nx.utils.graphs_equal(G_new, G_old)
