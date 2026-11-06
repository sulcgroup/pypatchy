"""
Tests for computing PLPSimulation energies, bonds and particle placement with oxDNA's own interactions (via oxpy),
which replaced pypatchy's Python re-implementations of the patchy potentials.
"""
import numpy as np
import pytest

from pypatchy.patchy.pl.oxpy_energy import PatchBondAmbiguityWarning
from pypatchy.patchy.pl.plpatchylib import polycube_rule_to_PL, polycube_to_pl
from pypatchy.patchy.pl.plparticleset import MultidentateConvertSettings
from pypatchy.patchy.pl.plscene import PLPSimulation
from pypatchy.polycubeutil.polycube_structure import PolycubeStructure
from pypatchy.polycubeutil.polycubesRule import PolycubesRule

from ptutils import energy_model, requires_energy_model
from test_polycube_lib import ducky

FORMATS = ["flavio", "lorenzo", "raspberry"]


def random_scene(patchy_format: str, n_per_type: int = 20, density: float = 0.1, seed: int = 0) -> PLPSimulation:
    """
    places particles of a two-species rule at random with PLPSimulation.add_particle_rand_positions
    """
    np.random.seed(seed)
    ptypes = polycube_rule_to_PL(PolycubesRule(rule_str="1|1|1|1|1|1_-1|-1|-1|-1|-1|-1"))
    n = n_per_type * ptypes.num_particle_types()
    box = (n / density) ** (1 / 3)
    scene = PLPSimulation()
    scene.set_particle_types(ptypes)
    scene.set_box_size(np.array([box] * 3))
    scene.compute_cell_size(n_particles=n)
    scene.apportion_cells()
    scene.set_temperature(0.1)
    scene.set_energy_model(energy_model(patchy_format))
    scene.add_particle_rand_positions([ptypes.particle(i // n_per_type).instantiate(i) for i in range(n)])
    return scene


@pytest.mark.parametrize("patchy_format", FORMATS)
def test_random_placement_and_energy(patchy_format: str):
    if not __import__("ptutils").plugin_available(patchy_format):
        pytest.skip(f"no {patchy_format} plugin")
    scene = random_scene(patchy_format)
    assert scene.num_particles() == 40
    # oxDNA's own total and the sum over pairs agree
    assert scene.get_potential_energy() == pytest.approx(scene.get_energy_full(), abs=1e-9)
    # placement rejects overlaps: no strongly repulsive pairs
    with scene.energy_session() as session:
        pair_energies = session.pair_energies()
    assert max(pair_energies.values(), default=0.) < 1.


@requires_energy_model("raspberry")
def test_check_for_particle_overlap():
    scene = random_scene("raspberry")
    particle = scene.particle_types().particle(0).instantiate(scene.num_particles())
    # right on top of an existing particle: always rejected
    particle.set_position(scene.get_particle(0).position())
    particle.set_random_orientation()
    assert scene.check_for_particle_overlap(particle, boltzmann=False)


@requires_energy_model("flavio")
def test_add_conf_clusters_keeps_cluster_bonds(ducky: PolycubeStructure):
    cluster = polycube_to_pl(ducky)
    cluster.set_energy_model(energy_model("flavio"))
    n_cluster_bonds = len(cluster.compute_scene_graph().edges)
    assert n_cluster_bonds == 7

    np.random.seed(1)
    scene = PLPSimulation()
    scene.set_particle_types(cluster.particle_types())
    scene.set_box_size(np.array([15., 15., 15.]))
    scene.compute_cell_size(n_cells=3)
    scene.apportion_cells()
    scene.set_temperature(0.1)
    scene.set_energy_model(energy_model("flavio"))
    scene.add_conf_clusters([cluster])
    assert scene.num_particles() == cluster.num_particles()
    # surround the cluster with free particles
    n_free = 20
    scene.add_particle_rand_positions([
        scene.particle_types().particle(i % scene.num_particle_types()).instantiate(scene.num_particles() + i)
        for i in range(n_free)])
    assert scene.num_particles() == cluster.num_particles() + n_free
    G = scene.compute_scene_graph()
    assert len(G.subgraph(range(cluster.num_particles())).edges) == n_cluster_bonds
    with scene.energy_session() as session:
        assert max(session.pair_energies().values()) < 1.


@requires_energy_model("flavio")
def test_multidentate_ambiguity_warns(ducky: PolycubeStructure):
    """
    a 4-tooth, wide (dental_radius=0.5) multidentate ducky has teeth equidistant from teeth on two neighbours;
    the patch-level bonds are still one-to-one, but the ambiguity is reported
    """
    scene = polycube_to_pl(ducky, MultidentateConvertSettings(n_teeth=4, dental_radius=0.5))
    scene.set_energy_model(energy_model("flavio"))
    with pytest.warns(PatchBondAmbiguityWarning, match="near-tie"):
        G = scene.compute_scene_multigraph()
    assert len(G.edges) == 7 * 4
    patch_uses = [end for _, _, ends in G.edges(data="patches") for end in ends]
    assert len(patch_uses) == len(set(patch_uses)), "a patch was used in more than one bond"


@requires_energy_model("flavio")
def test_unidentate_ducky_is_unambiguous(ducky: PolycubeStructure, recwarn):
    scene = polycube_to_pl(ducky)
    scene.set_energy_model(energy_model("flavio"))
    G = scene.compute_scene_multigraph()
    assert len(G.edges) == 7
    assert not [w for w in recwarn if issubclass(w.category, PatchBondAmbiguityWarning)]


@requires_energy_model("raspberry")
def test_raspberry_model_repulsion_sites_reach_oxdna():
    """
    the energy model's repulsion sites (and per-type site lists) are what oxDNA sees: two overlapping particles
    repel with one central site, and don't when neither type uses any site
    """
    ptypes = polycube_rule_to_PL(PolycubesRule(rule_str="1|1|1|1|1|1_-1|-1|-1|-1|-1|-1"))
    scene = PLPSimulation()
    scene.set_particle_types(ptypes)
    scene.set_box_size(np.array([5., 5., 5.]))
    scene.compute_cell_size(n_cells=1)
    scene.apportion_cells()
    scene.set_temperature(0.1)
    for i, pos in enumerate([[2., 2., 2.], [2.3, 2., 2.]]):
        p = ptypes.particle(i).instantiate(i)
        p.set_position(np.array(pos))
        p.set_rotation(np.identity(3))
        scene.add_particle(p)
    model = energy_model("raspberry")
    central = [((0., 0., 0.), 0.5)]
    model.repulsion_sites, model.type_rsite_lists = central, None
    scene.set_energy_model(model)
    assert scene.get_potential_energy() > 1.
    model.type_rsite_lists = {0: [], 1: []}
    assert scene.get_potential_energy() < 1e-6


def test_scene_without_energy_model_raises():
    scene = PLPSimulation()
    with pytest.raises(ValueError, match="energy model"):
        scene.compute_scene_graph()
