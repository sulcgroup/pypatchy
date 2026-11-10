"""
Tests for the HOOMD simulation engine (pypatchy.patchy.engines.hoomd_engine, pypatchy.patchy.hoomd_backend).

Parameter, model and rotation tests are offline. GSD tests need gsd; the energy check and the end-to-end ensemble
run need hoomd (and run on its CPU device, so they don't need a GPU).
"""
import datetime
import json
import math
import pickle

import numpy as np
import pytest

from ptutils import temp_dir  # noqa: F401 (fixture)
from test_polycube_lib import ducky  # noqa: F401 (fixture)
from pypatchy.patchy.engines import HOOMDEngine, get_engine
from pypatchy.patchy.hoomd_backend.model import (RigidPatchyModel, angmom_quat_from_body, body_angmom_from_quat,
                                                 matrix_from_quat, quat_from_matrix, rotation_matrix)
from pypatchy.patchy.hoomd_backend.params import HOOMDParameterError, resolve_hoomd_params
from pypatchy.patchy.pl.oxpy_energy import PatchBondAmbiguityWarning
from pypatchy.patchy.pl.plparticleset import MultidentateConvertSettings
from pypatchy.patchy.pl.plpatchylib import polycube_rule_to_PL, polycube_to_pl
from pypatchy.patchy.pl.plscene import PLPSimulation
from pypatchy.polycubeutil.polycubesRule import PolycubesRule
from pypatchy.server_config import PatchyServerConfig
from pypatchy.util import get_input_dir

REQUIRED = {"kT": 0.5, "dt": 0.002, "patch_epsilon": 8.0, "patch_sigma": 0.1, "print_conf_interval": 1000}
# a centre type with six +1 patches, and an arm type with one -1 patch
RULE = "1|1|1|1|1|1_-1|||||"


def hoomd_config(**kwargs) -> PatchyServerConfig:
    return PatchyServerConfig(name="pytest_hoomd", engine="hoomd", input_file_params={"device": "cpu"}, **kwargs)


def particle_set():
    return polycube_rule_to_PL(PolycubesRule(rule_str=RULE))


def model(**kwargs) -> RigidPatchyModel:
    args = dict(patch_epsilon=8.0, patch_sigma=0.1, patch_rcut=0.3)
    args.update(kwargs)
    return RigidPatchyModel.from_particle_set(particle_set(), **args)


# ------------------------------------------------------------------------------------------ server configs
def test_hoomd_server_configs_need_no_oxdna_settings():
    sc = hoomd_config()
    assert sc.oxdna_path is None and sc.patchy_format is None
    assert isinstance(get_engine(sc.engine), HOOMDEngine)
    assert PatchyServerConfig(**sc.to_dict()).engine == "hoomd"


def test_oxdna_server_configs_still_need_oxdna_settings():
    with pytest.raises(TypeError, match="patchy_format"):
        PatchyServerConfig(name="n", oxdna_path="/x")


# ------------------------------------------------------------------------------------------ parameters
def test_required_parameters_have_no_defaults():
    with pytest.raises(HOOMDParameterError) as e:
        resolve_hoomd_params({"patch_epsilon": 1.0})
    for name in ("kT", "dt", "patch_sigma", "print_conf_interval"):
        assert f"`{name}`" in str(e.value)


def test_oxdna_parameters_are_errors_not_translations():
    with pytest.raises(HOOMDParameterError) as e:
        resolve_hoomd_params({**REQUIRED, "T": 0.1, "interaction_type": "PatchyShapeInteraction", "CT0": 4},
                             particle_type_names={"CT0"})
    msg = str(e.value)
    assert "`T`" in msg and "`kT`" in msg  # names the parameter and says what to use instead
    assert "`interaction_type`" in msg
    assert "`CT0`" not in msg.split("\n")[0]  # particle counts are fine


def test_defaults_are_filled_in_and_reported():
    params, defaulted = resolve_hoomd_params({**REQUIRED, "steps": 1e6, "density": 0.1})
    assert params["patch_rcut"] == pytest.approx(0.3)  # 3 * patch_sigma
    assert params["print_energy_every"] == 1000
    assert params["device"] == "gpu"
    assert 0 <= params["seed"] < 65536
    assert {"patch_rcut", "print_energy_every", "seed", "gamma"} <= defaulted
    assert "kT" not in defaulted


def test_step_counts_must_be_whole_numbers():
    params, _ = resolve_hoomd_params({**REQUIRED, "print_conf_interval": 1e4})
    assert params["print_conf_interval"] == 10000 and isinstance(params["print_conf_interval"], int)
    with pytest.raises(HOOMDParameterError, match="whole number"):
        resolve_hoomd_params({**REQUIRED, "print_conf_interval": 2.5})


def test_device_must_be_a_known_one():
    with pytest.raises(HOOMDParameterError, match="device"):
        resolve_hoomd_params({**REQUIRED, "device": "CUDA"})


# ------------------------------------------------------------------------------------------ model
def test_one_site_type_per_colour_with_complementary_attraction():
    m = model()
    # polycube_rule_to_PL shifts rule colours by 20 (|colour| >= 20 binds its negative)
    c = max(patch.color() for patch in particle_set().patches())
    assert [s["name"] for s in m.sites] == [f"patch{-c:+d}", f"patch{c:+d}"]
    eps = {(a["color"], b["color"]): m.site_epsilon[i, j]
           for i, a in enumerate(m.sites) for j, b in enumerate(m.sites)}
    assert eps[(c, -c)] == pytest.approx(8.0)
    assert eps[(c, c)] == 0 and eps[(-c, -c)] == 0
    assert [len(b["sites"]) for b in m.bodies] == [6, 1]


def test_patch_strengths_scale_the_well():
    pset = particle_set()
    for patch in pset.patches():
        if patch.color() < 0:
            patch.set_strength(0.25)
    m = RigidPatchyModel.from_particle_set(pset, patch_epsilon=8.0, patch_sigma=0.1, patch_rcut=0.3)
    colors = [s["color"] for s in m.sites]
    i, j = colors.index(max(colors)), colors.index(min(colors))
    assert m.site_epsilon[i, j] == pytest.approx(8.0 * math.sqrt(0.25))


def test_model_round_trips_through_dict_json_and_pickle():
    m = model()
    again = RigidPatchyModel.from_dict(json.loads(json.dumps(m.to_dict())))
    assert again.to_dict() == m.to_dict()
    assert pickle.loads(pickle.dumps(m)).to_dict() == m.to_dict()


def test_quaternions_and_angular_momenta_round_trip():
    rng = np.random.default_rng(1)
    for _ in range(20):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        # q and -q are the same rotation
        assert np.allclose(matrix_from_quat(quat_from_matrix(matrix_from_quat(q))), matrix_from_quat(q))
        assert abs(quat_from_matrix(matrix_from_quat(q)) @ q) == pytest.approx(1.)
        l_body = rng.normal(size=3)
        assert np.allclose(body_angmom_from_quat(q, angmom_quat_from_body(q, l_body)), l_body)


def ducky_scene(ducky, mdt=None):
    """
    the ducky polycube (8 cubes, 7 bonds) as a bonded scene, built by pypatchy's own polycube conversion (and
    multidentate conversion, if `mdt` is given). Cubes sit 1.05 apart, so bonded patches are 0.05 apart.
    """
    scene = polycube_to_pl(ducky, mdt)
    # make room for HOOMD's neighbour list (box >= 2 * (cutoff + buffer)) in every direction
    old = scene.box_size().copy()
    big = np.maximum(old, 8.)
    scene.set_box_size(big)
    for p in scene.particles():
        p.set_position(p.position() + (big - old) / 2)
    scene.compute_cell_size(n_cells=1)
    scene.apportion_cells()
    m = RigidPatchyModel.from_particle_set(scene.particle_types(), patch_epsilon=8.0, patch_sigma=0.1,
                                           patch_rcut=0.3)
    scene.set_energy_model(m)
    return scene, m


MDT = MultidentateConvertSettings(n_teeth=4, dental_radius=0.2)


def bond_energy(gap: float, centre_distance: float, m: RigidPatchyModel) -> float:
    """energy of two particles whose bonded patches are `gap` apart (and no other patches in reach)"""
    return float(m.wca(centre_distance, 1.0)) - 8.0 * (math.exp(-gap ** 2 / 0.02) - math.exp(-0.3 ** 2 / 0.02))


def test_unidentate_ducky_bonds(ducky, recwarn):
    scene, m = ducky_scene(ducky)
    G = scene.compute_scene_graph()
    assert len(G.edges) == 7
    with scene.energy_session() as s:
        for u, v in G.edges:
            pu, pv = scene.get_particle(u), scene.get_particle(v)
            gap = min(np.linalg.norm(pu.patch_position(a) - pv.patch_position(b))
                      for a in pu.patches() for b in pv.patches())
            assert gap == pytest.approx(0.05)
            assert s.pair_energy(u, v) == pytest.approx(
                bond_energy(gap, np.linalg.norm(pu.position() - pv.position()), m))
    assert len(scene.compute_scene_multigraph().edges) == 7
    assert not [w for w in recwarn if issubclass(w.category, PatchBondAmbiguityWarning)]


def test_multidentate_ducky_bonds_tooth_by_tooth(ducky):
    """
    4 teeth per face, each with its own colour (so torsion matters) and a quarter of the patch strength: the teeth
    line up across every bond, and the bonds are exactly as strong as the unidentate ones
    """
    uni, _ = ducky_scene(ducky)
    scene, m = ducky_scene(ducky, MDT)
    assert len(m.sites) == 4 * len(RigidPatchyModel.from_particle_set(
        uni.particle_types(), patch_epsilon=1., patch_sigma=0.1, patch_rcut=0.3).sites)
    G = scene.compute_scene_graph()
    assert sorted(G.edges) == sorted(uni.compute_scene_graph().edges)
    with scene.energy_session() as s, uni.energy_session() as s_uni:
        for u, v in G.edges:
            assert s.pair_energy(u, v) == pytest.approx(s_uni.pair_energy(u, v))
    assert len(scene.compute_scene_multigraph().edges) == 7 * 4


@pytest.mark.parametrize("mdt", [None, MDT], ids=["unidentate", "multidentate"])
def test_only_multidentate_patches_resist_torsion(ducky, mdt):
    """twisting a cube 90 degrees about one of its bonds breaks a multidentate bond, but not a unidentate one"""
    from scipy.spatial.transform import Rotation
    scene, m = ducky_scene(ducky, mdt)
    u, v = sorted(scene.compute_scene_graph().edges)[0]
    with scene.energy_session() as s:
        bonded = s.pair_energy(u, v)
    pu, pv = scene.get_particle(u), scene.get_particle(v)
    axis = (pv.position() - pu.position()) / np.linalg.norm(pv.position() - pu.position())
    pv.set_rotation(Rotation.from_rotvec(math.pi / 2 * axis).as_matrix() @ rotation_matrix(pv))
    with scene.energy_session() as s:
        twisted = s.pair_energy(u, v)
    assert bonded < -5
    wca = float(m.wca(np.linalg.norm(pv.position() - pu.position()), 1.0))
    if mdt is None:
        assert twisted == pytest.approx(bonded)
    else:
        # the teeth no longer face same-coloured teeth: next to nothing is left but the centre-centre repulsion
        # (the nearest same-coloured teeth are dental_radius * sqrt(2) = 0.28 apart, just inside patch_rcut)
        assert twisted - wca > 0.01 * (bonded - wca)


# ------------------------------------------------------------------------------------------ GSD
def random_scene(m: RigidPatchyModel, n=27, spacing=1.05, seed=3) -> PLPSimulation:
    """particles on a jittered cubic lattice with random orientations: plenty of WCA and patch contacts"""
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    pset = particle_set()
    side = round(n ** (1 / 3))
    box = np.array([side * spacing + 0.3] * 3)
    scene = PLPSimulation()
    scene.set_particle_types(pset)
    scene.set_box_size(box)
    scene.compute_cell_size(n_cells=1)
    scene.apportion_cells()
    for uid, idx in enumerate(np.ndindex(side, side, side)):
        p = pset.particle(uid % 2).instantiate(uid)
        p.set_position(np.array(idx) * spacing + rng.normal(scale=0.03, size=3) + 0.15)
        p.set_random_orientation()
        p.v = rng.normal(size=3)
        p.L = rng.normal(size=3)
        scene.add_particle(p)
    scene.set_energy_model(m)
    return scene


def test_scenes_round_trip_through_gsd_frames():
    pytest.importorskip("gsd")
    from pypatchy.patchy.hoomd_backend import gsdio
    m = model()
    scene = random_scene(m)
    frame, unthermalized = gsdio.scene_to_frame(scene, m, step=1234)
    assert unthermalized == []
    assert frame.configuration.step == 1234
    assert np.all(np.abs(frame.particles.position) <= scene.box_size() / 2 + 1e-9)  # HOOMD's centred box
    again = gsdio.frame_to_scene(frame, particle_set(), m)
    assert again.time() == 1234
    for p, q in zip(scene.particles(), again.particles()):
        assert p.get_type() == q.get_type()
        assert np.allclose(np.mod(p.position(), scene.box_size()), q.position())
        assert np.allclose(rotation_matrix(p), rotation_matrix(q))
        assert np.allclose(p.v, q.v) and np.allclose(p.L, q.L)
        for patch in p.patches():
            assert np.allclose(p.patch_position(patch) - p.position(), q.patch_position(patch) - q.position())


def test_new_particles_are_marked_for_thermalizing():
    pytest.importorskip("gsd")
    from pypatchy.patchy.hoomd_backend import gsdio
    m = model()
    scene = random_scene(m)
    scene.particles()[4].v = None
    _, unthermalized = gsdio.scene_to_frame(scene, m, step=0)
    assert unthermalized == [4]


# ------------------------------------------------------------------------------------------ HOOMD
def hoomd_potential_energy(scene: PLPSimulation, m: RigidPatchyModel, tmp_path) -> float:
    hoomd = pytest.importorskip("hoomd")
    from hoomd import md
    from pypatchy.patchy.hoomd_backend import gsdio
    from pypatchy.patchy.hoomd_backend.run_stage import build_simulation
    params, _ = resolve_hoomd_params({**REQUIRED, "device": "cpu", "seed": 1})
    frame, _ = gsdio.scene_to_frame(scene, m, step=0)
    gsdio.write_frame(tmp_path / "init.gsd", frame)
    sim, ops = build_simulation({"params": params, "model": m.to_dict()}, tmp_path / "init.gsd",
                                device=hoomd.device.CPU())
    pe = md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
    sim.operations.computes.append(pe)
    sim.run(0)
    return pe.potential_energy


@pytest.mark.parametrize("mdt", [None, MDT], ids=["unidentate", "multidentate"])
def test_hoomd_energy_of_the_ducky_matches_numpy(ducky, mdt, tmp_path):
    scene, m = ducky_scene(ducky, mdt)
    with scene.energy_session() as s:
        expected = s.total_energy()
    assert expected < -40  # 7 bonds
    assert hoomd_potential_energy(scene, m, tmp_path) == pytest.approx(expected, rel=1e-5)


def test_hoomd_energy_of_a_dense_scene_matches_numpy(tmp_path):
    # wide patches, so randomly oriented neighbours have patch contacts; a 4x4x4 lattice keeps the box over twice
    # the interaction range (so every pair has a single nearest image)
    m = model(patch_sigma=0.25, patch_rcut=0.75)
    scene = random_scene(m, n=64)
    assert np.all(scene.box_size() > 2 * m.screen_radius())
    with scene.energy_session() as s:
        expected = s.total_energy()
    with model(patch_epsilon=0., patch_sigma=0.25, patch_rcut=0.75).session(scene) as s:
        wca_only = s.total_energy()
    assert wca_only > 0.5, "test scene should have some centre-centre overlaps"
    assert expected - wca_only < -0.5, "test scene should have some patch contacts"
    assert hoomd_potential_energy(scene, m, tmp_path) == pytest.approx(expected, rel=1e-5)


# ------------------------------------------------------------------------------------------ ensembles
def build_hoomd_ensemble(name: str, const_params: dict, stages: dict):
    from pypatchy.patchy.simulation_ensemble import (build_ensemble, EXPORT_NAME_KEY, ENSEMBLE_SETUP_DATE,
                                                     CONST_PARAMS_KEY, ENSEMBLE_PARAMS_KEY, SERVER_SETTINGS_KEY)
    return build_ensemble(cfg={
        EXPORT_NAME_KEY: name,
        "rule": RULE,
        CONST_PARAMS_KEY: const_params,
        ENSEMBLE_PARAMS_KEY: [("staging", [{"name": "two_stage", "type": "staging", "stages": stages}])],
        SERVER_SETTINGS_KEY: hoomd_config(),
    }, mdt={ENSEMBLE_SETUP_DATE: datetime.datetime.now()})


TWO_STAGES = {
    "first": {"add_method": {"type": "random", "particles": {"CT0": 1}}},
    "second": {"t": 2000, "kT": 0.4, "add_method": {"type": "random", "particles": {"CT1": 3}}},
}
CONST = {**REQUIRED, "num_assemblies": 4, "density": 0.05, "steps": 4000, "seed": 7}


def test_hoomd_ensembles_skip_the_oxdna_default_parameter_set(temp_dir):
    e = build_hoomd_ensemble("pytest_hoomd_defaults", CONST, TWO_STAGES)
    try:
        # the oxDNA default set (if the user has one) holds e.g. dt = 0.001; it must not reach a HOOMD ensemble
        assert e.get_param("dt") == REQUIRED["dt"]
        assert "thermostat" not in e.const_params
    finally:
        (get_input_dir() / e.metadata_file).unlink(missing_ok=True)


def test_setup_rejects_oxdna_parameters(temp_dir):
    pytest.importorskip("gsd")
    e = build_hoomd_ensemble("pytest_hoomd_reject", {**CONST, "T": 0.1}, TWO_STAGES)
    try:
        with pytest.raises(HOOMDParameterError, match="`T`"):
            e.do_setup()
    finally:
        (get_input_dir() / e.metadata_file).unlink(missing_ok=True)


def test_two_stage_hoomd_ensemble_runs_and_continues(temp_dir):
    pytest.importorskip("hoomd")
    from pypatchy.patchy.hoomd_backend import gsdio
    from pypatchy.patchy.hoomd_backend.run_stage import read_spec
    e = build_hoomd_ensemble("pytest_hoomd_staged", CONST, TWO_STAGES)
    try:
        sim = e.ensemble()[0]
        first, second = e.sim_get_stages(sim)
        e.start_simulations(stage="first")
        assert e.sim_stage_done(sim, first)
        assert e.sim_get_stage_last_step(sim, first) == 2000
        traj = e.folder_path(sim, first) / "trajectory.gsd"
        assert gsdio.frame_steps(traj) == [0, 1000, 2000]
        spec = read_spec(e.folder_path(sim, first))
        assert spec["params"]["kT"] == 0.5 and spec["params"]["seed"] == 7
        assert len(spec["thermalize"]) == 4  # all new particles

        scene = e.get_scene(sim, first)
        assert scene.num_particles() == 4 and scene.time() == 2000

        e.start_simulations(stage="second")
        assert e.sim_stage_done(sim, second)
        spec2 = read_spec(e.folder_path(sim, second))
        assert spec2["params"]["kT"] == 0.4  # stage-level parameter wins
        assert spec2["start_step"] == 2000 and spec2["end_step"] == 4000
        # the continued particles keep their momenta; only the 12 arms added in this stage are thermalized
        assert spec2["thermalize"] == list(range(4, 16))
        start = gsdio.read_frame(e.folder_path(sim, second) / "init.gsd")
        end_of_first = gsdio.read_frame(e.folder_path(sim, first) / "last.gsd")
        assert np.allclose(start.particles.velocity[:4], end_of_first.particles.velocity)

        assert e.time_length(sim) == 4000
        assert e.sim_done(sim)
        final = e.get_scene(sim)
        assert final.num_particles() == 16
        assert e.get_scene(sim, second, step=3000).time() == 3000
        # running again finds nothing to do
        e.start_simulations()
    finally:
        (get_input_dir() / e.metadata_file).unlink(missing_ok=True)
