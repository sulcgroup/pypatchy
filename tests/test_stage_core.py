"""
Fast, offline unit tests for the staged-assembly machinery: StageInfoParam
(param_val_types.py), RandParticleAdder (particle_adders.py), the Stage class and its
exception hierarchy (stage.py).

Stage itself wraps oxpy_utils's BuildSimulation/Simulation, but neither of those touch disk
at construction time, so a real Stage can be built against a real (but throwaway)
PatchySimulationEnsemble without any actual oxDNA install, compiled plugin, or simulation
data - see test_ensemble_core.py's stub_ensemble fixture for the same pattern.
"""
import datetime

import pytest

from pypatchy.patchy.base_param_val import ParameterValue, InvalidParameterTypeError
from pypatchy.patchy.ensemble_parameter import EnsembleParameter
from pypatchy.patchy.simulation_ensemble import PatchySimulationEnsemble
from pypatchy.patchy.simulation_specification import ParamSet
from pypatchy.patchy.param_val_types import StageInfoParam
from pypatchy.patchy.particle_adders import RandParticleAdder
from pypatchy.patchy.stage import (
    Stage,
    StagedAssemblyError,
    IncompleteStageError,
    StageTrajFileError,
    NoStageTrajError,
    StageTrajFileEmptyError,
)
from pypatchy.server_config import PatchyServerConfig


# --------------------------- RandParticleAdder --------------------------- #

def test_rand_particle_adder_from_dict():
    adder = RandParticleAdder({0: 4, 1: 2})
    assert adder.get_particle_counts() == {0: 4, 1: 2}
    assert adder.extern is True


def test_rand_particle_adder_from_list_filters_zero_and_negative_counts():
    adder = RandParticleAdder([4, 0, 2])
    assert adder.get_particle_counts() == {0: 4, 2: 2}


def test_rand_particle_adder_extern_false():
    adder = RandParticleAdder({0: 1}, extern=False)
    assert adder.extern is False


# --------------------------- StageInfoParam --------------------------- #

def test_stage_info_param_defaults():
    info = StageInfoParam("stage0", steps=1000, T=0.1)
    assert info.stage_name == "stage0"
    assert info.start_time == 0  # no "t" given
    assert info.add_method is None
    assert info.allow_shortfall is False
    assert dict(info) == {"steps": 1000, "T": 0.1}


def test_stage_info_param_start_time_from_t_and_excluded_from_info():
    info = StageInfoParam("stage1", t=500, steps=1000)
    assert info.start_time == 500
    assert "t" not in info


def test_stage_info_param_allow_shortfall_explicit():
    info = StageInfoParam("stage0", steps=1000, allow_shortfall=True)
    assert info.allow_shortfall is True


def test_stage_info_param_add_method_random_from_dict():
    info = StageInfoParam("stage0", steps=1000, add_method={"type": "random", "particles": {0: 4}})
    assert isinstance(info.add_method, RandParticleAdder)
    assert info.add_method.get_particle_counts() == {0: 4}
    assert "add_method" not in info


def test_stage_info_param_add_method_no_type_assumes_random():
    info = StageInfoParam("stage0", steps=1000, add_method={"particles": {0: 4}})
    assert isinstance(info.add_method, RandParticleAdder)


def test_stage_info_param_add_method_stage_particle_adder_instance():
    adder = RandParticleAdder({0: 4})
    info = StageInfoParam("stage0", steps=1000, add_method=adder)
    assert info.add_method is adder


def test_stage_info_param_add_method_string_type_raises():
    with pytest.raises(TypeError):
        StageInfoParam("stage0", steps=1000, add_method="random")


def test_stage_info_param_add_method_invalid_type_raises():
    with pytest.raises(InvalidParameterTypeError):
        StageInfoParam("stage0", steps=1000, add_method=42)


def test_stage_info_param_add_method_unknown_dict_type_raises():
    with pytest.raises(TypeError):
        StageInfoParam("stage0", steps=1000, add_method={"type": "bogus"})


def test_stage_info_param_get_and_set_end_time():
    info = StageInfoParam("stage0", steps=1000)
    assert info.get_end_time() == 1000
    info.set_end_time(2000)
    assert info.get_end_time() == 2000


def test_stage_info_param_get_start_time():
    info = StageInfoParam("stage0", t=250, steps=1000)
    assert info.get_start_time() == 250


def test_stage_info_param_mapping_protocol():
    info = StageInfoParam("stage0", steps=1000, T=0.1)
    assert len(info) == 2
    assert set(info.keys()) == {"steps", "T"}
    assert info["T"] == 0.1
    with pytest.raises(KeyError):
        info["nonexistent"]


# --------------------------- Stage exception hierarchy --------------------------- #

class _FakeStage:
    """Minimal duck-typed stand-in - these exceptions only ever call .name()/.start_time()/.end_time()."""

    def __init__(self, name="stage0", start=0, end=1000):
        self._name = name
        self._start = start
        self._end = end

    def name(self):
        return self._name

    def start_time(self):
        return self._start

    def end_time(self):
        return self._end


def _fake_sim() -> ParamSet:
    return ParamSet([ParameterValue("T", 0.1)])


def test_staged_assembly_error_accessors():
    stage = _FakeStage()
    sim = _fake_sim()
    err = StagedAssemblyError(stage, sim)
    assert err.stage() is stage
    assert err.sim() is sim


def test_incomplete_stage_error():
    stage = _FakeStage(name="stage1", start=0, end=1000)
    sim = _fake_sim()
    err = IncompleteStageError(stage, sim, 500)
    assert err.last_timestep() == 500
    s = str(err)
    assert "stage1" in s
    assert "500" in s
    assert repr(sim) in s


def test_stage_traj_file_error_accessor():
    stage = _FakeStage()
    sim = _fake_sim()
    err = StageTrajFileError(stage, sim, "/some/path/traj.dat")
    assert err.traj_file() == "/some/path/traj.dat"


def test_no_stage_traj_error_str():
    stage = _FakeStage(name="stage0")
    sim = _fake_sim()
    err = NoStageTrajError(stage, sim, "/some/path/traj.dat")
    s = str(err)
    assert "/some/path/traj.dat" in s
    assert "stage0" in s


def test_stage_traj_file_empty_error_str():
    stage = _FakeStage(name="stage0")
    sim = _fake_sim()
    err = StageTrajFileEmptyError(stage, sim, "/some/path/traj.dat")
    s = str(err)
    assert "/some/path/traj.dat" in s
    assert "stage0" in s


# --------------------------- Stage (real instances, throwaway ensemble) --------------------------- #

@pytest.fixture
def stub_stage_ensemble(tmp_path) -> PatchySimulationEnsemble:
    server = PatchyServerConfig(name="pytest_stub_stage_server", oxdna_path="/nonexistent", patchy_format="flavio")
    return PatchySimulationEnsemble(
        export_name="pytest_stub_stage_ensemble",
        setup_date=datetime.datetime(2024, 1, 1),
        metadata_file_name="pytest_stub_stage_ensemble_2024-01-01_metadata.json",
        const_params=ParamSet([
            ParameterValue("density", 0.1),
            ParameterValue("num_assemblies", 2),
        ]),
        ensemble_params=[EnsembleParameter("T", [ParameterValue("T", 0.1)])],
        observables={},
        metadata_dict={"ensemble_config": {}},
        server_settings=server,
        log_file_name=tmp_path / "pytest_stub_stage_ensemble.log",
    )


@pytest.fixture
def stage0(stub_stage_ensemble) -> Stage:
    sim = stub_stage_ensemble.get_simulation(T="0.1")
    info = StageInfoParam("stage0", t=0, steps=1000, T=0.1, add_method=RandParticleAdder({0: 4, 1: 2}))
    return Stage(sim, None, stub_stage_ensemble, info)


def test_stage_zero_indexing_and_naming(stage0):
    assert stage0.idx() == 0
    assert stage0.name() == "stage0"
    assert stage0.is_first()
    assert stage0.is_last()
    assert stage0.get_prev() is None
    assert stage0.get_next() is None


def test_stage_timing(stage0):
    assert stage0.start_time() == 0
    assert stage0.end_time() == 1000
    assert stage0.time_length() == 1000


def test_stage_rejects_zero_length(stub_stage_ensemble):
    sim = stub_stage_ensemble.get_simulation(T="0.1")
    info = StageInfoParam("stage0", t=1000, steps=1000)  # start == end
    with pytest.raises(AssertionError):
        Stage(sim, None, stub_stage_ensemble, info)


def test_stage_particles_to_add_incorporates_num_assemblies(stage0):
    # particles={0: 4, 1: 2}, num_assemblies=2 -> 8 of type 0, 4 of type 1
    added = stage0.particles_to_add()
    assert added.count(0) == 8
    assert added.count(1) == 4
    assert stage0.num_particles_to_add() == 12
    assert stage0.num_particles() == 12  # no previous stage


def test_stage_has_var_and_get_var(stage0):
    assert stage0.has_var("T")
    assert stage0.get_var("T") == 0.1
    assert not stage0.has_var("nonexistent")


def test_stage_params_yields_info_entries(stage0):
    params = list(stage0.params())
    assert ParameterValue("steps", 1000) in params
    assert ParameterValue("T", 0.1) in params


def test_stage_adjfn_unprefixed_for_first_stage(stage0):
    assert stage0.adjfn("trajectory.dat") == "trajectory.dat"


def test_stage_allow_shortfall_default_and_set(stage0):
    assert not stage0.allow_shortfall()
    stage0.set_allow_shortfall(True)
    assert stage0.allow_shortfall()


def test_stage_box_size_default_and_set(stage0):
    assert tuple(stage0.box_size()) == (0, 0, 0)
    stage0.set_box_size((5, 5, 5))
    assert tuple(stage0.box_size()) == (5, 5, 5)


def test_stage_str(stage0):
    assert str(stage0) == "Stage stage0 (#0)"


def test_stage_set_tend_requires_last_stage(stage0):
    stage0.set_tend(2000)
    assert stage0.end_time() == 2000


def test_stage_chaining_second_stage(stub_stage_ensemble, stage0):
    sim = stub_stage_ensemble.get_simulation(T="0.1")
    info1 = StageInfoParam("stage1", t=1000, steps=2000)
    stage1 = Stage(sim, stage0, stub_stage_ensemble, info1)

    assert stage1.idx() == 1
    assert not stage0.is_last()  # stage0 got linked to stage1 as a side effect of construction
    assert stage1.is_last()
    assert stage1.get_prev() is stage0
    assert stage0.get_next() is stage1
    assert stage1.adjfn("trajectory.dat") == "stage1/trajectory.dat"


def test_stage_set_tend_on_non_last_stage_raises(stub_stage_ensemble, stage0):
    sim = stub_stage_ensemble.get_simulation(T="0.1")
    info1 = StageInfoParam("stage1", t=1000, steps=2000)
    Stage(sim, stage0, stub_stage_ensemble, info1)  # links stage0 -> stage1, stage0 no longer "last"

    with pytest.raises(AssertionError):
        stage0.set_tend(5000)


def test_stage_num_particles_includes_previous_stages(stub_stage_ensemble, stage0):
    sim = stub_stage_ensemble.get_simulation(T="0.1")
    info1 = StageInfoParam("stage1", t=1000, steps=2000, add_method=RandParticleAdder({0: 1}))
    stage1 = Stage(sim, stage0, stub_stage_ensemble, info1)

    assert stage1.num_particles_to_add() == 2  # 1 particle * num_assemblies(2)
    assert stage1.num_particles() == 12 + 2  # stage0's 12 plus stage1's own 2
