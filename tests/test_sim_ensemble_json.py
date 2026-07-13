"""
Fast, offline-ish unit tests for SimEnsembleJsonEncoder (sim_ensemble_json.py) - the encoder used
to write/read PatchySimulationEnsemble metadata to/from JSON.

These tests exercise the encoder's `default()`/`encode_param_value()`/`encode_ensemble_parameter()`
dispatch and the analysis-pipeline-step read/write round trip. They don't touch disk or run oxDNA,
but importing sim_ensemble_json.py transitively imports oxpy (via analysis_lib -> patchyio ->
server_config -> oxpy_utils), so a working oxpy install is required to collect this file.

Note on pipes: AnalysisPipeline.pipes() returns tuples, but read_analysis_pipeline() only recognizes
pipes encoded as *lists* (`isinstance(pipe, list)`). This is fine in real usage because the pipeline
data always passes through actual json.dumps/json.loads (which turns tuples into lists) on its way
to/from disk - see test_analysis_pipeline_round_trip_through_real_json below, which mirrors that.
Calling write_analysis_pipeline()/read_analysis_pipeline() back-to-back on plain Python objects
(skipping the JSON text step) would silently drop the pipes; that's not how the encoder is used.
"""
import json

import pytest

from pypatchy.patchy.sim_ensemble_json import SimEnsembleJsonEncoder, encode_add_method
from pypatchy.patchy.base_param_val import ParameterValue, ParamValueGroup
from pypatchy.patchy.ensemble_parameter import EnsembleParameter
from pypatchy.patchy.param_val_types import (
    MDTConvertParams,
    StagedAssemblyParam,
    StageInfoParam,
    ParticleSetParam,
)
from pypatchy.patchy.particle_adders import RandParticleAdder, FromConfAdder
from pypatchy.patchy.pl.plparticleset import MultidentateConvertSettings
from pypatchy.patchy.pl.pljsonrw import encode_particle_set
from pypatchy.patchy.pl.plpatchylib import polycube_rule_to_PL
from pypatchy.patchy.simulation_specification import ParamSet
from pypatchy.polycubeutil.polycubesRule import PolycubesRule
from pypatchy.server_config import PatchyServerConfig
from pypatchy.patchy.analysis_lib import (
    LoadParticlesTraj,
    LoadEnergies,
    ComputeClusterYield,
    ComputeClusterSizeData,
    ClassifyClusters,
)
from pypatchy.analysis.analysis_pipeline import AnalysisPipeline
from pypatchy.structure import Structure
from oxpy_utils.utils.observable import Observable, ObservableColumn


@pytest.fixture
def enc() -> SimEnsembleJsonEncoder:
    return SimEnsembleJsonEncoder()


# --------------------------- default() primitive/container dispatch --------------------------- #

def test_default_primitives_passthrough(enc):
    assert enc.default(5) == 5
    assert enc.default(True) is True
    assert enc.default("x") == "x"
    assert enc.default(1.5) == 1.5


def test_default_dict_recurses(enc):
    assert enc.default({"a": 1, "b": 2}) == {"a": [1], "b": [2]}


def test_default_list_recurses(enc):
    assert enc.default([1, 2, 3]) == [1, 2, 3]


def test_default_paramset(enc):
    ps = ParamSet([ParameterValue("T", 0.1), ParameterValue("density", 0.2)])
    assert enc.default(ps) == {"T": 0.1, "density": 0.2}


def test_default_polycubes_rule(enc):
    rule = PolycubesRule(rule_str="1|1|1|1|1|1_-1|||||")
    encoded = enc.default(rule)
    assert encoded == rule.toJSON()


def test_default_patchy_server_config(enc):
    sc = PatchyServerConfig(name="n", oxdna_path="/x", patchy_format="flavio")
    assert enc.default(sc) == sc.to_dict()


def test_default_plain_parameter_value(enc):
    pv = ParameterValue("narrow_type", 1)
    assert enc.default(pv) == 1


def test_default_observable(enc):
    obs = Observable("clusters.txt", 100, ObservableColumn(name="PatchyBonds", show_types=1))
    assert enc.default(obs) == obs.export()["output"]


def test_default_unhandled_type_raises_typeerror(enc):
    with pytest.raises(TypeError):
        enc.default(object())


# --------------------------- encode_ensemble_parameter --------------------------- #

def test_encode_ensemble_parameter(enc):
    ep = EnsembleParameter("T", [ParameterValue("T", 0.1), ParameterValue("T", 0.2)])
    key, values = enc.encode_ensemble_parameter(ep)
    assert key == "T"
    assert values == [0.1, 0.2]


# --------------------------- encode_param_value --------------------------- #

def test_encode_param_value_plain(enc):
    pv = ParameterValue("density", 0.1)
    assert enc.encode_param_value(pv) == 0.1


def test_encode_param_value_group(enc):
    pvg = ParamValueGroup(
        param_name="grp",
        param_value={"a": ParameterValue("a", 1), "b": ParameterValue("b", 2)},
        valname="grpval",
    )
    assert enc.encode_param_value(pvg) == {"name": "grpval", "a": 1, "b": 2}


def test_encode_param_value_mdt_convert_defaults_omitted(enc):
    mdt = MDTConvertParams(MultidentateConvertSettings(n_teeth=4, dental_radius=0.5))
    assert enc.encode_param_value(mdt) == {"n_teeth": 4, "dental_radius": 0.5}


def test_encode_param_value_mdt_convert_nondefaults_included(enc):
    mdt = MDTConvertParams(MultidentateConvertSettings(
        n_teeth=4, dental_radius=0.5, torsion=False, follow_surf=True,
        energy_scale_method="none", alpha_scale_method=1,
    ))
    assert enc.encode_param_value(mdt) == {
        "n_teeth": 4,
        "dental_radius": 0.5,
        "torsion": False,
        "follow_surf": True,
        "energy_scale_method": "none",
        "alpha_scale_method": 1,
    }


def test_encode_param_value_mdt_convert_follow_surf_true_not_dropped(enc):
    # regression test: encode_param_value used to have the follow_surf condition inverted
    # (`if not obj.param_value.follow_surf`), silently dropping follow_surf=True on encode.
    mdt = MDTConvertParams(MultidentateConvertSettings(n_teeth=4, dental_radius=0.5, follow_surf=True))
    assert enc.encode_param_value(mdt)["follow_surf"] is True


def test_encode_param_value_mdt_convert_energy_scale_log(enc):
    mdt = MDTConvertParams(MultidentateConvertSettings(n_teeth=4, dental_radius=0.5, energy_scale_method="log"))
    assert enc.encode_param_value(mdt)["energy_scale_method"] == "log"


def test_encode_param_value_staged_assembly(enc):
    info = StageInfoParam("stage0", steps=1000, T=0.1)
    sap = StagedAssemblyParam({"stage0": info}, "default")
    assert enc.encode_param_value(sap) == {"stages": {"stage0": {"steps": 1000, "T": 0.1}}}


def test_encode_param_value_particle_set_param(enc):
    rule = PolycubesRule(rule_str="1|1|1|1|1|1_-1|||||")
    particles = polycube_rule_to_PL(rule)
    psp = ParticleSetParam(particles)
    assert enc.encode_param_value(psp) == encode_particle_set(particles)


# --------------------------- default() dispatch to StageInfoParam / ParameterValue subtypes --------------------------- #

def test_default_stage_info_param(enc):
    info = StageInfoParam("stage0", t=500, steps=1000, T=0.1)
    encoded = enc.default(info)
    assert encoded["steps"] == 1000
    assert encoded["T"] == 0.1
    assert encoded["t"] == 500


def test_default_stage_info_param_no_start_time_or_shortfall(enc):
    info = StageInfoParam("stage0", steps=1000)
    encoded = enc.default(info)
    assert "t" not in encoded
    assert "allow_shortfall" not in encoded


def test_default_particle_set_param_tags_type(enc):
    rule = PolycubesRule(rule_str="1|1|1|1|1|1_-1|||||")
    particles = polycube_rule_to_PL(rule)
    psp = ParticleSetParam(particles)
    encoded = enc.default(psp)
    assert encoded["type"] == "particle_types"


def test_default_mdt_convert_params_tags_type(enc):
    mdt = MDTConvertParams(MultidentateConvertSettings(n_teeth=4, dental_radius=0.5))
    encoded = enc.default(mdt)
    assert encoded["type"] == "mdt_convert"


def test_default_staged_assembly_param_tags_type(enc):
    info = StageInfoParam("stage0", steps=1000)
    sap = StagedAssemblyParam({"stage0": info}, "default")
    encoded = enc.default(sap)
    assert encoded["type"] == "staging"


# --------------------------- encode_add_method --------------------------- #

def test_encode_add_method_rand_extern_default():
    adder = RandParticleAdder({0: 4})
    assert encode_add_method(adder) == {"particles": {0: 4}}


def test_encode_add_method_rand_extern_false():
    adder = RandParticleAdder({0: 4}, extern=False)
    assert encode_add_method(adder) == {"particles": {0: 4}, "extern": False}


def test_encode_add_method_from_conf_adder_raises_not_implemented_error():
    # regression test: this used to `raise NotImplemented(...)`, which is a non-callable
    # singleton and blows up with a confusing TypeError instead of NotImplementedError.
    with pytest.raises(NotImplementedError):
        encode_add_method(FromConfAdder())


def test_encode_add_method_invalid_type_raises_typeerror():
    with pytest.raises(TypeError):
        encode_add_method("not an adder")


# --------------------------- analysis pipeline step read/write round trip --------------------------- #

def test_load_particles_traj_round_trip(enc):
    step = LoadParticlesTraj(name="traj", normalize_coords=False, input_tstep=10, output_tstep=100)
    d = enc.write_analysis_step(step)
    assert d == {
        "type": "LoadParticlesTraj",
        "name": "traj",
        "input_tstep": 10,
        "output_tstep": 100,
        "normalize_coords": False,
    }
    step2 = enc.read_analysis_step(d, context=None)
    assert isinstance(step2, LoadParticlesTraj)
    assert step2.name == "traj"
    assert step2.input_tstep == 10
    assert step2.output_tstep == 100
    assert step2.normalize_coords is False


def test_load_energies_round_trip(enc):
    step = LoadEnergies(step_name="energies", output_tstep=50)
    d = enc.write_analysis_step(step)
    assert d["input_tstep"] == 1  # hardcoded
    step2 = enc.read_analysis_step(d, context=None)
    assert isinstance(step2, LoadEnergies)
    assert step2.output_tstep == 50


def test_compute_cluster_yield_round_trip(enc):
    step = ComputeClusterYield(name="yield", cutoff=0.8, overreach=True, input_tstep=10, output_tstep=20)
    d = enc.write_analysis_step(step)
    step2 = enc.read_analysis_step(d, context=None)
    assert isinstance(step2, ComputeClusterYield)
    assert step2.cutoff == 0.8
    assert step2.overreach is True


def test_compute_cluster_size_data_round_trip(enc):
    step = ComputeClusterSizeData(name="sizedata", input_tstep=10, output_tstep=20, minsize=3)
    d = enc.write_analysis_step(step)
    step2 = enc.read_analysis_step(d, context=None)
    assert isinstance(step2, ComputeClusterSizeData)
    assert step2.minsize == 3


def test_classify_clusters_round_trip(enc):
    target = Structure(bindings=[(0, 0, 1, 3)])
    step = ClassifyClusters(name="classify", target=target, input_tstep=10, output_tstep=20)
    d = enc.write_analysis_step(step)
    assert d["target"] == [[0, 0, 1, 3]]
    step2 = enc.read_analysis_step(d, context=None)
    assert isinstance(step2, ClassifyClusters)
    assert step2.target.to_json() == [[0, 0, 1, 3]]


def test_read_analysis_step_unknown_type_raises(enc):
    with pytest.raises(Exception):
        enc.read_analysis_step({"type": "NotARealStep"}, context=None)


def test_analysis_pipeline_round_trip_through_real_json(enc):
    pipeline = AnalysisPipeline(
        LoadParticlesTraj(name="traj", input_tstep=10, output_tstep=100),
        ComputeClusterSizeData(name="sizedata", input_tstep=100, output_tstep=100, minsize=2),
        ("traj", "sizedata"),
    )
    pipeline_data = enc.write_analysis_pipeline(pipeline)
    # mirrors the real write path: json.dump(..., cls=SimEnsembleJsonEncoder) then json.load(...)
    round_tripped = json.loads(json.dumps(pipeline_data, cls=SimEnsembleJsonEncoder))

    pipeline2 = enc.read_analysis_pipeline(round_tripped, context=None)
    assert [s.name for s in pipeline2.steps()] == ["traj", "sizedata"]
    assert pipeline2.pipes() == [("traj", "sizedata")]


def test_default_analysis_pipeline(enc):
    pipeline = AnalysisPipeline(
        ComputeClusterSizeData(name="sizedata", input_tstep=100, output_tstep=100, minsize=2),
    )
    assert enc.default(pipeline) == enc.write_analysis_pipeline(pipeline)