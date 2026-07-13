"""
Fast, offline unit tests for PatchySimulationEnsemble's pure-logic surface and for
find_ensemble()'s dispatch tree.

These deliberately avoid do_setup()/start_simulations()/get_data() and anything else that
needs a real oxDNA binary, compiled plugin, or simulation data - see test_ensembles_basic.py
and test_analysis.py for that E2E coverage. Everything here should run in well under a second
and requires nothing but a configured ~/.pypatchy working directory (see test_setup.py).
"""
import datetime
import json

import pytest

from pypatchy.patchy import simulation_ensemble
from pypatchy.patchy.simulation_ensemble import PatchySimulationEnsemble, find_ensemble
from pypatchy.patchy.ensemble_parameter import EnsembleParameter
from pypatchy.patchy.base_param_val import ParameterValue
from pypatchy.patchy.simulation_specification import ParamSet, NoSuchParamError
from pypatchy.patchy.pl.patchyio import LWriter, FWriter
from pypatchy.server_config import PatchyServerConfig
from pypatchy.util import get_input_dir, simulation_run_dir


# --------------------------- PatchySimulationEnsemble fixtures --------------------------- #

@pytest.fixture
def stub_server_settings() -> PatchyServerConfig:
    return PatchyServerConfig(name="pytest_stub_server", oxdna_path="/nonexistent", patchy_format="flavio")


@pytest.fixture
def stub_ensemble(tmp_path, stub_server_settings) -> PatchySimulationEnsemble:
    return PatchySimulationEnsemble(
        export_name="pytest_stub_ensemble",
        setup_date=datetime.datetime(2024, 1, 1),
        metadata_file_name="pytest_stub_ensemble_2024-01-01_metadata.json",
        const_params=ParamSet([
            ParameterValue("density", 0.1),
            ParameterValue("topology", "initial.top"),
        ]),
        ensemble_params=[
            EnsembleParameter("T", [ParameterValue("T", 0.05), ParameterValue("T", 0.1)]),
            EnsembleParameter("duplicate", [ParameterValue("duplicate", 0), ParameterValue("duplicate", 1)]),
        ],
        observables={},
        metadata_dict={"ensemble_config": {}},
        server_settings=stub_server_settings,
        log_file_name=tmp_path / "pytest_stub_ensemble.log",
    )


# --------------------------- __init__ / server settings --------------------------- #

def test_init_with_explicit_server_settings(stub_ensemble, stub_server_settings):
    assert stub_ensemble.server_settings is stub_server_settings


def test_init_with_no_server_settings_falls_back_to_default(tmp_path):
    """
    regression test: PatchySimulationEnsemble's class body used to declare
    `server_settings = PatchyServerConfig` (an assignment, not an annotation), which made
    `self.server_settings` always truthy before __init__ ever set it - so the `None`
    fallback branch (the documented default) was unreachable and instead tried to
    load_server_settings(None), raising NoSpecJSONError.
    """
    e = PatchySimulationEnsemble(
        export_name="pytest_stub_ensemble_default_server",
        setup_date=datetime.datetime(2024, 1, 1),
        metadata_file_name="pytest_stub_ensemble_default_server_2024-01-01_metadata.json",
        const_params=ParamSet([]),
        ensemble_params=[],
        observables={},
        metadata_dict={},
        server_settings=None,
        log_file_name=tmp_path / "pytest_stub_ensemble_default_server.log",
    )
    assert isinstance(e.server_settings, PatchyServerConfig)


def test_set_server_settings_updates_writer(stub_ensemble):
    stub_ensemble.set_server_settings(
        PatchyServerConfig(name="pytest_stub_server_lorenzo", oxdna_path="/nonexistent2", patchy_format="lorenzo")
    )
    assert stub_ensemble.server_settings.patchy_format == "lorenzo"
    assert isinstance(stub_ensemble.writer, LWriter)


# --------------------------- ensemble() / get_simulation() / is_multiselect() --------------------------- #

def test_num_ensemble_parameters(stub_ensemble):
    assert stub_ensemble.num_ensemble_parameters() == 2


def test_ensemble_is_full_cartesian_product(stub_ensemble):
    sims = stub_ensemble.ensemble()
    assert len(sims) == 4
    combos = {(sim["T"], sim["duplicate"]) for sim in sims}
    assert combos == {(0.05, 0), (0.05, 1), (0.1, 0), (0.1, 1)}


def test_get_simulation_fully_specified_by_kwargs_returns_single(stub_ensemble):
    sim = stub_ensemble.get_simulation(T="0.05", duplicate="0")
    assert not isinstance(sim, list)
    assert sim["T"] == 0.05
    assert sim["duplicate"] == 0


def test_get_simulation_partially_specified_returns_multiselect_list(stub_ensemble):
    sims = stub_ensemble.get_simulation(T="0.05")
    assert isinstance(sims, list)
    assert len(sims) == 2
    assert all(sim["T"] == 0.05 for sim in sims)
    assert {sim["duplicate"] for sim in sims} == {0, 1}


def test_get_simulation_with_parameter_value_positional_arg(stub_ensemble):
    sim = stub_ensemble.get_simulation(ParameterValue("T", 0.05), duplicate="1")
    assert sim["T"] == 0.05
    assert sim["duplicate"] == 1


def test_get_simulation_with_tuple_positional_args(stub_ensemble):
    sim = stub_ensemble.get_simulation(("T", "0.05"), ("duplicate", "0"))
    assert sim["T"] == 0.05
    assert sim["duplicate"] == 0


def test_param_value_valid(stub_ensemble):
    assert stub_ensemble.param_value_valid(ParameterValue("T", 0.05))
    assert not stub_ensemble.param_value_valid(ParameterValue("T", 999))
    assert not stub_ensemble.param_value_valid(ParameterValue("nonexistent", 1))


def test_is_multiselect_false_for_patchysimulation_instance(stub_ensemble):
    sim = stub_ensemble.get_simulation(T="0.05", duplicate="0")
    assert not stub_ensemble.is_multiselect(sim)


def test_is_multiselect_true_for_partial_selector(stub_ensemble):
    assert stub_ensemble.is_multiselect((ParameterValue("T", 0.05),))


def test_is_multiselect_false_for_fully_specified_selector(stub_ensemble):
    assert not stub_ensemble.is_multiselect((ParameterValue("T", 0.05), ParameterValue("duplicate", 0)))


def test_is_multiselect_non_iterable_selector_raises(stub_ensemble):
    with pytest.raises(Exception):
        stub_ensemble.is_multiselect(42)


# --------------------------- const params / add_params / get_param --------------------------- #

def test_add_params_scalar(stub_ensemble):
    stub_ensemble.add_params(pressure=5)
    assert "pressure" in stub_ensemble.const_params
    assert stub_ensemble.get_param("pressure") == 5


def test_add_params_invalid_type_raises(stub_ensemble):
    with pytest.raises(Exception):
        stub_ensemble.add_params(bad=[1, 2])


def test_get_param_missing_raises_no_such_param_error(stub_ensemble):
    with pytest.raises(NoSuchParamError):
        stub_ensemble.get_param("nonexistent")


def test_sim_get_param_prefers_sim_override_over_const_params(stub_ensemble):
    sim = stub_ensemble.get_simulation(T="0.05", duplicate="0")
    assert stub_ensemble.sim_get_param(sim, "T") == 0.05  # from the sim itself
    assert stub_ensemble.sim_get_param(sim, "density") == 0.1  # falls through to const_params


def test_sim_get_param_missing_raises_and_records_sim(stub_ensemble):
    sim = stub_ensemble.get_simulation(T="0.05", duplicate="0")
    with pytest.raises(NoSuchParamError) as excinfo:
        stub_ensemble.sim_get_param(sim, "nonexistent")
    assert repr(sim) in str(excinfo.value)


# --------------------------- path helpers --------------------------- #

def test_datestr_and_long_name(stub_ensemble):
    assert stub_ensemble.datestr() == "2024-01-01"
    assert stub_ensemble.long_name() == "pytest_stub_ensemble_2024-01-01"


def test_tld(stub_ensemble):
    assert stub_ensemble.tld() == simulation_run_dir() / "pytest_stub_ensemble_2024-01-01"


def test_folder_path_matches_tld_plus_sim_folder(stub_ensemble):
    sim = stub_ensemble.get_simulation(T="0.05", duplicate="0")
    assert stub_ensemble.folder_path(sim) == stub_ensemble.tld() / sim.get_folder_path()


def test_paramfile_resolves_param_to_filename(stub_ensemble):
    sim = stub_ensemble.get_simulation(T="0.05", duplicate="0")
    assert stub_ensemble.paramfile(sim, "topology") == stub_ensemble.folder_path(sim) / "initial.top"


# --------------------------- find_ensemble() dispatch and error paths --------------------------- #

def test_find_ensemble_with_nothing_raises():
    with pytest.raises(Exception):
        find_ensemble()


def test_find_ensemble_name_ending_in_json_raises():
    with pytest.raises(Exception):
        find_ensemble(name="something.json")


def test_find_ensemble_metadata_name_without_date_raises():
    with pytest.raises(Exception):
        find_ensemble(mdt="pytest_metadata_without_date")


def test_find_ensemble_missing_cfg_file_raises_with_path_in_message():
    missing_name = "pytest_definitely_does_not_exist_9182"
    with pytest.raises(FileNotFoundError) as excinfo:
        find_ensemble(cfg=missing_name)
    assert missing_name in str(excinfo.value)


@pytest.fixture
def stub_cfg_file():
    """
    Writes a minimal cfg-lookalike json file into the real ~/.pypatchy/input directory
    (matching the convention used by test_ensembles_basic.ensemble()), so find_ensemble's
    "name" dispatch branch can find it via a filesystem lookup. Cleans up after itself.
    """
    name = "pytest_stub_dispatch_cfg_9f3a"
    path = get_input_dir() / f"{name}.json"
    path.write_text(json.dumps({"export_name": "pytest_stub_dispatch_export_9f3a"}))
    yield name
    if path.exists():
        path.unlink()


def test_find_ensemble_forwards_loader_through_name_dispatch(monkeypatch, stub_cfg_file):
    """
    regression test: every recursive find_ensemble() call used to hand-pick which kwargs to
    forward (name/date/set_server) and silently drop `loader`, so a caller-supplied
    encoder_decoder was discarded unless resolution happened to bottom out on the very
    first hop.
    """
    calls = []

    def fake_build_ensemble(cfg, mdt, mdtfile=None, encoder_decoder=None):
        calls.append(encoder_decoder)
        return "SENTINEL_ENSEMBLE"

    monkeypatch.setattr(simulation_ensemble, "build_ensemble", fake_build_ensemble)

    class CustomLoader:
        pass

    result = find_ensemble(name=stub_cfg_file, loader=CustomLoader)
    assert result == "SENTINEL_ENSEMBLE"
    assert calls == [CustomLoader]


def test_find_ensemble_forwards_loader_through_positional_dispatch(monkeypatch, stub_cfg_file):
    """
    Same regression as above, but entering through the single-positional-arg branch first
    (an extra hop before reaching the "name" branch covered above).
    """
    calls = []

    def fake_build_ensemble(cfg, mdt, mdtfile=None, encoder_decoder=None):
        calls.append(encoder_decoder)
        return "SENTINEL_ENSEMBLE"

    monkeypatch.setattr(simulation_ensemble, "build_ensemble", fake_build_ensemble)

    class CustomLoader:
        pass

    result = find_ensemble(stub_cfg_file, loader=CustomLoader)
    assert result == "SENTINEL_ENSEMBLE"
    assert calls == [CustomLoader]