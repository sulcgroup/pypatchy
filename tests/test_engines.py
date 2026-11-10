"""
Tests for simulation engine selection (pypatchy.patchy.engines): server configs pick the program a
PatchySimulationEnsemble builds, runs and reads simulations with. Offline; nothing is simulated.
"""
import datetime
import pickle

import pytest

from pypatchy.patchy.base_param_val import ParameterValue
from pypatchy.patchy.engines import OxDNAEngine, SimulationEngine, get_engine
from pypatchy.patchy.ensemble_parameter import EnsembleParameter
from pypatchy.patchy.simulation_ensemble import PatchySimulationEnsemble
from pypatchy.patchy.simulation_specification import ParamSet
from pypatchy.server_config import PatchyServerConfig


def test_server_configs_default_to_oxdna():
    sc = PatchyServerConfig(name="n", oxdna_path="/x", patchy_format="flavio")
    assert sc.engine == "oxdna"


def test_engine_round_trips_through_to_dict():
    sc = PatchyServerConfig(name="n", oxdna_path="/x", patchy_format="flavio", engine="oxdna")
    assert PatchyServerConfig(**sc.to_dict()).engine == "oxdna"


def test_saved_configs_without_an_engine_still_load():
    """server configs and ensemble metadata written before engines existed have no "engine" key"""
    saved = PatchyServerConfig(name="n", oxdna_path="/x", patchy_format="flavio").to_dict()
    del saved["engine"]
    assert PatchyServerConfig(**saved).engine == "oxdna"


def test_get_engine():
    engine = get_engine("oxdna")
    assert isinstance(engine, OxDNAEngine)
    assert isinstance(engine, SimulationEngine)
    assert engine.name == "oxdna"


def test_unknown_engine_names_are_rejected():
    with pytest.raises(ValueError, match="No simulation engine called 'gromacs'"):
        get_engine("gromacs")


def test_ensemble_uses_its_server_configs_engine(tmp_path):
    e = PatchySimulationEnsemble(
        export_name="pytest_engine_ensemble",
        setup_date=datetime.datetime(2024, 1, 1),
        metadata_file_name="pytest_engine_ensemble_2024-01-01_metadata.json",
        const_params=ParamSet([ParameterValue("density", 0.1)]),
        ensemble_params=[EnsembleParameter("T", [ParameterValue("T", 0.05)])],
        observables={},
        metadata_dict={"ensemble_config": {}},
        server_settings=PatchyServerConfig(name="pytest_engine", oxdna_path="/nonexistent",
                                           patchy_format="flavio"),
        log_file_name=tmp_path / "pytest_engine_ensemble.log",
    )
    assert isinstance(e.engine, OxDNAEngine)
    # engines hold no state, so they pickle with the ensemble (parallel analysis pickles ensembles)
    assert isinstance(pickle.loads(pickle.dumps(e.engine)), OxDNAEngine)
