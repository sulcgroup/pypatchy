"""
Tests for batched execution (server config `cuda_mps`): PatchySimulationEnsemble.start_simulations()
running all of its simulations as concurrent worker processes via oxpy_utils's SimulationManager.

Most of these replace SimulationManager with a recording fake and stub out the ensemble's
filesystem-facing methods, so they're offline and test only the ensemble's own decisions: which
simulations get queued, how they're set up, and how failures surface. The last test runs a real
(small, CPU) ensemble end to end, and is skipped unless oxDNA's romano plugin is built.
"""
import datetime
import pickle
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pypatchy.patchy.base_param_val import ParameterValue
from pypatchy.patchy.engines import oxdna as oxdna_engine_module
from pypatchy.patchy.ensemble_parameter import EnsembleParameter
from pypatchy.patchy.simulation_ensemble import PatchySimulationEnsemble
from pypatchy.patchy.simulation_specification import ParamSet
from pypatchy.patchy.stage import AllStagesCompleteError
from pypatchy.server_config import PatchyServerConfig
from pypatchy.util import get_input_dir
from ptutils import temp_dir  # noqa: F401 - pytest fixture


# --------------------------- server config --------------------------- #

def test_cuda_mps_is_a_plain_flag():
    """
    cuda_mps used to be swapped for a live SimulationManager (a multiprocessing.Manager server
    process, plus a global SIGTERM handler) the moment the config was constructed - including
    just from listing server configs.
    """
    sc = PatchyServerConfig(name="n", oxdna_path="/x", patchy_format="flavio", cuda_mps=True)
    assert sc.cuda_mps is True
    assert sc.is_batched()
    assert not PatchyServerConfig(name="n", oxdna_path="/x", patchy_format="flavio").is_batched()


def test_limits_default_to_unset_and_round_trip():
    default = PatchyServerConfig(name="n", oxdna_path="/x", patchy_format="flavio")
    assert default.max_sims_per_gpu is None and default.max_concurrent_sims is None
    sc = PatchyServerConfig(name="n", oxdna_path="/x", patchy_format="flavio", cuda_mps=True,
                            max_sims_per_gpu=8, max_concurrent_sims=32)
    again = PatchyServerConfig(**sc.to_dict())
    assert (again.cuda_mps, again.max_sims_per_gpu, again.max_concurrent_sims) == (True, 8, 32)
    assert pickle.loads(pickle.dumps(sc)).max_sims_per_gpu == 8  # still picklable, no special-casing


# --------------------------- start_simulations(), batched --------------------------- #

class FakeManager:
    """records what start_simulations() asks of a SimulationManager"""
    instances: list["FakeManager"] = []

    def __init__(self, n_processes=None, sleep_time=5.0, max_sims_per_gpu=None):
        self.init_args = dict(n_processes=n_processes, max_sims_per_gpu=max_sims_per_gpu)
        self.queued = []
        self.worker_manager_kwargs = None
        self.shut_down = False
        self.on_run = lambda: None
        FakeManager.instances.append(self)

    def queue_sim(self, sim):
        self.queued.append(sim)

    def worker_manager(self, **kwargs):
        self.worker_manager_kwargs = kwargs
        self.on_run()
        return []

    def shutdown(self):
        self.shut_down = True


class FakeIpySim:
    def __init__(self, backend):
        self.input = MagicMock()
        self.input.input_dict = {"backend": backend}


@pytest.fixture
def batched_ensemble(tmp_path, monkeypatch):
    """four-simulation ensemble whose filesystem-touching methods are stubbed out"""
    FakeManager.instances = []
    # the oxDNA engine's run_batch() is what creates the SimulationManager
    monkeypatch.setattr(oxdna_engine_module, "SimulationManager", FakeManager)
    e = PatchySimulationEnsemble(
        export_name="pytest_batched_ensemble",
        setup_date=datetime.datetime(2024, 1, 1),
        metadata_file_name="pytest_batched_ensemble_2024-01-01_metadata.json",
        const_params=ParamSet([ParameterValue("density", 0.1)]),
        ensemble_params=[
            EnsembleParameter("T", [ParameterValue("T", 0.05), ParameterValue("T", 0.1)]),
            EnsembleParameter("duplicate", [ParameterValue("duplicate", 0), ParameterValue("duplicate", 1)]),
        ],
        observables={},
        metadata_dict={"ensemble_config": {}},
        server_settings=PatchyServerConfig(name="pytest_batched", oxdna_path="/nonexistent",
                                           patchy_format="flavio", cuda_mps=True,
                                           max_sims_per_gpu=6, max_concurrent_sims=12),
        log_file_name=tmp_path / "pytest_batched_ensemble.log",
    )
    e.setup_calls = []
    e.backend = "CUDA"
    e.setup_done = set()
    e.stage_done = lambda sim, stage: True  # overridden per test
    stage = MagicMock()
    stage.name.return_value = "stage0"
    monkeypatch.setattr(e, "check_stage", lambda sim: stage)
    monkeypatch.setattr(e, "sim_get_stage", lambda sim, name: stage)
    monkeypatch.setattr(e, "is_setup_done", lambda sims, stage: all(s in e.setup_done for s in sims))
    monkeypatch.setattr(e, "do_setup", lambda sims, stage=None, clean_build=False:
                        (e.setup_calls.append((list(sims), stage)), e.setup_done.update(sims)))
    monkeypatch.setattr(e, "ok_to_run", lambda sim, stage: True)
    monkeypatch.setattr(e, "ipy", lambda sim, stage=None: FakeIpySim(e.backend))
    monkeypatch.setattr(e, "sim_stage_done", lambda sim, stage: e.stage_done(sim, stage))
    return e


def test_queues_every_simulation_with_the_configured_limits(batched_ensemble):
    batched_ensemble.start_simulations()
    (mgr,) = FakeManager.instances
    assert len(mgr.queued) == 4
    assert mgr.init_args == dict(n_processes=12, max_sims_per_gpu=6)
    assert mgr.worker_manager_kwargs["cpu_run"] is False
    assert mgr.shut_down


def test_cpu_backend_runs_without_gpus(batched_ensemble):
    # the backend is read from the built simulation, so it works wherever it was configured
    batched_ensemble.backend = "CPU"
    batched_ensemble.start_simulations()
    assert FakeManager.instances[0].worker_manager_kwargs["cpu_run"] is True


def test_remaining_simulations_run_even_if_one_fails(batched_ensemble):
    # a single bad simulation mustn't abort a large sweep half way
    batched_ensemble.start_simulations()
    assert FakeManager.instances[0].worker_manager_kwargs["run_when_failed"] is True


def test_sets_up_all_simulations_in_one_call_per_stage(batched_ensemble):
    # do_setup() rewrites the ensemble metadata each call, so not once per simulation. It is also
    # given the stage *name*: a Stage object belongs to one simulation, and handing sim 0's
    # stage to the others builds them against the wrong directory.
    batched_ensemble.start_simulations()
    assert len(batched_ensemble.setup_calls) == 1
    sims, stage = batched_ensemble.setup_calls[0]
    assert len(sims) == 4 and stage == "stage0"


def test_does_not_rebuild_simulations_that_are_already_set_up(batched_ensemble):
    batched_ensemble.setup_done.update(batched_ensemble.ensemble())
    batched_ensemble.start_simulations()
    assert batched_ensemble.setup_calls == []
    assert len(FakeManager.instances[0].queued) == 4


def test_skips_simulations_with_nothing_left_to_run(batched_ensemble, monkeypatch):
    sims = batched_ensemble.ensemble()
    finished = sims[0]

    def check_stage(sim):
        if sim == finished:
            raise AllStagesCompleteError(MagicMock(), sim)
        return MagicMock()

    monkeypatch.setattr(batched_ensemble, "check_stage", check_stage)
    batched_ensemble.start_simulations()
    assert len(FakeManager.instances[0].queued) == 3


def test_skips_simulations_that_are_not_ok_to_run(batched_ensemble, monkeypatch):
    sims = batched_ensemble.ensemble()
    monkeypatch.setattr(batched_ensemble, "ok_to_run", lambda sim, stage: sim != sims[0])
    batched_ensemble.start_simulations()
    assert len(FakeManager.instances[0].queued) == 3


def test_nothing_to_run_does_not_start_a_manager(batched_ensemble, monkeypatch):
    monkeypatch.setattr(batched_ensemble, "check_stage",
                        lambda sim: (_ for _ in ()).throw(AllStagesCompleteError(MagicMock(), sim)))
    batched_ensemble.start_simulations()
    assert FakeManager.instances == []


def test_incomplete_simulations_raise(batched_ensemble):
    sims = batched_ensemble.ensemble()
    batched_ensemble.stage_done = lambda sim, stage: sim not in (sims[1], sims[2])
    with pytest.raises(RuntimeError, match="2 of 4 simulations did not complete"):
        batched_ensemble.start_simulations()
    assert FakeManager.instances[0].shut_down


def test_manager_released_when_the_run_is_interrupted(batched_ensemble):
    def interrupted():
        raise KeyboardInterrupt

    original_init = FakeManager.__init__

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.on_run = interrupted

    FakeManager.__init__ = init
    try:
        with pytest.raises(KeyboardInterrupt):
            batched_ensemble.start_simulations()
    finally:
        FakeManager.__init__ = original_init
    assert FakeManager.instances[0].shut_down


def test_leaving_the_ensemble_context_needs_no_manager_teardown(batched_ensemble, monkeypatch):
    monkeypatch.setattr(batched_ensemble, "dump_metadata", lambda: None)
    with batched_ensemble:
        pass
    assert batched_ensemble.server_settings.cuda_mps is True  # used to be replaced with a bare True


# --------------------------- end to end --------------------------- #

OXDNA_PATH = Path("/root/Github/oxDNA")


@pytest.mark.skipif(not (OXDNA_PATH / "contrib" / "romano" / "PatchyShapeInteraction.so").exists(),
                    reason="needs oxDNA with contrib/romano built")
def test_batched_cpu_ensemble_runs_and_resumes(temp_dir):
    """real simulations, run concurrently by a real SimulationManager; a second call leaves them alone"""
    from pypatchy.patchy.simulation_ensemble import build_ensemble, EXPORT_NAME_KEY, ENSEMBLE_SETUP_DATE, \
        CONST_PARAMS_KEY, ENSEMBLE_PARAMS_KEY, SERVER_SETTINGS_KEY, PARTICLE_TYPE_LVLS_KEY

    e = build_ensemble(cfg={
        EXPORT_NAME_KEY: "pytest_batched_cpu",
        "rule": "1|1|1|1|1|1_-1|||||",
        CONST_PARAMS_KEY: {
            "num_assemblies": 2, "density": 0.3, "steps": 2e4, "print_conf_interval": 1e4,
            "print_energy_every": 1e4, "interaction_potentials": ["fr_excl_vol", "fr_patchy"],
            "narrow_type": 1, "patchy_file": "patches.txt", "particle_file": "particles.txt",
            PARTICLE_TYPE_LVLS_KEY: [1, 6], "backend": "CPU", "backend_precision": "double",
            "interaction_type": "PatchyShapeInteraction",
            "plugin_search_path": str(OXDNA_PATH / "contrib" / "romano"),
        },
        ENSEMBLE_PARAMS_KEY: [("T", [0.01, 0.025]), ("duplicate", [0, 1])],
        SERVER_SETTINGS_KEY: PatchyServerConfig(name="pytest_batched_cpu_server", oxdna_path=str(OXDNA_PATH),
                                                patchy_format="flavio", cuda_mps=True, max_concurrent_sims=2),
    }, mdt={ENSEMBLE_SETUP_DATE: datetime.datetime.now()})
    try:
        e.start_simulations()
        sims = e.ensemble()
        assert len(sims) == 4
        assert all(e.sim_stage_done(sim, e.sim_get_stage(sim, 0)) for sim in sims)

        last_confs = {sim: e.sim_get_stage_last_conf(sim, e.sim_get_stage(sim, 0)) for sim in sims}
        mtimes = {sim: path.stat().st_mtime_ns for sim, path in last_confs.items()}
        e.start_simulations()  # everything is finished: must not wipe and rerun it
        assert {sim: path.stat().st_mtime_ns for sim, path in last_confs.items()} == mtimes
    finally:
        # build_ensemble() wrote this to the user's input dir (the simulations themselves went
        # to temp_dir)
        (get_input_dir() / e.metadata_file).unlink(missing_ok=True)
