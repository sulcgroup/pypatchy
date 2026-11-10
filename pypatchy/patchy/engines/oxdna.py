from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Union

from oxDNA_analysis_tools.file_info import file_info
from oxpy_utils.oxdna_simulation import SimulationManager

from ...util import EXTERNAL_OBSERVABLES
from ..base_param_val import ParameterValue
from ..pl.oxpy_energy import OxpyEnergyModel
from ..pl.plscene import PLPSimulation
from ..simulation_specification import PatchySimulation
from ..stage import NoStageTrajError, Stage, StageTrajFileEmptyError
from .base import SimulationEngine


class OxDNAEngine(SimulationEngine):
    """
    Runs simulations with oxDNA. Stages are oxpy_utils simulations: each stage directory holds an oxDNA
    topology, configuration and input file, written by Stage.build_dat_top() / Stage.build_input(), and
    runs through oxpy (or the oxDNA executable, for per-simulation slurm jobs).
    """

    name = "oxdna"

    def build_stage(self,
                    ensemble: Any,
                    sim: PatchySimulation,
                    stage: Stage,
                    clean_build: Union[bool, str] = False):
        # build the stage directory, .top/.dat, and input file via oxpy_utils's Simulation.build(),
        # which Stage plugs into via build_dat_top()/build_input()
        ensemble.get_logger().info("Building simulation directory, .top/.dat, and input files...")
        ensemble.ipy(sim, stage).build(clean_build=clean_build)

        # write observables.json if applicble
        if EXTERNAL_OBSERVABLES:
            ensemble.get_logger().info("Writing observable json, as nessecary...")
            ensemble.write_sim_observables(sim)
        # only needed when each simulation is submitted as its own sbatch job -
        # batched (cuda_mps) and in-process runs never read this script
        if ensemble.server_settings.submits_per_sim_slurm_job():
            ensemble.get_logger().info("Writing sbatch scripts...")
            ensemble.write_run_script(sim, stage)

    def is_stage_built(self, ensemble: Any, sim: PatchySimulation, stage: Stage) -> bool:
        # check that folder exists
        if not stage.sim.sim_dir.exists():
            return False
        # check for various files
        if not ensemble.sim_get_stage_top(sim, stage).exists():
            return False
        if not (ensemble.folder_path(sim, stage) / ensemble.sim_stage_get_param(sim, stage, "conf_file")).exists():
            return False
        if not (ensemble.folder_path(sim, stage) / "input").exists():
            return False
        return True

    def check_ready_to_run(self, ensemble: Any, sim: PatchySimulation, stage: Stage):
        # shouldn't be nessecary anymore, since do_setup()/write_setup_files() always writes the
        # conf file -- but if it's somehow still missing here, fail loudly rather than silently
        # proceeding to run without a conf (there is no gen_conf() method to fall back on)
        if not ensemble.paramstagefile(sim, stage, "conf_file").exists():
            raise FileNotFoundError(
                f"Conf file {ensemble.paramstagefile(sim, stage, 'conf_file')} missing for sim {repr(sim)} "
                f"stage {stage.name()} after setup completed!")

    def has_output(self, ensemble: Any, sim: PatchySimulation) -> bool:
        return (ensemble.folder_path(sim) / ensemble.sim_get_param(sim, "lastconf_file")).exists()

    def stage_last_step(self, ensemble: Any, sim: PatchySimulation, stage: Stage) -> int:
        lastconf_file = ensemble.sim_get_stage_last_conf(sim, stage)
        if not lastconf_file.is_file():
            raise NoStageTrajError(stage, sim, str(lastconf_file))
        else:
            # return timepoint of last conf in traj
            try:
                return file_info([str(lastconf_file)])["t_end"][0]
            except IndexError as e:
                # trajectory file empty
                raise StageTrajFileEmptyError(stage, sim, str(lastconf_file))

    def stage_done(self, ensemble: Any, sim: PatchySimulation, stage: Stage) -> bool:
        last_conf_file = ensemble.sim_get_stage_last_conf(sim, stage)
        if not last_conf_file.is_file():
            return False
        else:
            # return timepoint of last conf in traj
            return file_info([str(last_conf_file)])["t_end"][0] == stage.end_time()

    def read_scene(self,
                   ensemble: Any,
                   sim: PatchySimulation,
                   stage: Stage,
                   step: Union[int, None] = None) -> PLPSimulation:
        ensemble.writer.set_directory(ensemble.folder_path(sim, stage))

        top_file, traj_file = ensemble.sim_get_stage_top_traj(sim, stage)
        particle_types = ensemble.writer.read_particle_types(**{key: ensemble.sim_stage_get_param(sim, stage, key)
                                                                for key in ensemble.writer.reqd_args()
                                                                })
        # particle files don't store type names, so restore them (e.g. "CT1") from the
        # ensemble's particle set, matching on type id
        named_types = {p.type_id(): p.name() for p in ensemble.sim_get_particles_set(sim).particles()}
        for ptype in particle_types.particles():
            if ptype.type_id() in named_types:
                ptype.set_name(named_types[ptype.type_id()])
        if traj_file.exists() and os.stat(traj_file).st_size > 0:
            if step is None:
                step = ensemble.time_length(sim)

            # if we have requested last conf
            if ensemble.time_length(sim) == step or ensemble.sim_stage_get_param(sim, stage, "steps") == step:
                traj_file = ensemble.sim_get_stage_last_conf(sim, stage)
                scene: PLPSimulation = ensemble.writer.read_scene(top_file.name, traj_file.name,
                                                                  particle_types)
            else:
                conf_interval = int(float(ensemble.sim_stage_get_param(sim, stage, "print_conf_interval")))
                assert step % conf_interval == 0, (f"Attempting to get a conf at a timepoint {step} that is not a "
                                                   f"multiple of the conf interval "
                                                   f"{ensemble.sim_stage_get_param(sim, stage, 'print_conf_interval')}")

                scene: PLPSimulation = ensemble.writer.read_scene(top_file.name, traj_file.name,
                                                                  particle_types,
                                                                  (step - stage.start_time()) // conf_interval)
        else:
            assert step is None, f"Could not find trajectory file {traj_file!s}, " \
                                 f"but timestep was explicitly specified."
            conf_file = ensemble.sim_stage_get_param(sim, stage, "conf_file")
            scene: PLPSimulation = ensemble.writer.read_scene(top_file.name,
                                                              conf_file,
                                                              particle_types)

        scene.set_temperature(ensemble.sim_stage_get_param(sim, stage, "T"))
        scene.set_energy_model(self.get_energy_model(ensemble, sim, stage))
        return scene

    def get_energy_model(self,
                         ensemble: Any,
                         sim: PatchySimulation,
                         stage: Union[Stage, None] = None) -> OxpyEnergyModel:
        """
        Describes the oxDNA interaction a simulation uses, so scenes from it can have their energies computed by
        oxDNA itself (via oxpy). Uses the same merged parameters (stage > sim > ensemble > server) the stage
        writes into its oxDNA input file.
        """
        if stage is None:
            stage = ensemble.sim_get_stages(sim)[0]
        input_params = {}
        for pv in ensemble.iter_params(sim, stage):
            if type(pv) is ParameterValue and not isinstance(pv.param_value, (dict, list, tuple)):
                input_params.setdefault(pv.param_name, pv.param_value)
        if "interaction_potentials" in {pv.param_name for pv in ensemble.iter_params(sim, stage)} \
                and not getattr(ensemble, "_warned_interaction_potentials", False):
            warnings.warn("`interaction_potentials` is deprecated and ignored: pypatchy now computes energies with "
                          "the simulation's own oxDNA interaction (via oxpy). Remove it from your ensemble and "
                          "server configs.", FutureWarning, stacklevel=3)
            ensemble._warned_interaction_potentials = True

        # a raspberry topology may define non-default (or per-type) repulsion sites; reuse them if this stage has
        # written one
        repulsion_sites, type_rsite_lists = None, None
        if ensemble.server_settings.patchy_format == "raspberry":
            top_file = ensemble.folder_path(sim, stage) / ensemble.sim_stage_get_param(sim, stage, "topology")
            if top_file.exists():
                ensemble.writer.set_directory(top_file.parent)
                top = ensemble.writer.read_top(top_file.name)
                repulsion_sites, type_rsite_lists = top.particle_repulsion_sites, top.type_rsite_lists

        return OxpyEnergyModel(ensemble.server_settings.patchy_format, input_params,
                               repulsion_sites=repulsion_sites, type_rsite_lists=type_rsite_lists)

    def run_stage(self, ensemble: Any, sim: PatchySimulation, stage: Stage):
        # run oxDNA in-process via oxpy, blocking until it finishes. This replaces the old
        # `bash {script_name} > simulation.log` + async Popen mechanism, which returned
        # immediately without waiting for completion, leaving the simulation running as an
        # orphaned background process with no way to detect success/failure.
        ipysim = ensemble.ipy(sim, stage)
        ipysim.oxpy_run.run(subprocess=True, join=False, verbose=False)
        ipysim.oxpy_run.process.join()

    def run_batch(self, ensemble: Any, to_run: list[tuple[PatchySimulation, Stage]]):
        """
        Uses an oxpy_utils SimulationManager: at most `max_concurrent_sims` at once, spread evenly
        across the GPUs (at most `max_sims_per_gpu` on each), sharing each GPU through an NVIDIA MPS daemon.
        """
        mgr = SimulationManager(n_processes=ensemble.server_settings.max_concurrent_sims,
                                sleep_time=1,
                                max_sims_per_gpu=ensemble.server_settings.max_sims_per_gpu)
        try:
            backends = set()
            for sim, stage in to_run:
                ipysim = ensemble.ipy(sim, stage)
                backends.add(ipysim.input.input_dict.get("backend"))
                mgr.queue_sim(ipysim)
            ensemble.get_logger().info(f"Let the simulating commence! ({len(to_run)} simulations)")
            # run_when_failed: one bad simulation shouldn't take the rest of a large sweep with
            # it - failures are reported together afterwards
            mgr.worker_manager(cpu_run=backends == {"CPU"}, run_when_failed=True)
        finally:
            mgr.shutdown()

    def stage_log_file(self, ensemble: Any, sim: PatchySimulation, stage: Stage) -> Path:
        return ensemble.folder_path(sim, stage) / "log.log"
