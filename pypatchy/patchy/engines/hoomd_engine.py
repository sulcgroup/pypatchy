from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Union

from ..base_param_val import ParameterValue
from ..hoomd_backend import gsdio
from ..hoomd_backend.model import RigidPatchyModel
from ..hoomd_backend.params import resolve_hoomd_params
from ..hoomd_backend.run_stage import (ENERGY_FILE, INIT_FILE, LAST_FILE, LOG_FILE, STAGE_SPEC_FILE,
                                       TRAJECTORY_FILE, read_spec)
from ..pl.plscene import PLPSimulation
from ..simulation_specification import PatchySimulation
from ..stage import NoStageTrajError, Stage, StageTrajFileEmptyError
from .base import SimulationEngine

# the name PatchySimulationEnsemble.start_simulation() submits per-simulation slurm jobs under
SLURM_SCRIPT_FILE = "slurm_script.sh"
# every file a HOOMD stage directory holds (build_stage(clean_build="force") removes these, and only these)
STAGE_FILES = (STAGE_SPEC_FILE, INIT_FILE, TRAJECTORY_FILE, LAST_FILE, ENERGY_FILE, LOG_FILE, SLURM_SCRIPT_FILE)


class HOOMDEngine(SimulationEngine):
    """
    Runs simulations with HOOMD-blue MD, using pypatchy's rigid-body patchy model
    (pypatchy.patchy.hoomd_backend.model). Each stage directory holds stage.json (the stage's resolved
    parameters, model and steps) and init.gsd (its starting configuration); running it
    (pypatchy.patchy.hoomd_backend.run_stage) writes trajectory.gsd, energy.txt and, at the end, last.gsd.
    A later stage starts from the previous stage's last.gsd, after its own particle additions.

    Parameters use HOOMD-native names (see pypatchy.patchy.hoomd_backend.params); oxDNA input parameters are an
    error, not translated.
    """

    name = "hoomd"

    # -------------------------------------------------------------------------------------- parameters
    def stage_params(self, ensemble: Any, sim: PatchySimulation, stage: Stage) -> tuple[dict[str, Any], set[str]]:
        """
        :return: (the stage's HOOMD parameters, with defaults filled in; names of the ones left at their default)
        :raises HOOMDParameterError: if the simulation has parameters the HOOMD engine doesn't use, or is
            missing required ones
        """
        values = {}
        # iter_params yields in order of precedence (stage, sim, ensemble, server): the first value seen applies
        for pv in ensemble.iter_params(sim, stage):
            if type(pv) is ParameterValue:
                values.setdefault(pv.param_name, pv.param_value)
        particle_names = {p.name() for p in ensemble.sim_get_particles_set(sim).particles()}
        return resolve_hoomd_params(values, particle_names)

    def get_energy_model(self,
                         ensemble: Any,
                         sim: PatchySimulation,
                         stage: Union[Stage, None] = None) -> RigidPatchyModel:
        if stage is None:
            stage = ensemble.sim_get_stages(sim)[0]
        params, _ = self.stage_params(ensemble, sim, stage)
        return RigidPatchyModel.from_particle_set(ensemble.sim_get_particles_set(sim),
                                                  patch_epsilon=params["patch_epsilon"],
                                                  patch_sigma=params["patch_sigma"],
                                                  patch_rcut=params["patch_rcut"],
                                                  particle_epsilon=params["particle_epsilon"])

    # -------------------------------------------------------------------------------------- building
    def build_stage(self,
                    ensemble: Any,
                    sim: PatchySimulation,
                    stage: Stage,
                    clean_build: Union[bool, str] = False):
        stage_dir = ensemble.folder_path(sim, stage)
        if self.is_stage_built(ensemble, sim, stage):
            if not clean_build:
                ensemble.get_logger().info(f"Stage {stage.name()} of {sim!r} is already built; leaving it alone")
                return
            for name in STAGE_FILES:
                (stage_dir / name).unlink(missing_ok=True)
        params, defaulted = self.stage_params(ensemble, sim, stage)
        if stage.is_first():
            scene = PLPSimulation()
            scene.set_particle_types(ensemble.sim_get_particles_set(sim))
        else:
            prev = stage.get_prev()
            if not (self.stage_done(ensemble, sim, prev) or prev.allow_shortfall()):
                raise RuntimeError(f"Can't build stage {stage.name()} of {sim!r}: the previous stage "
                                   f"({prev.name()}) hasn't finished")
            # start from where the previous stage finished
            scene = self.read_scene(ensemble, sim, prev)
        scene.set_temperature(params["kT"])
        # adds this stage's particles, resizes the box, ... (sets the scene's energy model)
        stage.apply(scene)
        model: RigidPatchyModel = scene.energy_model()

        frame, unthermalized = gsdio.scene_to_frame(scene, model, step=stage.start_time())
        stage_dir.mkdir(parents=True, exist_ok=True)
        spec = {
            "engine": self.name,
            "format": 1,
            "stage_name": stage.name(),
            "start_step": stage.start_time(),
            "end_step": stage.end_time(),
            "params": params,
            "defaulted": sorted(defaulted),
            "model": model.to_dict(),
            # particles with no momenta yet (added this stage): drawn from the Maxwell-Boltzmann distribution
            "thermalize": unthermalized,
        }
        gsdio.write_frame(stage_dir / INIT_FILE, frame)
        with (stage_dir / STAGE_SPEC_FILE).open("w") as f:
            json.dump(spec, f, indent=2)
        if defaulted:
            ensemble.get_logger().info(f"HOOMD parameters left at their defaults for {sim!r} stage {stage.name()}: "
                                       + ", ".join(f"{k}={params[k]}" for k in sorted(defaulted)))
        if ensemble.server_settings.submits_per_sim_slurm_job():
            self.write_run_script(ensemble, sim, stage)

    def write_run_script(self, ensemble: Any, sim: PatchySimulation, stage: Stage,
                         script_name: str = SLURM_SCRIPT_FILE) -> Path:
        """
        Writes the sbatch script that runs a stage as its own slurm job (PatchySimulationEnsemble.start_simulation()
        submits it). It runs the stage with the Python that built it, so the job's environment (slurm_includes)
        must provide the same environment. Runner output goes to the stage's log.log, as for in-process runs.
        """
        stage_dir = ensemble.folder_path(sim, stage)
        script = stage_dir / script_name
        with script.open("w") as f:
            ensemble.server_settings.write_sbatch_params(ensemble.export_name, f)
            f.write(f"{shlex.quote(sys.executable)} -m pypatchy.patchy.hoomd_backend.run_stage "
                    f"{shlex.quote(str(stage_dir))} > {shlex.quote(str(stage_dir / LOG_FILE))} 2>&1\n")
        return script

    def is_stage_built(self, ensemble: Any, sim: PatchySimulation, stage: Stage) -> bool:
        stage_dir = ensemble.folder_path(sim, stage)
        return (stage_dir / STAGE_SPEC_FILE).exists() and (stage_dir / INIT_FILE).exists()

    def check_ready_to_run(self, ensemble: Any, sim: PatchySimulation, stage: Stage):
        if not self.is_stage_built(ensemble, sim, stage):
            raise FileNotFoundError(f"Stage {stage.name()} of {sim!r} isn't built (no {STAGE_SPEC_FILE} and "
                                    f"{INIT_FILE} in {ensemble.folder_path(sim, stage)})")

    # -------------------------------------------------------------------------------------- progress
    def has_output(self, ensemble: Any, sim: PatchySimulation) -> bool:
        return (ensemble.folder_path(sim) / TRAJECTORY_FILE).exists()

    def stage_last_step(self, ensemble: Any, sim: PatchySimulation, stage: Stage) -> int:
        stage_dir = ensemble.folder_path(sim, stage)
        if (stage_dir / LAST_FILE).is_file():
            return gsdio.frame_steps(stage_dir / LAST_FILE)[-1]
        traj = stage_dir / TRAJECTORY_FILE
        if not traj.is_file():
            raise NoStageTrajError(stage, sim, str(traj))
        steps = gsdio.frame_steps(traj)
        if not steps:
            raise StageTrajFileEmptyError(stage, sim, str(traj))
        return steps[-1]

    def stage_done(self, ensemble: Any, sim: PatchySimulation, stage: Stage) -> bool:
        last = ensemble.folder_path(sim, stage) / LAST_FILE
        return last.is_file() and gsdio.frame_steps(last)[-1] == stage.end_time()

    def read_scene(self,
                   ensemble: Any,
                   sim: PatchySimulation,
                   stage: Stage,
                   step: Union[int, None] = None) -> PLPSimulation:
        stage_dir = ensemble.folder_path(sim, stage)
        last, traj, init = stage_dir / LAST_FILE, stage_dir / TRAJECTORY_FILE, stage_dir / INIT_FILE
        if step is None:
            # the most recent configuration: the final one, else the latest trajectory frame, else the start
            if last.is_file():
                frame = gsdio.read_frame(last)
            elif traj.is_file() and gsdio.frame_steps(traj):
                frame = gsdio.read_frame(traj)
            elif init.is_file():
                frame = gsdio.read_frame(init)
            else:
                raise NoStageTrajError(stage, sim, str(traj))
        else:
            if last.is_file() and gsdio.frame_steps(last)[-1] == step:
                frame = gsdio.read_frame(last)
            else:
                steps = gsdio.frame_steps(traj) if traj.is_file() else []
                if step not in steps:
                    raise ValueError(f"Stage {stage.name()} of {sim!r} has no configuration at step {step}" +
                                     (f" (trajectory frames span steps {steps[0]}-{steps[-1]}, every "
                                      f"{steps[1] - steps[0] if len(steps) > 1 else '?'})" if steps else
                                      " (no trajectory frames)"))
                frame = gsdio.read_frame(traj, steps.index(step))

        # use the model the stage was built with, so the scene matches what was simulated
        model = RigidPatchyModel.from_dict(read_spec(stage_dir)["model"]) \
            if (stage_dir / STAGE_SPEC_FILE).is_file() else self.get_energy_model(ensemble, sim, stage)
        scene = gsdio.frame_to_scene(frame, ensemble.sim_get_particles_set(sim), model)
        params, _ = self.stage_params(ensemble, sim, stage)
        scene.set_temperature(params["kT"])
        scene.set_energy_model(model)
        return scene

    def generate_stages(self, ensemble: Any, sim: PatchySimulation) -> list[Stage]:
        # every HOOMD stage directory is built from the ensemble's staging, so it describes the stages exactly
        return ensemble.sim_get_spec_stages(sim)

    # -------------------------------------------------------------------------------------- running
    def run_stage(self, ensemble: Any, sim: PatchySimulation, stage: Stage):
        stage_dir = ensemble.folder_path(sim, stage)
        ensemble.get_logger().info(f"Running HOOMD stage {stage.name()} of {sim!r} in {stage_dir}")
        with (stage_dir / LOG_FILE).open("w") as log:
            subprocess.run([sys.executable, "-m", "pypatchy.patchy.hoomd_backend.run_stage", str(stage_dir)],
                           stdout=log, stderr=subprocess.STDOUT, cwd=stage_dir)

    def run_batch(self, ensemble: Any, to_run: list[tuple[PatchySimulation, Stage]]):
        """
        Runs the stages as concurrent worker processes (pypatchy.patchy.hoomd_backend.batch): at most
        `max_concurrent_sims` at once, spread over the GPUs (at most `max_sims_per_gpu` on each), sharing each GPU
        through a dedicated NVIDIA MPS daemon.
        """
        from ..hoomd_backend.batch import run_stage_dirs
        settings = ensemble.server_settings
        run_stage_dirs([ensemble.folder_path(sim, stage) for sim, stage in to_run],
                       max_concurrent=settings.max_concurrent_sims,
                       max_per_gpu=settings.max_sims_per_gpu,
                       log=ensemble.get_logger().info)

    def stage_log_file(self, ensemble: Any, sim: PatchySimulation, stage: Stage) -> Path:
        return ensemble.folder_path(sim, stage) / LOG_FILE

    def trajectory_file(self, ensemble: Any, sim: PatchySimulation, stage: Stage) -> Path:
        return ensemble.folder_path(sim, stage) / TRAJECTORY_FILE
