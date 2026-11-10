from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Union

from ..pl.plscene import PLPSimulation
from ..simulation_specification import PatchySimulation
from ..stage import Stage


class SimulationEngine(ABC):
    """
    The simulation program a PatchySimulationEnsemble builds, runs and reads its simulations with.

    The ensemble handles everything that doesn't depend on the program: parameter sweeps, stage
    ordering, working out which stage runs next, and submitting slurm jobs. An engine handles the
    files and the program for a single stage of a single simulation.

    Engines hold no state of their own. Every method takes the ensemble as its first argument, so
    an engine pickles along with the ensemble that uses it.
    """

    # the server config "engine" value that selects this engine; also the job type in the slurm log
    name: str

    @abstractmethod
    def build_stage(self,
                    ensemble: Any,  # PatchySimulationEnsemble; not imported, to avoid a circular import
                    sim: PatchySimulation,
                    stage: Stage,
                    clean_build: Union[bool, str] = False):
        """
        Writes everything needed to run a stage: its directory, initial configuration, and input
        files, plus the sbatch script if the ensemble submits each simulation as its own slurm job.
        :param ensemble: the ensemble the simulation belongs to
        :param sim: simulation descriptor
        :param stage: the stage to build
        :param clean_build: False leaves an already-built stage alone; "force" rebuilds it
        """
        pass

    @abstractmethod
    def is_stage_built(self, ensemble: Any, sim: PatchySimulation, stage: Stage) -> bool:
        """
        :return: True if every file build_stage() writes for this stage exists
        """
        pass

    @abstractmethod
    def check_ready_to_run(self, ensemble: Any, sim: PatchySimulation, stage: Stage):
        """
        Raises an exception if a built stage is still missing something it needs to run.
        """
        pass

    @abstractmethod
    def has_output(self, ensemble: Any, sim: PatchySimulation) -> bool:
        """
        :return: True if this simulation has written any output yet, i.e. whether its stages are worth
        checking for progress at all
        """
        pass

    @abstractmethod
    def stage_last_step(self, ensemble: Any, sim: PatchySimulation, stage: Stage) -> int:
        """
        :return: the last step this stage has written a configuration for
        :raises NoStageTrajError: if the stage has no output yet
        :raises StageTrajFileEmptyError: if the stage's output exists but holds no configurations
        """
        pass

    @abstractmethod
    def stage_done(self, ensemble: Any, sim: PatchySimulation, stage: Stage) -> bool:
        """
        :return: True if the stage has run to its end time
        """
        pass

    @abstractmethod
    def read_scene(self,
                   ensemble: Any,
                   sim: PatchySimulation,
                   stage: Stage,
                   step: Union[int, None] = None) -> PLPSimulation:
        """
        Reads a configuration of a stage as a scene, with its energy model set.
        :param ensemble: the ensemble the simulation belongs to
        :param sim: simulation descriptor
        :param stage: stage to read from
        :param step: step to read, or None for the stage's last configuration (or its initial one,
        if the stage hasn't run)
        """
        pass

    @abstractmethod
    def get_energy_model(self, ensemble: Any, sim: PatchySimulation, stage: Union[Stage, None] = None):
        """
        :return: the energy model scenes from this simulation compute their energies with (see
        PLPSimulation.set_energy_model)
        """
        pass

    @abstractmethod
    def run_stage(self, ensemble: Any, sim: PatchySimulation, stage: Stage):
        """
        Runs a built stage in this process's job, blocking until it finishes. Doesn't check whether the
        stage completed; the ensemble does that afterwards with stage_done().
        """
        pass

    @abstractmethod
    def run_batch(self, ensemble: Any, to_run: list[tuple[PatchySimulation, Stage]]):
        """
        Runs built stages as concurrent workers in this process's job, sharing GPUs as the ensemble's
        server settings say. Blocks until all of them finish, and doesn't check whether they completed;
        the ensemble does that afterwards with stage_done().
        :param to_run: (simulation, stage) pairs to run
        """
        pass

    @abstractmethod
    def stage_log_file(self, ensemble: Any, sim: PatchySimulation, stage: Stage) -> Path:
        """
        :return: the file to look at when a stage fails
        """
        pass
