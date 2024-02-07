from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Union, Iterable, Any

import numpy as np

from ipy_oxdna.oxdna_simulation import BuildSimulation, Simulation, Input

from .pl.plpatchylib import to_PL
from .simulation_specification import NoSuchParamError
from ..patchy.simulation_specification import PatchySimulation
from .pl.plscene import PLPSimulation
from ..polycubeutil.polycube_structure import PolycubeStructure
from ..util import get_input_dir, is_write_abs_paths, get_server_config, EXTERNAL_OBSERVABLES, NUM_TEETH_KEY, \
    DENTAL_RADIUS_KEY


class Stage(BuildSimulation):
    """
    Staged-assembly stage class
    This class should be assumed to refer to a specific oxDNA simulation!
    """
    # the name of this stage
    _stage_name: str
    # the step (time) that this stage starts
    _stage_start_time: int
    # a list of PARTICLE TYPE IDS of particles to add
    # the length of this list should be the number of particles to add in
    # this stage and each item is a TYPE ID of a particle type to add.
    _particles_to_add: list[int]
    _add_method: str
    _stage_vars: dict
    _ctxt: Any  # PatchySimulationEnsemble
    _sim_spec: PatchySimulation

    input_param_dict: dict
    _prev_stage: Union[Stage, None]

    def __init__(self,
                 sim: PatchySimulation,
                 previous_stage: Union[Stage, None],
                 ctxt: Any,  # it's PatchySimulationEnsemble but if we tell it that we get a circular import
                 stagename: str,
                 t: int,
                 particles: list[Union[int, str]],
                 box_size: np.ndarray = np.array((0, 0, 0)),
                 add_method: str = "RANDOM",
                 stage_vars: dict = {},
                 tlen: int = 0,
                 tend: int = 0
                 ):
        super().__init__(Simulation(ctxt.folder_path(sim)) if previous_stage is None
                         else Simulation(previous_stage.sim_dir, ctxt.folder_path(sim) / stagename))
        self._ctxt = ctxt
        self._sim_spec = sim
        assert tlen or tend, "Specify stage length with "
        self._prev_stage = previous_stage
        if previous_stage is not None:
            self._prev_stage._next_stage = self
        self._next_stage = None
        self._stage_name = stagename
        self._stage_start_time = t
        self._particles_to_add = particles
        self._box_size = box_size
        self._add_method = add_method
        self._stage_vars = stage_vars
        self.input_param_dict = {}
        if tlen:
            self._stage_time_length = tlen
        else:
            self._stage_time_length = tend - t

    def idx(self) -> int:
        return self._prev_stage.idx() + 1 if self._prev_stage is not None else 0

    def name(self) -> str:
        return self._stage_name

    def getctxt(self):
        return self._ctxt

    def spec(self) -> PatchySimulation:
        return self._sim_spec

    def is_first(self) -> bool:
        return self.idx() == 0

    def get_prev(self) -> Union[None, Stage]:
        return self._prev_stage

    def get_next(self) -> Union[None, Stage]:
        return self._next_stage

    def box_size(self) -> np.ndarray:
        return self._box_size

    def set_box_size(self, box_size: Iterable):
        self._box_size = np.array(box_size)

    def particles_to_add(self) -> list[int]:
        return self._particles_to_add

    def num_particles_to_add(self) -> int:
        return len(self.particles_to_add())

    def start_time(self) -> int:
        return self._stage_start_time

    def time_length(self) -> int:
        return self._stage_time_length

    def end_time(self) -> int:
        return self.start_time() + self.time_length()

    def build_dat_top(self):
        if self.is_first():
            assert self.start_time() == 0, f"Stage {self} has idx 0 but nonzero start time!"

            # generate conf
            scene = PLPSimulation()
            particle_set = to_PL(self.getctxt().particle_set,
                                 self.getctxt().sim_get_param(self.spec(), NUM_TEETH_KEY),
                                 self.getctxt().sim_get_param(self.spec(), DENTAL_RADIUS_KEY))
            # patches will be added automatically
            scene.set_particle_types(particle_set)
        else:
            self.get_last_conf_top()
            scene: PLPSimulation = self.getctxt().get_scene(self.spec(), self)
        self.apply(scene)
        # grab args required by writer
        reqd_extra_args = {
            a: self.getctxt().sim_get_param(self.spec(), a) for a in self.getctxt().writer.reqd_args()
        }
        assert "conf_file" in reqd_extra_args, "Missing arg for req"
        self.getctxt().writer.set_directory(self.getctxt().folder_path(self.spec(), self))
        # write top, conf, and others
        files = self.getctxt().writer.write(scene,
                                            self,
                                            **reqd_extra_args)

        # update top and dat files in replacer dict
        self.input_param_dict.update(files)
        self.input_param_dict["steps"] = self.end_time()
        self.input_param_dict["trajectory_file"] = self.adjfn(
            self.getctxt().sim_get_param(self.spec(), "trajectory_file"))
        self.input_param_dict.update(self.getctxt().writer.get_input_file_data(scene, **reqd_extra_args))

    def build_input(self, production=False):
        input_json_name = self.adjfn("input.json")

        server_config = get_server_config()

        # write server config spec
        for key in server_config["input_file_params"]:
            val = server_config['input_file_params'][key]
            self.input_param_dict[key] = val

        # write default input file stuff

        # loop parameters in group
        for paramname in self.getctxt().default_param_set["input"]:
            # if we've specified this param in a replacer dict
            # if no override
            if paramname not in self.spec() and paramname not in self.getctxt().const_params:
                val = self.getctxt().default_param_set["input"][paramname]
            else:
                val = self.getctxt().sim_get_param(self.spec(), paramname)
            # check paths are absolute if applicable
            if is_write_abs_paths():
                # approximation for "is this a file?"
                if isinstance(val, str) and re.search(r'\.\w+$', val) is not None:
                    # if path isn't absolute
                    if not Path(val).is_absolute():
                        # prepend folder path
                        val = str(self.getctxt().folder_path(self.spec(), self) / val)
            self.input_param_dict[paramname] = val

        # write more parameters
        self.input_param_dict["T"] = self.getctxt().sim_get_param(self.spec(), 'T')
        try:
            self.input_param_dict["narrow_type"] = self.getctxt().sim_get_param(self.spec(), 'narrow_type')
        except NoSuchParamError as e:
            self.getctxt().get_logger().info(f"No narrow type specified for simulation {self.spec()}.")

        # write external observables file path
        if len(self.getctxt().observables) > 0:
            assert not is_write_abs_paths(), "Absolute file paths aren't currently compatible with observiables!" \
                                             " Get on it Josh!!!"
            if EXTERNAL_OBSERVABLES:
                self.input_param_dict["observables_file"] = self.adjfn("observables.json")
            else:
                for i, obsrv in enumerate(self.getctxt().observables.values()):
                    obsrv.write_input_dict(self.input_param_dict, i)

        # already ran adjfn on input_json_name, ignore stage info
        with open(self.getctxt().folder_path(self.spec()) / input_json_name, "w+") as f:
            json.dump(self.input_param_dict, f)

        assert (self.getctxt().folder_path(self.spec(),
                                           self) / "input.json").exists(), "Didn't correctly set up input file!"
        self.sim.input = Input(str(self.getctxt().folder_path(self.spec(), self)))
        self.sim.input.write_input(production=production)

    def apply(self, scene: PLPSimulation):
        scene.set_box_size(self.box_size())
        assert all(self.box_size()), "Box size hasn't been set!!!"
        if self._add_method == "RANDOM":
            particles = [copy.deepcopy(scene.particle_types().particle(i)) for i in self._particles_to_add]
            scene.add_particle_rand_positions(particles, overlap_min_dist=1)
        elif "=" in self._add_method:
            mode, src = self._add_method.split("=")
            if mode == "from_conf":
                raise Exception("If you're seeing this, this feature hasn't been implemented yet")
            elif mode == "from_polycubes":
                with open(get_input_dir() / src, "r") as f:
                    pc = PolycubeStructure(json.load(f))
                    # TODO: the rest of this

    def adjfn(self, file_name: str) -> str:
        if self.idx() > 0:
            return self.name() + os.sep + file_name
        else:
            return file_name


class StagedAssemblyError(Exception):
    _stage: Stage
    _sim: PatchySimulation

    def __init__(self,
                 stage: Stage,
                 sim: PatchySimulation):
        self._stage = stage
        self._sim = sim

    def stage(self) -> Stage:
        return self._stage

    def sim(self) -> PatchySimulation:
        return self._sim


class IncompleteStageError(StagedAssemblyError):
    def __init__(self,
                 stage: Stage,
                 sim: PatchySimulation,
                 last_timestep: int):
        StagedAssemblyError.__init__(self, stage, sim)
        self._last_timestep = last_timestep

    def last_timestep(self) -> int:
        return self._last_timestep

    def __str__(self):
        return f"Stage {self.stage().name()} of simulation {repr(self.sim())} is incomplete! Last timestep was " \
               f"{self._last_timestep} out of {self.stage().start_time()}:{self.stage().end_time()}"


class StageTrajFileError(StagedAssemblyError):
    def __init__(self,
                 stage: Stage,
                 sim: PatchySimulation,
                 traj_file_path: str):
        StagedAssemblyError.__init__(self, stage, sim)
        self._traj_file = traj_file_path

    def traj_file(self):
        return self._traj_file


class NoStageTrajError(StageTrajFileError):
    def __str__(self):
        return f"Stage {self.stage().name()} of simulation {repr(self.sim())} has no traj file. Traj file expected" \
               f"to be located at `{self.traj_file()}`."


class StageTrajFileEmptyError(StageTrajFileError):
    def __str__(self):
        return f"Stage {self.stage().name()} of simulation {repr(self.sim())} has empty traj file. Traj file " \
               f"located at `{self.traj_file()}`."
