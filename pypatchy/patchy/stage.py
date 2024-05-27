from __future__ import annotations

import copy
import json
import math
import os
import re
from pathlib import Path
from typing import Union, Iterable, Any, Generator

import numpy as np

from ipy_oxdna.oxdna_simulation import BuildSimulation, Simulation, Input
from .pl.plpotential import PLPatchyPotential, PLExclVolPotential

from .ensemble_parameter import MDT_CONVERT_KEY, StageInfoParam, ParameterValue
from .particle_adders import RandParticleAdder, FromPolycubeAdder, FromConfAdder
from .pl.plpatchylib import polycube_to_pl
from .simulation_specification import NoSuchParamError
from ..patchy.simulation_specification import PatchySimulation
from .pl.plscene import PLPSimulation
from ..util import get_input_dir, EXTERNAL_OBSERVABLES


class Stage(BuildSimulation):
    """
    Staged-assembly stage class
    This class should be assumed to refer to a specific oxDNA simulation!
    """
    # the name of this stage
    _stage_name: str
    # the step (time) that this stage starts
    # a list of PARTICLE TYPE IDS of particles to add
    # the length of this list should be the number of particles to add in
    # this stage and each item is a TYPE ID of a particle type to add.
    _particles_to_add: list[int]
    _ctxt: Any  # PatchySimulationEnsemble
    _sim_spec: PatchySimulation

    # this is a class member because I do NOT want to deal with multiple inheritance rn
    _param_info: StageInfoParam

    # why
    # input_param_dict: dict
    _prev_stage: Union[Stage, None]

    _allow_shortfall: bool

    def __init__(self,
                 sim: PatchySimulation,
                 previous_stage: Union[Stage, None],
                 ctxt: Any,  # it's PatchySimulationEnsemble but if we tell it that we get a circular import
                 paraminfo: StageInfoParam,
                 particles: list[Union[int, str]],
                 box_size: np.ndarray = np.array((0, 0, 0)),
                 tlen: int = 0,
                 tend: int = 0
                 ):
        super().__init__(Simulation(ctxt.folder_path(sim)) if previous_stage is None
                         else Simulation(previous_stage.sim.sim_dir, ctxt.folder_path(sim) / paraminfo.stage_name))
        self._ctxt = ctxt
        self._sim_spec = sim
        assert tlen or tend, "Specify stage length with "
        self._prev_stage = previous_stage
        if previous_stage is not None:
            self._prev_stage._next_stage = self
        self._next_stage = None
        self._particles_to_add = particles
        self._box_size = box_size
        self._param_info = paraminfo
        if tlen:
            self._stage_time_length = tlen
        else:
            self._stage_time_length = tend - paraminfo.start_time
        self._allow_shortfall = False
        # self.input_param_dict = {}

    def idx(self) -> int:
        return self._prev_stage.idx() + 1 if self._prev_stage is not None else 0

    def name(self) -> str:
        return self._param_info.stage_name

    def getctxt(self):
        return self._ctxt

    def spec(self) -> PatchySimulation:
        """
        Returns: the patchy simulation specification associated with this stage
        """
        return self._sim_spec

    def is_first(self) -> bool:
        """
        Returns: true if this is the first stage, false otherwise
        """
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
        return self._param_info.start_time

    def time_length(self) -> int:
        return self._stage_time_length

    def end_time(self) -> int:
        return self.start_time() + self.time_length()

    def has_var(self, key: str) -> bool:
        return key in self._param_info.info

    def get_var(self, key: str) -> Any:
        return self._param_info.info[key]

    def params(self) -> Generator[ParameterValue, None, None]:
        """
        iterates input file params specific to this stage
        """
        for key in self._param_info.info:
            yield ParameterValue(key, self._param_info.info[key])

    def build_dat_top(self):
        if self.is_first():
            assert self.start_time() == 0, f"Stage {self} has idx 0 but nonzero start time!"

            # generate conf
            scene = PLPSimulation()
            particle_set = self.getctxt().sim_get_particles_set(self.spec())
            # patches will be added automatically
            scene.set_particle_types(particle_set)
            scene.set_temperature(self.getctxt().sim_stage_get_param(self.spec(), self, "T"))
        else:
            self.get_last_conf_name()
            scene: PLPSimulation = self.getctxt().get_scene(self.spec(), self)
        self.apply(scene)
        # grab args required by writer
        reqd_extra_args = {
            a: self.getctxt().sim_get_param(self.spec(), a) for a in self.getctxt().writer.reqd_args()
        }
        assert "conf_file" in reqd_extra_args, "Missing arg for req"
        self.getctxt().writer.set_directory(self.getctxt().folder_path(self.spec(), self))
        self.getctxt().writer.set_abs_paths(self.getctxt().server_settings.absolute_paths)
        # write top, conf, and others
        files = self.getctxt().writer.write(scene,
                                            self,
                                            **reqd_extra_args)

        # update top and dat files in replacer dict
        self.input_param_dict.update(files)
        self.input_param_dict["steps"] = self.end_time()
        self.input_param_dict["trajectory_file"] = self.adjfn(
            self.getctxt().sim_get_param(self.spec(), "trajectory_file"))
        # include input file stuff required by writer
        self.input_param_dict.update(self.getctxt().writer.get_input_file_data(scene, **reqd_extra_args))
        for param in self.getctxt().server_settings.input_file_params:
            if param.param_name not in self.input_param_dict:
                # todo: assert to avoid complex params here
                self.input_param_dict[param.param_name] = param.param_value

    def build_input(self, production=False):
        """
        Builds the stage input file
        """

        for pv in self.getctxt().iter_params(self.spec(), self):
            # only write raw-data params to input file
            # todo: filter better, to use only actual oxdna params
            if type(pv) is ParameterValue:
                self.sim.input[pv.param_name] = pv.param_value

        # # write server config spec
        # for pv in self.getctxt().server_settings.input_file_params:
        #     self.sim.input[pv.param_name] = pv.param_value
        #
        # # write default input file stuff
        #
        # # loop parameters in group
        # for param in self.getctxt().default_param_set:
        #     paramname = param.param_name
        #     # if we've specified this param in a replacer dict
        #     # if no override
        #     if paramname not in self.spec() and paramname not in self.getctxt().const_params:
        #         val = self.getctxt().default_param_set[paramname]
        #     else:
        #         val = self.getctxt().sim_get_param(self.spec(), paramname)
        #     # check paths are absolute if applicable
        #     if self.getctxt().server_settings.absolute_paths:
        #         # approximation for "is this a file?"
        #         if isinstance(val, str) and re.search(r'\.\w+$', val) is not None:
        #             # if path isn't absolute
        #             if not Path(val).is_absolute():
        #                 # prepend folder path
        #                 val = str(self.getctxt().folder_path(self.spec(), self) / val)
        #     self.sim.input[paramname] = val
        #
        # # write more parameters
        # self.sim.input["T"] = self.getctxt().sim_get_param(self.spec(), 'T')
        # try:
        #     self.sim.input["narrow_type"] = self.getctxt().sim_get_param(self.spec(), 'narrow_type')
        # except NoSuchParamError as e:
        #     self.getctxt().get_logger().info(f"No narrow type specified for simulation {self.spec()}.")
        #
        # self.sim.input["steps"] = self.end_time()

        # write external observables file path
        if len(self.getctxt().observables) > 0:
            assert not self.getctxt().server_settings.absolute_paths, "Absolute file paths aren't currently compatible" \
                                                                      " with observiables! Get on it Josh!!!"
            for obs in self.getctxt().observables.values():
                self.sim.add_observable(obs)
            # if EXTERNAL_OBSERVABLES:
            #     self.sim.input["observables_file"] = self.adjfn("observables.json")
            # else:
            #     for i, obsrv in enumerate(self.getctxt().observables.values()):
            #         obsrv.write_input_dict(self.input_param_dict, i)

        # already ran adjfn on input_json_name, ignore stage info
        # with open(self.getctxt().folder_path(self.spec()) / input_json_name, "w+") as f:
        #     json.dump(self.input_param_dict, f)
        self.sim.input.write_input(production=production)

        assert (self.getctxt().folder_path(self.spec(),
                                           self) / "input.json").exists(), "Didn't correctly set up input file!"

    def apply(self, scene: PLPSimulation):
        scene.set_box_size(self.box_size())
        scene.compute_cell_size(n_particles=self.num_particles_to_add())
        scene.apportion_cells()
        # add excluded volume potential

        # # add patchy interaction
        # # TODO: i'm like 99% sure we can ignore patchy interaction for this purpose
        # if self.getctxt().sim_get_param(self.spec(), "use_torsion"):
        #     raise Exception("Torsional patches not yet supported in this confgen! Get on it Josh!")
        # else:
        #     patchy_potential = PLPatchyPotential(
        #         alpha=self.getctxt().sim_get_param(self.spec(), "PATCHY_alpha"),
        #         rmax=0.4 * 1.5  # lorenzo's code, =0.6
        #     )
        #
        # scene.add_potential(patchy_potential)
        scene.add_potential(PLExclVolPotential(
            rmax=0.4 * 1.5,  # from lorenzo's code, =0.6. particle radius added at runtiume
            rstar=2 ** (1 / 6),  # this is the rcut used in lorenzo's code, = 1.122
            b=677.505671539  # from flavio's code
        ))
        # TODO: compute cell sizes using something other than "pull from rectum"
        assert all(self.box_size()), "Box size hasn't been set!!!"

        if self._param_info.add_method is None:
            assert len(self._particles_to_add) == 0, "No add method specified but particles still " \
                                                        "queued to add!"
        elif isinstance(self._param_info.add_method, RandParticleAdder):
            start_particle_count = scene.num_particles()
            particles = [scene.particle_types().particle(i_type).instantiate(i + start_particle_count)
                         for i, i_type in enumerate(self._particles_to_add)]
            scene.add_particle_rand_positions(particles)

        elif isinstance(self._param_info.add_method, FromPolycubeAdder):
            if len(self._param_info.add_method.polycubes) == 1:
                pl = polycube_to_pl(self._param_info.add_method.polycubes[0],
                                    self.getctxt().sim_get_param(self.spec(), MDT_CONVERT_KEY), pad_cubes=0.13)
                scene.add(pl, cubize_box=True)
            else:
                print("WARNING: I HAVE NOT TESTED THIS YET")
                scene.add_conf_clusters([
                    polycube_to_pl(pc,
                                   self.getctxt().sim_get_param(self.spec(), MDT_CONVERT_KEY),
                                   pad_cubes=0.13)
                    for pc in self._param_info.add_method.polycubes
                ])

        elif isinstance(self._param_info.add_method, FromConfAdder):
            raise Exception("If you're seeing this, this feature hasn't been implemented yet although it can't be"
                            "THAT hard really")
            # TODO: write
            # step 1: split the conf to add up by clusters
            # step 2: add clusters using scene.add_conf_clusters
        else:
            raise Exception(f"Invalid add method {type(self._param_info.add_method)}")
        e = scene.get_potential_energy()
        assert e < 1e-4, "Scene energy too high!!"

    def adjfn(self, file_name: str) -> str:
        if self.idx() > 0:
            return self.name() + os.sep + file_name
        else:
            return file_name

    def allow_shortfall(self) -> bool:
        return self._allow_shortfall

    def set_allow_shortfall(self, bNewVal: bool):
        self._allow_shortfall = bNewVal

    def __str__(self) -> str:
        return f"Stage {self.name()} (#{self.idx()})"


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
