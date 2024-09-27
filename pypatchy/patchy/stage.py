from __future__ import annotations

import itertools
import os

from typing import Union, Iterable, Any, Generator

import numpy as np

from ipy_oxdna.oxdna_simulation import BuildSimulation, Simulation

from .patchy_scripts import add_standard_patchy_interaction
from .pl.plparticleset import PLParticleSet
from .pl.plpotential import PLFRPatchyPotential, PLFRExclVolPotential

from .ensemble_parameter import MDT_CONVERT_KEY, StageInfoParam, ParameterValue
from .particle_adders import RandParticleAdder, FromPolycubeAdder, FromConfAdder
from .pl.plpatchylib import polycube_to_pl
from ..patchy.simulation_specification import PatchySimulation, NoSuchParamError
from .pl.plscene import PLPSimulation

class Stage(BuildSimulation):
    """
    Staged-assembly stage class
    This class should be assumed to refer to a specific oxDNA simulation!
    """
    # the name of this stage
    _stage_name: str
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
                 box_size: np.ndarray = np.array((0, 0, 0)),
                 ):
        super().__init__(Simulation(ctxt.folder_path(sim)) if previous_stage is None
                         else Simulation(previous_stage.sim.sim_dir, ctxt.folder_path(sim) / paraminfo.stage_name))
        self._ctxt = ctxt
        self._sim_spec = sim
        self._prev_stage = previous_stage
        if previous_stage is not None:
            self._prev_stage._next_stage = self
        self._next_stage = None
        self._box_size = box_size
        self._param_info = paraminfo
        self._allow_shortfall = False

        dps_sigma = self.getctxt().sim_get_param(self.spec(), "DPS_sigma_ss") # todo: back-compatibility w/ flavio

        # check that add param info add method is valid
        if isinstance(self._param_info.add_method, FromPolycubeAdder):
            # check that the particle set matches up
            # todo: do this somewhere where i need to run the check fewer times
            try:
                mdt_settings = self.getctxt().sim_get_param(self.spec(), MDT_CONVERT_KEY)
            except NoSuchParamError:
                mdt_settings = None
            for pc in self._param_info.add_method.iter_polycubes():
                sim_particle_set: PLParticleSet = self.getctxt().sim_get_particles_set(sim)
                pc_particle_set: PLParticleSet = polycube_to_pl(pc.polycube_file_path,
                               mdt_settings,
                               pad_cubes=dps_sigma * pc.patch_distance_multiplier).particle_types()
                assert sim_particle_set == pc_particle_set
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
        if self._param_info.add_method is None:
            return []
        else:
            num_assemblies = self.getctxt().sim_stage_get_param(self.spec(), self, "num_assemblies")

            return list(itertools.chain.from_iterable(itertools.chain.from_iterable([
                [[key] * count * num_assemblies]
                for key, count in self._param_info.add_method.get_particle_counts().items()
            ])))

    def num_particles_to_add(self) -> int:
        return len(self.particles_to_add())

    def start_time(self) -> int:
        return self._param_info.start_time

    def time_length(self) -> int:
        return self._param_info.info["steps"] - self.start_time()

    def end_time(self) -> int:
        return self._param_info.info["steps"]

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
        self.input_param_dict["steps"] = self.time_length()
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
        self.sim.input["steps"] = self.time_length()
        self.sim.input["restart_step_counter"] = 0
        assert self.sim.input.get_conf_file() is not None

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
                                           self) / "input").exists(), "Didn't correctly set up input file!"
        assert (self.getctxt().folder_path(self.spec(),
                                           self) / "input.json").exists(), "Didn't correctly set up input file!"

    def apply(self, scene: PLPSimulation):
        if scene.num_particles() > 0:
            assert np.allclose(self.box_size(), scene.box_size())
        else:
            # set box size
            scene.set_box_size(self.box_size())
            scene.compute_cell_size(n_particles=self.num_particles_to_add())
            scene.apportion_cells()
        # add excluded volume potential
        dps_sigma = self.getctxt().sim_get_param(self.spec(), "DPS_sigma_ss") # todo: back-compatibility w/ flavio
        add_standard_patchy_interaction(scene,
                                        sigma=dps_sigma)

        # add patchy interaction
        # unfortunately i haven't implemented the swap interaction here yet
        # and even if I had it would have its own issues bc it's a state function
        # so while ideally the computed energy should always be less than zero in practice it will sometimes be low
        # but positive
        if scene.num_particles() > 0:
            e_start = scene.get_potential_energy()
            assert e_start < 1., f"Scene energy {e_start} too high!!"
        else:
            e_start = 0
        # TODO: compute cell sizes using something other than "pull from rectum"
        assert all(self.box_size()), "Box size hasn't been set!!!"

        if self._param_info.add_method is None:
            assert len(self.particles_to_add()) == 0, "No add method specified but particles still " \
                                                        "queued to add!"
        elif isinstance(self._param_info.add_method, RandParticleAdder):
            start_particle_count = scene.num_particles()
            # self.particles_to_add incorporates num_assmblies
            particles = [scene.particle_types().particle(i_type).instantiate(i + start_particle_count)
                         for i, i_type in enumerate(self.particles_to_add())]
            scene.add_particle_rand_positions(particles)

        # TODO: merge FromPolycubeAdder into FromConfAdder
        elif isinstance(self._param_info.add_method, FromPolycubeAdder):
            # try to get multidentate convert settings
            try:
                mdt_settings = self.getctxt().sim_get_param(self.spec(), MDT_CONVERT_KEY)
            except NoSuchParamError:
                mdt_settings = None
            # add polycubes
            # self._param_info.add_method.iter_polycubes() does NOT incorporate num_assemblies
            # so be explicit about that!
            for _ in range(self.getctxt().sim_get_param(self.spec(), "num_assemblies")):
                scene.add_conf_clusters([
                    polycube_to_pl(pc.polycube_file_path,
                                   mdt_settings,
                                   pad_cubes=dps_sigma * pc.patch_distance_multiplier)
                    for pc in self._param_info.add_method.iter_polycubes()])
        elif isinstance(self._param_info.add_method, FromConfAdder):
            raise Exception("If you're seeing this, this feature hasn't been implemented yet although it can't be"
                            "THAT hard really")
            # TODO: write
            # step 1: split the conf to add up by clusters
            # step 2: add clusters using scene.add_conf_clusters
        else:
            raise Exception(f"Invalid add method {type(self._param_info.add_method)}")
        # e = scene.get_potential_energy()
        # take starting energy into consideration
        # assert e < 0 or (e - e_start) < 1e-4, "Scene energy too high!!"

    def adjfn(self, file_name: str) -> str:
        if self.idx() > 0:
            return self.name() + os.sep + file_name
        else:
            return file_name

    def allow_shortfall(self) -> bool:
        return self._param_info.allow_shortfall

    def set_allow_shortfall(self, bNewVal: bool):
        self._param_info.allow_shortfall = bNewVal

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
