from __future__ import annotations

import itertools
import os
import subprocess
import tempfile
from pathlib import Path

from typing import Union, Iterable, Any, Generator

import numpy as np
from weakref import ref, ReferenceType

from ipy_oxdna.oxdna_simulation import BuildSimulation, Simulation, write_input_file

from .patchy_scripts import add_standard_patchy_interaction
from .pl.plparticleset import PLParticleSet

from .param_val_types import MDT_CONVERT_KEY, StageInfoParam
from .base_param_val import ParameterValue
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
    _ctxt: ReferenceType  # PatchySimulationEnsemble
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
        self._ctxt = ref(ctxt)
        self._sim_spec = sim
        self._prev_stage = previous_stage
        if previous_stage is not None:
            self._prev_stage._next_stage = self
        self._next_stage = None
        self._box_size = box_size
        self._param_info = paraminfo
        self._param_info.info["steps"] = int(float(self._param_info.info["steps"]))
        self._allow_shortfall = False

        # check that add param info add method is valid
        if isinstance(self._param_info.add_method, FromPolycubeAdder):
            # check that the particle set matches up
            # todo: do this somewhere where i need to run the check fewer times
            try:
                mdt_settings = self.getctxt().sim_get_param(self.spec(), MDT_CONVERT_KEY)
            except NoSuchParamError:
                mdt_settings = None
            # for pc in self._param_info.add_method.iter_polycubes():
                # sim_particle_set: PLParticleSet = self.getctxt().sim_get_particles_set(sim)
                # pc_particle_set: PLParticleSet = polycube_to_pl(pc.polycube_file_path,
                #                                                 mdt_settings,
                #                                                 pad_cubes=dps_sigma * pc.patch_distance_multiplier).particle_types()
                # assert sim_particle_set == pc_particle_set
        self.input_param_dict = {}
        assert self.time_length() > 0, \
                f"Stage cannot have zero (or negative) steps. Stage start time: {self.start_time()}, end time: {self.end_time()}"

    def idx(self) -> int:
        """
        returns index of this stage within staging
        :return: index of this stage
        """
        return self._prev_stage.idx() + 1 if self._prev_stage is not None else 0

    def name(self) -> str:
        """
        returns name of this stage
        """
        return self._param_info.stage_name

    def getctxt(self):
        """
        returns ensemble context
        :returns: a PatchySimulationEnsemble instance
        """
        return self._ctxt()

    def spec(self) -> PatchySimulation:
        """
        :returns: the patchy simulation specification associated with this stage
        """
        return self._sim_spec

    def is_first(self) -> bool:
        """
        :returns: true if this is the first stage, false otherwise
        """
        return self.idx() == 0

    def is_last(self) -> bool:
        """
        :returns: true if this stage has no stage after it, false otherwise
        """
        return self._next_stage is None

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

    def num_particles(self) -> int:
        return self.num_particles_to_add() + (self.get_prev().num_particles() if self.get_prev() is not None else 0)

    def start_time(self) -> int:
        return self._param_info.start_time

    # TODO: VERIFY THAT THIS WORKS CORRECTLY
    def time_length(self) -> int:
        return self._param_info.info["steps"] - self.start_time()

    def end_time(self) -> int:
        return self._param_info.info["steps"]

    def set_tend(self, new_val: int):
        """
        sets the end time of this stage.
        """
        assert self.is_last()
        self._param_info.info["steps"] = new_val


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
        inputs_vals = self.iter_input_dict()
        # todo: nicer version, free up stuff in input file
        del inputs_vals["interaction_potentials"]
        self.sim.input.modify_input(inputs_vals)
        assert self.sim.input.get_conf_file() is not None

        # write external observables file path
        if len(self.getctxt().observables) > 0:
            assert not self.getctxt().server_settings.absolute_paths, "Absolute file paths aren't currently compatible" \
                                                                      " with observiables! Get on it Josh!!!"
            for obs in self.getctxt().observables.values():
                self.sim.add_observable(obs)

        self.sim.input.write_input(production=production)

        assert (self.getctxt().folder_path(self.spec(),
                                           self) / "input").exists(), "Didn't correctly set up input file!"
        assert (self.getctxt().folder_path(self.spec(),
                                           self) / "input.json").exists(), "Didn't correctly set up input file!"

    def iter_input_dict(self) -> dict[str, Any]:
        """
        iterates through the values that belong in the input dict, at least the ones that have been assigned

        """
        inpt = dict()
        for pv in self.getctxt().iter_params(self.spec(), self):
            # only write raw-data params to input file
            # todo: filter better, to use only actual oxdna params
            if type(pv) is ParameterValue:
                inpt[pv.param_name] = pv.param_value
        inpt["steps"] = self.time_length()
        inpt["restart_step_counter"] = 0
        return inpt

    def apply(self, scene: PLPSimulation):
        """
        applies stage operations to patchy scene
        """
        # first, add patchy interactions
        patchy_potentials = self.getctxt().get_interaction_potentials(self.spec())
        for potential in patchy_potentials:
            scene.add_potential(potential)

        # if the scene box has no particles, this is stage 0
        if scene.num_particles() == 0:
            assert self.idx() == 0, "No particles present but nonzero stage!!"
            # set box size
            scene.set_box_size(self.box_size())
            scene.compute_cell_size(n_particles=self.num_particles_to_add())
            scene.apportion_cells()
        # otherwise we need to check if current scene box size is consistant with existing box size
        # at which point things become. tricky
        elif not np.allclose(self.box_size(), scene.box_size()):
            try:
                scene.alter_box_size(self.box_size(), self.getctxt().sim_get_param(self.spec(),
                                                                                   "cluster_bond_energy"))
            except NoSuchParamError:
                # if cluter bond eneregy isn't specified, include all interactions w/ bond ene  rgy
                scene.alter_box_size(self.box_size(), 0)

        if scene.num_particles() > 0 and self.num_particles_to_add() > 0:
            e_start = scene.get_potential_energy()
            assert e_start < 1., f"Scene energy {e_start} too high!!"
        else:
            e_start = 0
        # TODO: compute cell sizes using something other than "pull from rectum"
        assert all(self.box_size()), "Box size hasn't been set!!!"

        # ---------------- add particles -----------------------------------------
        if self._param_info.add_method is None:
            assert len(self.particles_to_add()) == 0, "No add method specified but particles still " \
                                                      "queued to add!"
        elif isinstance(self._param_info.add_method, RandParticleAdder):
            # particles should be added first whether we are are using built in or extern adder
            start_particle_count = scene.num_particles()
            # self.particles_to_add incorporates num_assmblies
            particles = [scene.particle_types().particle(i_type).instantiate(i + start_particle_count)
                         for i, i_type in enumerate(self.particles_to_add())]
            # default is now to use confGenerator.
            if self._param_info.add_method.extern:
                # we need particles to be sane for the stupid jurryrig i do later
                for p in particles:
                    p.set_rotation(np.identity(3))
                    p.set_position(np.array([0, 0, 0]))
                # add particles to scene so we can make top later
                scene.add_particles(particles, strict_check=False) # strict check currently does nothing but might someday?
                assert self.idx() == 0, "Cannot use extern adder on stages after zero!"
                # create temporary directory within the simulation directory
                with tempfile.TemporaryDirectory(dir=self.getctxt().folder_path(self.spec())) as tmpdir:
                    # eventually this should be a context manager
                    w = self.getctxt().writer
                    w.set_directory(tmpdir)
                    files = w.write(scene, conf_file="dummy.dat")
                    inpdct = self.iter_input_dict()
                    inpdct.update(dict(w.get_input_file_data(scene)))
                    inpdct.update(files)
                    write_input_file(Path(tmpdir) / "input", inpdct)

                    # w.write_top(w.get_scene_top(scene), self.getctxt().sim_stage_get_param(self.spec(), self, 'topology'))
                    # cmd = f"{str(Path(self.getctxt().server_settings.oxdna_path) / 'build' /'bin'/ 'confGenerator')} input {self.getctxt().sim_stage_get_param(self.spec(), self, 'density')}"
                    cmd = [str(Path(self.getctxt().server_settings.oxdna_path) / 'build' /'bin'/ 'confGenerator'),
                            "input",
                           f"{self.getctxt().sim_stage_get_param(self.spec(), self, 'density')}"]
                    # run and capture everything
                    result = subprocess.run(cmd, capture_output=True, text=True, cwd=tmpdir)
                    if result.returncode:
                        raise Exception("Failed to generate conf!\n" + result.stderr)

                    # log stdout & sterr
                    for line in result.stdout.splitlines():
                        self.getctxt().get_logger().info(line)
                    for line in result.stderr.splitlines():
                        self.getctxt().get_logger().error(line)
                    gend_conf = w.read_scene(inpdct["topology"], inpdct["conf_file"],
                                             scene.particle_types())
                    scene.clear_particles() # clear particles at the Singularity
                    scene.add_particles(gend_conf.particles(), False)
            else:

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
                                   pad_cubes=pc.patch_distance_multiplier)
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

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the unpicklable weakref
        state['_ctxt'] = None  # or store an id if needed
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._ctxt = None  # or restore from id if needed


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
