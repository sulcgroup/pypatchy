from __future__ import annotations

import datetime
import json
import os
import itertools
import pickle
import sys
import tempfile
from pathlib import Path
from typing import Any, Union

import networkx as nx
import pandas as pd
import subprocess
import re
import logging

from .analysis.analysis_pipeline import AnalysisPipeline, analysis_step_idx
from .analysis_pipeline_step import AnalysisPipelineStep, PipelineDataType
from .patchy_sim_observable import PatchySimObservable, observable_from_file
from ..util import get_param_set, simulation_run_dir, get_server_config, get_log_dir, get_input_dir, all_equal
from .ensemble_parameter import EnsembleParameter, ParameterValue
from .simulation_specification import PatchySimulation
from .plpatchy import PLPatchyParticle, export_interaction_matrix
from .UDtoMDt import convert_multidentate
from ..polycubeutil.polycubesRule import PolycubesRule
from oxDNA_analysis_tools.UTILS.oxview import from_path

EXPORT_NAME_KEY = "export_name"
PARTICLES_KEY = "particles"
DEFAULT_PARAM_SET_KEY = "default_param_set"
CONST_PARAMS_KEY = "const_params"
ENSEMBLE_PARAMS_KEY = "ensemble_params"
OBSERABLES_KEY = "observables"
PARTICLE_TYPE_LVLS_KEY = "particle_type_levels"
NUM_ASSEMBLIES_KEY = "num_assemblies"
DENSITY_KEY = "density"
NUM_TEETH_KEY = "num_teeth"
DENTAL_RADIUS_KEY = "dental_radius"

PATCHY_FILE_FORMAT_KEY = "patchy_format"

SUBMIT_SLURM_PATTERN = r"Submitted batch job (\d+)"

METADATA_FILE_KEY = "sim_metadata_file"


def describe_param_vals(*args) -> str:
    return "_".join([str(v) for v in args])


class PatchySimulationEnsemble:
    """
    Stores data for a group of related simulation
    class was originally written for setup but I've been folding analysis stuff from
    `PatchyResults` and `PatchyRunResult` in as well with the aim of eventually deprecating those
    preferably sooner tather than later
    """

    # --------------- GENERAL MEMBER VARS -------------- #

    # metadata regarding execution and analysis (TBD)
    metadata: dict[Any]

    # list of parameters that will be varied across this ensemble
    # the set of simulations constituting this ensemble is the cartesian
    # product of all possible values of each ensemble param
    # each simulation is defined by a list of ParameterValue objects where each
    # ParameterValue object corresponds to a value in a different EnsembleParameter
    # object in this list
    # TODO: some sort of "skip" list for
    ensemble_params: list[EnsembleParameter]

    # set of cube types that are used in this ensemble
    rule: PolycubesRule

    # logging
    logger: logging.Logger

    observables: dict[str: PatchySimObservable]

    # ------------ SETUP STUFF -------------#

    # simulation parameters which are constant over the entire ensemble
    const_params: dict

    # parameter values to use if not specified in `const_params` or in the
    # simulation specification params
    # load from `spec_files/input_files/[name].json`
    default_param_set: dict[str: Union[dict, Any]]

    # -------------- STUFF SPECIFIC TO ANALYSIS ------------- #

    # each "node" of the analysis pipeline graph is a step in the analysis pipeline
    analysis_pipeline: AnalysisPipeline

    # dict to store loaded analysis data
    analysis_data: dict[tuple[int, str]: PipelineDataType]

    def __init__(self, **kwargs):
        """
        Very flexable constructor
        Options are:
            PatchySimulationEnsemble(cfg_file_name="input_file_name.json")
            PatchySimulationEnsemble(sim_metadata_file="metadata_file.json)
            PatchySimulationEnsemble(export_name="a-simulation-name", particles=[{.....)
        """
        assert "cfg_file_name" in kwargs or METADATA_FILE_KEY in kwargs or "cfg_dict" in kwargs

        self.metadata: dict = {}
        sim_cfg = {}
        if METADATA_FILE_KEY in kwargs or "cfg_file_name" in kwargs:
            # if an exec metadata file was provided
            if METADATA_FILE_KEY in kwargs:
                self.metadata_file = get_input_dir() / kwargs[METADATA_FILE_KEY]
                with open(self.metadata_file) as f:
                    self.metadata.update(json.load(f))
                    sim_cfg = self.metadata["ensemble_config"]

            # if a config file name was provided
            elif "cfg_file_name" in kwargs:
                # if a cfg file name was provided
                cfg_file_name = kwargs["cfg_file_name"]
                with open(get_input_dir() / cfg_file_name) as f:
                    sim_cfg = json.load(f)

        # if no file key was provided, use function arg dict as setup dict
        else:
            sim_cfg = kwargs

        # if a date string was provided
        if "sim_date" in sim_cfg:
            self.sim_init_date = sim_cfg["sim_date"]
            if isinstance(self.sim_init_date, str):
                self.sim_init_date = datetime.datetime.strptime(self.sim_init_date, "%Y-%m-%d")

        else:
            # save current date
            self.sim_init_date: datetime.datetime = datetime.datetime.now()

        datestr = self.sim_init_date.strftime("%Y-%m-%d")

        if METADATA_FILE_KEY not in kwargs:
            self.metadata["setup_date"] = datestr

            self.metadata["ensemble_config"] = sim_cfg
            self.metadata_file = get_input_dir() / f"{sim_cfg[EXPORT_NAME_KEY]}_{datestr}.json"

        # assume standard file format json

        # name of simulation set
        self.export_name: str = sim_cfg[EXPORT_NAME_KEY]

        # configure logging
        self.logger: logging.Logger = logging.getLogger(self.export_name)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.FileHandler(get_log_dir() /
                                                   f"log_{self.export_name}_{self.sim_init_date.strftime('%Y-%m-%d')}"))
        self.logger.addHandler(logging.StreamHandler(sys.stdout))

        # load particles
        if "rule" not in sim_cfg:
            self.rule: PolycubesRule = PolycubesRule(rule_json=sim_cfg[PARTICLES_KEY])
        else:
            if isinstance(sim_cfg["rule"], PolycubesRule):
                self.rule: PolycubesRule = sim_cfg["rule"]
            else:
                self.rule: PolycubesRule = PolycubesRule(rule_str=sim_cfg["rule"])

        # default simulation parameters
        self.default_param_set = get_param_set(
            sim_cfg[DEFAULT_PARAM_SET_KEY] if DEFAULT_PARAM_SET_KEY in sim_cfg else "default")
        self.const_params = sim_cfg[CONST_PARAMS_KEY] if CONST_PARAMS_KEY in sim_cfg else {}
        self.ensemble_params = [EnsembleParameter(*p) for p in sim_cfg[ENSEMBLE_PARAMS_KEY]]

        # observables are optional
        # TODO: integrate oxpy
        self.observables: dict[str: PatchySimObservable] = []

        if OBSERABLES_KEY in sim_cfg:
            for obs_specifier in sim_cfg[OBSERABLES_KEY]:
                # if the name of an observable is provided
                if isinstance(obs_specifier, str):
                    try:
                        obs = observable_from_file(obs_specifier)
                    except FileNotFoundError as e:
                        print(f"No file {obs_specifier}!")
                else:  # assume observable is provided as raw json
                    obs = PatchySimObservable(**obs_specifier)
                self.observables[obs.name] = obs
        # handle potential weird stuff??

        # load analysis pipeline
        self.analysis_pipeline = AnalysisPipeline(self.tld() / "analysis_pipeline.pickle")

        if "analysis_file" in self.metadata:
            file_path = self.metadata["analysis_file"]
            self.analysis_pipeline = self.analysis_pipeline + pickle.load(file_path)
        else:
            self.metadata["analysis_file"] = self.analysis_pipeline.file_path

        # init slurm log dataframe
        self.slurm_log = pd.DataFrame(columns=["slurm_job_id", *[p.param_key for p in self.ensemble_params]])

    # --------------- Accessors and Mutators -------------------------- #
    def get_simulation(self, *args: list[ParameterValue]) -> PatchySimulation:
        """
        TODO: idk
        given a list of parameter values, returns a PatchySimulation object
        """
        return PatchySimulation(args)

    def long_name(self) -> str:
        return f"{self.export_name}_{self.sim_init_date.strftime('%Y-%m-%d')}"

    def is_do_analysis_parallel(self) -> bool:
        return self.metadata["parallel"]

    def get_sim_set_root(self) -> Path:
        return simulation_run_dir() / self.export_name

    """
    num_particle_types should be constant across all simulations. some particle
    types may have 0 instances but hopefully oxDNA should tolerate this
    """

    def num_particle_types(self) -> int:
        return len(self.rule)

    """
    Returns the value of the parameter specified by paramname in the 
    simulation specification sim
    The program first checks if the parameter exists
    """

    def sim_get_param(self, sim: PatchySimulation,
                      paramname: str) -> Any:
        if paramname in sim:
            return sim[paramname]
        # use default
        assert paramname in self.const_params
        return self.const_params[paramname]

    def get_sim_particle_count(self, sim: PatchySimulation,
                               particle_idx: int) -> int:
        particle_lvl = 1  # mainly to shut up my IDE
        # grab particle name
        particle_name = self.rule.particle(particle_idx).name()
        if PARTICLE_TYPE_LVLS_KEY in self.const_params and particle_name in self.const_params[PARTICLE_TYPE_LVLS_KEY]:
            particle_lvl = self.const_params[PARTICLE_TYPE_LVLS_KEY][particle_name]
        if PARTICLE_TYPE_LVLS_KEY in sim:
            spec = self.sim_get_param(sim, PARTICLE_TYPE_LVLS_KEY)
            if particle_name in spec:
                particle_lvl = spec[particle_name]
        return particle_lvl * self.sim_get_param(sim, NUM_ASSEMBLIES_KEY)

    def get_sim_total_num_particles(self, sim: PatchySimulation) -> int:
        return sum([self.get_sim_particle_count(sim, i) for i in range(self.num_particle_types())])

    def num_patch_types(self, sim: PatchySimulation) -> int:
        return self.rule.numPatches() * self.sim_get_param(sim, NUM_TEETH_KEY)

    def paths_list(self) -> list[str]:
        return [
            os.sep.join(combo)
            for combo in itertools.product(*[
                p.dir_names() for p in self.ensemble_params
            ])
        ]

    """
    Returns a list of lists of tuples,
    """

    def ensemble(self) -> list[PatchySimulation]:
        return [PatchySimulation(e) for e in itertools.product(*self.ensemble_params)]

    def tld(self) -> Path:
        return simulation_run_dir() / self.long_name()

    def folder_path(self, sim: PatchySimulation) -> Path:
        return self.tld() / sim.get_folder_path()

    def get_pipeline_step(self, step: Union[int, AnalysisPipelineStep]) -> AnalysisPipelineStep:
        return self.analysis_pipeline.get_pipeline_step(step)

    # ------------------------ Status-Type Stuff --------------------------------#

    def show_pipeline_graph(self):
        nx.draw(self.analysis_pipeline)

    def show_last_conf(self, sim: PatchySimulation):
        from_path(self.folder_path(sim) / "last_conf.dat",
                  self.folder_path(sim) / "init.top",
                  self.folder_path(sim) / "particles.txt",
                  self.folder_path(sim) / "patches.txt")

    # ----------------------- Setup Methods ----------------------------------- #
    def do_setup(self):
        self.logger.info("Setting up folder / file structure...")
        for sim in self.ensemble():
            self.logger.info(f"Setting up folder / file structure for {repr(sim)}...")
            # create nessecary folders
            if not os.path.isdir(self.folder_path(sim)):
                self.logger.info(f"Creating folder {self.folder_path(sim)}")
                Path(self.folder_path(sim)).mkdir(parents=True)
            else:
                self.logger.info(f"Folder {self.folder_path(sim)} already exists. Continuing...")
            # write input file
            self.logger.info("Writing input files...")
            self.write_input_file(sim)
            # write requisite top, patches, particles files
            self.logger.info("Writing .top, .txt, etc. files...")
            self.write_sim_top_particles_patches(sim)
            # write observables.json if applicble
            self.logger.info("Writing observable json, as nessecary...")
            self.write_sim_observables(sim)
            # write .sh script
            self.logger.info("Writing sbatch scripts...")
            self.write_confgen_script(sim)
            self.write_run_script(sim)



    def write_confgen_script(self, sim: PatchySimulation):
        with open(self.get_run_confgen_sh(sim), "w+") as confgen_file:
            self.write_sbatch_params(sim, confgen_file)
            confgen_file.write(
                f"{get_server_config()['oxdna_path']}/build/bin/confGenerator input {self.sim_get_param(sim, 'density')}\n")
        self.bash_exec(f"chmod u+x {self.get_run_confgen_sh(sim)}")

    def write_run_script(self, sim: PatchySimulation, input_file="input"):
        server_config = get_server_config()

        # write slurm script
        with open(self.folder_path(sim) / "slurm_script.sh", "w+") as slurm_file:
            # bash header

            self.write_sbatch_params(sim, slurm_file)

            # skip confGenerator call because we will invoke it directly later
            slurm_file.write(f"{server_config['oxdna_path']}/build/bin/oxDNA {input_file}\n")

        self.bash_exec(f"chmod u+x {self.folder_path(sim)}/slurm_script.sh")


    def write_sbatch_params(self, sim, slurm_file):
        server_config = get_server_config()

        slurm_file.write("#!/bin/bash\n")

        # slurm flags
        for flag_key in server_config["slurm_bash_flags"]:
            if len(flag_key) > 1:
                slurm_file.write(f"#SBATCH --{flag_key}=\"{server_config['slurm_bash_flags'][flag_key]}\"\n")
            else:
                slurm_file.write(f"#SBATCH -{flag_key} {server_config['slurm_bash_flags'][flag_key]}\n")
        run_oxdna_counter = 1
        slurm_file.write(f"#SBATCH --job-name=\"{self.export_name}\"\n")
        slurm_file.write(f"#SBATCH -o run{run_oxdna_counter}_%j.out\n")
        slurm_file.write(f"#SBATCH -e run{run_oxdna_counter}_%j.err\n")

        # slurm includes ("module load xyz" and the like)
        for line in server_config["slurm_includes"]:
            slurm_file.write(line + "\n")

    def write_input_file(self,
                         sim: PatchySimulation,
                         file_name: str = "input",
                         replacer_dict=None):
        """
        Writes an input file
        """
        if replacer_dict is None:
            replacer_dict = {}
        server_config = get_server_config()

        # create input file
        with open(self.folder_path(sim) / file_name, 'w+') as inputfile:
            # write server config spec
            inputfile.write("#" * 32 + "\n")
            inputfile.write(" SERVER PARAMETERS ".center(32, '#') + "\n")
            inputfile.write("#" * 32 + "\n")
            for key in server_config["input_file_params"]:
                if key in replacer_dict:
                    val = replacer_dict[key]
                else:
                    val = server_config['input_file_params'][key]
                inputfile.write(f"{key} = {val}\n")

            # newline
            inputfile.write("\n")

            # write default input file stuff
            for paramgroup_key in self.default_param_set['input']:
                paramgroup = self.default_param_set['input'][paramgroup_key]
                inputfile.write("#" * 32 + "\n")
                inputfile.write(f" {paramgroup_key} ".center(32, "#") + "\n")
                inputfile.write("#" * 32 + "\n\n")

                # loop parameters
                for paramname in paramgroup:
                    # if no override
                    if paramname not in sim and paramname not in self.const_params:
                        val = paramgroup[paramname]
                    elif key in replacer_dict:
                        val = replacer_dict[key]
                    else:
                        val = self.sim_get_param(sim, paramname)
                    inputfile.write(f"{key} = {val}\n")
            # write things specific to rule
            # if josh_flavio or josh_lorenzo
            if server_config[PATCHY_FILE_FORMAT_KEY].find("josh") > -1:
                patch_file_info = [
                    ("patchy_file", "patches.txt"),
                    ("particles_file", "particles.txt"),
                    ("particle_types_N", self.num_particle_types()),
                    ("patch_types_N", self.num_patch_types(sim))
                ]
                for key, val in patch_file_info:
                    if key in replacer_dict:
                        val = replacer_dict[key]
                    inputfile.write(f"{key} = {val}\n")
            elif server_config[PATCHY_FILE_FORMAT_KEY] == "lorenzo":
                key = "DPS_interaction_matrix_file"
                val = "interactions.txt"
                if key in replacer_dict:
                    val = replacer_dict[key]
                inputfile.write(f"{key} = {val}\n")
            else:
                # todo: throw exception
                pass

            # write more parameters
            ensemble_var_names = sim.var_names()

            for param in ["T", "narrow_type"]:
                inputfile.write(f"{param} = {self.sim_get_param(sim, param)}" + "\n")

            # write external observables file path
            if len(self.observables) > 0:
                inputfile.write(f"observables_file = observables.json" + "\n")

    def write_sim_top_particles_patches(self, sim: PatchySimulation):
        server_config = get_server_config()

        # write top and particles/patches spec files
        # first convert particle json into PLPatchy objects (cf plpatchy.py)
        particles = []
        for particle in self.rule.particles():
            particle_patches = [patch.to_pl_patch() for patch in particle.patches()]
            particle = PLPatchyParticle(type_id=particle.get_id(), index_=particle.get_id())
            particle.set_patches(particle_patches)

            particles.append(particle)

        if self.sim_get_param(sim, NUM_TEETH_KEY) > 1:
            particles, patches = convert_multidentate(particles,
                                                      self.sim_get_param(sim, DENTAL_RADIUS_KEY),
                                                      self.sim_get_param(sim, NUM_TEETH_KEY))
        else:
            patches = particle_patches
        # do any/all valid conversions
        # either josh_lorenzo or josh_flavio
        if server_config[PATCHY_FILE_FORMAT_KEY].find("josh") > -1:
            # write top file
            with open(self.folder_path(sim) / "init.top", "w+") as top_file:
                # first line of file
                top_file.write(f"{self.get_sim_total_num_particles(sim)} {len(particles)}\n")
                top_file.write(" ".join([
                    f"{i} " * self.get_sim_particle_count(sim, i) for i in range(len(particles))
                ]))
            # write patches.txt and particles.txt
            with open(self.folder_path(sim) / "patches.txt", "w+") as patches_file, open(
                    self.folder_path(sim) / "particles.txt", "w+") as particles_file:
                for particle_patchy, cube_type in zip(particles, self.rule.particles()):
                    # handle writing particles file
                    for i, patch_obj in enumerate(particle_patchy.patches()):
                        # we have to be VERY careful here with indexing to account for multidentate simulations
                        # adjust for patch multiplier from multidentate
                        polycube_patch_idx = int(i / self.sim_get_param(sim, NUM_TEETH_KEY))

                        extradict = {}
                        # if this is the "classic" format
                        if server_config[PATCHY_FILE_FORMAT_KEY] == "josh_flavio":
                            allo_conditional = cube_type.patch_conditional(
                                cube_type.get_patch_by_idx(polycube_patch_idx))
                            # allosteric conditional should be "true" for non-allosterically-controlled patches
                            extradict = {"allostery_conditional": allo_conditional if allo_conditional else "true"}
                        else:  # josh/lorenzo
                            # adjust for patch multiplier from multidentate
                            state_var = cube_type.get_patch_by_diridx(polycube_patch_idx).state_var()
                            activation_var = cube_type.get_patch_by_diridx(polycube_patch_idx).activation_var()
                            extradict = {
                                "state_var": state_var,
                                "activation_var": activation_var
                            }
                        patches_file.write(patch_obj.save_to_string(extradict))

                    if server_config[PATCHY_FILE_FORMAT_KEY] == "josh_flavio":
                        particles_file.write(particle_patchy.save_type_to_string())
                    else:  # josh/lorenzo
                        particles_file.write(
                            particle_patchy.save_type_to_string({"state_size": cube_type.state_size()}))

        else:  # lorenzian
            with open(self.folder_path(sim) / "init.top", "w+") as top_file:
                top_file.write(f"{self.get_sim_total_num_particles(sim)} {len(particles)}\n")
                # export_to_lorenzian_patchy_str also writes patches.dat file
                top_file.writelines([
                    particle.export_to_lorenzian_patchy_str(self.get_sim_particle_count(sim, particle),
                                                            self.folder_path(sim))
                    + "\n"
                    for particle in particles])
            export_interaction_matrix(patches)

    def write_sim_observables(self, sim: PatchySimulation):
        if len(self.observables) > 0:
            with open(self.folder_path(sim) / "observables.json", "w+") as f:
                json.dump({f"data_output_{i + 1}": obs.to_dict() for i, obs in enumerate(self.observables.values())}, f)

    def write_continue_files(self, sim: PatchySimulation, counter: int = 2):
        """
        writes input file and shell script to continue running the simulation after
        completion of first oxDNA execution
        """

        # construct an input file for the continuation execution
        # using previous conf as starting conf, adding new traj, writing new last_conf
        self.write_input_file(sim,
                              file_name=f"input_{counter}",
                              replacer_dict={
                                  "trajectory_file": f"trajectory_{counter}.dat",
                                  "conf_file": "last_conf.dat" if counter == 2 else f"last_conf_{counter - 1}.dat",
                                  "lastconf_file": f"last_conf_{counter}.dat"
                              })
        self.write_run_script(sim, input_file=f"input_{counter}")

    def exec_continue(self, sim: PatchySimulation, counter: int = 2):
        pass

    def gen_confs(self):
        for sim in self.ensemble():
            self.run_confgen(sim)

    def dump_slurm_log_file(self):
        self.slurm_log.to_csv(f"{self.tld()}/slurm_log.csv")

    def start_simulations(self):
        for sim in self.ensemble():
            self.start_simulation(sim)

    def start_simulation(self,
                         sim: PatchySimulation,
                         script_name: str = "slurm_script.sh"):
        command = f"sbatch --chdir={self.folder_path(sim)}"

        if not os.path.isfile(self.get_conf_file(sim)):
            confgen_slurm_jobid = self.run_confgen(sim)
            command += f" --dependency=afterok:{confgen_slurm_jobid}"
        command += f" {script_name}"
        submit_txt = self.bash_exec(command)

        jobid = int(re.search(SUBMIT_SLURM_PATTERN, submit_txt).group(1))
        self.slurm_log.loc[len(self.slurm_log.index)] = {
            "slurm_job_id": jobid,
            **{
                key: value for key, value in sim
            }
        }
        os.chdir(self.tld())

    def get_run_oxdna_sh(self, sim: PatchySimulation) -> Path:
        return self.folder_path(sim) / "slurm_script.sh"

    def get_run_confgen_sh(self, sim: PatchySimulation) -> Path:
        return self.folder_path(sim) / "gen_conf.sh"

    def run_confgen(self, sim: PatchySimulation) -> int:
        response = self.bash_exec(f"sbatch --chdir {self.folder_path()} {self.folder_path(sim)}/gen_conf.sh")
        jobid = int(re.search(SUBMIT_SLURM_PATTERN, response).group(1))
        return jobid

    def get_conf_file(self, sim: PatchySimulation) -> Path:
        return self.folder_path(sim) / "init.conf"

    # ------------- ANALYSIS FUNCTIONS --------------------- #

    def has_data_for_analysis_step(self,
                                   step: Union[int, AnalysisPipelineStep],
                                   sim: Union[tuple[ParameterValue], PatchySimulation],
                                   time_steps: range = None) -> bool:
        """
        Checks if the system has data for the given analysis step and simulation within the given
        timepoints
        Parameters:
            :param step
            :param sim
            :param time_steps
        """
        sim_key = str(sim) if isinstance(sim, PatchySimulation) else describe_param_vals(*sim)
        if not self.get_cache_file(step, sim).exists():
            return False
        else:
            # all read types are plain-text except
            read_type = "rb" if step.get_input_data_type() == PipelineDataType.PIPELINE_DATATYPE_GRAPH else "r"
            with open(self.get_cache_file(step, sim), read_type) as f:
                self.analysis_data[(analysis_step_idx(step), sim_key)] = step.load_cached_files(f)
                return step.data_matches_trange(self.analysis_data[(analysis_step_idx(step), sim_key)],
                                                time_steps)

    def get_data(self,
                 step: Union[int, AnalysisPipelineStep],
                 sim: Union[tuple[ParameterValue], PatchySimulation],
                 time_steps: range = None) -> PipelineDataType:
        """
        Returns data for a step, doing any/all required calculations
        Parameters:
            :param step
            :param sim
            :param time_steps
        """
        step = self.get_pipeline_step(step)

        # compute data for previous steps
        data_in = [
            self.get_data(prev_step,
                          sim,
                          time_steps)
            for prev_step in step.previous_steps
        ]

        if self.is_do_analysis_parallel() and step.can_parallelize():
            server_cfg = get_server_config()
            data_sources = [
                self.get_cache_file(data_source, sim)
                for data_source in step.previous_steps
            ]
            with tempfile.TemporaryDirectory() as temp_dir:
                jobid = step.exec_step_slurm(temp_dir,
                                             server_cfg["slurm_bash_flags"],
                                             server_cfg["slurm_includes"],
                                             data_sources,
                                             self.get_cache_file(step, sim))
                read_type = "rb" if step.get_input_data_type() == PipelineDataType.PIPELINE_DATATYPE_GRAPH else "r"

                with open(self.get_cache_file(step, sim), read_type) as f:
                    data = step.load_cached_files(f)

        else:
            # TODO: make sure this can handle the amount of args we're feeding in here
            data = step.exec(*data_in)

        sim_key = str(sim) if isinstance(sim, PatchySimulation) else describe_param_vals(*sim)
        self.analysis_data[(step.idx, sim_key)] = data
        return data

    def get_cache_file(self,
                       step: Union[int, AnalysisPipelineStep],
                       sim: Union[tuple[ParameterValue], PatchySimulation]) -> Path:
        step = self.get_pipeline_step(step)
        if isinstance(sim, PatchySimulation):
            return self.folder_path(sim) / step.get_cache_file_name()
        else:
            return self.tld() / describe_param_vals(*sim) / step.get_cache_file_name()

    def bash_exec(self, command: str):
        self.logger.info(f">`{command}`")
        response = subprocess.run(command, shell=True,
                                  capture_output=True, text=True, check=False)
        # response = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT, check=False,
        # universal_newlines=True)
        self.logger.info(f"`{response.stdout}`")
        return response.stdout


def ensemble_from_export_settings(export_settings_file_path: Union[Path, str],
                                  targets_file_path: Union[Path, str]) -> PatchySimulationEnsemble:
    """
    Constructs a PatchySimulationEnsemble object from the `patchy_export_settings.json` format
    """
    with open(export_settings_file_path, "r") as f:
        export_setup_dict = json.load(f)

    # Grab list of narrow types
    narrow_types = export_setup_dict['narrow_types']
    # grab list of temperatures
    temperatures = export_setup_dict['temperatures']

    rule = PolycubesRule(rule_str=export_setup_dict["rule_json"])

    # make sure that all export groups have the same set of narrow types, duplicates,
    # densities, and temperatures
    assert all([
        all_equal([
            export_group[key]
            for export_group in export_setup_dict['export_groups']
        ])
        for key in ["temperatures", "num_duplicates", "particle_density", "narrow_types"]
    ])

    particle_type_level_groups_list = [
        {
            "name": export_group["exportGroupName"],
            "value": {
                "particle_type_levels": {
                    p.name(): export_group["particle_type_levels"]
                    for p in rule.particles()
                }
            }
        }
        for export_group in export_setup_dict['export_groups']
    ]

    cfg_dict = {
        EXPORT_NAME_KEY: export_setup_dict['export_name'],
        PARTICLES_KEY: rule,
        DEFAULT_PARAM_SET_KEY: "default.json",
        OBSERABLES_KEY: ["plclustertopology"],
        CONST_PARAMS_KEY: {
            DENSITY_KEY: export_setup_dict["particle_density"] if "particle_density" else 0.1,
            DENTAL_RADIUS_KEY: 0,
            NUM_TEETH_KEY: 1,
            "torsion": True
        },
        ENSEMBLE_PARAMS_KEY: [
            (
                "T",
                temperatures
            ),
            (
                "narrow_type",
                narrow_types
            ),
            (
                "type_level_group",
                particle_type_level_groups_list
            ),
            (
                "duplicate",

            )
        ],
    }
    return PatchySimulationEnsemble(cfg_dict=cfg_dict)
