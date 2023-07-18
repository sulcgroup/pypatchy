from __future__ import annotations

import datetime
import glob
import json
import multiprocessing
import os
import itertools
import pickle
import sys
import tempfile
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Union

import networkx as nx
import pandas as pd
import subprocess
import re
import logging

from matplotlib import pyplot as plt

from oxDNA_analysis_tools.UTILS.oxview import from_path
from oxDNA_analysis_tools.file_info import file_info

from .analysis.analysis_pipeline import AnalysisPipeline
from .analysis_pipeline_step import AnalysisPipelineStep, PipelineData, AggregateAnalysisPipelineStep, \
    AnalysisPipelineHead, PipelineStepDescriptor
from .patchy_sim_observable import PatchySimObservable, observable_from_file
from ..util import get_param_set, simulation_run_dir, get_server_config, get_log_dir, get_input_dir, all_equal, \
    get_local_dir
from .ensemble_parameter import EnsembleParameter, ParameterValue
from .simulation_specification import PatchySimulation
from .plpatchy import PLPatchyParticle, export_interaction_matrix
from .UDtoMDt import convert_multidentate
from ..polycubeutil.polycubesRule import PolycubesRule

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

METADATA_FILE_KEY = "sim_metadata_file"
LAST_CONTINUE_COUNT_KEY = "continue_count"

SUBMIT_SLURM_PATTERN = r"Submitted batch job (\d+)"

# for forwards compatibility in case I ever get this working
EXTERNAL_OBSERVABLES = False

def describe_param_vals(*args) -> str:
    return "_".join([str(v) for v in args])


PatchySimDescriptor = Union[tuple[ParameterValue, ...],
                            PatchySimulation,
                            list[Union[tuple[ParameterValue, ...], PatchySimulation],
                            ]]


def get_descriptor_key(sim: PatchySimDescriptor):
    return sim if isinstance(sim, str) else str(sim) if isinstance(sim, PatchySimulation) else describe_param_vals(*sim)


def print_help():
    pass


def list_simulation_ensembles():
    print("Simulations:")
    sim_paths = [ensemble_dir
                 for ensemble_dir in simulation_run_dir().glob("*_*-*-*")
                 if ensemble_dir.is_dir() and re.match(r"[\w\d_]*_\d{4}-\d{2}-\d{2}", ensemble_dir.name)]
    for file in get_input_dir().glob("*.json"):
        try:
            with open(file, "r") as f:
                sim_json = json.load(f)
                if EXPORT_NAME_KEY in sim_json:
                    sim_name = sim_json[EXPORT_NAME_KEY]
                    print(f"\tEnsemble spec `{sim_name}` specified in file `{file.name}`:")
                    for sim_path in sim_paths:
                        if sim_path.name.startswith(sim_name):
                            print(f"\t\t{sim_path}")
        except JSONDecodeError as e:
            print(f"\tJSON file `{file.name} is malformed. Skipping...")


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
    analysis_data: dict[tuple[str, str]: PipelineData]

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
        sim_cfg = kwargs
        if METADATA_FILE_KEY in kwargs or "cfg_file_name" in kwargs:
            # if an execution metadata file was provided
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
                    sim_cfg.update(json.load(f))

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

        # whether it was just set from kwaargs or gen'd from today, make init date str
        # to identify this ensemble
        datestr = self.sim_init_date.strftime("%Y-%m-%d")

        # if no metadata file was provided in the keyword arguements
        if METADATA_FILE_KEY not in kwargs:
            self.metadata["setup_date"] = datestr
            self.metadata["ensemble_config"] = sim_cfg
            self.metadata_file = get_input_dir() / f"{sim_cfg[EXPORT_NAME_KEY]}_{datestr}_metadata.json"
            # if a metadata file exists at the default path
            if self.metadata_file.exists():
                # update metadata dict from file
                with open(self.metadata_file, "r") as f:
                    self.metadata.update(json.load(f))

        # name of simulation set
        self.export_name: str = sim_cfg[EXPORT_NAME_KEY]

        # configure logging
        logger: logging.Logger = logging.getLogger(self.export_name)
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.FileHandler(get_log_dir() /
                                              f"log_{self.export_name}_{self.sim_init_date.strftime('%Y-%m-%d')}.log"))
        logger.addHandler(logging.StreamHandler(sys.stdout))

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
        self.observables: dict[str: PatchySimObservable] = {}

        if OBSERABLES_KEY in sim_cfg:
            for obs_name in sim_cfg[OBSERABLES_KEY]:
                self.observables[obs_name] = observable_from_file(obs_name)

        # handle potential weird stuff??

        # load analysis pipeline
        self.analysis_pipeline = AnalysisPipeline()

        # if the metadata specifies a pickle file of a stored analywsis pathway, use it
        if "analysis_file" in self.metadata:
            file_path = Path(self.metadata["analysis_file"])
            if file_path.exists():
                with open(file_path, "rb") as f:
                    self.analysis_pipeline = self.analysis_pipeline + pickle.load(f)
        else:  # if not, use a default one and cache it
            file_path = self.tld() / "analysis_pipeline.pickle"
            self.metadata["analysis_file"] = str(file_path)

        # construct analysis data dict in case we need it
        self.analysis_data = dict()

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

    def get_logger(self) -> logging.Logger:
        return logging.getLogger(self.export_name)

    def is_do_analysis_parallel(self) -> bool:
        return "parallel" in self.metadata and self.metadata["parallel"]

    def n_processes(self):
        return self.metadata["parallel"]

    def set_metadata_attr(self, key, val):
        """
        Sets a metadata value, and
        saves the metadata file if a change has been made
        """
        oldVal = self.metadata[val]
        self.metadata[key] = val
        if val != oldVal:
            self.dump_metadata()

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
        # use const_params
        if paramname in self.const_params:
            return self.const_params[paramname]
        # if paramname is surface level in default param set
        if paramname in self.default_param_set:
            return self.default_param_set[paramname]
        # go deep
        for paramgroup in self.default_param_set["input"].values():
            if paramname in paramgroup:
                return paramgroup[paramname]
        assert False, f"Parameter {paramname} not found ANYWHERE!!!"

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

    def get_ensemble_parameter(self, ens_param_name: str) -> EnsembleParameter:
        param_match = [p for p in self.ensemble_params if p.param_key == ens_param_name]
        assert len(param_match) == 1, "ensemble parameter name problem bad bad bad!!!"
        return param_match[0]

    def tld(self) -> Path:
        return simulation_run_dir() / self.long_name()

    def folder_path(self, sim: PatchySimulation) -> Path:
        return self.tld() / sim.get_folder_path()

    def get_pipeline_step(self, step: Union[int, AnalysisPipelineStep]) -> AnalysisPipelineStep:
        return self.analysis_pipeline.get_pipeline_step(step)

    def time_length(self, sim: Union[PatchySimDescriptor, list[PatchySimDescriptor], None] = None) -> int:
        if sim is None:
            return self.time_length(self.ensemble())
        elif isinstance(sim, PatchySimulation):
            traj_file = self.folder_path(sim) / self.sim_get_param(sim, "trajectory_file")
            return file_info([str(traj_file)])["t_end"][0]
        elif isinstance(sim, tuple):
            return self.time_length(self.get_simulation(*sim))
        else:
            return min(file_info([
                self.folder_path(s) / self.sim_get_param(s, "trajectory_file")
                for s in sim
            ])["t_end"])

    # ------------------------ Status-Type Stuff --------------------------------#
    def info(self, infokey: str = "all"):
        """
        prints help text, for non-me people or if I forget
        """
        print(f"Ensemble of simulations of {self.export_name}")
        print("Ensemble Params")
        for param in self.ensemble_params:
            print(param)

        print("Function `has_pipeline`")
        print("Function `show_pipeline_graph`")
        print("Function `ensemble`")
        print("Function `show_last_conf`")
        print("Function `all_folders_exist`")

    def has_pipeline(self) -> bool:
        return len(self.analysis_pipeline) != 0

    def show_pipeline_graph(self):
        # Increase the figure size
        plt.figure(figsize=(8, 8))

        # Define the layout. Here we use spring_layout but you can try other layouts like shell_layout, random_layout etc.
        pos = nx.spring_layout(self.analysis_pipeline.pipeline_graph, 0.8)

        # Draw the nodes
        nx.draw_networkx_nodes(self.analysis_pipeline.pipeline_graph, pos, node_size=500)

        # Draw the edges
        nx.draw_networkx_edges(self.analysis_pipeline.pipeline_graph, pos, arrowstyle='->', arrowsize=20,
                               edge_cmap=plt.cm.Blues, width=2)

        # Draw the labels
        nx.draw_networkx_labels(self.analysis_pipeline.pipeline_graph, pos, font_size=8)

        plt.show()

    def show_last_conf(self, sim: PatchySimulation):
        from_path(self.folder_path(sim) / "last_conf.dat",
                  self.folder_path(sim) / "init.top",
                  self.folder_path(sim) / "particles.txt",
                  self.folder_path(sim) / "patches.txt")

    def show_analysis_status(self) -> pd.DataFrame:
        """
        Returns a Pandas dataframe showing the status of every simulation in the ensemble
        at each step on the analysis pipeline
        """
        return pd.DataFrame.from_dict({
            tuple(v.value for _, v in sim.param_vals):
                {
                    step_name: self.has_data_file(self.analysis_pipeline[step_name], sim)
                    for step_name in self.analysis_pipeline.name_map
                }
            for sim in self.ensemble()
        }, orient="index")

    def all_folders_exist(self):
        return all(self.folder_path(s).exists() for s in self.ensemble())

    def dna_analysis(self,
                     observable: Union[str, PatchySimObservable],
                     simulation_selector: Union[None, list[PatchySimulation], PatchySimulation] = None,
                     conf_file_name: Union[None, str] = None):
        """
        Runs the oxDNA utility DNAAnalysis, which allows the program to compute output for
        an observable for t
        """
        if isinstance(observable, str):
            observable = self.observables[observable]
        if simulation_selector is None:
            simulation_selector = self.ensemble()
        if conf_file_name is None:
            conf_file_name = "full_trajectory.dat"
        if isinstance(simulation_selector, list):
            for sim in simulation_selector:
                self.dna_analysis(observable, sim)
        else:
            self.write_input_file(simulation_selector,
                                  "input_dna_analysis",
                                  {
                                      "conf_file": conf_file_name,
                                      "analysis_data_output_1": observable
                                  })
            server_config = get_server_config()

            # write slurm script
            with open(self.folder_path(simulation_selector) / "dna_analysis.sh", "w+") as slurm_file:
                # bash header

                self.write_sbatch_params(simulation_selector, slurm_file)

                # skip confGenerator call because we will invoke it directly later
                slurm_file.write(f"{server_config['oxdna_path']}/build/bin/DNAnalysis input_dna_analysis\n")

            self.bash_exec(f"chmod u+x {self.folder_path(simulation_selector)}/slurm_script.sh")
            self.start_simulation(simulation_selector, "dna_analysis.sh")

    def merge_topologies(self,
                         sim_selector: Union[None, PatchySimulation, list[PatchySimulation]] = None,
                         topologies: Union[list[int], None] = None,
                         out_file_name: Union[str, None] = None):
        """
        Merges some topology files
        """
        if sim_selector is None:
            sim_selector = self.ensemble()
        if isinstance(sim_selector, list):
            for sim in sim_selector:
                self.merge_topologies(sim, topologies, out_file_name)
        else:
            # if no topology file specified
            if topologies is None:
                topologies = [f for f in self.folder_path(sim_selector).iterdir() if re.match(r"trajectory_\d+\.dat", f.name)]
                topologies = sorted(topologies, key=lambda f: int(re.search(r'trajectory_(\d+)\.dat', f.name).group(1)))
            if out_file_name is None:
                out_file_name = self.folder_path(sim_selector) / "full_trajectory.dat"

            #
            self.bash_exec(f"cat {' '.join(map(str, topologies))} > {str(out_file_name)}")

    def list_folder_files(self, sim: PatchySimulation):
        print([p.name for p in self.folder_path(sim).iterdir()])

    # ----------------------- Setup Methods ----------------------------------- #
    def do_setup(self):
        self.get_logger().info("Setting up folder / file structure...")
        for sim in self.ensemble():
            self.get_logger().info(f"Setting up folder / file structure for {repr(sim)}...")
            # create nessecary folders
            if not os.path.isdir(self.folder_path(sim)):
                self.get_logger().info(f"Creating folder {self.folder_path(sim)}")
                Path(self.folder_path(sim)).mkdir(parents=True)
            else:
                self.get_logger().info(f"Folder {self.folder_path(sim)} already exists. Continuing...")
            # write input file
            self.get_logger().info("Writing input files...")
            self.write_input_file(sim)
            # write requisite top, patches, particles files
            self.get_logger().info("Writing .top, .txt, etc. files...")
            self.write_sim_top_particles_patches(sim)
            # write observables.json if applicble
            if EXTERNAL_OBSERVABLES:
                self.get_logger().info("Writing observable json, as nessecary...")
                self.write_sim_observables(sim)
            # write .sh script
            self.get_logger().info("Writing sbatch scripts...")
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

    def write_sbatch_params(self, sim: PatchySimulation, slurm_file):
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

                # loop parameters in group
                for paramname in paramgroup:
                    # if we've specified this param in a replacer dict
                    if paramname in replacer_dict:
                        val = replacer_dict[paramname]
                    # if no override
                    elif paramname not in sim and paramname not in self.const_params:
                        val = paramgroup[paramname]
                    else:
                        val = self.sim_get_param(sim, paramname)
                    inputfile.write(f"{paramname} = {val}\n")
            # write things specific to rule
            # if josh_flavio or josh_lorenzo
            if server_config[PATCHY_FILE_FORMAT_KEY].find("josh") > -1:
                patch_file_info = [
                    ("patchy_file", "patches.txt"),
                    ("particle_file", "particles.txt"),
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
                if EXTERNAL_OBSERVABLES:
                    inputfile.write(f"observables_file = observables.json" + "\n")
                else:
                    for i, obsrv in enumerate(self.observables.values()):
                        obsrv.write_input(inputfile, i)

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

    def write_continue_files(self,
                             sim: Union[None, PatchySimulation] = None,
                             counter: int = -1,
                             continue_step_count: Union[int, None] = None):
        """
        writes input file and shell script to continue running the simulation after
        completion of first oxDNA execution
        """
        # continues start at 2; interpret anything lower as "figure it out"
        if counter < 2:
            # if metadata can help here, use it
            if LAST_CONTINUE_COUNT_KEY in self.metadata:
                counter = self.metadata[LAST_CONTINUE_COUNT_KEY] + 1
            else:  # use default first continue counter (2)
                counter = 2

        if sim is None:
            for sim in self.ensemble():
                self.write_continue_files(sim, counter)
        else:
            # construct an input file for the continuation execution
            # using previous conf as starting conf, adding new traj, writing new last_conf
            self.write_input_file(sim,
                                  file_name=f"input_{counter}",
                                  replacer_dict={
                                      "trajectory_file": f"trajectory_{counter}.dat",
                                      "conf_file": "last_conf.dat" if counter == 2 else f"last_conf_{counter - 1}.dat",
                                      "lastconf_file": f"last_conf_{counter}.dat",
                                      "steps": self.sim_get_param(sim, "steps") if continue_step_count is None else continue_step_count
                                  })
            # overwrite run script
            self.write_run_script(sim, input_file=f"input_{counter}")
        self.metadata[LAST_CONTINUE_COUNT_KEY] = counter \
            if self.metadata[LAST_CONTINUE_COUNT_KEY] not in self.metadata \
            or self.metadata["continue_count"] < counter \
            else self.metadata["continue_count"]

    def exec_continue(self, sim: PatchySimulation, counter: int = 2):
        # continues start at 2; interpret anything lower as "figure it out"
        if counter < 2:
            # if metadata can help here, use it
            if LAST_CONTINUE_COUNT_KEY in self.metadata:
                counter = self.metadata[LAST_CONTINUE_COUNT_KEY] + 1
            else:  # use default first continue counter (2)
                counter = 2

        if not (self.folder_path(sim) / f"input_{counter}").exists():
            # write new input file, update .sh file
            self.write_continue_files(sim, counter)
        # start the simulation
        self.start_simulation(sim)
        self.metadata[LAST_CONTINUE_COUNT_KEY] = counter
        self.dump_metadata()

    def exec_all_continue(self, counter: int = 2):
        for sim in self.ensemble():
            self.exec_continue(sim, counter)

    def gen_confs(self):
        for sim in self.ensemble():
            self.run_confgen(sim)

    def dump_slurm_log_file(self):
        self.slurm_log.to_csv(f"{self.tld()}/slurm_log.csv")

    def dump_metadata(self):
        # dump metadata dict to file
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, fp=f, indent=4)
        # dump analysis pipeline as pickle
        with open(self.metadata["analysis_file"], "wb") as f:
            pickle.dump(self.analysis_pipeline, f)

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
        response = self.bash_exec(f"sbatch --chdir={self.folder_path(sim)} {self.folder_path(sim)}/gen_conf.sh")
        jobid = int(re.search(SUBMIT_SLURM_PATTERN, response).group(1))
        return jobid

    def get_conf_file(self, sim: PatchySimulation) -> Path:
        return self.folder_path(sim) / "init.conf"

    # ------------- ANALYSIS FUNCTIONS --------------------- #
    def clear_pipeline(self):
        del self.metadata["analysis_file"]
        self.analysis_pipeline = AnalysisPipeline()

    def add_analysis_steps(self, *args):
        if isinstance(args[0], AnalysisPipeline):
            new_steps = args[0]
        else:
            new_steps = AnalysisPipeline(args[0], *args[1:])
        if new_steps not in self.analysis_pipeline:
            self.get_logger().info(f"Adding {len(new_steps)} steps "
                                   f"and {len(new_steps.pipeline_graph.edges)} to the analysis pipeline")
            self.analysis_pipeline = self.analysis_pipeline + new_steps
            self.dump_metadata()
        else:
            self.get_logger().info("The analysis pipeline you passed is already present")

    def has_data_file(self, step: PipelineStepDescriptor, sim: PatchySimDescriptor) -> bool:
        return self.get_cache_file(step, sim).exists()

    def get_data(self,
                 step: PipelineStepDescriptor,
                 sim: PatchySimDescriptor,
                 time_steps: range = None) -> PipelineData:
        """
        Returns data for a step, doing any/all required calculations
        Parameters:
            :param step an analysis step
            :param sim a patchy simulation object, descriptor of a PatchySimulation object,
            list of PatchySimulation object, or tuple of ParameterValues that indicates a PatchySimulation
            object
            It's VERY important to note that the list and tuple options have DIFFERENT behaviors!!!
            :param time_steps a range of timesteps to get data at. if None, the steps
            will be calculated automatically
        """
        # if we've provided a list of simulations
        if isinstance(sim, list):
            if self.is_do_analysis_parallel():
                self.get_logger().info(f"Assembling pool of {self.n_processes()} processes")
                with multiprocessing.Pool(self.n_processes()) as pool:
                    args = [(self, step, s, time_steps) for s in sim]
                    return pool.map(process_simulation_data, args)
            else:
                return [self.get_data(step, s, time_steps) for s in sim]

        step = self.get_pipeline_step(step)
        # if timesteps were not specified
        data = None
        if time_steps is None:
            time_steps = range(0, self.time_length(sim), step.output_tstep)
            self.get_logger().info(f"Constructed time steps {time_steps}")
        else:
            assert time_steps.step % step.output_tstep == 0, f"Specified step interval {time_steps} " \
                                                             f"not consistant with {step} output time " \
                                                             f"interval {step.output_tstep}"
        # check if we have cached data for this step already
        if self.has_data_file(step, sim):
            self.get_logger().info(
                f"Cache file for simulation {get_descriptor_key(sim)} and step {step} exists! Loading...")
            cache_file_path = self.get_cache_file(step, sim)
            cached_data = step.load_cached_files(cache_file_path)
            # if we already have the data needed for the required time range
            if step.data_matches_trange(data, time_steps):
                # that was easy!
                self.get_logger().info(f"All data in file! That was easy!")
                return cached_data
            else:
                self.get_logger().info(f"Cache file missing data!")

        step = self.get_pipeline_step(step)

        lock = multiprocessing.Lock()
        lock.acquire()
        try:
            # compute data for previous steps
            self.get_logger().info(f"Computing data for previous step {step.name} for simulation {str(sim)}...")
            data_in = self.get_step_input_data(step, sim, time_steps)
        finally:
            lock.release()
        # TODO: make sure this can handle the amount of args we're feeding in here
        # execute the step!
        # TODO: handle existing data that's incomplete over the required time interval
        data = step.exec(*data_in)
        lock.acquire()
        try:
            self.get_logger().info(f"Caching data in file `{self.get_cache_file(step, sim)}`")
            step.cache_data(data, self.get_cache_file(step, sim))
            self.analysis_data[(step.name, get_descriptor_key(sim))] = data
        finally:
            lock.release()

        return data

    def get_step_input_data(self,
                            step: PipelineStepDescriptor,
                            sim: PatchySimDescriptor,
                            time_steps: range) -> list[PipelineData]:
        step = self.get_pipeline_step(step)
        # if this step is an aggregate, things get... complecated
        if isinstance(step, AggregateAnalysisPipelineStep):
            # compute the simulation data required for this step
            # TODO: employ parallelization where applicable
            param_prev_steps = step.get_input_data_params(sim)
            return [
                self.get_data(prev_step,
                              param_prev_steps,
                              time_steps)
                for prev_step in self.analysis_pipeline.steps_before(step)
            ]
        elif isinstance(step, AnalysisPipelineHead):
            return [self.folder_path(sim) / file_name for file_name in step.get_data_in_filenames()]
        else:  # honestly this is still complecated but not as bad
            return [
                self.get_data(prev_step,
                              sim,
                              time_steps)
                for prev_step in self.analysis_pipeline.steps_before(step)
            ]

    def get_cache_file(self,
                       step: PipelineStepDescriptor,
                       sim: Union[tuple[ParameterValue, ...], PatchySimulation]) -> Path:
        """
        Retrieves a path to a file of analysis cache data for the given analysis step and
        simulation descriptor
        Parameters:
             step : an AnalysisPipelineStep object or an int indxer for such an object
             sim : either a PatchySimulation object or a tuple of ParameterValue objects specifying
             a set of PatchySimulation objects
        Return: a path to a data file
        """
        # get step object if it was passed as an index
        step = self.get_pipeline_step(step)
        # if single simulation
        if isinstance(sim, PatchySimulation):
            # cache analysis data in the simulation data folder
            return self.folder_path(sim) / step.get_cache_file_name()
        else:
            # cache analysis data in a folder in the top level directory
            return self.tld() / describe_param_vals(*sim) / step.get_cache_file_name()

    def step_targets(self, step: PipelineStepDescriptor):
        step = self.get_pipeline_step(step)
        if isinstance(step, AggregateAnalysisPipelineStep):
            return itertools.product(p for p in self.ensemble_params if p not in step.params_aggregate_over)
        else:
            return self.ensemble()

    def bash_exec(self, command: str):
        """
        Executes a bash command and returns the output
        """
        self.get_logger().info(f">`{command}`")
        response = subprocess.run(command, shell=True,
                                  capture_output=True, text=True, check=False)
        # response = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT, check=False,
        # universal_newlines=True)
        self.get_logger().info(f"`{response.stdout}`")
        return response.stdout


def process_simulation_data(args):
    ensemble, step, s, time_steps = args
    return ensemble.get_data(step, s, time_steps)

#
# def ensemble_from_export_settings(export_settings_file_path: Union[Path, str],
#                                   targets_file_path: Union[Path, str] = None) -> PatchySimulationEnsemble:
#     """
#     Constructs a PatchySimulationEnsemble object from the `patchy_export_settings.json` format
#     """
#     with open(export_settings_file_path, "r") as f:
#         export_setup_dict = json.load(f)
#
#     # Grab list of narrow types
#     narrow_types = export_setup_dict['narrow_types']
#     # grab list of temperatures
#     temperatures = export_setup_dict['temperatures']
#
#     rule = PolycubesRule(rule_str=export_setup_dict["rule_json"])
#
#     # make sure that all export groups have the same set of narrow types, duplicates,
#     # densities, and temperatures
#     assert all([
#         all_equal([
#             export_group[key]
#             for export_group in export_setup_dict['export_groups']
#         ])
#         for key in ["temperatures", "num_duplicates", "particle_density", "narrow_types"]
#     ])
#
#     particle_type_level_groups_list = [
#         {
#             "name": export_group["exportGroupName"],
#             "value": {
#                 "particle_type_levels": {
#                     p.name(): export_group["particle_type_levels"]
#                     for p in rule.particles()
#                 }
#             }
#         }
#         for export_group in export_setup_dict['export_groups']
#     ]
#
#     cfg_dict = {
#         EXPORT_NAME_KEY: export_setup_dict['export_name'],
#         PARTICLES_KEY: rule,
#         DEFAULT_PARAM_SET_KEY: "default.json",
#         OBSERABLES_KEY: ["plclustertopology"],
#         CONST_PARAMS_KEY: {
#             DENSITY_KEY: export_setup_dict["particle_density"] if "particle_density" else 0.1,
#             DENTAL_RADIUS_KEY: 0,
#             NUM_TEETH_KEY: 1,
#             "torsion": True
#         },
#         ENSEMBLE_PARAMS_KEY: [
#             (
#                 "T",
#                 temperatures
#             ),
#             (
#                 "narrow_type",
#                 narrow_types
#             ),
#             (
#                 "type_level_group",
#                 particle_type_level_groups_list
#             ),
#             (
#                 "duplicate",
#
#             )
#         ],
#     }
#     return PatchySimulationEnsemble(cfg_dict=cfg_dict)
