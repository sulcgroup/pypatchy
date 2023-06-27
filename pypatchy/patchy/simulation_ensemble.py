import datetime
import json
import os
import itertools
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import subprocess
import re
import logging

from ..util import get_param_set, simulation_run_dir, get_server_config, get_spec_json, get_log_dir, get_input_dir
from .ensemble_parameter import EnsembleParameter, ParameterValue
from .simulation_specification import PatchySimulationSpec
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

SUBMIT_SLURM_PATTERN = r"Submitted batch job (\d+)"


class PatchySimulationEnsemble:
    """
    Stores data for a group of related simulation
    class was originally written for setup but I've been folding analysis stuff from
    `PatchyResults` and `PatchyRunResult` in as well with the aim of eventually deprecating those
    preferably sooner tather than later
    """
    def __init__(self, **kwargs):
        if "cfg_file_name" in kwargs:
            with open(get_input_dir() / "wereflamingo_X2_scaling.json") as f:
                sim_cfg = json.load(f)
        else:
            assert "sim_cfg" in kwargs
            sim_cfg = kwargs["sim_cfg"]

        if "sim_date" in kwargs:
            self.sim_date = kwargs["sim_date"]
            if isinstance(self.sim_date, str):
                self.sim_date = datetime.datetime.strptime(self.sim_date, "%Y-%m-%d")
        else:
            # save current date
            self.sim_date: datetime.datetime = datetime.datetime.now()
        # assume standard file format json

        # name of simulation set
        self.export_name: str = sim_cfg[EXPORT_NAME_KEY]

        # configure logging
        self.logger: logging.Logger = logging.getLogger(self.export_name)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.FileHandler(get_log_dir() /
                                                   f"log_{self.export_name}_{self.sim_date.strftime('%Y-%m-%d')}"))
        self.logger.addHandler(logging.StreamHandler(sys.stdout))

        # load particles
        if "rule" not in sim_cfg:
            self.rule: PolycubesRule = PolycubesRule(rule_json=sim_cfg[PARTICLES_KEY])
        else:
            self.rule: PolycubesRule = PolycubesRule(rule_str=sim_cfg["rule"])

        # default simulation parameters
        self.default_param_set = get_param_set(
            sim_cfg[DEFAULT_PARAM_SET_KEY] if DEFAULT_PARAM_SET_KEY in sim_cfg else "default")
        self.const_params = sim_cfg[CONST_PARAMS_KEY] if CONST_PARAMS_KEY in sim_cfg else {}
        self.ensemble_params = [EnsembleParameter(*p) for p in sim_cfg[ENSEMBLE_PARAMS_KEY]]

        # observables are optional
        # TODO: integrate oxpy
        self.observables: list[dict] = []

        if OBSERABLES_KEY in sim_cfg:
            for obs in sim_cfg[OBSERABLES_KEY]:
                # if the name of an observable is provided
                if isinstance(obs, str):
                    try:
                        self.observables.append(get_spec_json(obs, "observables"))
                    except FileNotFoundError as e:
                        print(f"No file {obs}!")
                else:  # assume observable is provided as raw json
                    self.observables.append(obs)
        # handle potential weird stuff

        # init slurm log dataframe
        self.slurm_log = pd.DataFrame(columns=["slurm_job_id", *[p.param_key for p in self.ensemble_params]])

# --------------- Accessors and Mutators -------------------------- #
    def get_simulation(self, *args: list[ParameterValue]) -> PatchySimulationSpec:
        """
        TODO: idk
        given a list of parameter values, returns a PatchySimulation object
        """
        return PatchySimulationSpec(args)

    def long_name(self) -> str:
        return f"{self.export_name}_{self.sim_date.strftime('%Y-%m-%d')}"

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

    def sim_get_param(self, sim: PatchySimulationSpec,
                      paramname: str) -> Any:
        if paramname in sim:
            return sim[paramname]
        # use default
        assert paramname in self.const_params
        return self.const_params[paramname]

    def get_sim_particle_count(self, sim: PatchySimulationSpec,
                               particle_idx: int) -> int:
        # grab particle name
        particle_name = self.rule.particle(particle_idx).name()
        if PARTICLE_TYPE_LVLS_KEY in self.const_params and particle_name in self.const_params[PARTICLE_TYPE_LVLS_KEY]:
            particle_lvl = self.const_params[PARTICLE_TYPE_LVLS_KEY][particle_name]
        if PARTICLE_TYPE_LVLS_KEY in sim:
            spec = self.sim_get_param(sim, PARTICLE_TYPE_LVLS_KEY)
            if particle_name in spec:
                particle_lvl = spec[particle_name]
        return particle_lvl * self.sim_get_param(sim, NUM_ASSEMBLIES_KEY)

    def get_sim_total_num_particles(self, sim: PatchySimulationSpec) -> int:
        return sum([self.get_sim_particle_count(sim, i) for i in range(self.num_particle_types())])

    def num_patch_types(self, sim: PatchySimulationSpec) -> int:
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

    def ensemble(self) -> list[PatchySimulationSpec]:
        return [PatchySimulationSpec(e) for e in itertools.product(*self.ensemble_params)]

    def tld(self) -> Path:
        return simulation_run_dir() / self.long_name()

    def folder_path(self, sim: PatchySimulationSpec) -> Path:
        return self.tld() / sim.get_folder_path()

# ------------------------ Status-Type Stuff --------------------------------#


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

    def write_confgen_script(self, sim: PatchySimulationSpec):
        with open(self.get_run_confgen_sh(sim), "w+") as confgen_file:
            self.write_sbatch_params(sim, confgen_file)
            confgen_file.write(f"{get_server_config()['oxdna_path']}/build/bin/confGenerator input {self.sim_get_param(sim, 'density')}\n")
        self.bash_exec(f"chmod u+x {self.get_run_confgen_sh(sim)}")

    def write_run_script(self, sim: PatchySimulationSpec, input_file="input"):
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
                         sim: PatchySimulationSpec,
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
                if param in ensemble_var_names:
                    inputfile.write(f"{param} = {self.sim_get_param(sim, param)}" + "\n")

            # write external observables file path
            if len(self.observables) > 0:
                inputfile.write(f"observables_file = observables.json" + "\n")

    def write_sim_top_particles_patches(self, sim: PatchySimulationSpec):
        server_config = get_server_config()

        # write top and particles/patches spec files
        # first convert particle json into PLPatchy objects (cf plpatchy.py)
        particles = []
        for particle in self.rule.particles():
            particle_patches = [patch.to_pl_patch() for patch in particle.patches()]
            particle = PLPatchyParticle(type_id=particle.getID(), index_=particle.getID())
            particle.set_patches(particle_patches)

            particles.append(particle)

        if self.sim_get_param(sim, NUM_TEETH_KEY) > 1:
            particles, patches = convert_multidentate(particles,
                                                      self.sim_get_param(sim, DENTAL_RADIUS_KEY),
                                                      self.sim_get_param(sim, NUM_TEETH_KEY))
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

    def write_sim_observables(self, sim: PatchySimulationSpec):
        if len(self.observables) > 0:
            with open(self.folder_path(sim) / "observables.json", "w+") as f:
                json.dump({f"data_output_{i + 1}": obs for i, obs in enumerate(self.observables)}, f)

    def write_continue_files(self, sim: PatchySimulationSpec, counter: int = 2):
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
                                    "conf_file": "last_conf.dat" if counter == 2 else f"last_conf_{counter-1}.dat",
                                    "lastconf_file": f"last_conf_{counter}.dat"
                              })
        self.write_run_script(sim, input_file=f"input_{counter}")

    def exec_continue(self, sim: PatchySimulationSpec, counter: int = 2):
        pass

    def gen_confs(self):
        for sim in self.ensemble():
            os.chdir(self.folder_path(sim))
            cgpath = f"{get_server_config()['oxdna_path']}/build/confGenerator"
            self.bash_exec(f"{cgpath} input")
            os.chdir(self.tld())

    def dump_slurm_log_file(self):
        self.slurm_log.to_csv(f"{self.tld()}/slurm_log.csv")

    def start_simulations(self):
        for sim in self.ensemble():
            self.start_simulation(sim)

    def start_simulation(self,
                         sim: PatchySimulationSpec,
                         script_name: str = "slurm_script.sh"):
        command = f"sbatch {script_name}"

        if not os.path.isfile(self.get_conf_file(sim)):
            confgen_slurm_jobid = self.run_confgen(sim)
            command += f" --dependency=afterok:{confgen_slurm_jobid}"
        os.chdir(self.folder_path(sim))
        submit_txt = self.bash_exec(command)
        jobid = int(re.search(SUBMIT_SLURM_PATTERN, submit_txt).group(1))
        self.slurm_log.loc[len(self.slurm_log.index)] = {
            "slurm_job_id": jobid,
            **{
                key: value for key, value in sim
            }
        }
        os.chdir(self.tld())

    def get_run_oxdna_sh(self, sim: PatchySimulationSpec) -> Path:
        return self.folder_path(sim) / "slurm_script.sh"

    def get_run_confgen_sh(self, sim: PatchySimulationSpec) -> Path:
        return self.folder_path(sim) / "gen_conf.sh"

    def run_confgen(self, sim: PatchySimulationSpec) -> int:
        os.chdir(self.folder_path(sim))
        response = self.bash_exec("sbatch gen_conf.sh")
        os.chdir(self.tld())
        jobid = int(re.search(SUBMIT_SLURM_PATTERN, response).group(1))
        return jobid

    def get_conf_file(self, sim: PatchySimulationSpec) -> Path:
        return self.folder_path(sim) / "init.conf"

    def bash_exec(self, command: str):
        self.logger.info(f">`{command}`")
        response = subprocess.run(command, shell=True,
                                  capture_output=True, text=True, check=False)
        # response = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT, check=False,
                                           # universal_newlines=True)
        self.logger.info(f"`{response.stdout}`")
        return response.stdout

