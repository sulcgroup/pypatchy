import datetime
import os
import itertools
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import subprocess
import re
import logging

from patchy.util import get_param_set, sims_root, get_server_config, get_spec_json, get_root
from patchy.setup.ensemble_parameter import EnsembleParameter, SimulationSpecification
from polycubeutil.polycubesRule import load_rule
from patchy.plpatchy import Patch, PLPatchyParticle, export_interaction_matrix
from patchy.UDtoMDt import convert_multidentate

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


class PatchySimulationSetup:
    def __init__(self, json):
        # assume standard file format json

        # name of simulation set
        self.export_name = json[EXPORT_NAME_KEY]

        # save current date
        self.current_date = datetime.datetime.now()

        # configure logging
        self.logger = logging.getLogger(self.export_name)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.FileHandler(get_root() + os.sep +
                                                   "output" + os.sep +
                                                   f"log_{self.export_name}_{self.current_date.strftime('%Y-%m-%d')}"))
        self.logger.addHandler(logging.StreamHandler(sys.stdout))

        # load particles
        if "rule" not in json:
            self.particles = json[PARTICLES_KEY]
        else:
            self.particles = load_rule(json["rule"])
        # default simulation parameters
        self.default_param_set = get_param_set(
            json[DEFAULT_PARAM_SET_KEY] if DEFAULT_PARAM_SET_KEY in json else "default")
        self.const_params = json[CONST_PARAMS_KEY] if CONST_PARAMS_KEY in json else {}
        self.ensemble_params = [EnsembleParameter(*p) for p in json[ENSEMBLE_PARAMS_KEY]]
        # observables are optional
        # TODO: integrate oxpy
        self.observables = []

        if OBSERABLES_KEY in json:
            for obs in json[OBSERABLES_KEY]:
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

    def long_name(self):
        return f"{self.export_name}_{self.current_date.strftime('%Y-%m-%d')}"
        
    def get_sim_set_root(self):
        return sims_root() + self.export_name

    """
    num_particle_types should be constant across all simulations. some particle
    types may have 0 instances but hopefully oxDNA should tolerate this
    """
    
    def num_particle_types(self):
        return len(self.particles)

    """
    Returns the value of the parameter specified by paramname in the 
    simulation specification sim
    The program first checks if the parameter exists
    """
    def sim_get_param(self, sim, paramname):
        if paramname in sim:
            return sim[paramname]
        # use default
        assert paramname in self.const_params
        return self.const_params[paramname]

    def get_sim_particle_count(self, sim, particle_idx):
        # grab particle name
        particle_name = self.particles[particle_idx]["typeName"]
        if PARTICLE_TYPE_LVLS_KEY in self.const_params and particle_name in self.const_params[PARTICLE_TYPE_LVLS_KEY]:
            particle_lvl = self.const_params[PARTICLE_TYPE_LVLS_KEY][particle_name]
        if PARTICLE_TYPE_LVLS_KEY in sim:
            spec = self.sim_get_param(sim, PARTICLE_TYPE_LVLS_KEY)
            if particle_name in spec:
                particle_lvl = spec[particle_name]
        return particle_lvl * self.sim_get_param(sim, NUM_ASSEMBLIES_KEY)

    def get_sim_total_num_particles(self, sim):
        return sum([self.get_sim_particle_count(sim, i) for i in range(self.num_particle_types())])

    def num_patch_types(self, sim):
        return sum(len(p["patches"]) for p in self.particles) * self.sim_get_param(sim, NUM_TEETH_KEY)

    def paths_list(self):
        return [
            os.sep.join(combo)
            for combo in itertools.product(*[
                p.dir_names() for p in self.ensemble_params
            ])
        ]

    """
    Returns a list of lists of tuples,
    """
    def ensemble(self):
        return [SimulationSpecification(itm) for itm in itertools.product(*self.ensemble_params)]

    def folder_path(self, sim):
        return sims_root() + os.sep + self.long_name() + os.sep + sim.get_folder_path()
    
    def do_setup(self):
        for sim in self.ensemble():
            # sim should be a tuple of (string, ParameterValue) tuples
            # create nessecary folders
            Path(self.folder_path(sim)).mkdir(parents=True, exist_ok=True)

            self.write_input_file(sim)

            self.write_sim_top_particles_patches(sim)

            self.write_slurm_script(sim)

    def write_slurm_script(self, sim):
        server_config = get_server_config()

        # write slurm script
        with open(self.folder_path(sim) + os.sep + "slurm_script.sh", "w+") as slurm_file:
            # bash header
            slurm_file.write("#!/bin/bash")

            # slurm flags
            for flag_key in server_config["slurm_bash_flags"]:
                if len(flag_key) > 1:
                    slurm_file.write(f"#SBATCH --{flag_key}=\"{server_config['slurm_bash_flags'][flag_key]}\"")
                else:
                    slurm_file.write(f"#SBATCH -{flag_key} {server_config['slurm_bash_flags'][flag_key]}")

            # slurm includes ("module load xyz" and the like)
            for line in server_config["slurm_includes"]:
                slurm_file.write(line)

            # skip confGenerator call because we will invoke it directly later

            # run oxDNA!!!
            slurm_file.write(f"{server_config['oxdna_path']} input")

    def write_input_file(self, sim):
        server_config = get_server_config()

        # create input file
        with open(self.folder_path(sim) + os.sep + "input", 'w+') as inputfile:
            # write server config spec
            inputfile.write("#" * 32)
            inputfile.write(" SERVER PARAMETERS ".center(32, '#'))
            inputfile.write("#" * 32)
            for key in server_config["input_file_params"]:
                inputfile.write(f"{key} = {server_config['input_file_params'][key]}")

            # newline
            inputfile.write("")

            # write default input file stuff
            for paramgroup_key in self.default_param_set['input']:
                paramgroup = self.default_param_set['input'][paramgroup_key]
                inputfile.write("#" * 32)
                inputfile.write(f" {paramgroup_key} ".center(32, "#"))
                inputfile.write("#" * 32 + "\n")

                # loop parameters
                for paramname in paramgroup:
                    # skip parameters which are specified elsewhere
                    if paramname not in sim and paramname not in self.const_params:
                        inputfile.write(f"{paramname} = {paramgroup[paramname]}")

            # write things specific to rule
            # if josh_flavio or josh_lorenzo
            if server_config[PATCHY_FILE_FORMAT_KEY].find("josh") > -1:
                inputfile.write("patchy_file = patches.txt")
                inputfile.write("particle_file = particles.txt")
                inputfile.write(f"particle_types_N = {self.num_particle_types()}")
                inputfile.write(f"patch_types_N = {self.num_patch_types(sim)}")
            elif server_config[PATCHY_FILE_FORMAT_KEY] == "lorenzo":
                inputfile.write("DPS_interaction_matrix_file = interactions.txt")
            else:
                # todo: throw exception
                pass

            # write more parameters
            for param in ["T", "narrow_type"]:
                if param in sim:
                    inputfile.write(f"{param} = {self.sim_get_param(sim, param)}")

            # write external observables file
            if len(self.observables) > 0:
                inputfile.write(f"observables_file = observables.json")

    def write_sim_top_particles_patches(self, sim):
        server_config = get_server_config()

        # write top and particles/patches spec files
        # first convert particle json into PLPatchy objects (cf plpatchy.py)

        patch_counter = 0
        patches = []
        particles = []
        for i_particle, particle_json in enumerate(self.particles):
            patches_start = patch_counter
            for patch_obj in particle_json["patches"]:
                # TODO: resolve potential issues from passing particle info with a radius of 1
                a1 = np.array((patch_obj["dir"]["x"], patch_obj["dir"]["y"], patch_obj["dir"]["z"]))
                position = a1 / 2
                a2 = np.array((patch_obj["alignDir"]["x"], patch_obj["alignDir"]["y"], patch_obj["alignDir"]["z"]))
                patches.append(Patch(
                    type=patch_counter,
                    color=patch_obj["color"],
                    relposition=position,
                    a1=a1,
                    a2=a2,
                    strength=1))
                patch_counter += 1
            particle = PLPatchyParticle(type=i_particle, index_=i_particle)
            particle.set_patches(patches[patches_start:patch_counter])

        if self.sim_get_param(sim, NUM_TEETH_KEY) > 1:
            particles, patches = convert_multidentate(particles,
                                                      self.sim_get_param(sim, DENTAL_RADIUS_KEY),
                                                      self.sim_get_param(sim, NUM_TEETH_KEY))
        # do any/all valid conversions
        # either josh_lorenzo or josh_flavio
        if server_config[PATCHY_FILE_FORMAT_KEY].find("josh") > -1:
            with open(self.folder_path(sim) + os.sep + "init.top", "w+") as top_file:
                # first line of file
                top_file.write(f"{self.get_sim_total_num_particles(sim)} {len(particles)}")
                top_file.write(" ".join([
                    f"{i} " * self.get_sim_particle_count(sim, i) for i in range(len(particles))
                ]))
            all_patches_json = list(itertools.chain.from_iterable([p["patches"] for p in self.particles]))
            with open(self.folder_path(sim) + os.sep + "patches.txt", "w+") as patches_file:
                if server_config[PATCHY_FILE_FORMAT_KEY] == "josh_flavio":
                    for i, patch_obj in enumerate(patches):
                        # adjust for patch multiplier from multidentate
                        allo_conditional = patch_obj["conditional"][int(i / self.sim_get_param(sim, NUM_TEETH_KEY))]
                        patches_file.write(patch_obj.save_to_string(),
                                           {"allostery_conditional": allo_conditional})
                    else:  # josh/lorenzo
                        # adjust for patch multiplier from multidentate
                        state_var = all_patches_json[int(i / self.sim_get_param(sim, NUM_TEETH_KEY))]["state_var"]
                        activation_var = all_patches_json[int(i / self.sim_get_param(sim, NUM_TEETH_KEY))]["activation_var"]
                        patches_file.write(patch_obj.save_to_string(),
                                           {
                                               "state_var": state_var,
                                               "activation_var": activation_var
                                           })
            with open(self.folder_path(sim) + os.sep + "particles.txt", "w+") as particles_file:
                for particle_obj, particle_json in zip(particles, self.particles):
                    if server_config[PATCHY_FILE_FORMAT_KEY] == "josh_flavio":
                        particles_file.write(particle.save_type_to_string())
                    else:  # josh/lorenzo
                        particles_file.write(particle.save_type_to_string(), {particle_json["state_size"]})
        else:  # lorenzian
            with open(self.folder_path(sim) + os.sep + "init.top", "w+") as top_file:
                top_file.write(f"{self.get_sim_total_num_particles(sim)} {len(particles)}\n")
                # export_to_lorenzian_patchy_str also writes patches.dat file
                top_file.writelines([
                    particle.export_to_lorenzian_patchy_str(self.get_sim_particle_count(sim, particle),
                                                            self.folder_path(sim) + os.sep)
                    + "\n"
                    for particle in particles])
            export_interaction_matrix(patches)

    def start_simulations(self):
        for sim in self.ensemble():
            command = f"{self.folder_path()}"
            command = f"sbatch {self.folder_path(sim)}/slurm_script.sh"
            try:
                submit_txt = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT,
                                                     universal_newlines=True)
                pattern = r"Submitted slurm job (\d+)"
                jobid = int(re.search(pattern, submit_txt).group(1))
                self.slurm_log.append({
                    "slurm_job_id": jobid,
                    **{
                        key: value for key, value in sim
                    }
                })
            except subprocess.CalledProcessError as e:
                # If the command returns a non-zero exit status, you can handle the error here
                print(f"Error executing command: {e}")
                return None
