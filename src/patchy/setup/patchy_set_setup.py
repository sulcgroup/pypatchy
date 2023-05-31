import os
import itertools
from pathlib import Path
import numpy as np

from patchy.util import get_param_set, sims_root, get_server_config, get_spec_json
from patchy.setup.ensemble_parameter import EnsembleParameter
from polycubeutil.polycubesRule import load_rule
from patchy.plpatchy import Patch, PLPatchyParticle, export_interaction_matrix

EXPORT_NAME_KEY = "export_name"
PARTICLES_KEY = "particles"
DEFAULT_PARAM_SET_KEY = "default_param_set"
CONST_PARAMS_KEY = "const_params"
ENSEMBLE_PARAMS_KEY = "ensemble_params"
OBSERABLES_KEY = "observables"

PATCHY_FILE_FORMAT_KEY = "patchy_format"


class PatchySimulationSetup:
    def __init__(self, json):
        # assume standard file format json

        # name of simulation set
        self.export_name = json[EXPORT_NAME_KEY]

        # load particles
        if "rule" not in json:
            self.particles = json[PARTICLES_KEY]
        else:
            self.particles = load_rule(json["rule"])
        # default simulation parameters
        self.default_param_set = get_param_set(json[DEFAULT_PARAM_SET_KEY] if DEFAULT_PARAM_SET_KEY in json else "default")
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
                        self.observables.append(get_spec_json(obs))
                    except FileNotFoundError as e:
                        print (f"No file {obs}!")
                else:  # assume observable is provided as raw json
                    self.observables.append(obs)
        # handle potential weird stuff

    def get_sim_set_root(self):
        return sims_root() + self.export_name

    def num_particle_types(self):
        return len(self.particles)

    def num_patch_types(self):
        return sum(len(p["patches"]) for p in self.particles)

    def paths_list(self):
        return [
            os.sep.join(combo)
            for combo in itertools.product(*[
                p.dir_names() for p in self.ensemble_params
            ])
        ]

    def ensemble(self):
        return list(itertools.product(*self.ensemble_params))

    def get_sim_total_num_particles(self, sim):
        pass

    def get_sim_particle_count(self, sim, particle_idx):
        pass

    def do_setup(self):
        server_config = get_server_config()
        for sim in self.ensemble():
            # sim should be a tuple of (string, ParameterValue) tuples
            ensemble_var_names = [varname for varname,_ in sim]

            # create nessecary folders
            folderpath = os.sep.join([f"{key}_{str(val)}" for key, val in sim])
            Path.mkdir(folderpath, parents=True, exist_ok=True)

            # create input file
            with open(folderpath + os.sep + "input", 'w+') as inputfile:
                # write server config spec
                inputfile.write("#" * 32)
                inputfile.write(" SERVER PARAMETERS ".center(32, '#'))
                inputfile.write("#" * 32)
                for key in server_config["input_file_params"]:
                    inputfile.write(f"{key} = {server_config['input_file_params'][key]}")

                # newline
                inputfile.write("")

                # write default input file stuff
                for paramgroup in self.default_param_set['input']:
                    inputfile.write("#"*32)
                    inputfile.write(f" {paramgroup} ".center(32,"#"))
                    inputfile.write("#"*32 + "\n")

                    # loop parameters
                    for paramname in paramgroup:
                        # skip parameters which are specified elsewhere
                        if paramname not in ensemble_var_names and paramname not in self.const_params:
                            inputfile.write(f"{paramname} = {paramgroup[paramname]}")

                # write things specific to rule
                # if josh_flavio or josh_lorenzo
                if server_config[PATCHY_FILE_FORMAT_KEY].find("josh") > -1:
                    inputfile.write("patchy_file = patches.txt")
                    inputfile.write("particle_file = particles.txt")
                    inputfile.write(f"particle_types_N = {self.num_particle_types()}")
                    inputfile.write(f"patch_types_N = {self.num_patch_types()}")
                elif server_config[PATCHY_FILE_FORMAT_KEY] == "lorenzo":
                    inputfile.write("DPS_interaction_matrix_file = interactions.txt")
                else:
                    # todo: throw exception
                    pass

                # write more parameters
                for param in ["T", "narrow_type"]:
                    if param in ensemble_var_names:
                        inputfile.write(f"{param} = {sim_get_param(sim, param)}")

                # write external observables file
                if len(self.observables) > 0:
                    inputfile.write(f"observables_file = observables.json")

            # write particles/patches spec files
            # first convert particle json into PLPatchy objects (cf plpatchy.py)
            patch_counter = 0
            patches = []
            particles = []
            for i_particle, particle_json in enumerate(self.particles):
                patches_start = patch_counter
                for patch in particles["patches"]:
                    # TODO: resolve potential issues from passing particle info with a radius of 1
                    a1 = np.array((patch["dir"]["x"], patch["dir"]["y"], patch["dir"]["z"]))
                    position = a1 / 2
                    a2 = np.array((patch["alignDir"]["x"], patch["alignDir"]["y"], patch["alignDir"]["z"]))
                    patches.append(Patch(
                        type=patch_counter,
                        color=patch["color"],
                        relposition=position,
                        a1=a1,
                        a2=a2,
                        strength=1))
                    patch_counter += 1
                particle = PLPatchyParticle(type=i_particle, index_=i_particle)
                particle.set_patches(patches[patches_start:patch_counter])

            # do any/all valid conversions

            # either josh_lorenzo or josh_flavio
            if server_config[PATCHY_FILE_FORMAT_KEY].find("josh") > -1:
                with open(folderpath + os.sep + "init.top", "w+") as top_file:
                    # first line of file
                    top_file.write(f"{self.get_sim_total_num_particles(sim)} {len(particles)}")
                    top_file.write(" ".join([
                        f"{i} " * self.get_sim_particle_count(sim, i) for i in range(len(particles))
                    ]))
                with open(folderpath + os.sep + "patches.txt", "w+") as patches_file:
                    if server_config[PATCHY_FILE_FORMAT_KEY] == "josh_flavio":
                        for patch in patches:
                            patches_file.write(patch.save_to_string())  # TODO: allostery in josh/flavio format
                        else:
                            patches_file.write(patch.save_to_string())  # TODO: allostery in josh/lorenzo format
                with open(folderpath + os.sep + "particles.txt", "w+") as particles_file:
                    for particle in particles:
                        particles_file.write(particle.save_type_to_string())
            else:  # lorenzian
                with open(folderpath + os.sep + "init.top", "w+") as top_file:
                    top_file.write(f"{self.get_sim_total_num_particles(sim)} {len(particles)}\n")
                    # export_to_lorenzian_patchy_str also writes patches.dat file
                    top_file.writelines([
                        particle.export_to_lorenzian_patchy_str(self.get_sim_particle_count(sim, particle), folderpath + os.sep)
                        + "\n"
                        for particle in particles])
                export_interaction_matrix(patches)

def sim_get_param(sim, paramname):
    for p,v in sim:
        if p == paramname:
            return v
    # TODO: throw exception

