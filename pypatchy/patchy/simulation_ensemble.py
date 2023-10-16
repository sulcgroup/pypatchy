from __future__ import annotations

import datetime
import json
import multiprocessing
import os
import itertools
import pickle
import tempfile
import time
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd
import subprocess
import re
import logging

# import oat stuff
from oxDNA_analysis_tools.UTILS.oxview import from_path
from oxDNA_analysis_tools.file_info import file_info
from oxDNA_analysis_tools.UTILS.RyeReader import get_confs, describe, write_conf
from oxDNA_analysis_tools.UTILS.data_structures import Configuration

from pypatchy.analpipe.analysis_pipeline import AnalysisPipeline
from .stage import Stage, NoStageTrajError, IncompleteStageError, StageTrajFileEmptyError
from ..analpipe.analysis_data import PDPipelineData, TIMEPOINT_KEY
from ..analpipe.analysis_pipeline_step import AnalysisPipelineStep, PipelineData, AggregateAnalysisPipelineStep, \
    AnalysisPipelineHead, PipelineStepDescriptor, PipelineDataType
from .patchy_sim_observable import PatchySimObservable, observable_from_file
from ..patchy_base_particle import BaseParticleSet, Scene
from ..patchyio import NUM_TEETH_KEY, get_writer, BasePatchyWriter, DENTAL_RADIUS_KEY
from ..slurm_log_entry import SlurmLogEntry
from ..slurmlog import SlurmLog
from ..util import get_param_set, simulation_run_dir, get_server_config, get_log_dir, get_input_dir, is_slurm_job, \
    is_server_slurm, SLURM_JOB_CACHE, append_to_file_name
from .ensemble_parameter import EnsembleParameter, ParameterValue
from .simulation_specification import PatchySimulation, ParamSet
from .plpatchy import PLPSimulation
from .patchy_scripts import to_PL
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


# Custom LogRecord that includes 'long_name'
class PyPatchyLogRecord(logging.LogRecord):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.long_name = kwargs.get('extra', {}).get('long_name', 'N/A')


def get_descriptor_key(sim: PatchySimDescriptor) -> str:
    """
    returns a string representing the provided descriptor
    """
    return sim if isinstance(sim, str) else str(sim) if isinstance(sim, PatchySimulation) else describe_param_vals(*sim)


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


def metadata_file_exist(name: str, date: datetime.datetime) -> bool:
    return metadata_file(name, date).exists()


def metadata_file(name: str, date: datetime.datetime) -> Path:
    if name.endswith(".json"):
        name = name[:name.rfind(".")]
    return get_input_dir() / (name + "_" + date.strftime("%Y-%m-%d") + "_metadata.json")


def normalize_date(d):
    return d if isinstance(d, datetime.datetime) else datetime.datetime.strptime(d, "%Y-%m-%d")


def find_ensemble(*args: str, **kwargs) -> PatchySimulationEnsemble:
    """
    External method to construct PatchySimulationEnsemble objects
    """

    if len(args) > 0:
        simname = args[0]
        if len(args) == 2:
            sim_init_date: datetime.datetime = normalize_date(args[1])
        elif any([key in kwargs for key in ["sim_date", "date"]]):
            sim_init_date = normalize_date(
                [kwargs[key] for key in ["sim_date", "date"] if key in kwargs][0])
        else:
            return find_ensemble(name=simname)
        if not simname.endswith(".json"):
            return find_ensemble(name=simname, date=sim_init_date)
        elif metadata_file_exist(simname, sim_init_date):
            # if metadata file exists, load from it
            return find_ensemble(metadata=simname, date=sim_init_date)
        else:
            # load from cfg file
            return find_ensemble(cfg=simname, date=sim_init_date)

    elif "name" in kwargs:
        # flexable option. if name is provided, will look for metadata file but default to using the cfg file
        simname: str = kwargs["name"]

        if simname.endswith(".json"):
            raise Exception("Do not pass file name as the `name` parameter in this method! "
                            "Use `cfg_file` or `cfg` to specify a cfg file name.")

        if any([key in kwargs for key in ["sim_date", "date"]]):
            sim_init_date = normalize_date(
                [kwargs[key] for key in ["sim_date", "date"] if key in kwargs][0])
        # default: today
        else:
            sim_init_date: datetime.datetime = datetime.datetime.now()
        if metadata_file_exist(simname, sim_init_date):
            return find_ensemble(mdf=simname, date=sim_init_date)
        else:
            # try loading a cfg file with that name
            if (get_input_dir() / (simname + ".json")).exists():
                with open((get_input_dir() / (simname + ".json")), "r") as f:
                    exportname = json.load(f)["export_name"]
                if metadata_file_exist(exportname, sim_init_date):
                    return find_ensemble(exportname, sim_init_date)
            else:
                print(f"Warning: could not find metadata file for {simname} at {sim_init_date.strftime('%Y-%m-%d')}")
                return find_ensemble(cfg=simname, date=sim_init_date)

    elif any([key in kwargs for key in ["cfg_file_name", "cfg_file", "cfg"]]):
        # if the user is specifying a cfg file
        cfg_file_name: str = [kwargs[key] for key in ["cfg_file_name", "cfg_file", "cfg"] if key in kwargs][0]
        # if user passed a date
        if any([key in kwargs for key in ["sim_date", "date"]]):
            sim_init_date = normalize_date(
                [kwargs[key] for key in ["sim_date", "date"] if key in kwargs][0]
            )
        # default: today
        else:
            sim_init_date = datetime.datetime.now()
        if metadata_file_exist(cfg_file_name, sim_init_date):
            print("Warning! Metadata already exists for this ensemble but will NOT be loaded!")
        if not cfg_file_name.endswith(".json"):
            cfg_file_name = cfg_file_name + ".json"
        cfg_file_path = (get_input_dir() / cfg_file_name)
        if not cfg_file_path.exists():
            raise FileNotFoundError("Ensamble configureation file ")
        with cfg_file_path.open("r") as cfg_file:
            cfg = json.load(cfg_file)
            return build_ensemble(cfg, {
                "setup_date": sim_init_date.strftime("%Y-%m-%d"),
            })

    elif any([key in kwargs for key in ["metadata_file_name", "metadata_file", "mdf", "mdt", "metadata"]]):
        metadata_file_name: str = [kwargs[key] for key in ["metadata_file_name",
                                                           "metadata_file",
                                                           "mdf",
                                                           "mdt",
                                                           "metadata"] if key in kwargs][0]
        if metadata_file_name.endswith(".json"):
            # assume - incorrectly - that the user knows what they're doing
            metadata_file_path = get_input_dir() / metadata_file_name
            if not metadata_file_path.is_file():
                raise Exception(
                    f"No metadata file at for simulation {metadata_file_name}")
            with metadata_file_path.open("r") as mdt_file:
                mdt = json.load(mdt_file)
                return build_ensemble(mdt["ensemble_config"], mdt, metadata_file_path)
        else:
            # grab date arg
            if any([key in kwargs for key in ["sim_date", "date"]]):
                sim_init_date = normalize_date(
                    [kwargs[key] for key in ["sim_date", "date"] if key in kwargs][0]
                )
            else:
                # no default! we're assuming the user is looking for a SPECIFIC file!
                raise Exception("Missing date information for metadata sim lookup!")
            # two options: date-included and date-excluded
            if metadata_file_name.find(
                    "metadata") == -1:  # please please do not name a file that isn't metadata "metadata"
                metadata_file_name = metadata_file_name + "_" + sim_init_date.strftime("%Y-%m-%d") + "_metadata"
            metadata_file_name += ".json"
            return find_ensemble(mdt=metadata_file_name)  # recurse to strong literal behavior
    else:
        raise Exception("Missing required identifier for simulation!")


def build_ensemble(cfg: dict[str], mdt: dict[str, Union[str, dict]],
                   mdtfile: Union[Path, None] = None) -> PatchySimulationEnsemble:
    export_name = cfg[EXPORT_NAME_KEY]  # grab export name

    # normalize setup date (wrong word)
    setup_date: datetime.datetime = normalize_date(mdt["setup_date"])

    # if metadata filename wasn't manually provided
    if mdtfile is None:
        mdtfile = f"{export_name}_{setup_date.strftime('%Y-%m-%d')}_metadata.json"

    if "ensemble_config" not in mdt:
        mdt["ensemble_config"] = cfg

    if "analysis_file" in mdt:
        analysis_file = mdt["analysis_file"]
        if (get_input_dir() / analysis_file).is_file():
            with open(get_input_dir() / analysis_file, "rb") as f:
                analysis_pipeline = pickle.load(f)
        else:
            print(f"Analysis file specified in metadata but path {get_input_dir() / analysis_file} does not exist!")
            analysis_pipeline = AnalysisPipeline()
    else:
        analysis_file = f"{export_name}_analysis_pipeline.pickle"
        analysis_pipeline = AnalysisPipeline()

    if isinstance(mdt["setup_date"], datetime.datetime):
        mdt["setup_date"] = setup_date.strftime("%Y-%m-%d")

    # too difficult to make this one a ParamSet object
    default_param_set = get_param_set(
        cfg[DEFAULT_PARAM_SET_KEY] if DEFAULT_PARAM_SET_KEY in cfg else "default")

    # load const params from cfg
    if CONST_PARAMS_KEY in cfg:
        params = []
        counter = 1  # style
        for key, val in cfg[CONST_PARAMS_KEY].items():
            if isinstance(val, dict):
                # backwards compatibility w/ questionable life choices
                param_val = {
                    "name": f"const{counter}",
                    "value": val
                }
                counter += 1
            else:
                param_val = val
            params.append(ParameterValue(key, param_val))
        const_parameters = ParamSet(params)
    else:
        const_parameters = ParamSet([])

    # load ensemble params from cfg
    # there should always be ensemble params in the cfg
    ensemble_parameters = [
        EnsembleParameter(*p) for p in cfg[ENSEMBLE_PARAMS_KEY]
    ]

    # observables are optional
    # TODO: integrate oxpy
    observables: dict[str: PatchySimObservable] = {}

    if OBSERABLES_KEY in cfg:
        for obserable in cfg[OBSERABLES_KEY]:
            # legacy: string-expression of observable
            if isinstance(obserable, str):
                observables[obserable] = observable_from_file(obserable)
            else:
                assert isinstance(obserable, dict), f"Invalid type for observale {type(obserable)}"
                obserable = PatchySimObservable(**obserable)
                observables[obserable.observable_name] = obserable

    # load particles
    if PARTICLES_KEY in cfg:
        particles: BaseParticleSet = BaseParticleSet(cfg[PARTICLES_KEY])
        # particles: PolycubesRule = PolycubesRule(rule_json=cfg[PARTICLES_KEY])
    elif "cube_types" in cfg:
        if len(cfg["cube_types"]) > 0 and isinstance(cfg["cube_types"][0], dict):
            particles: PolycubesRule = PolycubesRule(rule_json=cfg["cube_types"])
        else:
            particles = cfg["cube_types"]
    elif "rule" in cfg:  # 'rule' tag assumes serialized rule string
        particles: PolycubesRule = PolycubesRule(rule_str=cfg["rule"])
    else:
        raise Exception("Missing particle info!")

    ensemble = PatchySimulationEnsemble(
        export_name,
        setup_date,
        particles,
        mdtfile,
        analysis_pipeline,
        default_param_set,
        const_parameters,
        ensemble_parameters,
        observables,
        analysis_file,
        mdt,
    )
    if "slurm_log" in mdt:
        for entry in mdt["slurm_log"]:
            sim = ensemble.get_simulation(**entry["simulation"])
            assert sim is not None, f"Slurm log included a record for invalid simulation {str(entry['simulation'])}"
            entry["simulation"] = sim
        ensemble.slurm_log = SlurmLog(*[SlurmLogEntry(**e) for e in mdt["slurm_log"]])
    ensemble.dump_metadata()
    return ensemble


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
    ensemble_param_name_map: dict[str, EnsembleParameter]

    # set of cube types that are used in this ensemble
    particle_set: BaseParticleSet

    observables: dict[str: PatchySimObservable]

    # ------------ SETUP STUFF -------------#

    # simulation parameters which are constant over the entire ensemble
    const_params: ParamSet

    # parameter values to use if not specified in `const_params` or in the
    # simulation specification params
    # load from `spec_files/input_files/[name].json`
    default_param_set: dict[str: Union[dict, Any]]

    # -------------- STUFF SPECIFIC TO ANALYSIS ------------- #

    # each "node" of the analysis pipeline graph is a step in the analysis pipeline
    analysis_pipeline: AnalysisPipeline

    # dict to store loaded analysis data
    analysis_data: dict[tuple[AnalysisPipelineStep, ParamSet], PipelineData]
    analysis_file: str

    # log of slurm jobs
    slurm_log: SlurmLog

    # output writer
    writer: BasePatchyWriter

    def __init__(self,
                 export_name: str,
                 setup_date: datetime.datetime,
                 particle_set: BaseParticleSet,
                 metadata_file_name: str,
                 analysis_pipeline: AnalysisPipeline,
                 default_param_set: dict,
                 const_params: ParamSet,
                 ensemble_params: list[EnsembleParameter],
                 observables: dict[str, PatchySimObservable],
                 analysis_file: str,
                 metadata_dict: dict,  # dict of serialized metadata, to preserve it
                 ):
        self.export_name = export_name
        self.sim_init_date = setup_date

        # configure logging ASAP
        # File handler with a higher level (ERROR)
        logger: logging.Logger = logging.getLogger(self.export_name)
        logger.setLevel(logging.DEBUG)

        file_handler = logging.FileHandler(
            get_log_dir() / f"log_{self.export_name}_{self.datestr()}_at{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}.log",
            mode="a")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Stream handler with a lower level (INFO)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        stream_handler.setFormatter(stream_formatter)
        logger.addHandler(stream_handler)

        # logger: logging.Logger = logging.getLogger(self.export_name)
        # logger.setLevel(logging.INFO)
        # logger.addHandler(logging.FileHandler(get_log_dir() /
        #                                       f"log_{self.export_name}_{self.datestr()}.log", mode="a"))
        # logger.addHandler(logging.StreamHandler(sys.stdout))

        self.particle_set = particle_set

        self.metadata = metadata_dict
        self.slurm_log = SlurmLog()
        self.analysis_pipeline = analysis_pipeline
        self.analysis_file = analysis_file

        self.metadata_file = metadata_file_name

        self.default_param_set = default_param_set
        self.const_params = const_params

        self.ensemble_params = ensemble_params
        self.ensemble_param_name_map = {p.param_key: p for p in self.ensemble_params}
        self.observables = observables

        # construct analysis data dict in case we need it
        self.analysis_data = dict()

        self.writer = get_writer()

    # def __init__(self, *args: str, **kwargs):
    #     """
    #     Very flexable constructor
    #     Options are:
    #         PatchySimulationEnsemble(cfg_file_name="input_file_name.json", [sim_date="yyyy-mm-dd"])
    #         PatchySimulationEnsemble(sim_metadata_file="metadata_file.json)
    #         PatchySimulationEnsemble(export_name="a-simulation-name", particles=[{.....)
    #     """
    #
    #     # optional positional arguements are a file name string and an optional date string
    #     if len(args) > 0:
    #         # assign vals in cfg dict
    #         kwargs["cfg_file_name"] = args[0]
    #         if len(args) == 2:
    #             kwargs["sim_date"] = args[1]
    #
    #     assert "cfg_file_name" in kwargs or METADATA_FILE_KEY in kwargs
    #
    #     # initialize metadata dict
    #     self.metadata: dict = {}
    #     self.slurm_log = SlurmLog()
    #     sim_cfg = kwargs
    #
    #     if METADATA_FILE_KEY in kwargs or "cfg_file_name" in kwargs:
    #         # if an execution metadata file was provided
    #         if METADATA_FILE_KEY in kwargs:
    #             sim_cfg = self.load_metadata_from(get_input_dir() / kwargs[METADATA_FILE_KEY])
    #
    #         # if a config file name was provided
    #         elif "cfg_file_name" in kwargs:
    #             cfg_file_name: str = kwargs["cfg_file_name"]
    #             # if filename was provided without a json extension
    #             if cfg_file_name.find(".") == -1:
    #                 cfg_file_name = cfg_file_name + ".json"
    #                 with open(get_input_dir() / cfg_file_name, 'r') as f:
    #                     sim_cfg.update(json.load(f))
    #                 if "sim_date" in sim_cfg:
    #                     # if metadata file exists for this date
    #                     metadata_file = get_input_dir() / (
    #                             sim_cfg[EXPORT_NAME_KEY] + "_" + sim_cfg["sim_date"] + "_metadata.json")
    #                     if "sim_date" in sim_cfg and metadata_file.is_file():
    #                         self.load_metadata_from(metadata_file)
    #                     else:
    #                         print(f"Warning: Metadata file {str(metadata_file)} not found!")
    #
    #             else:
    #                 with open(get_input_dir() / cfg_file_name, 'r') as f:
    #                     sim_cfg.update(json.load(f))
    #
    #     # if a date string was provided
    #     if "sim_date" in sim_cfg:
    #         self.sim_init_date = sim_cfg["sim_date"]
    #         if isinstance(self.sim_init_date, str):
    #             self.sim_init_date = datetime.datetime.strptime(self.sim_init_date, "%Y-%m-%d")
    #
    #     else:
    #         # save current date
    #         self.sim_init_date: datetime.datetime = datetime.datetime.now()
    #
    #     # whether it was just set from kwaargs or gen'd from today, make init date str
    #     # to identify this ensemble
    #     datestr = self.sim_init_date.strftime("%Y-%m-%d")
    #
    #     # if no metadata file was provided in the keyword arguements
    #     if METADATA_FILE_KEY not in kwargs:
    #         self.metadata["setup_date"] = datestr
    #         self.metadata["ensemble_config"] = sim_cfg
    #         self.metadata_file = get_input_dir() / f"{sim_cfg[EXPORT_NAME_KEY]}_{datestr}_metadata.json"
    #         # if a metadata file exists at the default path
    #         if self.metadata_file.exists():
    #             # update metadata dict from file
    #             with open(self.metadata_file, "r") as f:
    #                 self.metadata.update(json.load(f))
    #                 sim_cfg.update(self.metadata["ensemble_config"])
    #
    #     # name of simulation set
    #     self.export_name: str = sim_cfg[EXPORT_NAME_KEY]
    #
    #     # configure logging
    #     logger: logging.Logger = logging.getLogger(self.export_name)
    #     logger.setLevel(logging.INFO)
    #     logger.addHandler(logging.FileHandler(get_log_dir() /
    #                                           f"log_{self.export_name}_{self.sim_init_date.strftime('%Y-%m-%d')}.log"))
    #     logger.addHandler(logging.StreamHandler(sys.stdout))
    #
    #     # load particles
    #     if "rule" not in sim_cfg and "cube_types" not in sim_cfg:
    #         if isinstance(sim_cfg[PARTICLES_KEY], BaseParticleSet):
    #             self.particle_set = sim_cfg[PARTICLES_KEY]
    #         else:
    #             self.particle_set: PolycubesRule = PolycubesRule(rule_json=sim_cfg[PARTICLES_KEY])
    #     else:
    #         if "cube_types" in sim_cfg:
    #             if len(sim_cfg["cube_types"]) > 0 and isinstance(sim_cfg["cube_types"][0], dict):
    #                 self.particle_set: PolycubesRule = PolycubesRule(rule_json=sim_cfg["cube_types"])
    #             else:
    #                 self.particle_set = sim_cfg["cube_types"]
    #         else:
    #             self.particle_set: PolycubesRule = PolycubesRule(rule_str=sim_cfg["rule"])
    #
    #     # default simulation parameters
    #     self.default_param_set = get_param_set(
    #         sim_cfg[DEFAULT_PARAM_SET_KEY] if DEFAULT_PARAM_SET_KEY in sim_cfg else "default")
    #     self.const_params = sim_cfg[CONST_PARAMS_KEY] if CONST_PARAMS_KEY in sim_cfg else {}
    #
    #     self.ensemble_params = [EnsembleParameter(*p) if not isinstance(p, EnsembleParameter) else p for p in
    #                             sim_cfg[ENSEMBLE_PARAMS_KEY]]
    #     self.ensemble_param_name_map = {p.param_key: p for p in self.ensemble_params}
    #
    #     if "slurm_log" in self.metadata:
    #         for entry in self.metadata["slurm_log"]:
    #             entry["simulation"] = self.get_simulation(*entry["simulation"].items())
    #         self.slurm_log = SlurmLog(*[SlurmLogEntry(**e) for e in self.metadata["slurm_log"]])
    #
    #     # observables are optional
    #     # TODO: integrate oxpy
    #     self.observables: dict[str: PatchySimObservable] = {}
    #
    #     if OBSERABLES_KEY in sim_cfg:
    #         for obs_name in sim_cfg[OBSERABLES_KEY]:
    #             self.observables[obs_name] = observable_from_file(obs_name)
    #
    #     # handle potential weird stuff??
    #
    #     # load analysis pipeline
    #     self.analysis_pipeline = AnalysisPipeline()
    #
    #     # if the metadata specifies a pickle file of a stored analywsis pathway, use it
    #     if "analysis_file" in self.metadata:
    #         file_path = Path(self.metadata["analysis_file"])
    #         if file_path.exists():
    #             with open(file_path, "rb") as f:
    #                 self.analysis_pipeline = self.analysis_pipeline + pickle.load(f)
    #         else:
    #             self.get_logger().warning(f"Analysis file specified in metadata but path {file_path} does not exist!")
    #     else:  # if not, use a default one and cache it
    #         file_path = self.tld() / "analysis_pipeline.pickle"
    #         self.metadata["analysis_file"] = str(file_path)
    #
    #     # construct analysis data dict in case we need it
    #     self.analysis_data = dict()

    # def load_metadata_from(self, metadata_file_path: Path):
    #     assert metadata_file_path.is_file(), f"File {metadata_file_path} does not exist!"
    #     self.metadata_file = metadata_file_path
    #     try:
    #         with metadata_file_path.open("r") as f:
    #             self.metadata.update(json.load(f))
    #             return self.metadata["ensemble_config"]
    #     except JSONDecodeError as e:
    #         raise Exception(f"File {metadata_file_path} is malformed!")

    # --------------- Accessors and Mutators -------------------------- #
    def get_simulation(self, *args: Union[tuple[str, Any], ParameterValue], **kwargs) -> Union[
        PatchySimulation, list[PatchySimulation]]:
        """
        This is a very flexable method for returning PatchySimulation objects
        but is also very complex, due to the range of inputs accepted

        given a list of parameter values, returns a PatchySimulation object
        """
        # sim_params = args
        sim_params: list[Union[list[ParameterValue], EnsembleParameter]] = []
        multiselect = False
        for i_counter, param_type in enumerate(self.ensemble_params):
            pname = param_type.param_key
            if pname in kwargs:
                valname = str(kwargs[pname])
                sim_params.append([param_type[valname]])
            else:
                for a in args:
                    if isinstance(a, ParameterValue):
                        if a in param_type:
                            sim_params.append([a])
                            break
                    else:
                        assert isinstance(a, tuple) and len(a) == 2, f"Invalid parameter {str(a)}!"
                        k, v = a
                        if k == pname:
                            pv = ParameterValue(k, v)
                            assert pv in param_type
                            sim_params.append([pv])
                            break  # terminate loop of args
            if len(sim_params) == i_counter:  # if we've found value for param, len(sim_params) should be one greater than counter
                sim_params.append(param_type.param_value_set)
                multiselect = True
        if multiselect:
            return [PatchySimulation(e) for e in itertools.product(*sim_params)]
        else:
            return PatchySimulation([p for p, in sim_params])  # de-listify

    def datestr(self) -> str:
        return self.sim_init_date.strftime("%Y-%m-%d")

    def long_name(self) -> str:
        return f"{self.export_name}_{self.datestr()}"

    def get_logger(self) -> logging.Logger:
        return logging.getLogger(self.export_name)

    def is_do_analysis_parallel(self) -> bool:
        """
        Returns:
            true if the analysis is set up to run in parallel using multiprocessing.Pool
            and false otherwise.
        """
        return "parallel" in self.metadata and self.metadata["parallel"]

    def set_analysis_pipeline(self,
                              src: Union[str, Path],
                              clear_old_pipeline=True,
                              link_file=True):
        """
        Sets the ensemble's analysis pipeline from an existing analysis pipeline file
        Args:
            src: a string or path object indicating the file to set from
            clear_old_pipeline: if set to True, the existing analysis pipleine will be replaced with the pipeline from the file. otherwise, the new pipeline will be appended
            link_file: if set to True, the provided source file will be linked to this ensemble, so changes made to this analysis pipeline will apply to all ensembles that source from that file
        """
        if isinstance(src, str):
            if src.endswith(".pickle"):
                src = get_input_dir() / src
            else:
                src = get_input_dir() / (src + ".pickle")
        if clear_old_pipeline:
            self.analysis_pipeline = AnalysisPipeline()
        try:
            with open(src, "rb") as f:
                self.analysis_pipeline = self.analysis_pipeline + pickle.load(f)
            if link_file:
                self.analysis_file = src
        except FileNotFoundError:
            logging.error(f"No analysis pipeline found at {str(src)}.")
        self.dump_metadata()

    def n_processes(self):
        return self.metadata["parallel"]

    def is_nocache(self) -> bool:
        return "nocache" in self.metadata and self.metadata["nocache"]

    def set_nocache(self, bNewVal: bool):
        self.metadata["nocache"] = bNewVal

    def append_slurm_log(self, item: SlurmLogEntry):
        """
        Appends an entry to the slurm log, also writes a brief description
        to the logger
        """
        self.get_logger().debug(str(item))
        self.slurm_log.append(item)

    def set_metadata_attr(self, key: str, val: Any):
        """
        Sets a metadata value, and
        saves the metadata file if a change has been made
        """
        oldVal = self.metadata[val]
        self.metadata[key] = val
        if val != oldVal:
            self.dump_metadata()

    def get_step_counts(self) -> list[tuple[PatchySimulation, int]]:
        return [
            (sim, self.time_length(sim))
            for sim in self.ensemble()
        ]

    def get_stopped_sims(self) -> list[PatchySimulation]:
        """
        Returns a list of simulations which have been stopped by the slurm
        controller before they were complete
        """
        sims_that_need_attn = []
        for sim in self.ensemble():
            entries = self.slurm_log.by_subject(sim)
            if len(entries) == 0:
                continue
            desired_sim_length = self.sim_get_param(sim, "steps")
            last_entry = entries[-1]
            # get job info
            jobinfo = self.bash_exec(f"scontrol show job {last_entry.job_id} | grep JobState")
            job_stopped = len(jobinfo) == 0 or jobinfo.split()[0].split("=")[1] != "RUNNING"
            if job_stopped:
                if self.time_length(sim) < desired_sim_length:
                    sims_that_need_attn.append(sim)
        return sims_that_need_attn

    def num_particle_types(self) -> int:
        """
        num_particle_types should be constant across all simulations. some particle
        types may have 0 instances but hopefully oxDNA should tolerate this
        """
        return len(self.particle_set)

    def sim_get_param(self,
                      sim: PatchySimulation,
                      paramname: str) -> Any:
        """
        Returns the value of a parameter
        This method will first search in the simulation to see if this parameter
        has a specific value for this simulation, then in the ensemble const params,
        then the default parameter set
        """
        if paramname == "particle_types":
            return self.particle_set
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
        # deeper
        if paramname in get_server_config()["input_file_params"]:
            return get_server_config()["input_file_params"][paramname]
        # TODO: custom exception here
        raise NoSuchParamError(self, sim, paramname)
        # raise Exception(f"Parameter {paramname} not found ANYWHERE!!!")

    def paramfile(self, sim: PatchySimulation, paramname: str) -> Path:
        """
        Shorthand to get a simulation data file
        """
        return self.folder_path(sim) / self.sim_get_param(sim, paramname)

    def paramstagefile(self, sim: PatchySimulation, stage: Stage, paramname: str):
        return self.folder_path(sim) / stage.adjfn(self.sim_get_param(sim, paramname))

    def get_input_file_param(self, paramname: str):
        """
        Gets a parameter which is passed to input file and is constant
        """
        assert paramname not in [e.param_key for e in self.ensemble_params]
        if paramname in self.const_params:
            return self.const_params[paramname]
        for paramgroup in self.default_param_set["input"].values():
            if paramname in paramgroup:
                return paramgroup[paramname]
        raise Exception(f"Parameter {paramname} not found!")

    # def get_sim_particle_count(self, sim: PatchySimulation,
    #                            particle_idx: int) -> int:
    #     """
    #     Args:
    #         sim: the patchy simulation to count for
    #         particle_idx: the index of the particle to get the count for
    #     Returns:
    #         the int
    #     """
    #     # grab particle name
    #     particle_name = self.particle_set.particle(particle_idx).name()
    # if PARTICLE_TYPE_LVLS_KEY in self.const_params and particle_name in self.const_params[PARTICLE_TYPE_LVLS_KEY]:
    #     particle_lvl = self.const_params[PARTICLE_TYPE_LVLS_KEY][particle_name]
    # if PARTICLE_TYPE_LVLS_KEY in sim:
    #     spec = self.sim_get_param(sim, PARTICLE_TYPE_LVLS_KEY)
    #     if particle_name in spec:
    #         particle_lvl = spec[particle_name]
    #     # return self.sim_get_param(sim, particle_name) * self.sim_get_param(sim, NUM_ASSEMBLIES_KEY)
    #
    # def get_sim_particle_counts(self, sim: PatchySimulation):
    #     """
    #     Returns: the number of particles in the simulation
    #     """
    #     return {
    #         p.type_id(): self.get_sim_particle_count(sim, p.type_id()) for p in self.particle_set.particles()
    #     }
    #
    # def get_sim_total_num_particles(self, sim: PatchySimulation) -> int:
    #     return sum([self.get_sim_particle_count(sim, i) for i in range(self.num_particle_types())])

    def get_sim_particle_count(self, sim: PatchySimulation,
                               particle_idx: int) -> int:
        """
        Args:
            sim: the patchy simulation to count for
            particle_idx: the index of the particle to get the count for
        Returns:
            the int
        """
        # grab particle name
        particle_name = self.particle_set.particle(particle_idx).name()
        return self.sim_get_param(sim, particle_name) * self.sim_get_param(sim, NUM_ASSEMBLIES_KEY)

    def get_sim_particle_counts(self, sim: PatchySimulation):
        """
        Returns: the number of particles in the simulation
        """
        return {
            p.type_id(): self.get_sim_particle_count(sim, p.type_id()) for p in self.particle_set.particles()
        }

    def sim_get_stages(self, sim: PatchySimulation) -> list[Stage]:
        """
        Computes stages
        Stage objects require other ensemble parameters so must be constructed as needed and not on init

        """
        num_assemblies = self.sim_get_param(sim, "num_assemblies")
        try:
            stages_info = self.sim_get_param(sim, "stages")
            # incredibly cursed line of code incoming
            stages_info = [{"idx": i, "name": k, **v} for i, (k, v) in
                           enumerate(sorted(stages_info.items(), key=lambda x: x[1]["t"]))]
            stages = []
            # loop stages in stage list from spec
            for stage_info in stages_info:
                # get stage idx
                i = stage_info["idx"]
                # get list of names of particles to add
                stage_particles = list(itertools.chain.from_iterable([
                    [pname] * (stage_info["particles"][pname] * num_assemblies)
                    for pname in stage_info["particles"]]))
                stage_init_args = {
                    "t": stage_info["t"],
                    "particles": stage_particles,
                }
                if "length" not in stage_info and "tlen" not in stage_info:
                    if i + 1 != len(stages_info):
                        # if stage is not final stage, last step of this stage will be first step of next stage
                        stage_init_args["tend"] = stages_info[i + 1]["t"]
                    else:
                        # if the stage is the final stage, last step will be end of simulation
                        stage_init_args["tend"] = self.sim_get_param(sim, "steps")
                elif "length" in stage_info:
                    stage_init_args["tlen"] = stage_info["length"]
                else:
                    stage_init_args["tlen"] = stage_info["tlen"]
                # construct stage objects
                stages.append(Stage(i,
                                    stage_info["name"],
                                    **stage_init_args
                                    ))

        except NoSuchParamError as e:
            # if stages not found
            # default: 1 stage, density = starting density, add method=random
            particles = list(itertools.chain.from_iterable([
                [p.type_id()] * self.sim_get_param(sim, p.name()) * num_assemblies
                for p in self.particle_set.particles()
            ]))
            stages = [Stage(
                stagenum=0,
                stagename="default",
                t=0,
                tend=self.sim_get_param(sim, "steps"),
                particles=particles
            )]
            # box_size = (len(particles) / self.sim_get_param(sim, "density")) ** (1/3)
            # write stage dict for code below
            stages_info = {
                0: {
                    "density": self.sim_get_param(sim, "density")
                }
            }

        # assign box sizes
        for stage in stages:
            stage_info = stages_info[stage.idx()]
            # if box size is specified explicitly
            if "box_size" in stage_info:
                stage.set_box_size(stage_info["box_size"])
            # if box size is specified relative to number of particles
            else:
                # if we are to use all particles (not changing volume)
                # default to no calc fwd
                if "calc_fwd" not in stage_info or stage_info["calc_fwd"]:
                    # make sure there are stages after this one
                    num_particles = sum([len(s.particles_to_add()) for s in stages])
                else:
                    # only consider particles in past stages (incl. this one)
                    num_particles = sum([len(s.particles_to_add()) for s in stages[:stage.idx()]])

                # do NOT incorporate num assemblies - already did above!

                if "rel_volume" in stage_info:
                    relvol = stage_info["rel_volume"]
                    box_side = (relvol * num_particles) ** .3

                # if density format
                elif "density" in stage_info:
                    # box side length = cube root of n / density
                    density = stage_info["density"]
                    box_side = (num_particles / density) ** .3

                else:
                    density = self.sim_get_param(sim, "density")
                    box_side = (num_particles / density) ** (1 / 3)

                stage.set_box_size(np.array((box_side, box_side, box_side)))

            if stage.idx() > 0:
                assert (stages[stage.idx() - 1].box_size() <= stage.box_size()).all(), "Shrinking box size not allowed!"

        return stages

    def sim_get_stages_between(self, sim: PatchySimulation, tstart: int, tend: int) -> list[Stage]:
        return [stage for stage in self.sim_get_stages(sim) if stage.end_time() >= tstart and stage.start_time() < tend]

    def sim_get_stage(self, sim: PatchySimulation, stage_name: Union[str, int]) -> Stage:
        stages = self.sim_get_stages(sim)
        if isinstance(stage_name, int):
            return stages[stage_name]
        # inefficient search algorithm but len(stages) should never be more than like 10 tops
        for stage in stages:
            if stage.name() == stage_name:
                return stage
        raise Exception(f"No stage named {stage_name}!")

    def sim_get_stage_top_traj(self, sim: PatchySimulation, stage: Union[str, int, Stage]) -> tuple[Path, Path]:
        """
        Returns:
            the traj and top FILES for provided simulation and stage
        """
        if not isinstance(stage, Stage):
            stage = self.sim_get_stage(sim, stage)
        return (
            self.folder_path(sim) / stage.adjfn(self.sim_get_param(sim, "topology")),
            self.folder_path(sim) / stage.adjfn(self.sim_get_param(sim, "trajectory_file"))
        )

    def sim_get_stage_last_step(self, sim: PatchySimulation, stage: Union[str, int, Stage]) -> int:
        _, traj = self.sim_get_stage_top_traj(sim, stage)
        if not traj.is_file():
            raise NoStageTrajError(stage, sim, str(traj))
        else:
            # return timepoint of last conf in traj
            try:
                return file_info([str(traj)])["t_end"][0]
            except IndexError as e:
                # trajectory file empty
                raise StageTrajFileEmptyError(stage, sim, str(traj))

    def sim_most_recent_stage(self, sim: PatchySimulation) -> Stage:
        """
        Returns:
            a tuple where the first element is the most recent stage file with a trajectory
            and the second element is true if the last conf in the traj is at the last timepoint of the stage
            or none if no stage has begun
        """
        # increment in  reverse order so we check later stages first
        for stage in reversed(self.sim_get_stages(sim)):
            # if traj file exists
            if (self.folder_path(sim) / stage.adjfn(self.sim_get_param(sim, "trajectory_file"))).exists():
                try:
                    stage_last_step = self.sim_get_stage_last_step(sim, stage)
                    if stage_last_step == stage.end_time():
                        return stage
                    # if stage is incomplete, raise an exception
                    raise IncompleteStageError(
                        stage,
                        sim,
                        stage_last_step
                    )
                except StageTrajFileEmptyError as e:
                    # if the stage traj is empty just continue
                    pass

        # if no stage has a traj file, raise exception
        trajpath = self.folder_path(sim) / self.sim_get_stage(sim,
                                                              0).adjfn(self.sim_get_param(sim,
                                                                                          "trajectory_file"))
        raise NoStageTrajError(self.sim_get_stage(sim, 0),
                               sim,
                               str(trajpath))

    def sim_num_stages(self, sim: PatchySimulation) -> int:
        return len(self.sim_get_stages(sim))

    def sim_stage_done(self, sim: PatchySimulation, stage: Stage) -> bool:
        """
        similar to get_last_step but returns a boolean: true if stage traj exists and is correct length, false otherwise
        """
        _, traj = self.sim_get_stage_top_traj(sim, stage)
        if not traj.is_file():
            return False
        else:
            # return timepoint of last conf in traj
            return file_info([str(traj)])["t_end"][0] == stage.end_time()

    def num_patch_types(self, sim: PatchySimulation) -> int:
        """
        Returns: the total number of patches in the simulation
        """
        return self.particle_set.num_patches() * self.sim_get_param(sim, NUM_TEETH_KEY)

    def ensemble(self) -> list[PatchySimulation]:
        """
        Returns a list of all simulations in this ensemble
        """
        return [PatchySimulation(e) for e in itertools.product(*self.ensemble_params)]

    def num_ensemble_parameters(self) -> int:
        return len(self.ensemble_params)

    def get_ensemble_parameter(self, ens_param_name: str) -> EnsembleParameter:
        """
        Return
            the EnsembleParameter object with the provided name
        """
        param_match = [p for p in self.ensemble_params if p.param_key == ens_param_name]
        assert len(param_match) == 1, "ensemble parameter name problem bad bad bad!!!"
        return param_match[0]

    def tld(self) -> Path:
        return simulation_run_dir() / self.long_name()

    def folder_path(self, sim: PatchySimulation) -> Path:
        return self.tld() / sim.get_folder_path()

    def get_analysis_step(self, step: Union[str, AnalysisPipelineStep]) -> AnalysisPipelineStep:
        """
        Alias for PatchySimulationEnsemble.get_pipeline_step
        """
        return self.get_pipeline_step(step)

    def get_pipeline_step(self, step: Union[str, AnalysisPipelineStep]) -> AnalysisPipelineStep:
        """
        Returns a step in the analysis pipeline
        """
        return self.analysis_pipeline.get_pipeline_step(step)

    def link_analysis_pipeline(self, other: PatchySimulationEnsemble):
        """
        links this ensembles's analysis pipeline to another ensemble
        """
        if len(self.analysis_pipeline) == 0:
            self.get_logger().warning("Error: should not link from existing analysis pipeline! "
                                      "Use `clear_analysis_pipeline() to clear pipeline and try again.")
            return
        self.analysis_pipeline = other.analysis_pipeline
        self.analysis_file = other.analysis_file
        self.dump_metadata()

    def time_length(self,
                    sim: Union[PatchySimDescriptor,
                               list[PatchySimDescriptor],
                               None] = None
                    ) -> int:
        """
        Returns the length of a simulation, in steps
        """

        if sim is None:
            return self.time_length(self.ensemble())
        elif isinstance(sim, list):
            return min([self.time_length(s) for s in sim])
        else:
            stage = self.sim_most_recent_stage(sim)
            return self.sim_get_stage_last_step(sim, stage)

    # ------------------------ Status-Type Stuff --------------------------------#
    def info(self, infokey: str = "all"):
        """
        prints help text, for non-me people or if I forget
        might replace later with pydoc
        """
        print(f"Ensemble of simulations of {self.export_name} set up on {self.sim_init_date.strftime('%Y-%m-%d')}")
        print(f"Particle info: {str(self.particle_set)}")
        print(f"Metadata stored in file {self.metadata_file}")
        print(f"Simulation TLD: {self.tld()}")
        print("Ensemble Params")
        for param in self.ensemble_params:
            print("\t" + str(param))
        print(f"Const Simulation Params")
        for param in self.const_params:
            if param.is_grouped_params():
                print(f"\t{param.param_name}")
                for name in param.group_params_names():

            else:
                print(f"\t{param.param_name}: {param.value_name}")

        if len(self.analysis_pipeline) > 0:
            print(
                f"Has analysis pipeline with {self.analysis_pipeline.num_pipeline_steps()} steps and {self.analysis_pipeline.num_pipes()} pipes.")
            print(f"Pipeline is saved in file {self.analysis_file}")
            print(f"Pipeline steps")

        if len(self.analysis_data) > 0:
            print(f"Has {len(self.analysis_data)} entries of analysis data loaded (each entry is data for a specific "
                  f"analysis step and simulation)")

        # print("\nHelpful analysis functions:")
        # print("Function `has_pipeline`")
        # print("\ttell me if there's an analysis pipeline")
        # print("Function `show_pipeline_graph`")
        # print("missing_analysis_data")
        # print("\tdisplay a visual representation of the analysis pipeline graph")
        # print("Function `show_analysis_status`")
        # print("\tdisplay the status of the analysis pipeline")
        # print("Function `ensemble`")
        # print("Function `show_last_conf`")
        # print("Function `all_folders_exist`")
        # print("Function `folder_path`")
        # print("Function `tld`")
        # print("Function `list_folder_files`")

    def has_pipeline(self) -> bool:
        return len(self.analysis_pipeline) != 0

    def show_analysis_pipeline(self):
        return self.analysis_pipeline.draw_pipeline()

    # def babysit(self):
    #     """
    #     intermittantly checks whether any simulations have been stopped by the slurm
    #     controller before completion
    #     if it finds any, it starts the simulation again
    #     """
    #     finished = False
    #     # loop until all simulations are complete
    #     while not finished:
    #         # sleep until refresh
    #         time.sleep(get_babysitter_refresh())
    #         # find stopped simulations
    #         to_reup = self.get_stopped_sims()
    #         self.get_logger().info(f"Found {len(to_reup)} stopped simulations.")
    #         if len(to_reup) == 0:
    #             self.get_logger().info("All simulations complete. Babysitter exiting.")
    #             finished = True
    #         else:
    #             for sim in to_reup:
    #                 self.get_logger().info(f"Re-upping simulation {str(sim)}")
    #                 self.write_continue_files(sim)
    #                 self.exec_continue(sim)
    #         self.dump_metadata()

    def show_last_conf(self, sim: Union[PatchySimulation, None] = None, **kwargs):
        """
        Displays the final configuration of a simulation
        """
        if len(kwargs) > 0:
            self.sim_get_param(self.get_simulation(**kwargs))
        else:
            assert sim is not None, "No simulation provided!"
            self.show_conf(sim, self.time_length(sim))
            # from_path(self.paramfile(sim, "lastconf_file"),
            #           self.paramfile(sim, "topology"),
            #           self.folder_path(sim) / "particles.txt",
            #           self.folder_path(sim) / "patches.txt")

    def get_scene(self, sim: PatchySimulation, stage: Union[Stage, str]):
        if isinstance(stage, str):
            stage = self.sim_get_stage(sim, stage)

        top_file, traj_file = self.sim_get_stage_top_traj(sim, stage)
        return self.writer.read_scene(top_file,
                                      traj_file,
                                      to_PL(self.particle_set,
                                            self.sim_get_param(sim, NUM_TEETH_KEY),
                                            self.sim_get_param(sim, DENTAL_RADIUS_KEY)))
        # scene: PLPSimulation()

    def get_conf(self, sim: PatchySimulation, timepoint: int) -> Configuration:
        """
        Returns:
            a Configuration object showing the conf of the given simulation at the given timepoint
        """
        assert self.time_length(sim) >= timepoint, f"Specified timepoint {timepoint} exceeds simulation length" \
                                                   f"{self.time_length(sim)}"
        if timepoint > self.sim_get_param(sim, "print_conf_interval"):
            # this means that we're dealing with tidxs not step numbers
            return self.get_conf(sim, int(timepoint / self.sim_get_param(sim, "print_conf_interval")))
        else:
            stage = self.sim_get_timepoint_stage(sim, timepoint)
            # it's possible there's a better way to do this
            top_file, traj_file = self.sim_get_stage_top_traj(sim, stage)

            top_info, traj_info = describe(str(top_file), str(traj_file))
            # read only the conf we're looking for
            conf = get_confs(
                traj_info=traj_info,
                top_info=top_info,
                start_conf=timepoint,
                n_confs=1
            )[0]
            assert conf is not None
            return conf

    def show_conf(self, sim: PatchySimulation, timepoint: int):
        conf = self.get_conf(sim, timepoint)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".conf") as temp_conf:
            write_conf(temp_conf.name, conf)  # skip velocities for speed
            from_path(temp_conf.name,
                      self.paramfile(sim, "topology"),
                      self.folder_path(sim) / "particles.txt",
                      self.folder_path(sim) / "patches.txt")

    def analysis_status(self) -> pd.DataFrame:
        """
        Returns a Pandas dataframe showing the status of every simulation in the ensemble
        at each step on the analysis pipeline
        """
        return pd.DataFrame.from_dict({
            tuple(v.value_name for v in sim.param_vals):
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
        elif not isinstance(observable, PatchySimObservable):
            print("You definately forgot to put the observable first again. Gonna stop before you do any more damage")
            return
        if simulation_selector is None:
            simulation_selector = self.ensemble()
        if isinstance(simulation_selector, list):
            for sim in simulation_selector:
                self.dna_analysis(observable, sim)
        else:
            if conf_file_name is None:
                conf_file_name = self.sim_get_param(simulation_selector, "trajectory_file")
            self.write_setup_files(simulation_selector,
                                   "input_dna_analysis",
                                   {
                                       "conf_file": conf_file_name
                                   }, analysis=True)
            server_config = get_server_config()

            # write slurm script
            with open(self.folder_path(simulation_selector) / "dna_analysis.sh", "w+") as slurm_file:
                # bash header

                self.write_sbatch_params(simulation_selector, slurm_file)

                # skip confGenerator call because we will invoke it directly later
                slurm_file.write(f"{server_config['oxdna_path']}/build/bin/DNAnalysis input_dna_analysis\n")

            self.bash_exec(f"chmod u+x {self.folder_path(simulation_selector)}/slurm_script.sh")
            self.start_simulation(simulation_selector,
                                  script_name="dna_analysis.sh",
                                  job_type="dna_analysis")
            self.dump_metadata()

    def missing_analysis_data(self, step: Union[AnalysisPipelineStep, str]) -> pd.DataFrame:
        if isinstance(step, str):
            return self.missing_analysis_data(self.analysis_pipeline.name_map[step])
        else:
            return ~self.analysis_status().loc[~self.analysis_status()[step.name]]

    # def merge_topologies(self,
    #                      sim_selector: Union[None, PatchySimulation, list[PatchySimulation]] = None,
    #                      topologies: Union[list[int], None] = None,
    #                      out_file_name: Union[str, None] = None):
    #     """
    #     Merges some topology files
    #     """
    #     if sim_selector is None:
    #         sim_selector = self.ensemble()
    #     if isinstance(sim_selector, list):
    #         for sim in sim_selector:
    #             self.merge_topologies(sim, topologies, out_file_name)
    #     else:
    #         # if no topology file specified
    #         if topologies is None:
    #             topologies = [f for f in self.folder_path(sim_selector).iterdir() if
    #                           re.match(r"trajectory_\d+\.dat", f.name)]
    #             topologies = sorted(topologies,
    #                                 key=lambda f: int(re.search(r'trajectory_(\d+)\.dat', f.param_name).group(1)))
    #         if out_file_name is None:
    #             out_file_name = self.folder_path(sim_selector) / "full_trajectory.dat"
    #
    #         #
    #         self.bash_exec(f"cat {' '.join(map(str, topologies))} > {str(out_file_name)}")

    def list_folder_files(self, sim: PatchySimulation):
        print([p.name for p in self.folder_path(sim).iterdir()])

    # ----------------------- Setup Methods ----------------------------------- #
    def do_setup(self, sims: Union[list[PatchySimulation], None] = None, stage: Union[None, str, Stage] = None):
        if sims is None:
            sims = self.ensemble()
        self.get_logger().info("Setting up folder / file structure...")
        for sim in sims:
            assert self.sim_get_param(sim, "print_conf_interval") < self.sim_get_param(sim, "steps")
            self.get_logger().info(f"Setting up folder / file structure for {repr(sim)}...")
            # create nessecary folders
            if not os.path.isdir(self.folder_path(sim)):
                self.get_logger().info(f"Creating folder {self.folder_path(sim)}")
                Path(self.folder_path(sim)).mkdir(parents=True)
            else:
                self.get_logger().info(f"Folder {self.folder_path(sim)} already exists. Continuing...")

            # write requisite top, patches, particles files
            self.get_logger().info("Writing .top, .txt, input, etc. files...")
            self.write_setup_files(sim, stage)
            # write observables.json if applicble
            if EXTERNAL_OBSERVABLES:
                self.get_logger().info("Writing observable json, as nessecary...")
                self.write_sim_observables(sim)
            # write .sh script
            self.get_logger().info("Writing sbatch scripts...")
            # self.write_confgen_script(sim)
            self.write_run_script(sim)

    def write_confgen_script(self, sim: PatchySimulation):
        with open(self.get_run_confgen_sh(sim), "w+") as confgen_file:
            self.write_sbatch_params(sim, confgen_file)
            confgen_file.write(
                f"{get_server_config()['oxdna_path']}/build/bin/confGenerator input {self.sim_get_param(sim, 'density')}\n")
        self.bash_exec(f"chmod u+x {self.get_run_confgen_sh(sim)}")

    def write_run_script(self, sim: PatchySimulation, input_file="input"):
        # if no stage name provided use first stage
        try:
            stage = self.sim_most_recent_stage(sim)
            if stage.idx() + 1 < self.sim_num_stages(sim):
                stage = self.sim_get_stage(sim, stage.idx() + 1)
            else:
                self.get_logger().info(f"Simulation {sim} has no more stages to execute!")
                return -1
        except NoStageTrajError:
            stage = self.sim_get_stage(sim, 0)

        slurm_script_name = "slurm_script.sh"
        slurm_script_name = stage.adjfn(slurm_script_name)

        input_file = stage.adjfn(input_file)

        server_config = get_server_config()

        # write slurm script
        with open(self.folder_path(sim) / slurm_script_name, "w+") as slurm_file:
            # bash header

            self.write_sbatch_params(sim, slurm_file)

            # skip confGenerator call because we will invoke it directly later
            slurm_file.write(f"{server_config['oxdna_path']}/build/bin/oxDNA {input_file}\n")

        self.bash_exec(f"chmod u+x {self.folder_path(sim)}/{slurm_script_name}")

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
        slurm_file.write(f"#SBATCH -o run%j.out\n")
        # slurm_file.write(f"#SBATCH -e run{run_oxdna_counter}_%j.err\n")

        # slurm includes ("module load xyz" and the like)
        for line in server_config["slurm_includes"]:
            slurm_file.write(line + "\n")

    def write_setup_files(self,
                          sim: PatchySimulation,
                          stage: Union[str, Stage, None] = None,
                          replacer_dict: Union[dict, None] = None,
                          extras: Union[dict, None] = None,
                          analysis: bool = False):
        """
        Writes any/all nessecary files
        """

        if stage is None:
            try:
                # get most recent stage
                stage = self.sim_most_recent_stage(sim)
            # if no stage exists
            except NoStageTrajError:
                # stage 0
                stage = self.sim_get_stages(sim)[0]
            else:
                # find next stage and if it's one
                stage, done = stage
                if not done:
                    # if mid-stage
                    self.get_logger().error(f"{stage.name()} incomplete!")
                    return
                stages = self.sim_get_stages(sim)
                if stage.idx() + 1 != len(stages):
                    stage = stages[stage.idx() + 1]
                else:
                    self.get_logger().info(f"Final stage {stage.name()} is already complete!")
        elif isinstance(stage, str):
            stage = self.sim_get_stage(sim, stage)

        if extras is None:
            extras = {}
        # set writer directory to simulation folder path
        self.writer.set_write_directory(self.folder_path(sim))

        # get server config
        if replacer_dict is None:
            replacer_dict = {}
        server_config = get_server_config()

        # if this is the first conf
        if stage.idx() == 0:
            assert stage.start_time() == 0, "This method should not be invoked for stages other than the first!"

            # generate conf
            scene = PLPSimulation()
            particle_set = to_PL(self.particle_set,
                                 self.sim_get_param(sim, NUM_TEETH_KEY),
                                 self.sim_get_param(sim, DENTAL_RADIUS_KEY))
            # patches will be added automatically
            scene.set_particle_types(particle_set)

        else:
            last_complete_stage, isdone = self.sim_most_recent_stage(sim)
            assert isdone
            scene = self.get_scene(sim, last_complete_stage)

        stage.apply(scene)

        # grab args required by writer
        reqd_extra_args = {
            a: self.sim_get_param(sim, a) for a in self.writer.reqd_args()
        }
        assert "conf_file" in reqd_extra_args
        # update file names
        # if applciable, particles.txt and patches.txt should not change!
        # if stage.idx() > 0:
        #     reqd_extra_args["conf_file"] = append_to_file_name(reqd_extra_args["conf_file"], stage.name())
        #     reqd_extra_args["topology"] = append_to_file_name(reqd_extra_args["topology"], stage.name())

        # write top, conf, and others
        files = self.writer.write(scene,
                                  stage,
                                  **reqd_extra_args)

        # update top and dat files in replacer dict
        replacer_dict.update(files)
        replacer_dict["steps"] = stage.end_time()
        replacer_dict["trajectory_file"] = stage.adjfn(self.sim_get_param(sim, "trajectory_file"))
        extras.update(self.writer.get_input_file_data(scene, **reqd_extra_args))

        input_file_name = stage.adjfn("input")

        # create input file
        with open(self.folder_path(sim) / input_file_name, 'w+') as inputfile:
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
            # write extras
            for key, val in extras.items():
                if key in replacer_dict:
                    val = replacer_dict[key]
                inputfile.write(f"{key} = {val}\n")

            # writerargs = self.writer.get_input_file_data(self.particle_set, {
            #     p.type_id(): self.get_sim_particle_count(sim, p.type_id()) for p in self.particle_set.particles()
            # }, **{
            #     a: self.sim_get_param(sim, a) for a in self.writer.reqd_args()
            # })
            # for key, val in writerargs:
            #     inputfile.write(f"{key} = {val}\n")

            # deprecated in favor of patchyio
            # # if josh_flavio or josh_lorenzo
            # if server_config[PATCHY_FILE_FORMAT_KEY].find("josh") > -1:
            #     patch_file_info = [
            #         ("patchy_file", "patches.txt"),
            #         ("particle_file", "particles.txt"),
            #         ("particle_types_N", self.num_particle_types()),
            #         ("patch_types_N", self.num_patch_types(sim))
            #     ]
            #     for key, val in patch_file_info:
            #         if key in replacer_dict:
            #             val = replacer_dict[key]
            #         inputfile.write(f"{key} = {val}\n")
            # elif server_config[PATCHY_FILE_FORMAT_KEY] == "lorenzo":
            #     key = "DPS_interaction_matrix_file"
            #     val = "interactions.txt"
            #     if key in replacer_dict:
            #         val = replacer_dict[key]
            #     inputfile.write(f"{key} = {val}\n")
            # else:
            #     # todo: throw exception
            #     pass

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
                        obsrv.write_input(inputfile, i, stage, analysis)
        # return {
        #     "input": input_file_name,
        #     **files
        # }

    #
    # def write_sim_top_particles_patches(self, sim: PatchySimulation):
    #     """
    #     Writes the topology file (.top) and the files speficying particle
    #     and patch behavior for a simulation in the ensemble
    #     This method was written to be called from `do_setup()` and it is not
    #     recommended to be used in other contexts
    #     """
    #

    # deleted in favor of patchy io
    # server_config = get_server_config()
    #
    # # write top and particles/patches spec files
    # # first convert particle json into PLPatchy objects (cf plpatchy.py)
    # particles, patches = to_PL(self.particle_set,
    #                            self.sim_get_param(sim, NUM_TEETH_KEY),
    #                            self.sim_get_param(sim, DENTAL_RADIUS_KEY))
    #
    # # do any/all valid conversions
    # # either josh_lorenzo or josh_flavio
    # if server_config[PATCHY_FILE_FORMAT_KEY].find("josh") > -1:
    #     # write top file
    #     with open(self.folder_path(sim) / "init.top", "w+") as top_file:
    #         # first line of file
    #         top_file.write(f"{self.get_sim_total_num_particles(sim)} {len(particles)}\n")
    #         top_file.write(" ".join([
    #             f"{i} " * self.get_sim_particle_count(sim, i) for i in range(len(particles))
    #         ]))
    #     # write patches.txt and particles.txt
    #     with open(self.folder_path(sim) / "patches.txt", "w+") as patches_file, open(
    #             self.folder_path(sim) / "particles.txt", "w+") as particles_file:
    #         for particle_patchy, cube_type in zip(particles, self.particle_set.particles()):
    #             # handle writing particles file
    #             for i, patch_obj in enumerate(particle_patchy.patches()):
    #                 # we have to be VERY careful here with indexing to account for multidentate simulations
    #                 # adjust for patch multiplier from multidentate
    #                 polycube_patch_idx = int(i / self.sim_get_param(sim, NUM_TEETH_KEY))
    #
    #                 extradict = {}
    #                 # if this is the "classic" format
    #                 if server_config[PATCHY_FILE_FORMAT_KEY] == "josh_flavio":
    #                     allo_conditional = cube_type.patch_conditional(
    #                         cube_type.get_patch_by_idx(polycube_patch_idx), minimize=True)
    #                     # allosteric conditional should be "true" for non-allosterically-controlled patches
    #                     extradict = {"allostery_conditional": allo_conditional if allo_conditional else "true"}
    #                 else:  # josh/lorenzo
    #                     # adjust for patch multiplier from multiparticale_patchesdentate
    #                     state_var = cube_type.get_patch_by_diridx(polycube_patch_idx).state_var()
    #                     activation_var = cube_type.get_patch_by_diridx(polycube_patch_idx).activation_var()
    #                     extradict = {
    #                         "state_var": state_var,
    #                         "activation_var": activation_var
    #                     }
    #                 patches_file.write(patch_obj.save_to_string(extradict))
    #
    #             if server_config[PATCHY_FILE_FORMAT_KEY] == "josh_flavio":
    #                 particles_file.write(particle_patchy.save_type_to_string())
    #             else:  # josh/lorenzo
    #                 particles_file.write(
    #                     particle_patchy.save_type_to_string({"state_size": cube_type.state_size()}))
    #
    # else:  # lorenzian
    #     with open(self.folder_path(sim) / "init.top", "w+") as top_file:
    #         top_file.write(f"{self.get_sim_total_num_particles(sim)} {len(particles)}\n")
    #         # export_to_lorenzian_patchy_str also writes patches.dat file
    #         top_file.writelines([
    #             particle.export_to_lorenzian_patchy_str(self.get_sim_particle_count(sim,
    #                                                                                 particle.type_id()),
    #                                                     self.folder_path(sim))
    #             + "\n"
    #             for particle in particles])
    #     export_interaction_matrix(patches)

    def write_sim_observables(self, sim: PatchySimulation):
        if len(self.observables) > 0:
            with open(self.folder_path(sim) / "observables.json", "w+") as f:
                json.dump({f"data_output_{i + 1}": obs.to_dict() for i, obs in enumerate(self.observables.values())}, f)

    def gen_confs(self):
        """
        DEPRECATED
        """
        for sim in self.ensemble():
            self.gen_conf(sim)
        # run_confgen does NOT dump metadata
        self.dump_metadata()

    def dump_metadata(self):
        """
        Saves metadata stored in `self.metadata` to a metadata file
        Also saves the analysis pathway
        """
        self.metadata["slurm_log"] = self.slurm_log.to_list()
        self.metadata["analysis_file"] = self.analysis_file
        # dump metadata dict to file
        with open(get_input_dir() / self.metadata_file, "w") as f:
            json.dump(self.metadata, fp=f, indent=4)
        # dump analysis pipevline as pickle
        # if "analysis_file" not in self.metadata:
        #     self.metadata["analysis_file"] = get_input_dir() / (self.export_name + "_analysis_pipeline.pickle")
        with open(get_input_dir() / self.analysis_file, "wb") as f:
            pickle.dump(self.analysis_pipeline, f)

    def continue_simulation(self, sims: Union[list[PatchySimulation], PatchySimulation, None]):
        if sims is None:
            self.continue_simulation(sims)
        elif isinstance(sims, list):
            for sim in sims:
                self.continue_simulation(sim)
        else:
            pass

    def start_simulations(self, e: Union[None, list[PatchySimulation]] = None, stage: Union[str, None] = None):
        """
        Starts all simulations
        """
        if e is None:
            e = self.ensemble()
        for sim in e:
            self.start_simulation(sim, stage_name=stage)
        self.dump_metadata()

    def ok_to_run(self, sim: PatchySimulation, stage: Stage) -> bool:
        try:
            most_recent_stage = self.sim_most_recent_stage(sim)
            if most_recent_stage.idx() >= stage.idx():
                self.get_logger().warning(
                    f"Already passed stage {stage.name()} for sim {repr(sim)}! Aborting")
                return False
            elif stage.idx() - most_recent_stage.idx() > 1:
                self.get_logger().warning(
                    f"Cannot exec stage {stage.name()} for sim {repr(sim)} when most recent stage is "
                    f"{most_recent_stage.name()}! Aborting")
                return False
            else:
                assert stage.idx() - most_recent_stage.idx() == 1
        except IncompleteStageError as e:
            # previous stage is incomplete; warn and return False
            self.get_logger().warning(
                f"Cannot execute stage {stage.name()} for sim {repr(sim)} when stage {e.stage().name()}"
                f" is incomplete! aborting!")
            return False
        except NoStageTrajError as e:
            # if most recent stage has no traj, that means current stage is first stage, which is completely fine
            pass

        if stage.idx() > 0 and not stage.name().startswith("continue"):
            # if not first stage
            if not self.sim_stage_done(sim, self.sim_get_stage(sim, stage.idx() - 1)):
                # if previous stage is incomplete
                self.get_logger().warning(f"Stage {stage.name()} for sim {repr(sim)} "
                                          f"cannot execute because stage "
                                          f"{self.sim_get_stage(sim, stage.idx() - 1).name()} "
                                          f"is incomplete")
                return False

        if is_server_slurm():
            most_recent_stage = self.sim_most_recent_stage(sim)
            if most_recent_stage.idx() > stage.idx():
                self.get_logger().warning(f"Already passed stage {stage.name()} for sim {repr(sim)}! Aborting")
                return False
            # include some extra checks to make sure we're not making a horrible mistake
            if not self.slurm_log.by_subject(sim).by_type("oxdna").by_other("stage", stage.name()).empty():
                # if slurm log shows a job with this sim, job type, and stage already exists
                jid = self.slurm_log.by_subject(sim).by_type("oxdna").by_other("stage", stage.name())[
                    0].job_id
                job_info = self.slurm_job_info(jid)
                if job_info is not None and job_info["JobState"] == "RUNNING":
                    # if job is currently running
                    logging.warning(
                        f"Already running job for sim {repr(sim)}, stage {stage.name()} (jobid={jid}! Skipping...")
                    return False
        return True

    def start_simulation(self,
                         sim: PatchySimulation,
                         script_name: str = "slurm_script.sh",
                         stage_name: Union[None, str] = None,
                         is_analysis: bool = False,
                         force_ignore_ok_check=False,
                         retries=3,
                         backoff_factor=2) -> int:
        """
        Starts an oxDNA simulation. direct invocation is not suggested;
        use `start_simulations` instead
        Parameters:
             sim : simulation to start
             script_name : the name of the slurm script file
             job_type : the label of the job type, for logging purposes
        """

        if stage_name is not None:
            stage = self.sim_get_stage(sim, stage_name)
        else:
            # if no stage name provided use first stage
            try:
                stage = self.sim_most_recent_stage(sim)
                if stage.idx() + 1 < self.sim_num_stages(sim):
                    stage = self.sim_get_stage(sim, stage.idx() + 1)
                else:
                    self.get_logger().info(f"Simulation {sim} has no more stages to execute!")
                    return -1
            except NoStageTrajError:
                # default to stage 0 if no previous stages found
                stage = self.sim_get_stage(sim, 0)

        if not force_ignore_ok_check and not self.ok_to_run(sim, stage):
            self.get_logger().warning(f"Stage {stage.name()} not ok to run for sim {repr(sim)}")
            return -1

        # get slurm log jobname
        if not is_analysis:
            job_type_name = "oxdna"
        else:
            job_type_name = "analysis"
        script_name = stage.adjfn(script_name)

        if is_server_slurm():
            command = f"sbatch --chdir={self.folder_path(sim)}"
        # for non-slurm servers
        else:
            command = f"bash {script_name} > simulation.log"

        # shouldn't be nessecary anymore but whatever
        if not self.paramstagefile(sim, stage, "conf_file").exists():
            confgen_slurm_jobid = self.gen_conf(sim)
            if is_server_slurm():
                command += f" --dependency=afterok:{confgen_slurm_jobid}"
        if is_server_slurm():
            command += f" {script_name}"
        submit_txt = ""
        for i in range(retries):
            submit_txt = self.bash_exec(command, is_async=not is_server_slurm(), cwd=self.folder_path(sim))
            if submit_txt:
                break
            time.sleep(backoff_factor ** i)

        if is_server_slurm():
            if not submit_txt:
                raise Exception(f"Submit slurm job failed for simulation {sim}")

            jobid = int(re.search(SUBMIT_SLURM_PATTERN, submit_txt).group(1))
            self.append_slurm_log(SlurmLogEntry(
                job_type=job_type_name,
                pid=jobid,
                simulation=sim,
                script_path=self.folder_path(sim) / script_name,
                log_path=self.folder_path(sim) / f"run{jobid}.out"
            ))
            return jobid
        else:
            return -1

    # def get_run_oxdna_sh(self, sim: PatchySimulation) -> Path:
    #     """
    #
    #     """
    #     return self.folder_path(sim) / "slurm_script.sh"

    def get_run_confgen_sh(self, sim: PatchySimulation) -> Path:
        return self.folder_path(sim) / "gen_conf.sh"

    def gen_conf(self, sim: PatchySimulation) -> int:

        """
        Runs a conf generator. These are run as slurm jobs if you're on a slurm server,
        or as non-background tasks otherwise
        """
        if is_server_slurm():
            response = self.bash_exec(f"sbatch --chdir={self.folder_path(sim)} {self.folder_path(sim)}/gen_conf.sh")
            jobid = int(re.search(SUBMIT_SLURM_PATTERN, response).group(1))
            self.append_slurm_log(SlurmLogEntry(
                pid=jobid,
                simulation=sim,
                job_type="confgen",
                script_path=self.folder_path(sim) / "gen_conf.sh",
                log_path=self.folder_path(sim) / f"run{jobid}.out"
            ))
            return jobid
        else:
            self.bash_exec(f"bash gen_conf.sh > confgenlog.out", cwd=self.folder_path(sim))
            # jobid = re.search(r'\[\d+\]\s+(\d+)', response).group(1)
            # slurm logs aren't valid when not on a slurm server
            return -1

    def get_conf_file(self, sim: PatchySimulation) -> Path:
        return self.folder_path(sim) / "init.conf"

    # ------------- ANALYSIS FUNCTIONS --------------------- #
    def clear_pipeline(self):
        """
        deletes all steps from the analysis pipeline
        """
        self.analysis_pipeline = AnalysisPipeline()

    def add_analysis_steps(self, *args):
        """
        Adds steps to the analysis pipeline
        """
        if isinstance(args[0], AnalysisPipeline):
            new_steps = args[0]
        else:
            new_steps = AnalysisPipeline(args[0], *args[1:])
            # if the above line didn't work
        newPipes = [a for a in args if isinstance(a, tuple)]
        newSteps = [a for a in args if issubclass(type(a), AnalysisPipelineStep)]
        if new_steps.num_pipes() != len(newPipes) or new_steps.num_pipeline_steps() != len(newSteps):
            self.analysis_pipeline = self.analysis_pipeline.extend(newSteps, newPipes)
            self.dump_metadata()
        elif new_steps not in self.analysis_pipeline:
            self.get_logger().info(f"Adding {len(new_steps)} steps "
                                   f"and {len(new_steps.pipeline_graph.edges)} pipes to the analysis pipeline")
            self.analysis_pipeline = self.analysis_pipeline + new_steps
            self.dump_metadata()
        else:
            self.get_logger().info("The analysis pipeline you passed is already present")

    def has_data_file(self, step: PipelineStepDescriptor, sim: PatchySimDescriptor) -> bool:
        return self.get_cache_file(step, sim).exists()

    def param_value_valid(self, pv: ParameterValue):
        return any([pv in ep for ep in self.ensemble_params])

    def is_multiselect(self, selector: PatchySimDescriptor, exceptions: tuple[EnsembleParameter, ...] = ()) -> bool:
        """
        Returns true if the provided selector will match multiple PatchySimulation objects, false otherwise
        """
        if isinstance(selector, PatchySimulation):
            return False
        try:
            assert all([self.param_value_valid(pv) for pv in selector]), f"Invalid selector {selector}"
            # if all provided items in selector are either in the ensemble parameters or are used in aggregation
            if len(selector) + len(exceptions) == self.num_ensemble_parameters():
                return False
            assert len(selector) + len(
                exceptions) < self.num_ensemble_parameters(), f"Too many specifiers found between {selector} and {exceptions} (expected {self.num_ensemble_parameters()} specifiers)"
            return True
        except TypeError as e:
            raise Exception(f"{selector} is not iterable!")

    def get_data(self,
                 step: PipelineStepDescriptor,
                 sim: Union[PatchySimDescriptor, None] = None,
                 time_steps: range = None) -> Union[PipelineData, list[PipelineData]]:
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

        # preprocessing: MAKE SURE THIS IS A STEP OBJECT
        step = self.get_pipeline_step(step)

        #  if we've provided a list of simulations
        if sim is None:
            return self.get_data(step, tuple(), time_steps)

        if isinstance(sim, list):
            if self.is_do_analysis_parallel():
                self.get_logger().info(f"Assembling pool of {self.n_processes()} processes")
                with multiprocessing.Pool(self.n_processes()) as pool:
                    args = [(self, step, s, time_steps) for s in sim]
                    return pool.map(process_simulation_data, args)
            else:
                return [self.get_data(step, s, time_steps) for s in sim]

        # DATA AGGREGATION!!!
        # second thing: check if the provided simulation selector is incomplete, a
        # nd that this isn't an aggregate step (which expects incomplete selectors)
        if (isinstance(step, AggregateAnalysisPipelineStep) and self.is_multiselect(sim, step.params_aggregate_over)) \
                or (not isinstance(step, AggregateAnalysisPipelineStep) and self.is_multiselect(sim)):
            # if the simulation selector provided doesn't line up with the aggregation params
            # (or the step isn't an aggregation step), this method will return a grouping of results. somehow.
            simulations_list: list[PatchySimulation] = self.get_simulation(*sim)
            # get data for each simulation
            data: list[PipelineData] = self.get_data(step, simulations_list, time_steps)
            # ugh wish python had switch/case
            if step.get_output_data_type() == PipelineDataType.PIPELINE_DATATYPE_DATAFRAME:
                self.get_logger().info(f"Merging data from simulations: {sim}")
                timerange = np.array(list(set.intersection(*[set(s.trange()) for s in data])))
                # iter simulation data
                for s, sim_data in zip(simulations_list, data):  # I'm GREAT at naming variables!
                    for p in s.param_vals:
                        sim_data.get()[p.param_name] = p.value_name
                df = pd.concat([d.get() for d in data], axis=0)
                df = df.loc[np.isin(df[TIMEPOINT_KEY], timerange)]
                return PDPipelineData(df, timerange)
            elif step.get_output_data_type() == PipelineDataType.PIPELINE_DATATYPE_GRAPH:
                raise Exception("I haven't bothered trying to join graph data yet")
            else:
                raise Exception("Attempting to merge non-mergable data type")

        # check if this is a slurm job (should always be true I guess? even if it's a jupyter notebook)
        if is_slurm_job():
            slurm_job_info = self.slurm_job_info()
            self.append_slurm_log(SlurmLogEntry(
                job_type="analysis",
                pid=int(slurm_job_info["JobId"]),
                simulation=sim,
                script_path=slurm_job_info["Command"],
                start_date=datetime.datetime.strptime(slurm_job_info["SubmitTime"], "%Y-%m-%dT%H:%M:%S"),
                log_path=slurm_job_info["StdOut"],
                additional_metadata={
                    "step": step.name
                }
            ))

        # TIMESTEPS!!!!
        # if timesteps were not specified
        if time_steps is None:
            # time steps generally do not start from step 0
            time_steps = range(step.output_tstep, self.time_length(sim), step.output_tstep)
            self.get_logger().info(f"Constructed time steps {time_steps}")
        else:
            assert time_steps.step % step.output_tstep == 0, f"Specified step interval {time_steps} " \
                                                             f"not consistant with {step} output time " \
                                                             f"interval {step.output_tstep}"

        self.get_logger().info(
            f"Retrieving data for analysis step {step.name} and simulation(s) {str(sim)} over timeframe {time_steps}")
        # DATA CACHING
        # check if data is already loaded
        if not self.is_nocache() and (step, sim,) in self.analysis_data and self.analysis_data[
            (step, sim)].matches_trange(time_steps):
            self.get_logger().info("Data already loaded!")
            return self.analysis_data[(step, sim)]  # i don't care enough to load partial data

        # check if we have cached data for this step already
        if not self.analysis_pipeline.is_force_recompute(step) and self.has_data_file(step, sim):
            self.get_logger().info(
                f"Cache file for simulation {get_descriptor_key(sim)} and step {step} exists! Loading...")
            cache_file_path = self.get_cache_file(step, sim)
            cached_data = step.load_cached_files(cache_file_path)
            # if we already have the data needed for the required time range
            if cached_data.matches_trange(time_steps):
                # that was easy!
                self.get_logger().info(f"All data in file! That was easy!")
                return cached_data
            else:
                self.get_logger().info(f"Cache file missing data!")
        if self.is_do_analysis_parallel():
            lock = multiprocessing.Lock()
            lock.acquire()
        try:
            # compute data for previous steps
            self.get_logger().info(f"Computing data for step {step.name} for simulation {str(sim)}...")
            data_in = self.get_step_input_data(step, sim, time_steps)
        finally:
            if self.is_do_analysis_parallel():
                lock.release()
        # TODO: make sure this can handle the amount of args we're feeding in here
        # execute the step!
        # TODO: handle existing data that's incomplete over the required time interval
        data: PipelineData = step.exec(*data_in)
        if self.is_do_analysis_parallel():
            lock.acquire()
        try:
            self.get_logger().info(f"Caching data in file `{self.get_cache_file(step, sim)}`")
            step.cache_data(data, self.get_cache_file(step, sim))
        finally:
            if self.is_do_analysis_parallel():
                lock.release()
        if not self.is_nocache():
            self.analysis_data[step, sim] = data
        return data

    def get_step_input_data(self,
                            step: PipelineStepDescriptor,
                            sim: PatchySimDescriptor,
                            time_steps: range) -> Union[list[PipelineData],
                                                        list[Union[
                                                            PatchySimulationEnsemble,
                                                            PatchySimulation,
                                                            list[Path]]]]:
        step = self.get_pipeline_step(step)
        # if this step is an aggregate, things get... complecated
        if isinstance(step, AggregateAnalysisPipelineStep):
            # compute the simulation data required for this step
            param_prev_steps = step.get_input_data_params(sim)
            return [
                self.get_data(prev_step,
                              param_prev_steps,
                              time_steps)
                for prev_step in self.analysis_pipeline.steps_before(step)
            ]
        # if this is a head node, it will take ensemble info, sim info, and
        # file paths instead of previous step data
        elif isinstance(step, AnalysisPipelineHead):
            assert isinstance(sim, PatchySimulation), "Analysis pipeline head nodes should only take single simulations"
            stages: list[Stage] = self.sim_get_stages_between(sim, time_steps.start, time_steps.stop)
            files = [self, sim, stages]
            for file_name_in in step.get_data_in_filenames():
                assert isinstance(file_name_in, str), f"Unexpected file_name_in parameter type {type(file_name_in)}." \
                                                   f"Regexes no longer supported."
                # add all files of this type, in an order corresponding to `stages`
                try:
                    # for files that are parameters (topology, conf, etc.)
                    file_name = self.sim_get_param(sim, file_name_in)
                except NoSuchParamError as e:
                    # for files that are observable outputs
                    file_name = file_name_in  # i don't think any modification is actually needed here?
                file_names = [self.folder_path(sim) / file_name for stage in stages]
                files.append(file_names)
            return files
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

    def slurm_job_info(self, jobid: int = -1, ignore_cache=False, retries=3, backoff_factor=2) -> Union[dict, None]:
        if jobid == -1:
            retr_id = os.environ.get("SLURM_JOB_ID")
            assert retr_id is not None
            jobid = int(retr_id)

        for i in range(retries):
            jobinfo = self.bash_exec(f"scontrol show job {jobid}")

            if jobinfo and jobinfo != "slurm_load_jobs error: Invalid job id specified":
                jobinfo = jobinfo.split()
                jobinfo = {key: value for key, value in [x.split("=", 1) for x in jobinfo if len(x.split("=", 1)) == 2]}
                # Cache it
                SLURM_JOB_CACHE[jobid] = jobinfo
                return jobinfo

            time.sleep(backoff_factor ** i)

        return None

    def bash_exec(self, command: str, is_async=False, cwd=None):
        """
        Executes a bash command and returns the output
        """
        self.get_logger().debug(f">`{command}`")
        if not is_async:
            response = subprocess.run(command,
                                      shell=True,
                                      capture_output=True,
                                      text=True,
                                      check=False)
        else:
            response = subprocess.Popen(command.split(), cwd=cwd)
        # response = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT, check=False,
        # universal_newlines=True)
        self.get_logger().debug(f"`{response.stdout}`")
        return response.stdout

    def sim_get_timepoint_stage(self, sim: PatchySimulation, timepoint: int) -> Union[Stage, None]:
        for stage in self.sim_get_stages(sim):
            if stage.start_time() < timepoint < stage.end_time():
                return stage
        return None  # raise exception?


class NoSuchParamError(Exception):
    def __init__(self,
                 e: PatchySimulationEnsemble,
                 sim: PatchySimulation,
                 param_name: str):
        self._ensemble = e
        self._sim = sim
        self._param_name = param_name

    def __str__(self):
        return f"No value specified for parameter \"{self._param_name}\" " \
               f"simulation {repr(self._sim)} in " \
               f"ensemble {self._ensemble.long_name()} "


def process_simulation_data(args: tuple[PatchySimulationEnsemble, AnalysisPipelineStep, PatchySimulation, range]):
    ensemble, step, s, time_steps = args
    return ensemble.get_data(step, s, time_steps)


def shared_ensemble(es: list[PatchySimulationEnsemble], ignores: set = set()) -> Union[
    list[list[PatchySimulation]], None]:
    """
    Computes the simulation specs that are shared between the provided ensembles
    Args:
        es: a list of simulation ensembles
        ignores: a set of const parameter names to ignore when constructing simulation overlap
    Returns:
    """

    names = set()
    name_vals: dict[str, set] = dict()
    for e in es:
        names.update([p.param_name for p in e.const_params if p.param_name not in ignores])
        for p in e.const_params:
            param_key = p.param_name
            if param_key in ignores:
                continue
            if param_key not in name_vals:
                name_vals[param_key] = {e.const_params[param_key]}
            else:
                name_vals[param_key] = name_vals[param_key].intersection([e.const_params[param_key]])
        names.update([p.param_key for p in e.ensemble_params])
        for p in e.ensemble_params:
            if p.param_key not in name_vals:
                name_vals[p.param_key] = {pv.value_name for pv in p.param_value_set}
            else:
                name_vals[p.param_key] = name_vals[p.param_key].intersection(
                    [pv.value_name for pv in p.param_value_set])

    # if any simulation is missing a parameter
    if not all([len(name_vals[name]) > 0 for name in names]):
        return None

    ensembles = []
    for e in es:
        valset: list[list[ParameterValue]] = []
        # make sure to order ensemble parameters correctly
        for p in e.ensemble_params:
            union_vals = name_vals[p.param_key]
            vals = [pv for pv in p if pv.value_name in union_vals]
            valset.append(vals)

        ensembles.append([PatchySimulation(sim) for sim in itertools.product(*valset)])

    return ensembles
