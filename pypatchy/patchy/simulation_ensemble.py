from __future__ import annotations

import copy
import datetime
import itertools
import multiprocessing
import random
import shutil
import tempfile
import time
from collections import Counter
from json import JSONDecodeError
from typing import Generator, Type, Optional, Any

import subprocess
import re
import logging

import oxpy
# import oat stuff
from oxDNA_analysis_tools.UTILS.oxview import from_path
from oxDNA_analysis_tools.file_info import file_info
from oxDNA_analysis_tools.UTILS.RyeReader import get_confs, describe, write_conf
from oxDNA_analysis_tools.UTILS.data_structures import Configuration

from ipy_oxdna.oxdna_simulation import Simulation
from ipy_oxdna.utils.observable import Observable, ObservableColumn
from .param_val_types import StageInfoParam
from .particle_adders import RandParticleAdder

from .pl.plparticle import PLPatchyParticle, PLPatch
from .pl.plparticleset import PLParticleSet
from .pl.plpotential import PLPotential, PLFRExclVolPotential, PLFRTorsionalPatchyPotential, PLLRPatchyPotential, \
    PLLRExclVolPotential, PLFRPatchyPotential
from .sim_ensemble_json import SimEnsembleJsonEncoder
from .simulation_specification import get_param_set
from ..analysis.analyzable import Analyzable

from ..patchy.patchy_scripts import lorenzian_to_flavian
from ..analysis.analysis_pipeline import AnalysisPipeline
from .pl.plscene import PLPSimulation
from .stage import Stage, NoStageTrajError, IncompleteStageError, StageTrajFileEmptyError
from ..analysis.analysis_data import PDPipelineData, TIMEPOINT_KEY, MissingDataError
from ..analysis.analysis_pipeline_step import *
from .patchy_sim_observable import PatchySimObservable, observable_from_file
from .pl.patchyio import get_writer, PLBaseWriter, FWriter
from ..server_config import load_server_settings, PatchyServerConfig, get_server_config
from ..slurm_log_entry import SlurmLogEntry
from ..slurmlog import SlurmLog
from ..util import *
from .ensemble_parameter import *
from .simulation_specification import PatchySimulation, ParamSet, NoSuchParamError
from .pl.plpatchylib import polycube_rule_to_PL, load_pl_particles
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
ENSEMBLE_SETUP_DATE = "setup_date"

METADATA_FILE_KEY = "sim_metadata_file"
LAST_CONTINUE_COUNT_KEY = "continue_count"

SUBMIT_SLURM_PATTERN = r"Submitted batch job (\d+)"

SERVER_SETTINGS_KEY = "server_settings"

def describe_param_vals(*args: ParameterValue) -> str:
    """
    describes parameters
    :param args: a list of parameter values
    :return: string desccripton of list of parameters
    """
    return "_".join([str(v) for v in args])


# i'm well beyond the point where i understand this type
PatchySimDescriptor = Union[tuple[Union[ParameterValue, tuple[str, Any]], ...],
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
    :return: a string representing the provided descriptor
    """
    return sim if isinstance(sim, str) else str(sim) \
        if isinstance(sim, PatchySimulation) else describe_param_vals(*sim)


def list_simulation_ensembles():
    """
    lists all simulation ensembles?
    """
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
    """
    tests if metadata exists for a given name and date
    :return: true if a metadata file exists for the provided ensemble name + date, false otherwise
    """
    return metadata_file(name, date).exists()


def metadata_file(name: str, date: datetime.datetime) -> Path:
    if name.endswith(".json"):
        name = name[:name.rfind(".")]
    return get_input_dir() / (name + "_" + date.strftime("%Y-%m-%d") + "_metadata.json")


def normalize_date(d):
    return d if isinstance(d, datetime.datetime) else datetime.datetime.strptime(d, "%Y-%m-%d")

class MissingEnembleSetupInfoError(BaseException):
    """
    custom exception for trying to construct a simulation ensemble without a required setup value (e.g. name, date)
    Note that this is distinct from issues with simulation parameters, which are handled by MissingSimParamError
    """

    __missing_info_key: str

    def __init__(self, missing_info_key: str):
        self.__missing_info_key = missing_info_key

    def __str__(self):
        return f"Missing ensemble setup key {self.__missing_info_key}"

class MissingSimParamError(BaseException):
    """
    Exception for missing simulation parameters (e.g. temperature, particle types, etc.
    """
    __missing_param_name: str

    def __init__(self, missing_name: str):
        self.__missing_param_name = missing_name

    def __str__(self):
        return f"Missing parameter {self.__missing_param_name}"

def find_ensemble(*args: str, **kwargs) -> PatchySimulationEnsemble:
    """
    External method to construct PatchySimulationEnsemble objects. nightmare,
    :param args: who knows
    :param kwargs: no seriously, DM me if you know
    :returns: a patchy simulation ensemble object
    """
    if "set_server" not in kwargs:
        kwargs["set_server"] = None

    # sometimes for custom pipeline we need custom read/write
    if "loader" not in kwargs:
        kwargs["loader"] = SimEnsembleJsonEncoder

    if len(args) > 0:
        simname = args[0]
        if len(args) == 2:
            sim_init_date: datetime.datetime = normalize_date(args[1])
        elif any([key in kwargs for key in ["sim_date", "date"]]):
            sim_init_date = normalize_date(
                [kwargs[key] for key in ["sim_date", "date"] if key in kwargs][0])
        else:
            return find_ensemble(name=simname,
                                 set_server=kwargs["set_server"])
        if not simname.endswith(".json"):
            return find_ensemble(name=simname,
                                 date=sim_init_date,
                                 set_server=kwargs["set_server"])
        elif metadata_file_exist(simname, sim_init_date):
            # if metadata file exists, load from it
            return find_ensemble(metadata=simname,
                                 date=sim_init_date,
                                 set_server=kwargs["set_server"])
        else:
            # load from cfg file
            return find_ensemble(cfg=simname,
                                 date=sim_init_date,
                                 set_server=kwargs["set_server"])

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
            try:
                cfg = json.load(cfg_file)
            except JSONDecodeError as e:
                raise JSONDecodeError(msg=f"Error parsing patchy ensemble spec file {str(cfg_file_path)}! {e.msg}",
                                      doc=e.doc,
                                      pos=e.pos)
            if kwargs["set_server"]:
                cfg[SERVER_SETTINGS_KEY] = kwargs["set_server"]
            # if we pass a "name" arg, export as that name instead of whatever's in the cfg
            if "name" in kwargs:
                cfg[EXPORT_NAME_KEY]  = kwargs["name"]
            return build_ensemble(cfg, {
                "setup_date": sim_init_date.strftime("%Y-%m-%d"),
            }, encoder_decoder=kwargs["loader"])

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
            return find_ensemble(mdf=simname,
                                 date=sim_init_date,
                                 set_server=kwargs["set_server"])
        else:
            # try loading a cfg file with that name
            if (get_input_dir() / (simname + ".json")).exists():
                with open((get_input_dir() / (simname + ".json")), "r") as f:
                    exportname = json.load(f)["export_name"]
                if metadata_file_exist(exportname, sim_init_date):
                    return find_ensemble(exportname,
                                         sim_init_date,
                                         set_server=kwargs["set_server"])
            print(f"Warning: could not find metadata file for {simname} at {sim_init_date.strftime('%Y-%m-%d')}")
            return find_ensemble(cfg=simname,
                                 date=sim_init_date,
                                 set_server=kwargs["set_server"])



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
                raise FileNotFoundError(
                    f"No metadata file at for simulation {metadata_file_name}")
            with metadata_file_path.open("r") as mdt_file:
                mdt = json.load(mdt_file)
                cfg = mdt["ensemble_config"]
                if kwargs["set_server"]:
                    cfg[SERVER_SETTINGS_KEY] = kwargs["set_server"]
                return build_ensemble(
                    cfg,
                    mdt,
                    metadata_file_path,
                    kwargs["loader"]
                )
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
            return find_ensemble(mdt=metadata_file_name,
                                 set_server=kwargs["set_server"])  # recurse to strong literal behavior
    else:
        raise Exception("Missing required identifier for simulation!")


def build_ensemble(cfg: dict[str: Any],
                   mdt: dict[str: Union[str, dict]],
                   mdtfile: Optional[Path] = None,
                   encoder_decoder: Type = SimEnsembleJsonEncoder) -> PatchySimulationEnsemble:
    """
    External patchy simulation constructor
    :param cfg: a key-value dict
    :param mdt: a metadata dict
    :param mdtfile: a path to a file for metadata
    :return: a PatchySimulationEnsemble object
    """

    # check for required keys
    if ENSEMBLE_SETUP_DATE not in mdt:
        raise MissingEnembleSetupInfoError(ENSEMBLE_SETUP_DATE)
    # standarize datatype of setup date
    setup_date: datetime.datetime = normalize_date(mdt[ENSEMBLE_SETUP_DATE])

    if EXPORT_NAME_KEY not in cfg:
        raise MissingEnembleSetupInfoError(EXPORT_NAME_KEY)
    export_name = cfg[EXPORT_NAME_KEY]  # grab export name

    # for debugging purporses, let me seed random
    if "randseed" in cfg:
        random.seed(cfg["randseed"])
        np.random.seed(cfg["randseed"])
    # if metadata filename wasn't manually provided
    if mdtfile is None:
        mdtfile = f"{export_name}_{setup_date.strftime('%Y-%m-%d')}_metadata.json"

    if "ensemble_config" not in mdt:
        mdt["ensemble_config"] = cfg

    # if "analysis_file" in mdt:
    #     analysis_file = mdt["analysis_file"]
    #     if (get_input_dir() / analysis_file).is_file():
    #         with open(get_input_dir() / analysis_file, "rb") as f:
    #             analysis_pipeline = pickle.load(f)
    #     else:
    #         print(f"Analysis file specified in metadata but path {get_input_dir() / analysis_file} does not exist!")
    #         analysis_pipeline = AnalysisPipeline()
    # else:
    #     analysis_file = f"{export_name}_analysis_pipeline.pickle"
    #     analysis_pipeline = AnalysisPipeline()

    if isinstance(mdt[ENSEMBLE_SETUP_DATE], datetime.datetime):
        mdt[ENSEMBLE_SETUP_DATE] = setup_date.strftime("%Y-%m-%d")

    # too difficult to make this one a ParamSet object
    # todo: kill this, replace with server config
    default_param_set_key = cfg[DEFAULT_PARAM_SET_KEY] if DEFAULT_PARAM_SET_KEY in cfg else "default"
    default_param_set = get_param_set(default_param_set_key) if default_param_set_key else {}

    params = []

    # load particles
    # todo: revise
    # default: if PARTICLE_TYPES_KEY is in const_params, ignore all else
    # todo: ensemble params
    if PARTICLE_TYPES_KEY in cfg[CONST_PARAMS_KEY]:
        pass #??
        # params.append(ParticleSetParam(load_pl_particles(**cfg[CONST_PARAMS_KEY][PARTICLE_TYPES_KEY])))
    elif any(key in cfg for key in [PARTICLE_TYPES_KEY, "rule", "cube_types"]):
        if PARTICLE_TYPES_KEY in cfg:
            if isinstance(cfg[PARTICLE_TYPES_KEY], PLParticleSet):
                particles = cfg[PARTICLE_TYPES_KEY]
            else:
                assert isinstance(cfg[PARTICLE_TYPES_KEY], dict), "Must include load info"
                ptypedict = {**cfg[PARTICLE_TYPES_KEY]}
                particles = load_pl_particles(**ptypedict)
        elif "cube_types" in cfg or "rule" in cfg:
            if "cube_types" in cfg:
                if len(cfg["cube_types"]) > 0 and isinstance(cfg["cube_types"][0], dict):
                    rule: PolycubesRule = PolycubesRule(rule_json=cfg["cube_types"])
                else:  # please do not
                    rule: PolycubesRule = PolycubesRule(rule_str=cfg["cube_types"])
                particles = polycube_rule_to_PL(rule)
            elif "rule" in cfg:  # 'rule' tag assumes serialized rule string
                # please for the love of god use this one
                if isinstance(cfg["rule"], str):
                    rule: PolycubesRule = PolycubesRule(rule_str=cfg["rule"])
                elif isinstance(cfg["rule"], PolycubesRule):
                    rule = cfg["rule"]
                else:
                    raise TypeError("Invalid type for rule!")
            else:
                raise Exception("wtf")
            particles = polycube_rule_to_PL(rule)
        params.append(ParticleSetParam(particles))
    else:
        # todo: should this be an exception?
        print("Warning: No particle info specified!")

    # handle multidentate params
    if NUM_TEETH_KEY in cfg[CONST_PARAMS_KEY]:
        if cfg[CONST_PARAMS_KEY][NUM_TEETH_KEY] > 1:
            raise Exception("Direct setting of teeth is no longer supported!")
        # assert DENTAL_RADIUS_KEY in cfg[CONST_PARAMS_KEY]
        # num_teeth = cfg[CONST_PARAMS_KEY][NUM_TEETH_KEY]
        # dental_radius = cfg[CONST_PARAMS_KEY][DENTAL_RADIUS_KEY]
        # only bothering to support legacy conversion params here
        # mdt_convert = MultidentateConvertSettings(num_teeth, dental_radius)
        # params.append(MDTConvertParams(mdt_convert))

    # load const params from cfg
    if CONST_PARAMS_KEY in cfg:
        for key, val in cfg[CONST_PARAMS_KEY].items():
            if key == NUM_TEETH_KEY or key == DENTAL_RADIUS_KEY:
                del cfg[CONST_PARAMS_KEY][key] # for compatability, please do not
                continue  # skip in const_params
            param_val = parameter_value(key, val)
            params.append(param_val)
    const_parameters = ParamSet(params)

    # observables are optional
    observables: dict[str: PatchySimObservable] = {}

    if OBSERABLES_KEY in cfg:
        # iter observables in cfg
        if all([type(obs) == dict for obs in cfg[OBSERABLES_KEY]]):
            for obs in cfg[OBSERABLES_KEY]:
                observable_obj = Observable(
                    obs["name"],
                    obs["print_every"],
                    *[
                        ObservableColumn(name=col["type"], **col)
                        for col in obs["cols"]
                    ]
                )
                observables[observable_obj.file_name] = observable_obj
        elif all([type(obs) == str for obs in cfg[OBSERABLES_KEY]]):
            # oh no
            for obs in cfg[OBSERABLES_KEY]:
                if isinstance(obs, str):
                    observables[obs] = observable_from_file(obs)
        else:
            assert all([type(obs) == Observable for obs in cfg[OBSERABLES_KEY]]), \
                f"Unrecognized type {str(type(next(o for o in cfg[OBSERABLES_KEY] if not isinstance(o, (str, dict)))))}"
            for obs in cfg[OBSERABLES_KEY]:
                observables[obs.file_name] = obs


    # load server spec
    server_spec_file = next((cfg[key] for key in [SERVER_SETTINGS_KEY,
                                              "server_spec",
                                              "server_config"] if key in cfg), None)
    if server_spec_file:
        if isinstance(server_spec_file, str):
            server_settings = load_server_settings(server_spec_file)
            if server_settings is None:
                raise Exception(f"No server setting set called `{cfg['server_settings']}`")
        elif isinstance(server_spec_file,dict):
            server_settings = PatchyServerConfig(**server_spec_file)
        elif isinstance(server_spec_file,PatchyServerConfig):
            server_settings = server_spec_file
        else:
            raise TypeError(f"Invalid type for key 'server_settings': {type(cfg['server_settings'])}")

    else:
        server_settings = get_server_config()
    # in case we need this

    # load ensemble params from cfg
    # there should always be ensemble params in the cfg
    if ENSEMBLE_PARAMS_KEY not in cfg:
        raise MissingEnembleSetupInfoError(ENSEMBLE_PARAMS_KEY)
    ensemble_parameters = [
        EnsembleParameter(key, [
            parameter_value(key, val) for val in paramData
        ])
        for key, paramData in cfg[ENSEMBLE_PARAMS_KEY]
    ]

    ensemble = PatchySimulationEnsemble(export_name=export_name,
                                        setup_date=setup_date,
                                        metadata_file_name=mdtfile,
                                        const_params=const_parameters + default_param_set,
                                        ensemble_params=ensemble_parameters,
                                        observables=observables,
                                        metadata_dict=mdt,
                                        server_settings=server_settings,
                                        json_rw=encoder_decoder)


    if "analysis_pipeline" in mdt:
        ensemble.analysis_pipeline = SimEnsembleJsonEncoder().read_analysis_pipeline(mdt["analysis_pipeline"],
                                                                                     ensemble)
        analysis_pipeline_data = mdt["analysis_pipeline_data"] if "analysis_pipeline_data" in mdt else None
        # read pipeline data *after* constructing the ensemble object
        if analysis_pipeline_data is not None:
            # analysis pipeline is encoded in json as a mixed-type list
            if isinstance(analysis_pipeline_data, list):
                ensemble.analysis_pipeline = encoder_decoder().read_analysis_pipeline(analysis_pipeline_data, ensemble)
            elif isinstance(analysis_pipeline_data, AnalysisPipeline):
                ensemble.analysis_pipeline = analysis_pipeline_data
            else:
                raise TypeError(f"Invalid type {type(analysis_pipeline_data)} for analysis pipeline")

    if "no_oxpy_check" in cfg:
        if cfg["no_oxpy_check"]:
            ensemble.no_oxpy_check_energies = True
    else:
        ensemble.no_oxpy_check_energies = server_settings.no_oxpy_check

    # I've removed dump_metadata from the initializer, can dump metadata later as needed

    return ensemble


class PatchySimulationEnsemble(Analyzable):
    """
    Stores data for a group of related simulations.
    This is the main object for running patchy particle simulations with PyPatchy
    The term "Ensemble" isn't used here in the same way as thermodynamics.
    """

    # --------------- GENERAL MEMBER VARS -------------- #

    # metadata regarding execution and analysis (TBD)
    metadata: dict[str,Any]
    metadata_file: str

    # list of parameters that will be varied across this ensemble
    # the set of simulations constituting this ensemble is the cartesian
    # product of all possible values of each ensemble param
    # each simulation is defined by a list of ParameterValue objects where each
    # ParameterValue object corresponds to a value in a different EnsembleParameter
    # object in this list
    ensemble_params: EnsembleParameters

    observables: dict[str, Observable]
    # skip oxpy checking, reqd for interactions like Romano
    no_oxpy_check_energies: bool = False

    # ------------ SETUP STUFF -------------#

    # simulation parameters which are constant over the entire ensemble
    const_params: ParamSet

    # log of slurm jobs
    slurm_log: SlurmLog

    # customizable server settings
    server_settings = PatchyServerConfig

    # output writer
    writer: PLBaseWriter

    json_rw: Type = SimEnsembleJsonEncoder

    _log_file_name: Path

    def __init__(self,
                 export_name: str,
                 setup_date: datetime.datetime,
                 metadata_file_name: str,
                 const_params: ParamSet,
                 ensemble_params: list[EnsembleParameter],
                 observables: dict[str, Observable],
                 metadata_dict: dict,
                 server_settings: Union[PatchyServerConfig, None] = None,
                 json_rw: Type = SimEnsembleJsonEncoder,
                 analysis_pipeline: Optional[AnalysisPipeline] = None,
                 log_file_name: Optional[Union[str, Path]] = None):
        """
        :param export_name: the name to use for this ensemble.
        :param setup_date: the date that this group of simulations was set up
        :param metadata_file_name: name of metadata file
        :param analysis_pipeline: analysis pipeline object
        :param const_params: the parameters that will be constant in all simulations
        :param ensemble_params: parameters which will be varied between different simulations
        :param observables: observables to use for this ensemble
        :param metadata_dict: metadata for this simulation grouo
        :param server_settings: "server settings" (aka simulation group preset) for the simulations
        :param json_rw: class type for JSON Encoder/Decoder. it's actually just an encoder, with class methods
        specifically to convert dict -> AnalysisPipelineStep
        :param log_file_name: name of file to log pypatchy stuff, or none to auto-generate
        """
        if analysis_pipeline is None:
            analysis_pipeline = AnalysisPipeline()
        super().__init__(analysis_pipeline)

        self.export_name = export_name
        self.sim_init_date = setup_date

        # load server settings ASAP
        if self.server_settings is not None:
            self.set_server_settings(server_settings)
        else:
            self.server_settings = get_server_config()
            self.writer = get_writer()

        # configure logging ASAP
        # File handler with a higher level (DEBUG)
        logger: logging.Logger = logging.getLogger(self.export_name)
        logger.setLevel(logging.INFO)
        # if the logger associated with this dataset already has handlers, clear them
        if len(logger.handlers) > 0:
            for handler in logger.handlers[::-1]:
                logger.removeHandler(handler)
        # setup logger file handler
        if not log_file_name:
            log_file_name = get_log_dir() / f"log_{self.export_name}_{self.datestr()}_at{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}.log"
        self._log_file_name = log_file_name

        file_handler = logging.FileHandler( log_file_name, mode="a")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        print(f"Appending logs to log file {str(log_file_name)}")

        # Stream handler with a lower level (INFO)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        stream_handler.setFormatter(stream_formatter)
        logger.addHandler(stream_handler)

        self.metadata = metadata_dict

        # slurm log stuff currently deprecated
        self.slurm_log = SlurmLog()

        self.metadata_file = metadata_file_name

        self.const_params = const_params

        self.ensemble_params = EnsembleParameters(ensemble_params)
        self.observables = observables
        self.json_rw = json_rw

    def __enter__(self):
        """
        Context manager entrypoint. required to make PatchySimulationEnsemble
        fuction as a context manager but doesn't do anything else
        :return: self
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        exit for context manager. writes updated ensemble info
        to metadata file
        :param exc_type: type of exception
        :param exc_val: value of exception
        :param exc_tb: traceback
        """
        # TODO: do we need to handle exceptions better?
        # if we had a simulation manager, turn it off.
        if self.server_settings.cuda_mps:
            # todo: make sure this is adeqquate
            self.server_settings.cuda_mps.manager.shutdown()
            self.server_settings.cuda_mps = True
        # on exit, dump metadata file
        self.dump_metadata()
        # if this block exited by raising an exception, re-raise it here
        if exc_type:
            raise exc_val

    # --------------- Accessors and Mutators -------------------------- #
    def get_simulation(self, *args: Union[tuple[str, Any], ParameterValue], **kwargs) -> Union[
        PatchySimulation,
        list[PatchySimulation]]:
        """
        This is a very flexable method for returning PatchySimulation objects
        but is also very complex, due to the range of inputs accepted

        given a list of parameter values, returns a PatchySimulation object

        :param args: group of ParameterValue objects
        :param kwargs: key-value pairs where each key is a parameter name in self.ensemble_params
        """
        # sim_params = args
        sim_params: list[Union[list[ParameterValue], EnsembleParameter]] = []
        multiselect = False
        for i_counter, param_type in enumerate(self.ensemble_params):
            pname = param_type.param_key
            if pname in kwargs:
                # use string to look up param value name
                sim_params.append([param_type[str(kwargs[pname])]])
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
                            pv = self.ensemble_params[k].lookup(v)
                            # pv = ParameterValue(k, v)
                            assert pv in param_type
                            sim_params.append([pv])
                            break  # terminate loop of args
            if len(sim_params) == i_counter:
                sim_params.append(param_type.param_value_set)
                multiselect = True
        if multiselect:
            return [PatchySimulation(e) for e in itertools.product(*sim_params)]
        else:
            return PatchySimulation([p for p, in sim_params])  # de-listify

    def datestr(self) -> str:
        """
        :return: the initialization date of the simulation, as a date string
        """
        return self.sim_init_date.strftime("%Y-%m-%d")

    def long_name(self) -> str:
        """
        :return: the full name of the simulation, including the date
        """
        return f"{self.export_name}_{self.datestr()}"

    def get_logger(self) -> logging.Logger:
        """
        implementation of Analyzable.get_logger()
        :returns: the simulation logger
        """
        return logging.getLogger(self.export_name)

    def is_do_analysis_parallel(self) -> bool:
        """
        :returns: true if the analysis is set up to run in parallel using multiprocessing.Pool and false otherwise.
        """
        return "parallel" in self.metadata and self.metadata["parallel"]

    def n_processes(self) -> int:
        """
        :returns: the number of times the patchy simulatinos are allowe to run in parallel
        """
        return self.metadata["parallel"]

    def is_nocache(self) -> bool:
        """
        implementation of Analyzable.is_nocache()
        :returns: true if analysis should skip cache
        """
        return "nocache" in self.metadata and self.metadata["nocache"]

    def set_server_settings(self, stgs: Union[PatchyServerConfig, str]):
        """
        sets the server settings that will be used to run simulations
        :param stgs: server settings
        """
        if isinstance(stgs, PatchyServerConfig):
            self.server_settings = stgs
        else:
            self.server_settings = load_server_settings(stgs)
        self.writer = get_writer(self.server_settings.patchy_format)
        # todo: better?
        self.no_oxpy_check_energies = self.server_settings.no_oxpy_check

    def set_nocache(self, bNewVal: bool):
        """
        Assigns the analysis system to not cache analysis data
        :param bNewVal: nev value for nocache. if True, analysis will skip cached data
        """
        self.metadata["nocache"] = bNewVal

    def append_slurm_log(self, item: SlurmLogEntry):
        """
        todo: make slurm logging work again
        Appends an entry to the slurm log, also writes a brief description
        to the logger
        """
        self.get_logger().debug(str(item))
        self.slurm_log.append(item)

    def get_stopped_sims(self) -> list[PatchySimulation]:
        """
        todo: unit test of this, somehow
        :return: a list of simulations which have been stopped by the slurm controller before they were complete
        """
        sims_that_need_attn = []
        for sim in self.ensemble():
            entries = self.slurm_log.by_subject(sim)
            if len(entries) == 0:
                continue
            desired_sim_length = self.sim_get_param(sim, "steps")
            last_entry = entries[-1]
            # get job info
            jobinfo, _ = self.bash_exec(f"scontrol show job {last_entry.job_id} | grep JobState")
            job_stopped = len(jobinfo) == 0 or jobinfo.split()[0].split("=")[1] != "RUNNING"
            if job_stopped:
                if self.time_length(sim) < desired_sim_length:
                    sims_that_need_attn.append(sim)
        return sims_that_need_attn

    def sim_get_param(self,
                      sim: PatchySimulation,
                      paramname: str) -> Any:
        """
        This method will first search in the simulation to see if this parameter has a specific value for this simulation, then in the ensemble const params, then the default parameter set
        :param sim: descriptor for a simulation for which to access a parameter
        :param paramname: name of the parameter
        :return: the value of a parameter.
        """
        if paramname in sim:
            return sim[paramname]
        try:
            paramval = self.get_param(paramname)
            assert not isinstance(paramval, ParameterValue)
            return paramval
        except NoSuchParamError as e:
            e.set_sim(sim)
            raise e

    def sim_stage_get_param(self, sim: PatchySimulation, stage: Stage, param_name: str) -> Any:
        """
        :param sim: descriptor for a simulation for which to access a parameter
        :param stage: Stage object for which to access a parameter
        :param param_name: name of the parameter
        :return: the value of the provided parameter for the given simulation and stage
        """
        if stage is not None and stage.has_var(param_name):
            return stage.get_var(param_name)
        else:
            return self.sim_get_param(sim, param_name)

    def sim_get_particles_set(self, sim: PatchySimulation) -> PLParticleSet:
        """
        :return: the particle set used by the simulation passed as a parameter, with multidentate conversion if appropriate
        """
        particles: PLParticleSet = self.sim_get_param(sim, PARTICLE_TYPES_KEY)
        try:
            mdt_convert: MultidentateConvertSettings = self.sim_get_param(sim, MDT_CONVERT_KEY)
            return particles.to_multidentate(mdt_convert)
        except NoSuchParamError as e:
            # if no multidentate convert, return particle set as-is
            assert isinstance(particles, PLParticleSet)
            return particles

    def get_param(self, paramname: str) -> Any:
        """
        :param paramname: the name of the simulation ensemble parameter to access
        :return: the inner value of the ParameterValue (NOT the ParameterValue object) with the given name
        """
        # use const_params
        if paramname in self.const_params:
            return self.const_params[paramname]
        # # go deep
        if paramname in self.server_settings.input_file_params:
            return self.server_settings.input_file_params[paramname]
        raise NoSuchParamError(self, paramname)

    def iter_params(self, sim: PatchySimulation, stage: Stage) -> Generator[ParameterValue, None, None]:
        """
        Iterates through parameter values for the given simulation and stage
        :param sim: descriptor for a simulation for which to access a parameter
        :param stage: Stage object for which to access a parameter
        :return: generator which will iterate through the parameters
        """
        pvs: set[ParameterValue] = set()
        # first, iter stage vals
        for pv in stage.params():
            if isinstance(pv, ParamValueGroup):
                for pvv in pv:
                    pvs.add(pvv)
                    yield pvv
            else:
                pvs.add(pv)
                yield pv
        # iter sim vals
        for pv in sim.param_vals:
            if isinstance(pv, ParamValueGroup):
                for pvv in pv:
                    pvs.add(pvv)
                    yield pvv
            else:
                pvs.add(pv)
                yield pv
        # iter const params
        for pv in self.const_params:
            if isinstance(pv, ParamValueGroup):
                for pvv in pv:
                    pvs.add(pvv)
                    yield pvv
            else:
                pvs.add(pv)
                yield pv
        # iter server params
        for pv in self.server_settings.input_file_params:
            if isinstance(pv, ParamValueGroup):
                for pvv in pv:
                    pvs.add(pvv)
                    yield pvv
            else:
                pvs.add(pv)
                yield pv
            # iter defaultsa
            if isinstance(pv, ParamValueGroup):
                for pvv in pv:
                    pvs.add(pvv)
                    yield pvv
            else:
                pvs.add(pv)
                yield pv

    def paramfile(self, sim: PatchySimulation, paramname: str) -> Path:
        """
        Shorthand to get a simulation data file
        IIRC This method needs to be distinct from paramstagefile (see below) for use in methods
        that generate staging info
        :param sim: A PatchySimulation object describing a specific simulation
        :param paramname: a param name associated with a simulation file (topoplogy, conf, energy, etc.)
        :return: a path to a simulation data file for the provided simulation
        """
        return self.folder_path(sim) / self.sim_get_param(sim, paramname)

    def paramstagefile(self, sim: PatchySimulation, stage: Stage, paramname: str):
        """
        Shorthand to get a simulation data file associated with a specific simulation and stage
        :param sim: A PatchySimulation object describing a specific simulation
        :param stage: The stage for this file
        :param paramname: a param name associated with a simulation file (topoplogy, conf, energy, etc.)
        :return: a path to a simulation data file for the provided simulation and stage
        """
        return self.folder_path(sim) / stage.adjfn(self.sim_get_param(sim, paramname))

    def get_sim_particle_count(self,
                               sim: PatchySimulation,
                               particle_idx: int) -> int:
        """
        :param sim: the patchy simulation to count for
        :param particle_idx: the index of the particle to get the count for
        :return: the number of particles of the provided types in the simulation
        """
        # grab particle name
        particle_name = self.sim_get_particles_set(sim).particle(particle_idx).name()
        return self.sim_get_param(sim, particle_name) * self.sim_get_param(sim, NUM_ASSEMBLIES_KEY)

    def get_stages_info(self, sim: Union[None, PatchySimulation] = None) -> dict[str, StageInfoParam]:
        """
        Returns staging information parameters
        :param sim: the simulation to access, or None to access for the entire ensemble
        :return: a map of stage name to StageInfoParam object
        """
        if sim is not None:
            stages_info: dict[str, StageInfoParam] = self.sim_get_param(sim, STAGES_KEY)
        else:
            stages_info: dict[str, StageInfoParam] = self.get_param(STAGES_KEY)
        return stages_info

    def sim_get_stages(self, sim: PatchySimulation, auto=False) -> list[Stage]:
        """
        Computes stages
        horrible nightmare code
        Stage objects require other ensemble parameters so must be constructed as needed and not on init
        :param sim: a PatchySimulation for which to access a stage list
        :return: list of stages to execute for this simulation
        """
        if not auto:
            return self.sim_get_spec_stages(sim)
        else:
            return self.sim_generate_stages(sim)

    def sim_get_spec_stages(self, sim: PatchySimulation) -> list[Stage]:
        assert isinstance(sim, PatchySimulation), "Cannot invoke this method with a parameter list!"
        # ---------- STEP ONE: LOAD STAGE INFO ------------------------------
        cumulative_type_counts = Counter()
        try:
            # load stage info parameters for simulation
            # sort by start time
            stages_info: list[StageInfoParam] = copy.deepcopy(list(sorted(self.get_stages_info(sim).values(),
                                                                          key=lambda x: x.start_time)))

            stages: list[Stage] = []
            # loop stages in stage list from spec
            # we need to initially loop them forward in order to compute run times
            for i, stage_info in enumerate(stages_info):
                stage_init_args = {}
                # compute stage runtime
                assert stage_info.get_start_time() != 0 or i == 0, \
                    "Stages other than start stage must have specified start time"
                # if this is not the last stage
                if i + 1 != len(stages_info):
                    # if stage is not final stage, last step of this stage will be first step of next stage
                    # assign stage endtime
                    stage_info.set_end_time(stages_info[i + 1].get_start_time())
                elif "steps" not in stage_info:
                    # if the stage is the final stage, last step will be end of simulation
                    stage_info.set_end_time(self.sim_get_param(sim, "steps"))
                # construct stage objects
                if stage_info.add_method is not None:
                    cumulative_type_counts.update(stage_info.add_method.get_particle_counts())
                stages.append(Stage(sim,
                                    stages[i - 1] if i else None,
                                    self,
                                    stage_info,
                                    **stage_init_args
                                    ))

        except NoSuchParamError as e:
            # if we are missing a parameter other than "staging", we are going to have problems
            if e.param_name() != STAGES_KEY:
                raise e
            # if stages not found
            # list particle type counts
            stages = [self._gen_default_stage(sim, cumulative_type_counts)]

        # assign box sizes
        for stage in stages:
            # if box size is specified explicitly
            # i've never used this and don't know why anyone ever would
            if stage.has_var("box_size"):
                stage.set_box_size(stage.get_var("box_size"))
            else:
                # please do not use density or num_assemblies as a staging param. please. please.
                # by default, use the old behavior. compute the box size at final stage concentration and use that
                if stage.has_var("density"):
                    density = stage.get_var("density")
                    # stage.num_particles = number of particles that will be present in this stage's simulation
                    num_particles = stage.num_particles()
                else:
                    num_particles = sum(cumulative_type_counts.values()) * self.sim_get_param(sim, "num_assemblies")
                    density = self.sim_get_param(sim, "density")
                box_side = (num_particles / density) ** (1 / 3)

                stage.set_box_size(np.array((box_side, box_side, box_side)))

        return stages

    def _gen_default_stage(self, sim, cumulative_type_counts: Union[dict, None]=None) -> Stage:
        particles = {
            p.name(): self.sim_get_param(sim, p.name())
            for p in self.sim_get_particles_set(sim).particles()
        }
        if cumulative_type_counts is not None:
            cumulative_type_counts.update(particles)
        return Stage(
            sim,
            None,
            self,
            StageInfoParam(
                "default",
                add_method=RandParticleAdder(particles=particles),
                steps=self.sim_get_param(sim, "steps"),
                allow_shortfall=True
            )
        )

    def sim_generate_stages(self, sim: PatchySimulation) -> list[Stage]:
        """
        generates staging information based on existing simulation data files, useful for more ad-hoc simulations
        :param sim: a PatchySimulation for which to generate a stage list
        """
        sim_subfolders = []
        # generate first stage. we can't use the _gen_default_stage method
        input_file_params = {}
        with (self.folder_path(sim) / "input").open("r") as f:
            for line in f:
                if not line: continue
                if line.strip().startswith("#"): continue
                try:
                    key, value = line.split("=")
                except ValueError:
                    self.get_logger().warning(f"Could not parse line in input file: {line}")
                    continue
                input_file_params[key.strip()] = value.strip()
        stages = [Stage(sim,
                    None,
                    self,
                    StageInfoParam("default",
                                   t=0,
                                   **input_file_params
                                   ))
        ]
        for fname in self.folder_path(sim).iterdir():
            if fname.is_dir() and (fname/"input").exists(): # we will open fname.input later
                try:
                    # do last_conf first since it's more likely to be absent
                    with (fname / self.sim_get_param(sim, "lastconf_file")).open("r") as f:
                        t_end = int(f.readline().split("=")[1].strip())
                    with (fname / self.sim_get_param(sim, "trajectory_file")).open("r") as f:
                        first_line = f.readline()
                        t_start = int(first_line.split("=")[1].strip())
                except FileNotFoundError as e:
                    self.get_logger().info(f"Folder {fname} does not have file {e.filename}, skipping....")
                    continue
                sim_subfolders.append((fname, t_start, t_end))
        if len(sim_subfolders) == 0:
            with (self.folder_path(sim) / self.sim_get_param(sim, "lastconf_file")).open("r") as f:
                last_step = int(f.readline().split("=")[1].strip())
            stages[0].set_tend(last_step)
            return stages
        else:
            for i, (fp, tstart, tend) in enumerate(sorted(sim_subfolders, key=lambda x: x[1])):
                # load params from input file, to make sure that params are considered
                # correctly
                input_file_params = {}
                with (fp / "input").open("r") as f:
                    for line in f:
                        key, value = line.split("=")
                        input_file_params[key.strip()] = value.strip()
                # overwrite steps key to actual last timestep
                input_file_params["steps"] = tend
                stages.append(Stage(
                    sim,
                    stages[i], # +1 offset to indexing, so i - 1 + 1 = i
                    self,
                    StageInfoParam(fp.name,
                                   t=tstart,
                                   **input_file_params
                                   )
                ))
            return stages


    def sim_get_stages_between(self, sim: PatchySimulation, tstart: int, tend: int, auto: bool=False, inclusive: bool=False) -> list[Stage]:
        """
        todo a unit test for this?
        :param sim: simulation descriptor
        :param tstart: start time
        :param tend: end time
        :return: list of stages for the provided simulation between the start and end
        """
        return [stage for stage in self.sim_get_stages(sim, auto) if stage.end_time() >= tstart
                and (stage.start_time() < tend or (inclusive and stage.start_time() == tend))]

    def sim_get_stage(self, sim: PatchySimulation, stage_name: Union[str, int]) -> Stage:
        """
        Constructs and returns a Stage object for the given simulation corresponding to the given stage
        :param sim: simulation descriptor
        :param stage_name: name of a stage
        :return: a Stage object
        """
        stages = self.sim_get_stages(sim)
        if isinstance(stage_name, int):
            return stages[stage_name]
        # inefficient search algorithm but len(stages) should never be more than like 10 tops
        for stage in stages:
            if stage.name() == stage_name or f"stage_{stage_name}" == stage.name():
                return stage
        raise Exception(f"No stage named {stage_name}!")

    def sim_get_stage_top_traj(self, sim: PatchySimulation, stage: Union[str, int, Stage]) -> tuple[Path, Path]:
        """
        gets top and traj file paths
        :param sim: simulation descriptor
        :param stage: name of a stage
        :return: the traj and top FILES for provided simulation and stage
        """
        return (
            self.sim_get_stage_top(sim, stage),
            self.folder_path(sim, stage) / self.sim_stage_get_param(sim, stage, "trajectory_file")
        )

    def sim_get_stage_top(self, sim: PatchySimulation, stage: Union[str, int, Stage]) -> Path:
        """
        Finds a path to a top file associated with a sim and stage
        :param sim: simulation descriptor
        :param stage: name of a stage
        :return: a path to the topology file for the given simulation and stage
        """
        return self.folder_path(sim, stage) / self.sim_stage_get_param(sim, stage, "topology")

    def sim_get_stage_last_conf(self, sim: PatchySimulation, stage: Union[str, int, Stage]) -> Path:
        """
        Gets the path to a last conf file for simulation and stage
        :param sim: simulation descriptor
        :param stage: name of a stage
        :return: a path to the last_conf file for the given simulation and stage
        """
        return self.folder_path(sim, stage) / self.sim_stage_get_param(sim, stage, "lastconf_file")

    def sim_get_stage_last_step(self, sim: PatchySimulation, stage: Union[str, int, Stage]) -> int:
        """
        :param sim: simulation descriptor
        :param stage: name of a stage
        :return: the last step for the given simulation and stage
        """
        lastconf_file = self.sim_get_stage_last_conf(sim, stage)
        if not lastconf_file.is_file():
            raise NoStageTrajError(stage, sim, str(lastconf_file))
        else:
            # return timepoint of last conf in traj
            try:
                return file_info([str(lastconf_file)])["t_end"][0]
            except IndexError as e:
                # trajectory file empty
                raise StageTrajFileEmptyError(stage, sim, str(lastconf_file))

    def sim_get_total_stage_particles(self, sim: PatchySimulation, stage: Stage) -> int:
        """
        Computes the number of particles in a stage. Includes particles added in this stage and
        all previous stages
        :param sim: simulation descriptor
        :param stage: name of a stage
        :return: the number of particles in the stage
        """
        return sum([s.num_particles_to_add() for s in self.sim_get_stages(sim)[:stage.idx() + 1]])  # incl passed stage

    def sim_most_recent_stage(self, sim: PatchySimulation, autogenerate_stages: bool = False) -> Stage:
        """
        constructs a Stage object for the most recent stage
        :param sim: simulation descriptor
        :return: a tuple where the first element is the most recent stage file with a trajectory and the second element
         is true if the last conf in the traj is at the last timepoint of the stage or none if no stage has begun
        """
        # increment in  reverse order so we check later stages first
        for stage in reversed(self.sim_get_stages(sim, autogenerate_stages)):
            # if traj file exists
            if (self.folder_path(sim) / self.sim_get_param(sim, "lastconf_file")).exists():
                try:
                    stage_last_step = self.sim_get_stage_last_step(sim, stage)
                    if stage_last_step == stage.end_time() or \
                            (stage_last_step < stage.end_time() and stage.allow_shortfall()):
                        return stage
                    elif stage_last_step > stage.end_time():
                        return stage  # more to do here?
                    # if stage is incomplete, raise an exception
                    raise IncompleteStageError(
                        stage,
                        sim,
                        stage_last_step
                    )
                except NoStageTrajError as e:
                    # if there's no traj for this stage it means the stage hasn't started yet
                    pass
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
        """
        Gets the number of stages in a simulation
        :param sim: simulation descriptor
        :return: the number of stages for a provided simulation
        """
        return len(self.sim_get_stages(sim))

    def get_interaction_potentials(self, sim: PatchySimulation) -> list[PLPotential]:
        """
        Contructs and returns a set of interaction potentials for the given simulation
        Much of this is currently hardcoded because making is more programmatic is not worth my time
        :param sim: simulation descriptor
        :returns: a list of patchy potentials
        """
        # todo: include interaction potential spec in server spec
        interaction_potential_names: list[str] = self.sim_get_param(sim, "interaction_potentials")
        potentials: list[PLPotential] = list()
        for name in interaction_potential_names:
            if name == "lr_excl_vol":
                # in Lorenzo's interaction the potential is hardcoded
                potential = PLLRExclVolPotential(rmax=2.01421)
            elif name == "fr_excl_vol":
                potential = PLFRExclVolPotential(
                    # rmax = 0.9988 if PATCHY_radius = 0.5
                    rmax=0.9988 * (self.sim_get_param(sim, "PATCHY_radius") * 2), # ????
                     rstar=0.90530002117156982 * (self.sim_get_param(sim, "PATCHY_radius") * 2),
                     b=677.505671539  # from flavio's code
                                                 )
                # potential = PLFRExclVolPotential(rmax=2.01421 * self.sim_get_param(sim, "PATCHY_radius") * 2,
                #                                  rstar=self.sim_get_param(sim, "PATCHY_radius") * 2 ** (1 / 6),
                #                                  b=677.505671539  # from flavio's code
                #                                  )

            elif name == "lr_patchy":
                particle_types: PLParticleSet = self.sim_get_particles_set(sim)
                potential = PLLRPatchyPotential(rmax=2.01421,
                                                sigma_ss=self.sim_get_param(sim, "DPS_sigma_ss"),
                                                interaction_matrix=particle_types.interaction_matrix())
            elif name == "fr_patchy":
                patchy_radius = self.sim_get_param(sim, "PATCHY_radius")
                if self.sim_get_param(sim, "use_torsion"):
                    potential = PLFRTorsionalPatchyPotential(
                        rmax=0.4 * 3 * 2 * patchy_radius,
                        alpha=self.sim_get_param(sim, "PATCHY_alpha"),
                        narrow_type=self.sim_get_param(sim, "narrow_type")
                    )
                    # potential = PLFRTorsionalPatchyPotential(
                    #     rmax=0.4 * 3 * patchy_radius,
                    #     alpha=self.sim_get_param(sim, "PATCHY_alpha"),
                    #     narrow_type=self.sim_get_param(sim, "narrow_type")
                    # )
                else:
                    potential = PLFRPatchyPotential(
                        rmax=0.4 * 3 * 2* patchy_radius,
                        alpha=self.sim_get_param(sim, "PATCHY_alpha")
                    )
            else:
                raise Exception(f"No python-implemented interaction potential called {name}, options are"
                                f" `lr_excl_vol`, `fr_excl_vol`, `lr_patchy`, `fr_patchy`.")
            potentials.append(potential)
        return potentials

    def sim_stage_done(self, sim: PatchySimulation, stage: Stage) -> bool:
        """
        :param sim: simulation descriptor
        :param stage: a stage object
        similar to get_last_step but returns a boolean
        :return: true if stage traj exists and is correct length, false otherwise
        """
        last_conf_file = self.sim_get_stage_last_conf(sim, stage)
        if not last_conf_file.is_file():
            return False
        else:
            # return timepoint of last conf in traj
            return file_info([str(last_conf_file)])["t_end"][0] == stage.end_time()

    def ensemble(self) -> list[PatchySimulation]:
        """
        :return: a list of all simulations in this ensemble
        """
        return [PatchySimulation(e) for e in itertools.product(*self.ensemble_params)]

    def num_ensemble_parameters(self) -> int:
        """
        :return: the number of ensemble parameters
        """
        return len(self.ensemble_params)

    def add_params(self, **kwargs):
        """
        Adds parameters to the simulation ensemble
        :param kwargs: key-value args parameters to add
        """
        for key, value in kwargs.items():
            if isinstance(value, (str, float, bool, int)):
                self.const_params.append(ParameterValue(key, value))
            elif isinstance(value, ParameterValue):
                self.const_params.append(value)
            else:
                raise Exception(f"invalid data type {type(value)} provided to add_params")

    def tld(self) -> Path:
        """
        :return: The top-level directory for simulation files for this ensemble
        """
        return simulation_run_dir() / self.long_name()

    def folder_path(self, sim: PatchySimulation, stage: Union[Stage, None] = None) -> Path:
        """
        Gets a path to a simulation folder associated with a simulation and stage
        :param sim: simulation descriptor
        :param stage: stage object or None
        :return: the path to the working directory of the given simulation, at the given stage
        """
        if stage is None or stage.idx() == 0:
            return self.tld() / sim.get_folder_path()
        else:
            return self.tld() / sim.get_folder_path() / stage.name()

    def save_pipeline_data(self):
        """

        """
        self.dump_metadata()

    def time_length(self,
                    sim: Union[PatchySimDescriptor,
                               list[PatchySimDescriptor],
                               None] = None
                    ) -> int:
        """
        :param sim: simulation descriptor or a list thereof
        :return: the length of a simulation, in steps
        """

        if sim is None:
            return self.time_length(self.ensemble())
        elif isinstance(sim, list):
            return min([self.time_length(s) for s in sim])
        elif isinstance(sim, tuple):  # todo: check single descriptor
            if len(sim) > 0 and isinstance(sim[0], tuple):
                return self.time_length(self.get_simulation(*sim))
            else:
                return self.time_length(self.get_simulation(sim))
        else:
            try:
                # for time length purposes autoidentify stages
                stage = self.sim_most_recent_stage(sim, True)
                return self.sim_get_stage_last_step(sim, stage)
            except IncompleteStageError as e:
                return e.last_timestep()

    # ------------------------ Status-Type Stuff --------------------------------#
    def info(self, infokey: str = "all"):
        """
        prints help text, for non-me people or if I forget
        might replace later with pydoc
        """
        print(f"Ensemble of simulations of {self.export_name} set up on {self.sim_init_date.strftime('%Y-%m-%d')}")
        try:
            print(f"Particle info: {str(self.get_param('particle_types'))}")
        except NoSuchParamError as e:
            print("Particle types are not constant")
        print(f"Metadata stored in file {self.metadata_file}")
        print(f"Simulation TLD: {self.tld()}")
        print("Ensemble Params")
        for param in self.ensemble_params:
            print("\t" + str(param))
        print(f"Const Simulation Params")
        for param in self.const_params:
            print(param.str_verbose())

        if len(self.analysis_pipeline) > 0:
            print(
                f"Has analysis pipeline with {self.analysis_pipeline.num_pipeline_steps()} steps and {self.analysis_pipeline.num_pipes()} pipes.")
            print(f"Pipeline steps")

        if len(self.analysis_data) > 0:
            print(f"Has {len(self.analysis_data)} entries of analysis data loaded (each entry is data for a specific "
                  f"analysis step and simulation)")

    def show_last_conf(self, sim: Union[PatchySimulation, None] = None, **kwargs):
        """
        Displays the final configuration of a simulation using OxView
        TODO: ppview
        :param sim: a patchy simulation descriptor, or None
        :param kwargs: key-value pairs to pass to get_simulation
        """
        if len(kwargs) > 0:
            self.sim_get_param(self.get_simulation(**kwargs))
        else:
            assert sim is not None, "No simulation provided!"
            assert isinstance(get_writer(), FWriter), "Can only show confs for FlavioWriter!!!"
            self.show_conf(sim, self.time_length(sim))
            # from_path(self.paramfile(sim, "lastconf_file"),
            #           self.paramfile(sim, "topology"),
            #           self.folder_path(sim) / "particles.txt",
            #           self.folder_path(sim) / "patches.txt")

    def get_scene(self, sim: PatchySimulation,
                  stage: Union[Stage, str, None] = None,
                  step: Union[int, None] = None) -> PLPSimulation:
        """
        Gets a patchy simulation conf as a patchy simulation object
        :param sim: patchy simulation descriptor
        :param stage: stage object or None
        :param step: step or None
        :return: a PLPSimulation object
        """
        if isinstance(stage, str):
            stage = self.sim_get_stage(sim, stage)
        elif stage is None:
            # if stage is not defined, auto-locate it
            stage = self.sim_most_recent_stage(sim, True)
        self.writer.set_directory(self.folder_path(sim, stage))

        top_file, traj_file = self.sim_get_stage_top_traj(sim, stage)
        particle_types = self.writer.read_particle_types(**{key: self.sim_stage_get_param(sim, stage, key)
                                                            for key in self.writer.reqd_args()
                                                            })
        # top_file, traj_file = self.sim_get_stage_top(sim, stage), self.sim_get_stage_last_conf(sim, stage)
        if traj_file.exists() and os.stat(traj_file).st_size > 0:
            if step is None:
                step = self.time_length(sim)

            # if we have requested last conf
            if self.time_length(sim) == step or self.sim_stage_get_param(sim, stage, "steps") == step:
                traj_file = self.sim_get_stage_last_conf(sim, stage)
                scene: PLPSimulation = self.writer.read_scene(top_file.name, traj_file.name,
                                                              # self.sim_get_particles_set(sim))
                                                              particle_types)
            else:
                conf_interval = int(float(self.sim_stage_get_param(sim, stage, "print_conf_interval")))
                assert step % conf_interval == 0, (f"Attempting to get a conf at a timepoint {step} that is not a "
                                                   f"multiple of the conf interval "
                                                   f"{self.sim_stage_get_param(sim, stage, 'print_conf_interval')}")

                scene: PLPSimulation = self.writer.read_scene(top_file.name, traj_file.name,
                                                              # self.sim_get_particles_set(sim),
                                                              particle_types,
                                                              (step - stage.start_time()) // conf_interval)
        else:
            assert step is None, f"Could not find trajectory file {str(traj_file)}, " \
                                 f"but timestep was explicitly specified."
            conf_file = self.sim_stage_get_param(sim, stage, "conf_file")
            scene: PLPSimulation = self.writer.read_scene(top_file.name,
                                                          conf_file,
                                                          # self.sim_get_particles_set(sim))
                                                          particle_types)

        # check particle files
        # todo: check that this line works for all writers


        # if ptypes_dir != self.sim_get_particles_set(sim):
        #     scene.set_particle_types(ptypes_dir)
        #     # todo: custom exception
        #     raise Exception("Mismatch between particle types written in directory "
        #                                                      " and particle types specified in setup files!"
        #                                                      " Or old data, that is also possible.")

        scene.set_temperature(self.sim_stage_get_param(sim, stage, "T"))
        for potential in self.get_interaction_potentials(sim):
            scene.add_potential(potential)
        return scene
        # scene: PLPSimulation()

    def is_traj_valid(self, sim: PatchySimulation, stage: Union[Stage, None] = None) -> bool:
        """
        Checks if a trajectory file is valid by counting the lines
        A valid traj file should have a line count which is a multiple of (number of particles + 3)
        I would like to depreacate this method ASAP but right now oat segfaults with no warning when a traj
        file is corrupted
        :param sim:
        :param stage:

        :return: True if the traj file is not corrupted, false otherwise
        """
        if stage is None:
            stage = self.sim_get_stages(sim)[0]
        traj_file = stage.adjfn(self.sim_get_param(sim, "trajectory_file"))
        num_particles = self.sim_get_total_stage_particles(sim, stage)
        assert num_particles > 0

    def get_conf(self, sim: PatchySimulation, timepoint: int = -1) -> Configuration:
        """
        do not add a stage parameter!
        :return: a Configuration object showing the conf of the given simulation at the given timepoint
        """
        if timepoint == -1:
            timepoint = self.time_length(sim)
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
        """
        displays a
        :param sim: a patchy simulation descriptor
        :param timepoint: a timepoint index
        """
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
            tuple(v.value_name() for v in sim.param_vals):
                {
                    step_name: self.has_data_file(self.analysis_pipeline[step_name], sim)
                    for step_name in self.analysis_pipeline.name_map
                }
            for sim in self.ensemble()
        }, orient="index")

    def all_folders_exist(self) -> bool:
        """
        tests if all folders exist
        :return: true if all folders exist, false otherwise
        """
        return all(self.folder_path(s).exists() for s in self.ensemble())

    # ----------------------- Setup Methods ----------------------------------- #
    def do_setup(self,
                 sims: Union[list[PatchySimulation], None] = None,
                 stage: Union[None, str, Stage] = None):
        """
        Performs setup, for the entire ensemble or for specific simulations.
        :param sims: None, or a list of simulation descriptors, If None, will use self.ensemble()
        :param stage: None, a stage name string, or a Stage object
        """

        # check for mps stuff
        if sims is None:
            sims = self.ensemble()
        self.get_logger().info("Setting up folder / file structure...")
        for sim in sims:
            assert isinstance(sim, PatchySimulation), "Invalid type!"
            assert self.sim_get_param(sim, "print_conf_interval") < self.sim_get_param(sim, "steps")
            self.get_logger().info(f"Setting up folder / file structure for {repr(sim)}...")
            # create nessecary folders
            if isinstance(stage, str):
                stage_obj = self.sim_get_stage(sim, stage)
            else:
                stage_obj = stage
            if not os.path.isdir(self.folder_path(sim, stage_obj)):
                self.get_logger().info(f"Creating folder {self.folder_path(sim, stage_obj)}")
                Path(self.folder_path(sim, stage_obj)).mkdir(parents=True)
            else:
                self.get_logger().info(f"Folder {self.folder_path(sim, stage_obj)} already exists. Continuing...")

            # write requisite top, patches, particles files
            self.get_logger().info("Writing .top, .txt, input, etc. files...")
            self.write_setup_files(sim, stage_obj)
            # write observables.json if applicble
            if EXTERNAL_OBSERVABLES:
                self.get_logger().info("Writing observable json, as nessecary...")
                self.write_sim_observables(sim)
            # skip writing sbatch script if mps is off
            # write .sh script
            self.get_logger().info("Writing sbatch scripts...")
            self.write_run_script(sim, stage_obj)
        # save metadata to file
        self.dump_metadata()

    def write_setup_files(self,
                          sim: PatchySimulation,
                          stage: Union[str, Stage, None] = None):
        """
        Writes any/all nessecary files for a simulation
        :param sim: A simulation descriptor
        :param stage: a stage object, or None
        """
        if stage is None:
            try:
                # get most recent stage
                stage = self.sim_most_recent_stage(sim)
                # stage 0 will produce a NoStageTrajError (caught below)
                stages = self.sim_get_stages(sim)
                if stage.idx() + 1 != len(stages):
                    stage = stages[stage.idx() + 1]
                else:
                    self.get_logger().info(f"Final stage {stage.name()} is already complete!")
            # if no stage exists
            except NoStageTrajError:
                # stage 0
                stage = self.sim_get_stages(sim)[0]
            except IncompleteStageError:
                self.get_logger().error(f"{stage.name()} incomplete!")
                return

        elif isinstance(stage, str):
            stage = self.sim_get_stage(sim, stage)

        # if this is the first conf
        if stage.idx() == 0:
            assert stage.start_time() == 0, f"Stage {stage} has idx 0 but nonzero start time!"

            # generate conf
            scene = PLPSimulation()

            particle_set = self.sim_get_particles_set(sim)
            # patches will be added automatically
            scene.set_particle_types(particle_set)

        else:
            last_complete_stage = stage.get_prev()
            # don't catch exxeption here
            step = last_complete_stage.end_time()
            scene = self.get_scene(sim, last_complete_stage, step)
        scene.set_temperature(self.sim_stage_get_param(sim, stage, "T"))

        nTries = 0
        # what should be the energy cutoff???
        # if we're doing this right it should be the outer end of statistical distribution dependant
        # on kB * T, with the variance dependant on the simulation size
        # or we can just make up a coefficient like 3
        # in oxDNA units kB = 1
        stage.apply(scene)

        # Todo: cut next line once we have this working
        if not self.no_oxpy_check_energies:
            scene_computed_energy = scene.get_potential_energy()

        # initialize a local writer from ensemble writer
        # todo: standardize this as the method to get writer object
        writer = copy.deepcopy(self.writer)

        # grab args required by writer
        reqd_extra_args = {
            a: self.sim_stage_get_param(sim, stage, a) for a in writer.reqd_args()
        }
        for key, value in writer.get_input_file_data(scene):
            # if we assign to input directly it will update input file, causing its own issues
            stage.sim.input.input_dict[key] = value

        assert "conf_file" in reqd_extra_args, "Missing conf file info!"

        # write top, conf, and others
        # set writer directory to simulation folder path
        writer.set_directory(self.folder_path(sim, stage))
        writer.set_abs_paths(self.server_settings.absolute_paths)
        self.folder_path(sim, stage).mkdir(exist_ok=True)
        files = writer.write(scene,
                             **reqd_extra_args)

        # update top and dat files in replacer dict
        # replacer_dict.update(files)
        # replacer_dict["steps"] = stage.end_time()
        # replacer_dict["trajectory_file"] = self.sim_get_param(sim, "trajctory_file")

        # add additional scene-specific args required b writer

        # create input file
        stage.build_input()
        # self.write_input_file(sim, stage, replacer_dict, extras, analysis)

        # check energies
        if not self.no_oxpy_check_energies:
            with oxpy.Context():
                assert (self.folder_path(sim, stage) / "input").exists()
                manager = oxpy.OxpyManager(str(self.folder_path(sim, stage))+"/input")
                e = manager.system_energy()
                del manager

                # oxpy manager gets total energy rather than per particle
                if e > 0 and abs(e / scene.num_particles() - scene_computed_energy) > 1e-4:
                    print(
                        f"Mismatch between python-computed energy {scene_computed_energy} and oxDNA-computed energy {e}")

    def write_run_script(self, sim: PatchySimulation, stage: Stage, input_file="input"):
        """
        Writes a script named `slurm_script.sh` which will be executed with `sbatch` to run the simulation
        :param sim: A simulation descriptor
        :param stage: stage
        :param input_file: name of input file
        """
        slurm_script_name = "slurm_script.sh"
        if stage:
            slurm_script_name = stage.adjfn(slurm_script_name)

        # input file is in same directory as slurm script so don't change the file name

        # write slurm script
        with open(self.folder_path(sim) / slurm_script_name, "w+") as slurm_file:
            # bash header

            self.server_settings.write_sbatch_params(self.export_name, slurm_file)

            # skip confGenerator call because we will invoke it directly later
            slurm_file.write(f"{self.server_settings.oxdna_path}/build/bin/oxDNA {input_file}\n")

        self.bash_exec(f"chmod u+x {self.folder_path(sim)}/{slurm_script_name}",is_async=False)

    def run_simulation(self, sim: PatchySimulation):
        """
        Runs a patchy simulation using oxpy
        :param sim: simulation descriptor
        """
        stage = self.sim_most_recent_stage(sim)
        with oxpy.Context():
            assert (self.folder_path(sim, stage) / "input").exists()
            os.chdir(self.folder_path(sim, stage))
            manager = oxpy.OxpyManager("input")
            manager.run(steps=int(stage.time_length()), print_output=True)
            del manager

    def lorenzian_to_flavian(self,
                             write_path: Union[Path, str],
                             sims: Union[None, list[PatchySimulation]] = None):
        """
        Converts lorenzian-type files to flavian
        """
        # standardize io args
        write_path = os.path.expanduser(write_path)
        if isinstance(write_path, str):
            write_path = Path(write_path)
        assert write_path.exists(), f"Location to contain ensemble copy data {str(write_path)} does not exist!"
        assert write_path != self.tld(), "Cannot format-translate to ensemble directory"
        if sims is None:
            sims = self.ensemble()

        for sim in sims:
            for stage in self.sim_get_stages(sim):
                # read data
                sim_folder_path = write_path / (self.long_name() + "_flav") / sim.get_folder_path()
                if stage.idx() > 0:
                    sim_folder_path = sim_folder_path / stage.name()
                try:
                    sim_folder_path.mkdir(parents=True)
                    if not (self.folder_path(sim, stage).exists()):
                        continue
                    lorenzian_to_flavian(self.folder_path(sim, stage), sim_folder_path,
                                         conf_name=self.sim_get_param(sim, "conf_file"))
                    if (self.paramstagefile(sim, stage, "lastconf_file")).exists():
                        shutil.copy(self.paramstagefile(sim, stage, "lastconf_file"),
                                    sim_folder_path / self.sim_get_param(sim, "lastconf_file"))
                    if (self.paramstagefile(sim, stage, "trajectory_file")).exists():
                        shutil.copy(self.paramstagefile(sim, stage, "trajectory_file"),
                                    sim_folder_path / self.sim_get_param(sim, "trajectory_file"))
                    else:
                        print(f"No last_conf.dat file for simulation {str(sim)} stage {stage.name()}")
                except FileExistsError:
                    print("Warning: Simulation directory already exists!")
        print(f"Exported to {str(write_path)}, everything - or something at least - went ok")


    def write_sim_observables(self, sim: PatchySimulation):
        """
        DEPRECATED
        write observables for a simulation in an observables.json file
        todo: use ipy_oxdna methods
        """
        if len(self.observables) > 0:
            with open(self.folder_path(sim) / "observables.json", "w+") as f:
                json.dump({f"data_output_{i + 1}": obs.to_dict() for i, obs in enumerate(self.observables.values())}, f)

    def dump_metadata(self):
        """
        Saves metadata stored in `self.metadata` to a metadata file
        Also saves the analysis pathway
        todo: make this comprehensive
        """

        # there's a bunch of metadata["ensemble_confg"] stuff that gets passed as raw text but should be saved as json to be comprehensive
        # todo: are there any more of these?
        ec = self.metadata["ensemble_config"]
        ec[OBSERABLES_KEY] = list(self.observables.values())
        ec[CONST_PARAMS_KEY] = self.const_params
        ec[ENSEMBLE_PARAMS_KEY] = self.ensemble_params.to_list()
        ec[SERVER_SETTINGS_KEY] = self.server_settings

        # dump metadata dict to file, using temp file for atomic write
        metadata_path = get_input_dir() / self.metadata_file
        temp_path = metadata_path.with_suffix('.tmp')


        try:
            if "analysis_pipeline" in self.metadata:
                del self.metadata["analysis_pipeline"]
            with open(temp_path, "w") as f:
                json.dump({
                    "analysis_pipeline": self.analysis_pipeline,
                    "slurm_log": self.slurm_log.to_list(),
                    **self.metadata
                }, fp=f, indent=4, cls=self.json_rw)

            # Only replace the original file if serialization succeeded
            temp_path.replace(metadata_path)
        except Exception:
            # Clean up temp file if it exists
            if temp_path.exists():
                temp_path.unlink()
            raise

    def sim_get_next_stage(self, sim: PatchySimulation) -> Stage:
        try:
            return self.sim_most_recent_stage(sim).get_next()
        except NoStageTrajError:
            # if no stages have been run
            return self.sim_get_stages(sim)[0]  # first stage

    def ipy(self, sim: PatchySimulation, stage: Union[Stage, None] = None) -> Simulation:
        """
        :param sim: simulation descriptor
        :param stage: stage object or None
        :return: an ipy_oxdna simulation object for a simulation
        """
        if stage is None:
            try:
                stage = self.sim_most_recent_stage(sim, True)
            except IncompleteStageError as e: # totally allowed
                stage = e.stage()
        if self.sim_num_stages(sim) == 1 or stage.idx() == 0:
            sim_obj = Simulation(str(self.folder_path(sim, stage)))
        else:
            # parameterize stage from previous stage
            sim_obj = Simulation(str(self.folder_path(sim, self.sim_get_stage(sim, stage.idx() - 1))),
                                 str(self.folder_path(sim, stage)))
        sim_obj.set_builder(stage)  # assign stage object as sim builder
        return sim_obj

    def ipy_all(self, sim: PatchySimulation) -> list[Simulation]:
        return [self.ipy(sim, stage) for stage in self.sim_get_stages(sim)]

    def start_simulations(self,
                          e: Union[None, list[PatchySimulation]] = None,
                          stage: Union[str, None] = None):
        """
        :param e: None or a list of simulation descriptor
        :param stage: stage object or None
        Starts simulations
        If no simulation param is specificed will start all
        """
        if e is None:
            e = self.ensemble()

        # normal circumstances - no batch exec, do the old way
        if not self.server_settings.is_batched():
            if not self.is_setup_done(e, stage):
                self.do_setup(e, stage)
            for sim in e:
                self.start_simulation(sim, stage_name=stage)
        else:
            # if the number of simulations to run requires more than one task

            mgr = self.server_settings.cuda_mps
            for sim in e:
                if stage is None:
                    sim_stage = self.sim_get_next_stage(sim)
                else:
                    sim_stage = self.sim_get_stage(sim, stage)
                assert sim_stage is not None
                if not self.folder_path(sim, sim_stage).exists():
                    self.folder_path(sim, sim_stage).mkdir(parents=True)
                # if stage unspecified

                ipysim = self.ipy(sim, sim_stage)
                ipysim.build(clean_build="force")  # todo: integrate stage assembly
                mgr.queue_sim(ipysim)
            self.get_logger().info("Let the simulating commence!")
            # batch execution for CUDA + MPS
            # matt says to use worker_manager instead of run
            mgr.worker_manager()
            # TODO: better slurm logging!

    def is_setup_done(self, sims: list[PatchySimulation], stage: Union[None, str, Stage]) -> bool:
        """
        tests if setup is complete for the given simulations and stages
        :param sims: list of patchy simulation descriptors
        :param stage: None, a Stage object, or a stage name
        :return: true if all the sims have the files required to run, false otherwise
        """
        # loop sims
        for sim in sims:
            # input sanitization
            if isinstance(stage, Stage):
                # stage info provided as Stage object - good!
                sim_stage = stage
            elif isinstance(stage, str):
                # stage info provided as stage name identifier string - acceptable!
                sim_stage = self.sim_get_stage(sim, stage)
            else:
                # stage info not provided! let's improvise!
                try:
                    sim_stage = self.sim_most_recent_stage(sim)
                except NoStageTrajError as e:
                    # if we don't have traj for any stage that means either setup is incomplet3
                    # or the stage is stage 0
                    sim_stage = self.sim_get_stages(sim)[0]

            # check that folder exists
            if not sim_stage.sim.sim_dir.exists():
                return False
            # check for various files
            if not self.sim_get_stage_top(sim, sim_stage).exists():
                return False
            if not (self.folder_path(sim, sim_stage) / self.sim_stage_get_param(sim, sim_stage, "conf_file")).exists():
                return False
            if not (self.folder_path(sim, sim_stage) / "input").exists():
                return False
        return True

    def ok_to_run(self, sim: PatchySimulation, stage: Stage) -> bool:
        """
        tests if a simulation is ready to run
        :param sim: simulation identifier
        :param stage: Stage object
        :return: true if the simulation (and stage) is ready to run, false otherwise
        """
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
        except IncompleteStageError:
            # previous stage is incomplete; warn and return False
            self.get_logger().warning(
                f"Cannot execute stage {stage.name()} for sim {repr(sim)} when stage {self.stage().name()}"
                f" is incomplete! aborting!")
            return False
        except NoStageTrajError:
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

        if self.server_settings.is_server_slurm():
            try:
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
            except NoStageTrajError as e:
                pass  # if no stage has a traj error everything is probably fine, just needs to run 1st stage
        return True

    def check_stage(self, sim: PatchySimulation) -> Stage:
        """
        Checks staging status of this simulation
        :param sim: simulation identifier
        :return: a Stage object
        """
        # if no stage name provided use first stage
        try:
            stage = self.sim_most_recent_stage(sim)
            if stage.idx() + 1 < self.sim_num_stages(sim):
                stage = self.sim_get_stage(sim, stage.idx() + 1)
            else:
                # TODO: replace with proper exception
                self.get_logger().info(f"Simulation {sim} has no more stages to execute!")
                return -1
        except NoStageTrajError:
            # default to stage 0 if no previous stages found
            stage = self.sim_get_stage(sim, 0)
        return stage

    def start_simulation(self,
                         sim: PatchySimulation,
                         script_name: str = "slurm_script.sh",
                         stage_name: Union[None, str] = None,
                         is_analysis: bool = False,
                         force_ignore_ok_check=False,
                         retries=3,
                         backoff_factor=2) -> int:
        """
        Starts an oxDNA simulation. direct invocation is not suggested; run from start_simulations()

        NOTE: even for non-slurm jobs we use subprocess.run on an execution file rather than
        oxpy.run for backwards compatibility with oxDNA_torsion, which isn't compatible with oxpy
        because oxDNA versions are a horrible nightmare world

        todo: clean this up

        :param sim: simulation to start
        :param script_name: the name of the slurm script file, why would this be something other than the default?
        :param stage_name: name of stage to start, or None to find a default one
        :param is_analysis: false
        :param force_ignore_ok_check: skip checking if the simulation is okay to run
        :param retries: number of times to retry submitting the slurm script
        :param backoff_factor: factor to increase time between job submissions, to avoid antagonizing the Slurm-Demon
        """

        if stage_name is not None:
            stage = self.sim_get_stage(sim, stage_name)
        else:
            stage = self.check_stage(sim)
        assert isinstance(stage, Stage)

        if not isinstance(script_name, str):
            raise TypeError("script_name must be a string")

        if not self.is_setup_done([sim], stage):
            self.do_setup([sim], stage)

        if not force_ignore_ok_check and not self.ok_to_run(sim, stage):
            self.get_logger().warning(f"Stage {stage.name()} not ok to run for sim {repr(sim)}")
            return -1

        # get slurm log jobname
        if not is_analysis:
            job_type_name = "oxdna"
        else:
            raise Exception("Since implementing this I have learned that dnaAnalysis doesn't work for patchy particles")

        if not (self.folder_path(sim, stage) / script_name).exists():
            raise FileNotFoundError(f"No executable file {str(self.folder_path(sim, stage) / script_name)}")
        
        if self.server_settings.is_server_slurm():
            command = f"sbatch --chdir={self.folder_path(sim, stage)}"
        # for non-slurm servers
        else:
            command = f"bash {script_name} > simulation.log"

        # shouldn't be nessecary anymore but whatever
        if not self.paramstagefile(sim, stage, "conf_file").exists():
            confgen_slurm_jobid = self.gen_conf(sim)
            if self.server_settings.is_server_slurm():
                command += f" --dependency=afterok:{confgen_slurm_jobid}"
        if self.server_settings.is_server_slurm():
            command += f" {script_name}"
        submit_txt = ""

        # DO NOT DO RETRIES ON NON SLURM MACHINE!!! THIS IS SUSPECTED OF DESTROYING MY ENTIRE LIFE!!!!
        if not self.server_settings.is_server_slurm():
            retries = 1
        # for nonslurm servers retries should equal 1
        for i in range(retries):
            # always run async for submitting slurm jobs, optionally also run serially
            do_async = not self.server_settings.is_server_slurm() and not self.is_run_serial()
            # execute or sbatch slurm script
            # todo something with error
            submit_txt, _ = self.bash_exec(command,
                                        is_async=do_async,
                                        cwd=self.folder_path(sim, stage)
                                        )
            if submit_txt:
                break
            time.sleep(backoff_factor ** i)

        # if we're running this on a slurm server, log the slurm job info (for some reason?)
        if self.server_settings.is_server_slurm():
            if not submit_txt:
                raise Exception(f"Submit slurm job failed for simulation {sim}\n"
                                f"Submit command: `{command}`")

            jobid = int(re.search(SUBMIT_SLURM_PATTERN, submit_txt).group(1))
            self.append_slurm_log(SlurmLogEntry(
                job_type=job_type_name,
                pid=jobid,
                simulation=sim,
                script_path=self.folder_path(sim, stage) / script_name,
                log_path=self.folder_path(sim, stage) / f"run{jobid}.out"
            ))
            return jobid
        else:
            return -1

    def is_run_serial(self):
        """
        what the fucking shit is this
        """
        try:
            # if param exists, return serial param val
            return self.get_param("run_serial")
        except NoSuchParamError as e:
            # if not, default to false
            return False

    # ------------- ANALYSIS FUNCTIONS --------------------- #

    def has_data_file(self, step: PipelineStepDescriptor, sim: PatchySimDescriptor) -> bool:
        """
        Test if there's a cache file for the given analysis step and patchy simulation
        :param sim: simulation identifier
        :param step: analysis pipeline step descriptor
        :return: true if a cache file exists for this simulation and step, false otherwise
        """
        return self._get_cache_file(step, sim).exists()

    def param_value_valid(self, pv: ParameterValue) -> bool:
        """
        tests if a parameter value exists in ensemble params
        :param pv: parameter value
        :return: true if the ParameterValue exists in any of the ensemble parameters
        """
        return any([pv in ep for ep in self.ensemble_params])

    def is_multiselect(self, selector: PatchySimDescriptor, exceptions: tuple[EnsembleParameter, ...] = ()) -> bool:
        """
        :return: true if the provided selector will match multiple PatchySimulation objects, false otherwise
        """
        if isinstance(selector, PatchySimulation):
            return False
        try:
            assert all([
                ParameterValue(param_name=pv[0], param_value=pv[1]) if isinstance(pv, tuple)
                else self.param_value_valid(pv) for pv in selector]), f"Invalid selector {selector}"
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
                 time_steps: Union[range, None, int] = None) -> Union[PipelineData, list[PipelineData]]:
        """
        horrific monster of a class method. Acquires data for a step in the analysis pipeline,
        doing any/all required calculations. If passed a list of simultions, it will analyze
         them and return a list of the data outputs. if passed a tuple, it will reutnr them as a single output
        :param step: an analysis step
        :param sim: a patchy simulation object, descriptor of a PatchySimulation object, list of PatchySimulation
        object, or tuple of ParameterValues that indicates a PatchySimulation object It's VERY important to note that
        the list and tuple options have DIFFERENT behaviors!!!
        :param time_steps: a range of timesteps to get data at. if None, the steps will be calculated automatically
        :return: data for a step in the analysis pipeline and a simulation, at the given time range

        """

        # preprocessing: MAKE SURE THIS IS A STEP OBJECT
        step = self.get_pipeline_step(step)

        # standardize timepoint input
        if type(time_steps) == int:
            time_steps = range(time_steps, time_steps+step.output_tstep, step.output_tstep)
            assert len(time_steps) == 1

        #  if we've provided a list of simulations
        if sim is None:
            # TODO: SHOULD DEFAULT BEHAVIOR BE E.ENSEMBLE() INSTEAD ??
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

        # if you try to actually use an aggregate step the sun will expand and swallow the earth
        is_aggregate_params: bool = isinstance(step, AggregateAnalysisPipelineStep) and self.is_multiselect(sim,
                                                                                                            step.params_aggregate_over)
        is_multiparam: bool = not isinstance(step, AggregateAnalysisPipelineStep) and self.is_multiselect(sim)
        if is_aggregate_params or is_multiparam:
            return self.get_data_multiselector(sim, step, time_steps)

        # check if this is a slurm job (should always be true I guess? even if it's a jupyter notebook)
        # I don't remember why this code even exists
        if is_slurm_job():
            slurm_job_info = self.slurm_job_info()
            if slurm_job_info is None:
                # todo: a better exception type
                raise Exception(f"Slurm job not found")
            try:
                self.append_slurm_log(SlurmLogEntry(
                    job_type="analysis",
                    pid=int(slurm_job_info["JobId"]),
                    simulation=sim,
                    script_path=slurm_job_info["Command"],
                    start_date=datetime.datetime.strptime(slurm_job_info["SubmitTime"], "%Y-%m-%dT%H:%M:%S"),
                    log_path=self._log_file_name,
                    additional_metadata={
                        "step": step.name
                    }
                ))
            except KeyError as e:
                self.get_logger().error(f"Slurm job info missing key {e.args[0]}\nSlurm job keys = {str(slurm_job_info)}")
                raise e

        # TIMESTEPS!!!!
        # if timesteps were not specified
        # todo: replace with a better descriptor
        if time_steps is None:
            simlength = self.time_length(sim)
            # time steps generally do not start from step 0
            time_steps = range(
                # step.output_tstep,
                0,
                simlength - (simlength % int(step.output_tstep)),
                int(step.output_tstep))
            self.get_logger().info(f"Constructed time steps {time_steps}")
        else:
            if not isinstance(time_steps, range) or isinstance(time_steps, slice):
                raise TypeError(f"Invalid type `{type(time_steps).__name__} for parameter `time_steps`")
            if (time_steps.step % step.output_tstep != 0
                    or time_steps.start % step.output_tstep != 0
                    or time_steps.stop % step.output_tstep != 0):
                raise ValueError(f"Specified step interval {time_steps} not consistant with {step} output time " \
                                                             f"interval {step.output_tstep}")

        self.get_logger().info(
            f"Retrieving data for analysis step {step.name} and simulation(s) {str(sim)} over timeframe {time_steps}")
        # DATA CACHING
        # check if data is already loaded
        if self.is_data_loaded(sim, step, time_steps):
            cached_data = self.get_cached_analysis_data(sim, step)
            return cached_data[time_steps]

        # check if we have cached data for this step already
        if not self.analysis_pipeline.is_force_recompute(step) and self.has_data_file(step, sim):
            self.get_logger().info(
                f"Cache file for simulation {get_descriptor_key(sim)} and step {step} exists! Loading...")
            cache_file_path = self._get_cache_file(step, sim)
            cached_data = step.load_cached_files(cache_file_path)
            # if we already have the data needed for the required time range
            if cached_data.matches_trange(time_steps):
                # that was easy!
                self.get_logger().info(f"All data in file! That was easy!")
                return cached_data[time_steps]
            else:
                self.get_logger().info(f"Cache file missing data! Missing timepoints are "
                                       f"{np.argwhere(~cached_data.compare_tranges(time_steps)) * step.input_tstep} ")
        if self.is_do_analysis_parallel():
            lock = multiprocessing.Lock()
            lock.acquire()
        try:
            # compute data for previous steps
            self.get_logger().info(f"Computing input data for step {step.name} for simulation {repr(sim)}...")
            data_in = self._get_step_input_data(step, sim, time_steps)
        finally:
            if self.is_do_analysis_parallel():
                lock.release()
        # TODO: make sure this can handle the amount of args we're feeding in here
        # execute the step!
        # TODO: handle existing data that's incomplete over the required time interval
        self.get_logger().info(f"Executing computations for step {step.name} on simulation data {repr(sim)}...")
        try:
            data: PipelineData = step(*data_in)[time_steps]
            if step.get_output_data_type() == PipelineDataType.PIPELINE_DATATYPE_DATAFRAME:
                for p in sim.param_vals:
                    data.get()[p.param_name] = p.value_name()
        except MissingDataError as e:
            e.other = f" while processing simulation {repr(sim)} for step {step.name}"
            raise e
        except TypeError as e:
            raise TypeError(f"Pipeline configuration error: incorrect number of arguements passed to __call__ method for step {step.name}") from e
        if self.is_do_analysis_parallel():
            lock.acquire()
        try:
            self.get_logger().info(f"Caching data in file `{self._get_cache_file(step, sim)}`")
            if isinstance(sim, tuple):
                assert not self.is_multiselect(sim), "Can't cache multiselected data in a single cache file!"
                sim = self.get_simulation(*sim)
            step.cache_data(data, self._get_cache_file(step, sim))
        finally:
            if self.is_do_analysis_parallel():
                lock.release()
        if not self.is_nocache():
            self.analysis_data[step, sim] = data
        return data

    def get_data_multiselector(self,
                               sim: PatchySimDescriptor,
                               step: AnalysisPipelineStep,
                               time_steps) -> PDPipelineData:
        """
        gets analysis step data for a multi-simulation selector
        :param sim: patchy simulation identifier
        :param step: an analysis pipeline step
        :param time_steps: timepoints in the simulation for which to get data
        """
        # if the simulation selector provided doesn't line up with the aggregation params
        # (or the step isn't an aggregation step), this method will return a grouping of results. somehow.
        simulations_list: list[PatchySimulation] = self.get_simulation(*sim)
        # get data for each simulation
        data: list[PipelineData] = self.get_data(step, simulations_list, time_steps)
        # ugh wish python had switch/case
        if step.get_output_data_type() == PipelineDataType.PIPELINE_DATATYPE_DATAFRAME:
            self.get_logger().info(f"Merging data from simulations: {describe_param_vals(*sim)}")
            timerange = np.array(list(set.intersection(*[set(s.trange()) for s in data])))
            # iter simulation data
            # todo: custom exception for merging data with inconsistant sim params
            df = pd.concat([d.get() for d in data], axis=0)
            df = df.loc[np.isin(df[TIMEPOINT_KEY], timerange)]
            return PDPipelineData(df, timerange)
        elif step.get_output_data_type() == PipelineDataType.PIPELINE_DATATYPE_GRAPH:
            raise Exception("I haven't bothered trying to join graph data yet")
        else:
            raise Exception("Attempting to merge non-mergable data type")

    def _get_step_input_data(self,
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
            assert isinstance(sim, PatchySimulation) or not self.is_multiselect(
                sim), "Analysis pipeline head nodes should only take single simulations"
            if not isinstance(sim, PatchySimulation):
                sim = self.get_simulation(*sim)
            # todo: watch this space!! it may react VERY BADLY to non-automated stages!!!
            stages: list[Stage] = self.sim_get_stages_between(sim,
                                                              time_steps.start,
                                                              time_steps.stop,
            # autogenerate stages, since we're trying to read actual data
                                                              auto=True,
                                                              inclusive=True)
            files = [self, sim, stages]
            # get list of files that are needed for this file
            for file_name_in in step.get_data_in_filenames():
                # I don't remember when regexes WERE supported
                assert isinstance(file_name_in, str), f"Unexpected file_name_in parameter type {type(file_name_in)}." \
                                                      f"Regexes no longer supported."
                # add all files of this type, in an order corresponding to `stages`
                try:
                    # for files that are parameters (topology, conf, etc.)
                    file_names = [
                        self.folder_path(sim, stage) / self.sim_stage_get_param(sim, stage, file_name_in)
                        for stage in stages
                    ]
                except NoSuchParamError as e:
                    # for files that are observable outputs
                    file_names = [
                        self.folder_path(sim, stage) / file_name_in
                        for stage in stages
                    ]
                files.append(file_names)
            return files
        else:  # honestly this is still complecated but not as bad
            return [
                self.get_data(prev_step,
                              sim,
                              time_steps)
                for prev_step in self.analysis_pipeline.steps_before(step)
            ]

    def _get_cache_file(self,
                        step: PipelineStepDescriptor,
                        sim: Union[tuple[ParameterValue, ...], PatchySimulation]) -> Path:
        """
        Retrieves a path to a file of analysis cache data for the given analysis step and
        simulation descriptor

        :param step : an AnalysisPipelineStep object or an int indxer for such an object
        :param sim : either a PatchySimulation object or a tuple of ParameterValue objects specifying a set of PatchySimulation objects
        :return: a path to a data file
        """
        # get step object if it was passed as an index
        step = self.get_pipeline_step(step)
        # if single simulation
        if isinstance(sim, PatchySimulation):
            # cache analysis data in the simulation data folder
            # don't incorporate staging info into analysis pipeline cache file paths
            return self.folder_path(sim) / step.get_cache_file_name()
        else:
            # cache analysis data in a folder in the top level directory
            return self.tld() / describe_param_vals(*sim) / step.get_cache_file_name()

    def slurm_job_info(self, jobid: int = -1, retries=3, backoff_factor=2) -> Union[dict, None]:
        """
        todo: this cannot be the optimal way to d oths
        :return: a dict of data relating to a slurm job
        """
        if jobid == -1:
            retr_id = os.environ.get("SLURM_JOB_ID")
            assert retr_id is not None
            jobid = int(retr_id)

        for i in range(retries):
            jobinfo, _ = self.bash_exec(f"scontrol show job {jobid}")

            if jobinfo and jobinfo != "slurm_load_jobs error: Invalid job id specified":
                jobinfo = jobinfo.split()
                jobinfo = {key: value for key, value in [x.split("=", 1) for x in jobinfo if len(x.split("=", 1)) == 2]}
                # Cache it
                SLURM_JOB_CACHE[jobid] = jobinfo
                return jobinfo

            time.sleep(backoff_factor ** i)

        return None

    def bash_exec(self, command: str, is_async=False, cwd: Union[None, str]=None):
        """
        Executes a bash command and returns the output
        :param command: a string to run with subprocess
        :param is_async: whether to run the command asynchrously
        :param cwd: directory to change to before running the command, or None to not change the directory
        """
        if cwd:
            self.get_logger().info(f"Executing `{command}`, in working directory {cwd}")
        else:
            self.get_logger().info(f"Executing `{command}`")
        if not is_async:
            response = subprocess.run(command,
                                      shell=True,
                                      capture_output=True,
                                      text=True,
                                      check=False,
                                      cwd=cwd)
        else:
            response = subprocess.Popen(command.split(), cwd=cwd)
        if response.stdout:
            self.get_logger().info(f"Command Stdout: `{response.stdout}`")
        if response.stderr:
            self.get_logger().error(f"Command Stderr: `{response.stderr}`")
        return response.stdout, response.stderr

    def sim_get_timepoint_stage(self, sim: PatchySimulation, timepoint: int) -> Union[Stage, None]:
        for stage in self.sim_get_stages(sim):
            if stage.start_time() < timepoint < stage.end_time():
                return stage
        return None  # raise exception?

    def dnaAnalysis(self,
                    obs: Observable, # todo: multiple observables, probably not hard but also not needed
                    sim: Union[None, PatchySimDescriptor] = None,
                    stage: Union[None, Stage, str] = None):
        """
        runs oxDNA's DNAnalysis tool on a simulation and stage, producing the output file(s)
        specified by the observable
        :param obs: an Observable object specifying the analysis to run
        :param sim: a PatchySimulation object or a descriptor of one, or None to run on all simulations
        :param stage: a Stage object or stage name string, or None to run on all stages
        """
        if sim is None:
            sim = self.ensemble()
        if isinstance(sim, list):
            for s in sim:
                self.dnaAnalysis(obs, s, stage)
        else:
            if stage is None:
                # automatically run dnaAnalysis for all sim-like stages
                for stage in self.sim_get_stages(sim, True):
                    self.dnaAnalysis(obs, sim, stage)
            elif isinstance(stage, str):
                stage = self.sim_get_stage(sim, stage)
            # make temporary directory to run dnaAnalysis
            with tempfile.TemporaryDirectory() as tmpdir:
                shutil.copyfile(self.folder_path(sim, stage) / "input",
                                tmpdir + "/input")
                with oxpy.Context():
                    infile = oxpy.InputFile()
                    infile.init_from_filename(str(self.folder_path(sim, stage) / "input"))
                    # add analyis observable
                    infile["trajectory_file"] = str(self.paramstagefile(sim, stage, "trajectory_file"))
                    # file paths to pypatchy directory
                    for key in self.writer.reqd_args():
                        infile[key] = str(self.folder_path(sim, stage) / self.sim_stage_get_param(sim, stage, key))
                    infile["plugin_search_path"] = self.get_param("plugin_search_path")
                    # with (Path(tmpdir) / infile["observables_file"]).open("w") as f:
                    #     json.dump({"analysis_data_output_1": obs.export()["output"]}, f)
                    with open(tmpdir + "/input", "w") as f:
                        for key in infile.keys():
                            f.write(f"{key} = {infile[key]}\n")
                        f.write("analysis_data_output_1 = " + repr(obs))
                    if any([column.type_name == "PatchyBonds" for column in obs.cols]):
                        # if we are using lorenzo's format, this gets much harder because
                        # of all the additional required files
                        # hardcoding, but let me state for the record i hate it
                        for i, _ in enumerate(self.sim_get_particles_set(sim)):
                            patches_file = self.folder_path(sim, stage) / f"patches_{i}.dat"
                            if not patches_file.exists():
                                raise FileNotFoundError(f"Could not find patches file `patches_{i}.dat in {sim}")
                            else:
                                shutil.copyfile(patches_file, tmpdir + f"/patches_{i}.dat")
                    cmd_out, cmd_err = self.bash_exec(self.server_settings.oxdna_path + "/build/bin/DNAnalysis input",
                                                          cwd=tmpdir)
                    self.get_logger().info("DNAnalysis OUT: " + cmd_out)
                try:
                    shutil.copyfile(tmpdir + f"/{obs.file_name}",
                                self.folder_path(sim, stage) / obs.file_name)
                except FileNotFoundError as e:
                    self.get_logger().error("ERROR: " + cmd_err)
                    self.get_logger().error(f"DNAnalysis of simulation {str(sim)}, stage {stage.name()} did not "
                                            f"produce the expected file with name {obs.file_name}")
                    if not Path(tmpdir).exists():
                        self.get_logger().error(f"Temporary directory {tmpdir} does not exist! WHAT THE HELL")
                    self.get_logger().error(f"Copying temporary directory to {get_output_dir() / 'logs' / Path(tmpdir).name}...")
                    shutil.copytree(Path(tmpdir), get_output_dir() / "logs" / Path(tmpdir).name)
                    raise e

def process_simulation_data(args: tuple[PatchySimulationEnsemble, AnalysisPipelineStep, PatchySimulation, range]):
    ensemble, step, s, time_steps = args
    return ensemble.get_data(step, s, time_steps)


def shared_ensemble(es: list[PatchySimulationEnsemble],
                    ignores: set = set()) -> Union[list[list[PatchySimulation]], None]:
    """
    TODO: unit test
    Computes the simulation specs that are shared between the provided ensembles
    :param es: a list of simulation ensembles
    :param ignores: a set of const parameter names to ignore when constructing simulation overlap
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
        names.update([p.param_key for p in e.__ensemble_params])
        for p in e.__ensemble_params:
            if p.param_key not in name_vals:
                name_vals[p.param_key] = {pv.value_name() for pv in p.param_value_set}
            else:
                name_vals[p.param_key] = name_vals[p.param_key].intersection(
                    [pv.value_name() for pv in p.param_value_set])

    # if any simulation is missing a parameter
    if not all([len(name_vals[name]) > 0 for name in names]):
        return None

    ensembles = []
    for e in es:
        valset: list[list[ParameterValue]] = []
        # make sure to order ensemble parameters correctly
        for p in e.__ensemble_params:
            union_vals = name_vals[p.param_key]
            vals = [pv for pv in p if pv.value_name() in union_vals]
            valset.append(vals)

        ensembles.append([PatchySimulation(sim) for sim in itertools.product(*valset)])

    return ensembles
