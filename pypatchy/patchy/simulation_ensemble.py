from __future__ import annotations

import datetime
import itertools
import multiprocessing
import shutil
import tempfile
import time
from json import JSONDecodeError
from typing import Any

import subprocess
import re
import logging

# import oat stuff
from oxDNA_analysis_tools.UTILS.oxview import from_path
from oxDNA_analysis_tools.file_info import file_info
from oxDNA_analysis_tools.UTILS.RyeReader import get_confs, describe, write_conf
from oxDNA_analysis_tools.UTILS.data_structures import Configuration

from ipy_oxdna.oxdna_simulation import SimulationManager, Simulation

from .simulation_specification import get_param_set
from ..analpipe.analyzable import Analyzable

from ..patchy.patchy_scripts import lorenzian_to_flavian
from ..analpipe.analysis_pipeline import AnalysisPipeline
from .pl.plscene import PLPSimulation
from .stage import Stage, NoStageTrajError, IncompleteStageError, StageTrajFileEmptyError
from ..analpipe.analysis_data import PDPipelineData, TIMEPOINT_KEY
from ..analpipe.analysis_pipeline_step import *
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

METADATA_FILE_KEY = "sim_metadata_file"
LAST_CONTINUE_COUNT_KEY = "continue_count"

SUBMIT_SLURM_PATTERN = r"Submitted batch job (\d+)"


def describe_param_vals(*args) -> str:
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
    returns a string representing the provided descriptor
    """
    return sim if isinstance(sim, str) else str(sim) \
        if isinstance(sim, PatchySimulation) else describe_param_vals(*sim)


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
    External method to construct PatchySimulationEnsemble objects. nightmare,
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
            try:
                cfg = json.load(cfg_file)
            except JSONDecodeError as e:
                raise JSONDecodeError(msg=f"Error parsing patchy ensemble spec file {str(cfg_file_path)}! {e.msg}",
                                      doc=e.doc,
                                      pos=e.pos)
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

    params = []

    # load particles
    # todo: revise
    if any(key in cfg for key in [PARTICLE_TYPES_KEY, "rule", "cube_types"]):
        if PARTICLE_TYPES_KEY in cfg:
            assert isinstance(cfg[PARTICLE_TYPES_KEY], dict), "Must include load info"
            ptypedict = {**cfg[PARTICLE_TYPES_KEY]}
            particles = load_pl_particles(**ptypedict)

        elif "cube_types" in cfg:
            if len(cfg["cube_types"]) > 0 and isinstance(cfg["cube_types"][0], dict):
                rule: PolycubesRule = PolycubesRule(rule_json=cfg["cube_types"])
            else:
                rule: PolycubesRule = PolycubesRule(rule_str=cfg["cube_types"])
            particles = polycube_rule_to_PL(rule)
        elif "rule" in cfg:  # 'rule' tag assumes serialized rule string
            # please for the love of god use this one
            rule: PolycubesRule = PolycubesRule(rule_str=cfg["rule"])
            particles = polycube_rule_to_PL(rule)
        else:
            raise Exception("wtf")
        params.append(ParticleSetParam(particles))
    else:
        print("Warning: No particle info specified!")

    # handle multidentate params
    if NUM_TEETH_KEY in cfg[CONST_PARAMS_KEY]:
        assert DENTAL_RADIUS_KEY in cfg[CONST_PARAMS_KEY]
        num_teeth = cfg[CONST_PARAMS_KEY][NUM_TEETH_KEY]
        dental_radius = cfg[CONST_PARAMS_KEY][DENTAL_RADIUS_KEY]
        # only bothering to support legacy conversion params here
        mdt_convert = MultidentateConvertSettings(num_teeth, dental_radius)
        params.append(MDTConvertParams(mdt_convert))

    # load const params from cfg
    if CONST_PARAMS_KEY in cfg:
        for key, val in cfg[CONST_PARAMS_KEY].items():
            if key == NUM_TEETH_KEY or key == DENTAL_RADIUS_KEY:
                continue  # skip in const_params
            param_val = parameter_value(key, val)
            params.append(param_val)
    const_parameters = ParamSet(params)

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

    if "server_settings" in cfg:
        if isinstance(cfg["server_settings"], str):
            server_settings = load_server_settings(cfg["server_settings"])
        else:
            assert isinstance(cfg["server_settings"],
                              dict), f"Invalid type for key 'server_settings': {type(cfg['server_settings'])}"
            server_settings = PatchyServerConfig(**cfg["server_settings"])
    else:
        server_settings = get_server_config()
    # in case we need this

    # load ensemble params from cfg
    # there should always be ensemble params in the cfg
    ensemble_parameters = [
        EnsembleParameter(key, [
            parameter_value(key, val) for val in paramData
        ])
        for key, paramData in cfg[ENSEMBLE_PARAMS_KEY]
    ]

    ensemble = PatchySimulationEnsemble(export_name, setup_date, mdtfile, analysis_pipeline, default_param_set,
                                        const_parameters, ensemble_parameters, observables, analysis_file, mdt,
                                        server_settings)
    # TODO: verify presence of required params
    if "slurm_log" in mdt:
        for entry in mdt["slurm_log"]:
            sim = ensemble.get_simulation(**entry["simulation"])
            assert isinstance(sim,
                              PatchySimulation), f"Selector {entry['simulation']} selects more than one simulation."
            assert sim is not None, f"Slurm log included a record for invalid simulation {str(entry['simulation'])}"
            entry["simulation"] = sim
        ensemble.slurm_log = SlurmLog(*[SlurmLogEntry(**e) for e in mdt["slurm_log"]])
    ensemble.dump_metadata()
    return ensemble


class PatchySimulationEnsemble(Analyzable):
    """
    Stores data for a group of related simulations
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

    observables: dict[str: PatchySimObservable]

    # ------------ SETUP STUFF -------------#

    # simulation parameters which are constant over the entire ensemble
    const_params: ParamSet

    # parameter values to use if not specified in `const_params` or in the
    # simulation specification params
    # load from `spec_files/input_files/[name].json`
    default_param_set: ParamSet

    # log of slurm jobs
    slurm_log: SlurmLog

    # customizable server settings
    server_settings = PatchyServerConfig

    # output writer
    writer: PLBaseWriter

    def __init__(self,
                 export_name: str,
                 setup_date: datetime.datetime,
                 metadata_file_name: str,
                 analysis_pipeline: AnalysisPipeline,
                 default_param_set: ParamSet,
                 const_params: ParamSet,
                 ensemble_params: list[EnsembleParameter],
                 observables: dict[str, PatchySimObservable],
                 analysis_file: str,
                 metadata_dict: dict,
                 server_settings: Union[PatchyServerConfig, None] = None):
        super().__init__(analysis_file, analysis_pipeline)
        self.export_name = export_name
        self.sim_init_date = setup_date

        # load server settings ASAP
        if self.server_settings is not None:
            self.set_server_settings(server_settings)
        else:
            self.server_settings = get_server_config()
            self.writer = get_writer()

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

        self.metadata = metadata_dict
        self.slurm_log = SlurmLog()

        self.metadata_file = metadata_file_name

        self.default_param_set = default_param_set
        self.const_params = const_params

        self.ensemble_params = ensemble_params
        self.ensemble_param_name_map = {p.param_key: p for p in self.ensemble_params}
        self.observables = observables

    # --------------- Accessors and Mutators -------------------------- #
    def get_simulation(self, *args: Union[tuple[str, Any], ParameterValue], **kwargs) -> Union[
                                                                                               PatchySimulation,
                                                                                               list[PatchySimulation]]:
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
                sim_params.append([param_type[kwargs[pname]]])
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
                            pv = self.ensemble_param_name_map[k].lookup(v)
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
        Returns: the initialization date of the simulation, as a date string
        """
        return self.sim_init_date.strftime("%Y-%m-%d")

    def long_name(self) -> str:
        """
        Returns:
             the full name of the simulation
        """
        return f"{self.export_name}_{self.datestr()}"

    def get_logger(self) -> logging.Logger:
        """
        Returns: the simulation logger
        """
        return logging.getLogger(self.export_name)

    def is_do_analysis_parallel(self) -> bool:
        """
        Returns:
            true if the analysis is set up to run in parallel using multiprocessing.Pool
            and false otherwise.
        """
        return "parallel" in self.metadata and self.metadata["parallel"]

    def n_processes(self) -> int:
        return self.metadata["parallel"]

    def is_nocache(self) -> bool:
        return "nocache" in self.metadata and self.metadata["nocache"]

    def set_server_settings(self, stgs: Union[PatchyServerConfig, str]):
        if isinstance(stgs, PatchyServerConfig):
            self.server_settings = stgs
        else:
            self.server_settings = load_server_settings(stgs)
        self.writer = get_writer(self.server_settings.patchy_format)

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

    # def get_step_counts(self) -> list[tuple[PatchySimulation, int]]:
    #     return [
    #         (sim, self.time_length(sim))
    #         for sim in self.ensemble()
    #     ]

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

    def sim_get_param(self,
                      sim: PatchySimulation,
                      paramname: str) -> Any:
        """
        Returns the value of a parameter
        This method will first search in the simulation to see if this parameter
        has a specific value for this simulation, then in the ensemble const params,
        then the default parameter set
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
        if stage.has_var(param_name):
            return stage.get_var(param_name)
        else:
            return self.sim_get_param(sim, param_name)

    def sim_get_particles_set(self, sim: PatchySimulation) -> PLParticleSet:
        particles: PLParticleSet = self.sim_get_param(sim, PARTICLE_TYPES_KEY)
        try:
            mdt_convert: MultidentateConvertSettings = self.sim_get_param(sim, MDT_CONVERT_KEY)
            return particles.to_multidentate(mdt_convert)
        except NoSuchParamError as e:
            # if no multidentate convert, return particle set as-is
            return particles

    def get_param(self, paramname: str) -> Any:
        # use const_params
        if paramname in self.const_params:
            return self.const_params[paramname]
        # if paramname is surface level in default param set
        if paramname in self.default_param_set:
            return self.default_param_set[paramname]
        # # go deep
        if paramname in self.server_settings.input_file_params:
            return self.server_settings.input_file_params[paramname]
        # TODO: custom exception here
        raise NoSuchParamError(self, paramname)

    def paramfile(self, sim: PatchySimulation, paramname: str) -> Path:
        """
        Shorthand to get a simulation data file
        """
        return self.folder_path(sim) / self.sim_get_param(sim, paramname)

    def paramstagefile(self, sim: PatchySimulation, stage: Stage, paramname: str):
        return self.folder_path(sim) / stage.adjfn(self.sim_get_param(sim, paramname))

    def get_sim_particle_count(self,
                               sim: PatchySimulation,
                               particle_idx: int) -> int:
        """
        Args:
            sim: the patchy simulation to count for
            particle_idx: the index of the particle to get the count for
        Returns:
            the int
        """
        # grab particle name
        particle_name = self.sim_get_particles_set(sim).particle(particle_idx).name()
        return self.sim_get_param(sim, particle_name) * self.sim_get_param(sim, NUM_ASSEMBLIES_KEY)

    def sim_get_stages(self, sim: PatchySimulation) -> list[Stage]:
        """
        Computes stages
        Stage objects require other ensemble parameters so must be constructed as needed and not on init
        # TODO: make this dynamic, use a generator
        """
        assert isinstance(sim, PatchySimulation), "Cannot invoke this method with a parameter list!"
        try:
            stages_info: dict = self.sim_get_param(sim, STAGES_KEY)
            # incredibly cursed line of code incoming
            stages_info = [{"idx": i, **stage_info} for i, (stage_name, stage_info) in
                           enumerate(sorted(stages_info.items(), key=lambda x: x[1]["t"]))]
            stages = []
            # loop stages in stage list from spec
            for stage_info in stages_info:
                # get stage idx
                i = stage_info["idx"]

                # find num assemblies
                if "num_assemblies" in stage_info:
                    num_assemblies = stage_info["num_assemblies"]
                else:
                    num_assemblies = self.sim_get_param(sim, "num_assemblies")

                # get list of names of particles to add
                # if add_method is specified as RANDOM or is unspecified (default to RANDOM)
                if "add_method" not in stage_info or stage_info["add_method"].upper() == "RANDOM":
                    if "particles" in stage_info:
                        particle_id_lists = [
                            [pname] * (stage_info["particles"][pname] * num_assemblies)
                            for pname in stage_info["particles"]]

                    else:
                        particle_id_lists = [
                            [pidx] * self.get_sim_particle_count(sim, pidx) * num_assemblies
                            for pidx in range(self.sim_get_particles_set(sim).num_particle_types())
                        ]
                    stage_particles = list(itertools.chain.from_iterable(particle_id_lists))
                else:
                    mode, src = stage_info["add_method"].split("=")
                    if mode == "from_conf":
                        raise Exception(
                            "If you're seeing this, this feature hasn't been implemented yet although it can't be"
                            "THAT hard really")
                    elif mode == "from_polycubes":
                        with open(get_input_dir() / src, "r") as f:
                            pcinfo = json.load(f)
                            stage_particles = [
                                # for now only cube types are important
                                cube_info["type"] for cube_info in pcinfo["cubes"]
                            ]
                stage_init_args = {
                    "t": stage_info["t"],
                    "particles": stage_particles,
                }
                if "length" not in stage_info and "tlen" not in stage_info and "tend" not in stage_info:
                    if i + 1 != len(stages_info):
                        # if stage is not final stage, last step of this stage will be first step of next stage
                        stage_init_args["tend"] = stages_info[i + 1]["t"]
                    else:
                        # if the stage is the final stage, last step will be end of simulation
                        stage_init_args["tend"] = self.sim_get_param(sim, "steps")
                elif "tend" in stage_info:
                    stage_init_args["tlen"] = stage_info["tend"] - stage_info["t"]
                elif "length" in stage_info:
                    stage_init_args["tlen"] = stage_info["length"]
                else:
                    stage_init_args["tlen"] = stage_info["tlen"]

                if "name" in stage_info:
                    stage_name: str = stage_info["name"]
                else:
                    stage_name = f"stage{i}"
                # construct stage objects
                stages.append(Stage(sim,
                                    stages[i - 1] if i else None,
                                    self,
                                    stage_name,
                                    add_method=stage_info["add_method"] if "add_method" in stage_info else "RANDOM",
                                    **stage_init_args
                                    ))

        except NoSuchParamError as e:
            if e.param_name() != STAGES_KEY:
                raise e
            # if stages not found
            # default: 1 stage, density = starting density, add method=random
            num_assemblies = self.sim_get_param(sim, "num_assemblies")
            particles = list(itertools.chain.from_iterable([
                [p.type_id()] * self.sim_get_param(sim, p.name()) * num_assemblies
                for p in self.sim_get_particles_set(sim).particles()
            ]))

            stages = [Stage(
                sim,
                None,
                self,
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
                    box_side = (relvol * num_particles) ** (1 / 3)

                # if density format
                elif "density" in stage_info:
                    # box side length = cube root of n / density
                    density = stage_info["density"]
                    box_side = (num_particles / density) ** (1 / 3)

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

    def sim_get_total_stage_particles(self, sim: PatchySimulation, stage: Stage) -> int:
        """
        Computes the number of particles in a stage. Includes particles added in this stage and
        all previous stages
        """
        return sum([s.num_particles_to_add() for s in self.sim_get_stages(sim)[:stage.idx() + 1]])  # incl passed stage

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

    def ensemble(self) -> list[PatchySimulation]:
        """
        Returns a list of all simulations in this ensemble
        """
        return [PatchySimulation(e) for e in itertools.product(*self.ensemble_params)]

    def num_ensemble_parameters(self) -> int:
        """
        Returns:
            the number of ensemble parameters
        """
        return len(self.ensemble_params)

    def tld(self) -> Path:
        """
        Returns:
            ensemble root directory
        """
        return simulation_run_dir() / self.long_name()

    def folder_path(self, sim: PatchySimulation, stage: Union[Stage, None] = None) -> Path:
        """
        Returns:
            the path to the working directory of the given simulation, at the given stage
        """
        if stage is None or stage.idx() == 0:
            return self.tld() / sim.get_folder_path()
        else:
            return self.tld() / sim.get_folder_path() / stage.name()

    def save_pipeline_data(self):
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
        elif isinstance(sim, tuple):  # todo: check single descriptor
            if len(sim) > 0 and isinstance(sim[0], tuple):
                return self.time_length(self.get_simulation(*sim))
            else:
                return self.time_length(self.get_simulation(sim))
        else:
            try:
                stage = self.sim_most_recent_stage(sim)
                return self.sim_get_stage_last_step(sim, stage)
            except IncompleteStageError as e:
                return e.last_timestep()

    # ------------------------ Status-Type Stuff --------------------------------#
    def info(self, infokey: str = "all"):
        """
        prints help text, for non-me people or if I forget
        might replace later with pydoc
        """

        # this is something useful that will help us later
        def print_stages(stages_dict: dict, pad=0):
            # stage dicts not stage objects!
            stages = stages_dict["stages"]
            for stage_name in stages:
                stage = stages[stage_name]
                print("\t" * pad + f"Stage {stage_name}:")
                print("\t" * (pad + 1) + f"Starting Timestep: {stage['t']}")
                if "tend" in stage:
                    print("\t" * (pad + 1) + f"Ending Timestep: {stage['tend']}")
                print("\t" * (pad + 1) + f"Add Method: {stage['add_method']}")
                if len(stage["particles"]) > 0:
                    print("\t" * (pad + 1) + "Particles to add per assembly unit:")
                    for particle_type_name in stage["particles"]:
                        print("\t" * (pad + 1) + f"  {particle_type_name}: {stage['particles'][particle_type_name]}")
                else:
                    print("No particles added")

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
                if param.param_name == "stages":
                    print_stages(param.param_value, 2)

                else:
                    print(f"\t{param.param_name}:")
                    for name in param.group_params_names():
                        print(f"\t\t{name}: {param.param_value[name]}")
            else:
                print(f"\t{param.param_name}: {param.value_name()}")

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
            assert isinstance(get_writer(), FWriter), "Can only show confs for FlavioWriter!!!"
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
                                      self.sim_get_particles_set(sim))
        # scene: PLPSimulation()

    def is_traj_valid(self, sim: PatchySimulation, stage: Union[Stage, None] = None) -> bool:
        """
        Checks if a trajectory file is valid by counting the lines
        A valid traj file should have a line count which is a multiple of (number of particles + 3)
        I sould like to depreacate this method ASAP but right now oat segfaults with no warning when a traj
        file is corrupted
        Args:
            :param sim
            :param stage

        Return: True if the traj file is not corrupted, false otherwise
        """
        if stage is None:
            stage = self.sim_get_stages(sim)[0]
        traj_file = stage.adjfn(self.sim_get_param(sim, "trajectory_file"))
        num_particles = self.sim_get_total_stage_particles(sim, stage)
        assert num_particles > 0

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
            tuple(v.value_name() for v in sim.param_vals):
                {
                    step_name: self.has_data_file(self.analysis_pipeline[step_name], sim)
                    for step_name in self.analysis_pipeline.name_map
                }
            for sim in self.ensemble()
        }, orient="index")

    def all_folders_exist(self):
        return all(self.folder_path(s).exists() for s in self.ensemble())

    # def dna_analysis(self,
    #                  observable: Union[str, PatchySimObservable],
    #                  simulation_selector: Union[None, list[PatchySimulation], PatchySimulation] = None,
    #                  conf_file_name: Union[None, str] = None):
    #     """
    #     Runs the oxDNA utility DNAAnalysis, which allows the program to compute output for
    #     an observable for t
    #     """
    #     if isinstance(observable, str):
    #         observable = self.observables[observable]
    #     elif not isinstance(observable, PatchySimObservable):
    #         print("You definately forgot to put the observable first again. Gonna stop before you do any more damage")
    #         return
    #     if simulation_selector is None:
    #         simulation_selector = self.ensemble()
    #     if isinstance(simulation_selector, list):
    #         for sim in simulation_selector:
    #             self.dna_analysis(observable, sim)
    #     else:
    #         if conf_file_name is None:
    #             conf_file_name = self.sim_get_param(simulation_selector, "trajectory_file")
    #         self.write_setup_files(simulation_selector,
    #                                "input_dna_analysis",
    #                                {
    #                                    "conf_file": conf_file_name
    #                                }, analysis=True)
    #         server_config = get_server_config()
    #
    #         # write slurm script
    #         with open(self.folder_path(simulation_selector) / "dna_analysis.sh", "w+") as slurm_file:
    #             # bash header
    #
    #             self.write_sbatch_params(simulation_selector, slurm_file)
    #
    #             # skip confGenerator call because we will invoke it directly later
    #             slurm_file.write(f"{server_config['oxdna_path']}/build/bin/DNAnalysis input_dna_analysis\n")
    #
    #         self.bash_exec(f"chmod u+x {self.folder_path(simulation_selector)}/slurm_script.sh")
    #         self.start_simulation(simulation_selector,
    #                               script_name="dna_analysis.sh",
    #                               job_type="dna_analysis")
    #         self.dump_metadata()

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
    def do_setup(self,
                 sims: Union[list[PatchySimulation], None] = None,
                 stage: Union[None, str, Stage] = None):
        """

        """
        if stage is not None:
            self.get_logger().info(f"Stages other than zero don't require setup anymore! I hope!")
            return

        # check for mps stuff
        if sims is None:
            sims = self.ensemble()
        self.get_logger().info("Setting up folder / file structure...")
        for sim in sims:
            assert self.sim_get_param(sim, "print_conf_interval") < self.sim_get_param(sim, "steps")
            self.get_logger().info(f"Setting up folder / file structure for {repr(sim)}...")
            # create nessecary folders
            if not os.path.isdir(self.folder_path(sim, stage)):
                self.get_logger().info(f"Creating folder {self.folder_path(sim, stage)}")
                Path(self.folder_path(sim, stage)).mkdir(parents=True)
            else:
                self.get_logger().info(f"Folder {self.folder_path(sim, stage)} already exists. Continuing...")

            # write requisite top, patches, particles files
            self.get_logger().info("Writing .top, .txt, input, etc. files...")
            self.write_setup_files(sim, stage)
            # write observables.json if applicble
            if EXTERNAL_OBSERVABLES:
                self.get_logger().info("Writing observable json, as nessecary...")
                self.write_sim_observables(sim)
            # skip writing sbatch script if mps is off
            # write .sh script
            self.get_logger().info("Writing sbatch scripts...")
            self.write_run_script(sim)

    def write_run_script(self, sim: PatchySimulation, input_file="input"):
        # if no stage name provided use first stage
        stage = self.check_stage(sim)

        slurm_script_name = "slurm_script.sh"
        slurm_script_name = stage.adjfn(slurm_script_name)

        input_file = stage.adjfn(input_file)

        # write slurm script
        with open(self.folder_path(sim) / slurm_script_name, "w+") as slurm_file:
            # bash header

            self.server_settings.write_sbatch_params(self.export_name, slurm_file)

            # skip confGenerator call because we will invoke it directly later
            slurm_file.write(f"{self.server_settings.oxdna_path}/build/bin/oxDNA {input_file}\n")

        self.bash_exec(f"chmod u+x {self.folder_path(sim)}/{slurm_script_name}")

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

        if extras is None:
            extras = {}
        # set writer directory to simulation folder path
        self.writer.set_directory(self.folder_path(sim, stage))
        self.writer.set_abs_paths(self.server_settings.absolute_paths)

        # get server config
        if replacer_dict is None:
            replacer_dict = {}

        # if this is the first conf
        if stage.idx() == 0:
            assert stage.start_time() == 0, f"Stage {stage} has idx 0 but nonzero start time!"

            # generate conf
            scene = PLPSimulation()
            scene.set_temperature(self.sim_get_param(sim, "T"))
            particle_set = self.sim_get_particles_set(sim)
            # patches will be added automatically
            scene.set_particle_types(particle_set)

        else:
            # don't catch exxeption here
            last_complete_stage = self.sim_most_recent_stage(sim)
            scene = self.get_scene(sim, last_complete_stage)

        stage.apply(scene)

        # grab args required by writer
        reqd_extra_args = {
            a: self.sim_get_param(sim, a) for a in self.writer.reqd_args()
        }
        assert "conf_file" in reqd_extra_args, "Missing conf file info!"

        # write top, conf, and others
        files = self.writer.write(scene,
                                  **reqd_extra_args)

        # update top and dat files in replacer dict
        replacer_dict.update(files)
        replacer_dict["steps"] = stage.end_time()
        replacer_dict["trajectory_file"] = stage.adjfn(self.sim_get_param(sim, "trajectory_file"))
        extras.update(self.writer.get_input_file_data(scene, **reqd_extra_args))

        # create input file
        self.write_input_file(sim, stage, replacer_dict, extras, analysis)

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
                    else:
                        print(f"No last_conf.dat file for simulation {str(sim)} stage {stage.name()}")
                except FileExistsError:
                    print("Warning: Simulation directory already exists!")

    # def rw(self,
    #        write_writer: Union[BasePatchyWriter, str],
    #        write_path: Union[Path, str],
    #        read_writer: Union[BasePatchyWriter, str, None] = None,
    #        sims: Union[None, list[PatchySimulation]] = None):
    #     """
    #     Reads the data from the ensemble and writes it in a different place using a different format
    #     """
    #     # standardize io args
    #     if isinstance(write_writer, str):
    #         write_writer = get_writer(write_writer)
    #     if isinstance(read_writer, str):
    #         read_writer = get_writer(read_writer)
    #     elif read_writer is None:
    #         read_writer = self.writer
    #     assert type(write_writer) != type(read_writer), "Trying to read and write in the same format, which seems " \
    #                                               "pretty pointless"
    #     if isinstance(write_path, str):
    #         write_path = Path(write_path)
    #     assert write_path.exists(), f"Location to contain ensemble copy data {str(write_path)} does not exist!"
    #     assert write_path != self.tld(), "Cannot format-translate to ensemble directory"
    #     if sims is None:
    #         sims = self.ensemble()
    #
    #     (write_path / (self.long_name() + "_copy")).mkdir(parents=True)
    #
    #     for sim in sims:
    #         for stage in self.sim_get_stages(sim):
    #             # read data
    #             read_writer.set_directory(self.folder_path(sim, stage))
    #             input_file = Input(str(read_writer.directory()))
    #
    #             # TODO: automate params
    #             top_file = input_file.input["topology"]
    #             if Path(top_file).is_absolute():
    #                 top_file = Path(top_file).suffix
    #
    #             if isinstance(read_writer, FWriter) or isinstance(read_writer, JFWriter):
    #                 # handle absolute vs relative file paths
    #                 patchy_file_path = input_file.input["patchy_file"]
    #                 if Path(patchy_file_path).is_absolute():
    #                     patchy_file_path = Path(patchy_file_path).suffix
    #                 assert (read_writer.directory() / patchy_file_path).exists(), "Missing patchy file!"
    #                 particle_file_path = input_file.input["particle_file"]
    #                 if Path(particle_file_path).is_absolute():
    #                     particle_file_path = Path(particle_file_path).suffix
    #
    #                 ptypes = read_writer.read_particle_types(patchy_file_path, particle_file_path)
    #                 # top = read_writer.read_top(top_file)
    #             elif isinstance(read_writer, LWriter):
    #                 # handle absolute vs relative file paths
    #                 int_file = input_file.input["DPS_interaction_matrix_file"]
    #                 if Path(int_file).is_absolute():
    #                     int_file = Path(int_file).suffix
    #                 ptypes = read_writer.read_particle_types(top_file,
    #                                                          int_file)
    #                 # top = read_writer.read_top(top_file)
    #             else:
    #                 raise Exception(f"Invalid or unsupported writer type {type(read_writer)}")
    #             scene = read_writer.read_scene(top_file,
    #                                            self.sim_get_param(sim, "conf_file"),
    #                                            ptypes)
    #             # write new files
    #             sim_folder_path = write_path / (self.long_name() + "_copy") / sim.get_folder_path() / stage.name()
    #             sim_folder_path.mkdir(parents=True)
    #             write_writer.write(scene)

    # imminant deprecation
    def write_input_file(self,
                         sim: PatchySimulation,
                         stage: Stage,
                         replacer_dict: [str, str],
                         extras: [str, str],
                         analysis: bool = False):
        # honestly think this is everything lmao
        # input_file = self.input_file(sim, stage, replacer_dict, extras, analysis)
        with open(self.folder_path(sim) / "input", 'w+') as inputfile:
            # write server config spec
            for key in self.server_settings.input_file_params.var_names():
                if key in replacer_dict:
                    val = replacer_dict[key]
                else:
                    val = self.server_settings.input_file_params[key]
                inputfile.write(f"{key} = {val}\n")

            # newline
            inputfile.write("\n")

            # loop parameters in group
            for paramname in self.default_param_set.var_names():
                # if we've specified this param in a replacer dict
                if paramname in replacer_dict:
                    val = replacer_dict[paramname]
                # if no override
                elif paramname not in sim and paramname not in self.const_params:
                    val = self.default_param_set[paramname]
                else:
                    val = self.sim_get_param(sim, paramname)
                # check paths are absolute if applicable
                if self.server_settings.absolute_paths:
                    # approximation for "is this a file?"
                    if isinstance(val, str) and re.search(r'\.\w+$', val) is not None:
                        # if path isn't absolute
                        if not Path(val).is_absolute():
                            # prepend folder path
                            val = str(self.folder_path(sim) / val)
                inputfile.write(f"{paramname} = {val}\n")

            # write extras
            for key, val in extras.items():
                if key in replacer_dict:
                    val = replacer_dict[key]
                inputfile.write(f"{key} = {val}\n")

            # write more parameters

            inputfile.write(f"T = {self.sim_get_param(sim, 'T')}" + "\n")
            try:
                inputfile.write(f"narrow_type = {self.sim_get_param(sim, 'narrow_type')}" + "\n")
            except NoSuchParamError as e:
                self.get_logger().info(f"No narrow type specified for simulation {sim}.")

            # write external observables file path
            if len(self.observables) > 0:
                assert not self.server_settings.absolute_paths, "Absolute file paths aren't currently compatible with observiables!" \
                                                                " Get on it Josh!!!"
                if EXTERNAL_OBSERVABLES:
                    inputfile.write(f"observables_file = observables.json" + "\n")
                else:
                    for i, obsrv in enumerate(self.observables.values()):
                        obsrv.write_input(inputfile, i, stage, analysis)

    def write_sim_observables(self, sim: PatchySimulation, rel_paths: bool = True):
        if len(self.observables) > 0:
            with open(self.folder_path(sim) / "observables.json", "w+") as f:
                json.dump({f"data_output_{i + 1}": obs.to_dict() for i, obs in enumerate(self.observables.values())}, f)

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

    def sim_get_next_stage(self, sim: PatchySimulation) -> Stage:
        try:
            return self.sim_most_recent_stage(sim).get_next()
        except NoStageTrajError:
            # if no stages have been run
            return self.sim_get_stages(sim)[0]  # first stage

    def ipy(self, sim: PatchySimulation, stage: Union[Stage, None] = None) -> Simulation:
        if stage is None:
            stage = self.sim_get_next_stage(sim)
        if self.sim_num_stages(sim) == 1 or stage.idx() == 0:
            sim_obj = Simulation(str(self.folder_path(sim, stage)))
        else:
            # parameterize stage from previous stage
            sim_obj = Simulation(str(self.folder_path(sim, self.sim_get_stage(sim, stage.idx() - 1))),
                                 str(self.folder_path(sim, stage)))
        sim_obj.build_sim = stage  # assign stage object as sim builder
        return sim_obj

    def ipy_all(self, sim: PatchySimulation) -> list[Simulation]:
        return [self.ipy(sim, stage) for stage in self.sim_get_stages(sim)]

    def start_simulations(self,
                          e: Union[None, list[PatchySimulation]] = None,
                          stage: Union[str, None] = None):
        """
        Starts all simulations
        """
        if e is None:
            e = self.ensemble()

        # normal circumstances - no batch exec, do the old way
        if not self.server_settings.is_batched():
            self.do_setup(e)
            for sim in e:
                self.start_simulation(sim, stage_name=stage)
        else:
            # if the number of simulations to run requires more than one task

            mgr = SimulationManager()
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
            print("Let the simulating commence!")
            mgr.run(join=True, gpu_mem_block=False)
            # batch execution for CUDA + MPS

            # TODO: better slurm logging!

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

    def check_stage(self, sim: PatchySimulation):
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
        Starts an oxDNA simulation. direct invocation is not suggested;
        use `start_simulations` instead
        Parameters:
             sim : simulation to start
             script_name : the name of the slurm script file
        """

        if stage_name is not None:
            stage = self.sim_get_stage(sim, stage_name)
        else:
            stage = self.check_stage(sim)

        if not force_ignore_ok_check and not self.ok_to_run(sim, stage):
            self.get_logger().warning(f"Stage {stage.name()} not ok to run for sim {repr(sim)}")
            return -1

        # get slurm log jobname
        if not is_analysis:
            job_type_name = "oxdna"
        else:
            job_type_name = "analysis"
        script_name = stage.adjfn(script_name)

        if self.server_settings.is_server_slurm():
            command = f"sbatch --chdir={self.folder_path(sim)}"
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
        for i in range(retries):
            submit_txt = self.bash_exec(command, is_async=not self.server_settings.is_server_slurm(),
                                        cwd=self.folder_path(sim))
            if submit_txt:
                break
            time.sleep(backoff_factor ** i)

        if self.server_settings.is_server_slurm():
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

    # def get_run_confgen_sh(self, sim: PatchySimulation) -> Path:
    #     return self.folder_path(sim) / "gen_conf.sh"

    def gen_conf(self, sim: PatchySimulation) -> int:

        """
        Runs a conf generator. These are run as slurm jobs if you're on a slurm server,
        or as non-background tasks otherwise
        """
        if self.server_settings.is_server_slurm():
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

    # def get_conf_file(self, sim: PatchySimulation) -> Path:
    #     return self.folder_path(sim) / self.sim_get_param(sim, "conf_file")

    # ------------- ANALYSIS FUNCTIONS --------------------- #

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

        # if you try to actually use an aggregate step the sun will expand and swallow the earth
        is_aggregate_params: bool = isinstance(step, AggregateAnalysisPipelineStep) and self.is_multiselect(sim,
                                                                                                            step.params_aggregate_over)
        is_multiparam: bool = not isinstance(step, AggregateAnalysisPipelineStep) and self.is_multiselect(sim)
        if is_aggregate_params or is_multiparam:
            return self.get_data_multiselector(sim, step, time_steps)

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
            time_steps = range(
                # step.output_tstep,
                0,
                self.time_length(sim),
                int(step.output_tstep))
            self.get_logger().info(f"Constructed time steps {time_steps}")
        else:
            assert time_steps.step % step.output_tstep == 0, f"Specified step interval {time_steps} " \
                                                             f"not consistant with {step} output time " \
                                                             f"interval {step.output_tstep}"

        self.get_logger().info(
            f"Retrieving data for analysis step {step.name} and simulation(s) {str(sim)} over timeframe {time_steps}")
        # DATA CACHING
        # check if data is already loaded
        if self.is_data_loaded(sim, step, time_steps):
            return self.get_cached_analysis_data(sim, step)

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
            if isinstance(sim, tuple):
                assert not self.is_multiselect(sim)
                sim = self.get_simulation(*sim)
            step.cache_data(data, self.get_cache_file(step, sim))
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
                    sim_data.get()[p.param_name] = p.value_name()
            df = pd.concat([d.get() for d in data], axis=0)
            df = df.loc[np.isin(df[TIMEPOINT_KEY], timerange)]
            return PDPipelineData(df, timerange)
        elif step.get_output_data_type() == PipelineDataType.PIPELINE_DATATYPE_GRAPH:
            raise Exception("I haven't bothered trying to join graph data yet")
        else:
            raise Exception("Attempting to merge non-mergable data type")

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
            assert isinstance(sim, PatchySimulation) or not self.is_multiselect(
                sim), "Analysis pipeline head nodes should only take single simulations"
            if not isinstance(sim, PatchySimulation):
                sim = self.get_simulation(*sim)
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
                file_names = [self.folder_path(sim) / stage.adjfn(file_name) for stage in stages]
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
            # don't incorporate staging info into analysis pipeline cache file paths
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
        if cwd is not None:
            self.get_logger().debug(f"`{cwd}$ {command}`")
        else:
            self.get_logger().debug(f"`{os.getcwd()}$ {command}`")
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


def process_simulation_data(args: tuple[PatchySimulationEnsemble, AnalysisPipelineStep, PatchySimulation, range]):
    ensemble, step, s, time_steps = args
    return ensemble.get_data(step, s, time_steps)


def shared_ensemble(es: list[PatchySimulationEnsemble],
                    ignores: set = set()) -> Union[list[list[PatchySimulation]], None]:
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
        for p in e.ensemble_params:
            union_vals = name_vals[p.param_key]
            vals = [pv for pv in p if pv.value_name() in union_vals]
            valset.append(vals)

        ensembles.append([PatchySimulation(sim) for sim in itertools.product(*valset)])

    return ensembles
