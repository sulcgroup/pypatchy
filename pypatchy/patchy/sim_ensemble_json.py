from __future__ import annotations
from dataclasses import asdict

# pypatchy imports
from .analysis_lib import *
from .ensemble_parameter import EnsembleParameter
from .param_val_types import PARTICLE_TYPES_KEY, MDT_CONVERT_KEY, STAGES_KEY, ParticleSetParam, MDTConvertParams, \
    StagedAssemblyParam, StageInfoParam
from .base_param_val import ParameterValue, ParamValueGroup
from .particle_adders import StageParticleAdder, RandParticleAdder, FromPolycubeAdder, FromConfAdder
from ..polycubeutil.polycubesRule import PolycubesRule
from ..structure import Structure
from .pl.pljsonrw import PLJSONEncoder
from .pl.plparticleset import MultidentateConvertSettings
from .simulation_specification import ParamSet
from ..analysis.analysis_pipeline import AnalysisPipeline
from ..analysis.analysis_pipeline_step import AnalysisPipelineStep
from ..polycubeutil.polycube_structure import PolycubeStructure
from ..server_config import PatchyServerConfig
from ..structure import GenericTypedStructure


class SimEnsembleJsonEncoder(PLJSONEncoder):
    """
    JSON encoder class for all stuff required by PatchySimulationEnsemble
    """

    def default(self, obj: Any) -> Union[dict, list, float, int, bool, str]:
        if type(obj) in [int, bool, str, float]:
            return obj
        elif type(obj) == dict:
            return {
                key: self.default([obj[key]]) for key in obj
            }
        elif type(obj) == list:
            return [
                self.default(val) for val in obj
            ]
        elif isinstance(obj, ParamSet):
            return {
                pv.param_name: self.default(pv)
                for pv in obj.param_vals
            }
        elif isinstance(obj, EnsembleParameter):
            return self.encode_ensemble_parameter(obj)
        elif isinstance(obj, PolycubesRule):
            return obj.toJSON()
        elif isinstance(obj, PatchyServerConfig):
            return obj.to_dict()
        elif isinstance(obj, ParameterValue):
            pvdict = self.encode_param_value(obj)
            if isinstance(obj, ParticleSetParam):
                pvdict["type"] = PARTICLE_TYPES_KEY
            elif isinstance(obj, StagedAssemblyParam):
                pvdict["type"] = STAGES_KEY
            elif isinstance(obj, MDTConvertParams):
                pvdict["type"] = MDT_CONVERT_KEY
            return pvdict
        elif isinstance(obj, Observable):
            return obj.export()["output"] # compatibility with ipy_oxDNA code
        # StageInfoParam should probably inherit from ParameterValue, but it doesn't right now
        # and i'm afraid to change that
        elif isinstance(obj, StageInfoParam):
            infodict = {
                # "stage_name": obj.stage_name,
                **obj.info
            }
            if obj.start_time > 0:
                infodict["t"] = obj.start_time
            if obj.allow_shortfall:
                infodict["allow_shortfall"] = True
            if obj.add_method is not None:
                infodict["add_method"] = encode_add_method(obj.add_method)
            return infodict
        elif isinstance(obj, AnalysisPipeline):
            return self.write_analysis_pipeline(obj)
        else:
            return super().default(obj)

    def encode_ensemble_parameter(self, obj: EnsembleParameter) -> list[Union[str, list[Any]]]:
        # important that dict keys match constructor args
        return [
            obj.param_key,
            [self.encode_param_value(v) for v in obj.param_value_set]
        ]

    def encode_param_value(self, obj: ParameterValue) -> dict:
        if isinstance(obj, ParamValueGroup):
            return {
                "name": obj.value_name(),
                ** {
                    pv.param_name: self.encode_param_value(pv) for pv in obj.param_value.values()
                }
            }
        # specific versions of parameter value group
        elif isinstance(obj, MDTConvertParams):
            mdt_dict = {
                "n_teeth": obj.param_value.n_teeth,
                "dental_radius": obj.param_value.dental_radius
            }
            # things with default value; don't waste space with defaults
            if not obj.param_value.torsion:
                mdt_dict["torsion"] = obj.param_value.torsion
            if not obj.param_value.follow_surf:
                mdt_dict["follow_surf"] = obj.param_value.follow_surf
            if obj.param_value.energy_scale_method != MultidentateConvertSettings.ENERGY_SCALE_LINEAR:
                if obj.param_value.energy_scale_method == MultidentateConvertSettings.ENERGY_SCALE_NONE:
                    mdt_dict["energy_scale_method"] = "none"
                elif obj.param_value.energy_scale_method == MultidentateConvertSettings.ENERGY_SCALE_LOG:
                    mdt_dict["energy_scale_method"] = "log"
                else:
                    # energy scale number is a scalar value
                    mdt_dict["energy_scale_method"] = obj.param_value.energy_scale_method
            if obj.param_value.alpha_scale_method != MultidentateConvertSettings.ALPHA_SCALE_NONE:
                # TODO: lipari-szabo method or something
                if obj.param_value.alpha_scale_method == -1:
                    raise Exception("lipari-szabo method not supported!")
                else:
                    # energy scale number is a scalar value
                    mdt_dict["alpha_scale_method"] = obj.param_value.alpha_scale_method
            return mdt_dict
        elif isinstance(obj, ParticleSetParam):
            return super().default(obj.param_value)
        elif isinstance(obj, StagedAssemblyParam):
            return {
                "stages": {
                    stage.stage_name: self.default(stage) for stage in obj.param_value.values()
                }
            }

        # default parameter value
        else:
            return obj.param_value
    def read_analysis_pipeline(self, pipeline_data: list[Union[list, dict]], context) -> AnalysisPipeline:
        """
        read analysis pipeline data from a json
        :param pipeline_data: a list of pipeline steps (represented by json objs) and pipes (represented by 2-length lists)
        :param context: a PatchySimulationEnseble object
        """
        pipes = [tuple(pipe) for pipe in pipeline_data if isinstance(pipe, list)]
        steps = [self.read_analysis_step(step,context) for step in pipeline_data if isinstance(step, dict)]
        return AnalysisPipeline(*pipes, *steps)

    def write_analysis_pipeline(self, pipeline: AnalysisPipeline) -> list:
        """
        write analysis pipeline data into a json'
        :return: a list of pipeline steps (represented by json objs) and pipes (represented by 2-length lists)
        """
        return [
            *[
                self.write_analysis_step(step) for step in pipeline.steps()
            ],
            *pipeline.pipes()
        ]

    def read_analysis_step(self, json_data: dict, context) -> AnalysisPipelineStep:
        """
        read data from json file
        this method consists of a chained if-elif to determine step type aka which constructor
        to call, going through each class in analysis_lib
        override this method in a subclass of SimEnsembleJsonEncoder when
        writing a subclass of AnalysisPipelineStep. add your own if-elif chain and then call super()
        :param json_data: data about an analysis pipeline step, in json form
        :param context: a PatchySimEnsemble object

        """
        step_type = json_data["type"]
        if step_type == "LoadParticlesTraj":
            return LoadParticlesTraj(
                name=json_data["name"],
                input_tstep=json_data["input_tstep"],
                output_tstep=json_data["output_tstep"],
                normalize_coords=json_data["normalize_coords"]
            )
        elif step_type == "LoadEnergies":
            return LoadEnergies(
                step_name=json_data["name"],
                # don't and can't set input tstep!
                output_tstep=json_data["output_tstep"]
            )
        elif step_type == "BlobsFromClusters":
            return BlobsFromClusters(
                name=json_data["name"],
                output_tstep=json_data["output_tstep"],
                source=context.observables[json_data["source"]],
            )
        elif step_type == "GraphsFromPatchyBonds":
            return GraphsFromPatchyBonds(
                name=json_data["name"],
                output_tstep=json_data["output_tstep"],
                source=context.observables[json_data["source"]]
            )
        elif step_type == "GraphsFromClusterTxt":
            return GraphsFromClusterTxt(
                name=json_data["name"],
                output_tstep=json_data["output_tstep"],
                source=context.observables[json_data["source"]]
            )
        elif step_type == "MatchGraphToCrystal":
            return MatchGraphToCrystal(
                name=json_data["name"],
                target=GenericTypedStructure(
                    bindings=json_data["target"]["bindings"],
                    types=json_data["target"]["types"],
                ),
                output_tstep=json_data["output_tstep"],
                input_tstep=json_data["input_tstep"]
            )
        elif step_type == "ClassifyClusters":
            return ClassifyClusters(
                name=json_data["name"],
                target=Structure(bindings=json_data["target"]),
                input_tstep=json_data["input_tstep"],
                output_tstep=json_data["output_tstep"]
            )
        elif step_type == "ClassifyPolycubeClusters":
            return ClassifyPolycubeClusters(
                name=json_data["name"],
                target_name=PolycubeStructure(**json_data["target"]),
                expected_edge_length=json_data["expected_edge_length"],
                edge_distance_tolerance=json_data["edge_distance_tolerance"],
                input_tstep=json_data["input_tstep"],
                output_tstep=json_data["output_tstep"]
            )
        elif step_type == "ComputeClusterYield":
            return ComputeClusterYield(
                name=json_data["name"],
                input_tstep=json_data["input_tstep"],
                output_tstep=json_data["output_tstep"],
                cutoff=json_data["cutoff"],
                overreach=json_data["overreach"]
            )
        elif step_type == "ComputeClusterSizeData":
            return ComputeClusterSizeData(
                name=json_data["name"],
                input_tstep=json_data["input_tstep"],
                output_tstep=json_data["output_tstep"],
                minsize=json_data["minsize"]
            )
        elif step_type == "ComputeSpecGroupClusterYield":
            return ComputeSpecGroupClusterYield(
                name=json_data["name"],
                input_tstep=json_data["input_tstep"],
                output_tstep=json_data["output_tstep"],
                aggregate_over=context.ensemble_params[json_data["aggregate_over"]],
            )
        else:
            raise Exception(f"unknown step type: {step_type}")

    def write_analysis_step(self, step: AnalysisPipelineStep):
        """
        writes an analysis step as a dict
        in overriding classes, overload and include a super() call for non-custom step types
        """
        step_dict = {}
        step_dict["type"] = type(step).__name__
        step_dict["name"] = step.name
        # can still write input/output tsteps for steps that don't use them
        step_dict["input_tstep"] = step.input_tstep
        step_dict["output_tstep"] = step.output_tstep
        if type(step) == LoadParticlesTraj:
            step_dict["normalize_coords"] = step.normalize_coords
        elif isinstance(step, (BlobsFromClusters, GraphsFromPatchyBonds, GraphsFromClusterTxt)):
            # my god how to load these
            step_dict["source"] = step.source_observable.file_name
        elif isinstance(step, (MatchGraphToCrystal, ClassifyClusters, ClassifyPolycubeClusters)):
            step_dict["target"] = step.target.to_json()
            if type(step) == ClassifyPolycubeClusters:
                step_dict["expected_edge_length"] = step.graphedgetolerence
                step_dict["edge_distance_tolerance"] = step.graphedgelen
        elif type(step) == ComputeClusterYield:
            step_dict["cutoff"] = step.cutoff
            step_dict["overreach"] = step.overreach
        elif type(step) == ComputeClusterSizeData:
            step_dict["minsize"] = step.minsize
        elif type(step) == ComputeSpecGroupClusterYield:
            step_dict["aggregate_over"] = step.aggregate_over.param_key

        return step_dict

def encode_add_method(add_method: StageParticleAdder) -> dict:
    infodict = {}
    if isinstance(add_method, RandParticleAdder):
        infodict["particles"] = add_method.particles
        if not add_method.extern:
            infodict["extern"] = False
    elif isinstance(add_method, FromPolycubeAdder):
        infodict["polycubes"] = [
            {
                "n_copies": pc.n_copies,
                "patch_distance_multiplier": pc.patch_distance_multiplier,
                "polycube_file_path": {
                    'cube_types': pc.polycube_file_path.rule.toJSON(),
                    'cubes': [
                        cube.toJSON() for cube in pc.polycube_file_path.particles()
                    ]
                }
            } for pc in add_method.polycubes
        ]
    elif isinstance(add_method, FromConfAdder):
        raise NotImplemented("FromConfAdder not yet implemented!")
    else:
        raise TypeError(f"Invalid adder type {type(add_method)}")
    return infodict
