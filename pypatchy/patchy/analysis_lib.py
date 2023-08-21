import re
from pathlib import Path
from typing import Union

import networkx as nx
import numpy as np
import pandas as pd
from pypatchy.patchy.simulation_specification import PatchySimulation

from .ensemble_parameter import EnsembleParameter
from .simulation_ensemble import PatchySimulationEnsemble
from ..analpipe.analysis_pipeline_step import AnalysisPipelineStep, AggregateAnalysisPipelineStep, AnalysisPipelineHead, \
    PipelineDataType, PipelineData
from .patchy_sim_observable import PatchySimObservable
from pypatchy.analpipe.yield_analysis_target import ClusterCategory

from pypatchy.analpipe.yield_analysis_target import YieldAnalysisTarget

from ..analpipe.analysis_data import PDPipelineData, GraphPipelineData, TIMEPOINT_KEY, load_cached_pd_data, \
    load_cached_graph_data


# this file contains classes that are useful in analpipe, but aren't required by other PyPatchy modules
# all classes in this document should extend AnalysisPipelineStep

# class LoadSimulationInfo(AnalysisPipelineHead, ABC):
#     """
#     This is a "helper" analysis pipeline step abstract base class for loading
#     simulation parameters
#     """
#     pass

class LoadParticlesTraj(AnalysisPipelineHead):
    """
    Loads the simulation trajectory.

    Executing this step should produce a pd.Dataframe with columns `timepoint`, `pidx`, `r`, `a1`, `a3`, `v`, `angv`, and `typeid`
    Each row of the dataframe is particle info at time (particle specified by `pid`, time specified by `timepoint`

    Note that I have not yet tested this class so it will likely collapse in spectacular fashion if used
    """

    def get_output_data_type(self):
        """
        Returns:
            PipelineDataType.PIPELINE_DATATYPE_DATAFRAME
        """
        return PipelineDataType.PIPELINE_DATATYPE_DATAFRAME

    trajfile: re.Pattern
    first_conf: re.Pattern

    # pandas array keys
    PARTICLE_IDX_KEY = "pidx"
    P_POS_KEY = "r"
    P_A1_KEY = "a1"
    P_A3_KEY = "a3"
    P_V_KEY = "v"
    P_ANGV_KEY = "angv"
    P_TYPE_ID_KEY = "typeid"

    def __init__(self,
                 name,
                 input_tstep: Union[int, None],
                 traj_file_regex=r"trajectory_\d+\.dat",
                 first_conf_file_name="init.conf"):
        """
        Constructor for traj read step
        I strongly advise initializing with:
        input_tstep = ensemble.get_input_file_param("print_conf_interval")
        traj_file_regex = ensemble.get_input_file_param("trajectory_file")
        first_conf_file_name = ensemble.get_input_file_param("conf_file")
        """
        super().__init__(name, input_tstep)
        self.trajfile = re.Pattern(traj_file_regex)
        self.first_conf = re.Pattern(re.escape(first_conf_file_name))

    def get_data_in_filenames(self) -> list[re.Pattern]:
        """
        Returns:
            a list of
        """
        return [self.first_conf, self.trajfile]

    load_cached_files = load_cached_pd_data

    def exec(self,
             ensemble: PatchySimulationEnsemble,
             sim: PatchySimulation,
             *args: Path) -> PipelineData:
        top_file = ensemble.folder_path(sim) / ensemble.sim_get_param(sim, "topology")
        with open(top_file) as f:
            f.readline()  # skip line
            particle_type_ids = f.readline().split()
        time_data = []
        t = 0
        timepoints = []
        particle_idx = 0
        for file_path in args:
            with open(file_path, 'r') as f:
                for line in f:
                    if line.startswith("t"):
                        t = int(line.split("=")[1].strip())
                        timepoints.append(t)
                        particle_idx = 0
                    # skip bound box size and whatever "E" is
                    elif not line.startswith("b") and not line.startswith("E"):
                        line_data = np.array([int(x) for x in line.split()])

                        # particle data
                        time_data.append({
                            TIMEPOINT_KEY: t,
                            self.PARTICLE_IDX_KEY: particle_idx,
                            self.P_POS_KEY: line_data[:3],
                            self.P_A1_KEY: line_data[3:6],
                            self.P_A3_KEY: line_data[6:9],
                            self.P_V_KEY: line_data[9:12],
                            self.P_ANGV_KEY: line_data[12:15],
                            self.P_TYPE_ID_KEY: particle_type_ids[particle_idx]
                        })
                        particle_idx += 1
        data = pd.DataFrame.from_records(time_data)
        return PDPipelineData(data, tr=np.array(timepoints))


class GraphsFromClusterTxt(AnalysisPipelineHead):
    """
    Analysis operation that reads in the text file produced by the observable
    PLPatchyTopology and outputs a dict where they keys are timepoints and the
    values are the list of cluster graphs at each of those timepoints
    """

    source_observable: PatchySimObservable

    def __init__(self,
                 name: str,
                 source: PatchySimObservable,
                 output_tstep: Union[int, None] = None):
        super().__init__(name, int(source.print_every), output_tstep)
        self.source_observable = source

    load_cached_files = load_cached_graph_data

    def exec(self, _, __, din: Path) -> GraphPipelineData:
        graphs = {}
        stepcounter = 0
        with open(din, "r") as f:
            # iterate lines in the graph file
            for line in f:
                # skip timepoints that aren't multiples of the specified timestep
                if stepcounter % self.output_tstep == 0:

                    clusterGraphs = []
                    # regex for a single cluster
                    clusters = re.finditer(r'\[.+?\]', line)

                    # iter regex matches
                    for cluster in clusters:
                        G = nx.Graph()
                        # iter entries within cluster
                        # entries are in format "[source-particle] -> ([space-seperated-list-of-connected-particles])
                        matches = re.finditer(
                            r'(\d+) -> \(((?:\d+ ?)+)\)', cluster.group()
                        )
                        # loop matches
                        for m in matches:
                            # grab source particle ID
                            source = m.group(1)
                            # loop destination particle IDs
                            for dest in m.group(2).split(' '):
                                # add edge between source and connected particle
                                G.add_edge(int(source), int(dest))
                        clusterGraphs.append(G)
                    graphs[stepcounter] = clusterGraphs
                stepcounter += self.source_observable.print_every
        return GraphPipelineData(graphs)

    def get_data_in_filenames(self):
        return [self.source_observable.name]

    def get_output_data_type(self):
        return PipelineDataType.PIPELINE_DATATYPE_GRAPH

    def can_parallelize(self):
        return True


class ClassifyClusters(AnalysisPipelineStep):
    """
    Compares graphs of clusters to a specified target graph, and
    produces a Pandas DataFrame of results
    each row in the dataframe corresponds to a cluster graph at a timepoint
    The dataframe has four columns:
        an integer index
        timepoint (int)
        size ratio (size of graph / size of target)
        category (see ClusterCategory enum at the top of this file)
    """

    target_name: YieldAnalysisTarget

    def __init__(self,
                 name: str,
                 target: Union[str, YieldAnalysisTarget],
                 input_tstep: Union[int, None] = None,
                 output_tstep: Union[int, None] = None):
        super().__init__(name, input_tstep, output_tstep)
        if isinstance(target, str):
            target = YieldAnalysisTarget(target)
        self.target = target

    CLUSTER_CATEGORY_KEY = "clustercategory"
    SIZE_RATIO_KEY = "sizeratio"

    load_cached_files = load_cached_pd_data

    def exec(self, input_data: GraphPipelineData) -> PDPipelineData:
        cluster_cats_data = {
            TIMEPOINT_KEY: [],
            self.CLUSTER_CATEGORY_KEY: [],
            self.SIZE_RATIO_KEY: []
        }
        # loop timepoints in input graph data
        for timepoint in input_data.get():
            # check output tstep
            if timepoint % self.output_tstep == 0:
                # loop cluster graphs at this timepoint
                for g in input_data.get()[timepoint]:
                    cat, sizeFrac = self.target.compare(g)

                    # assign stuff
                    cluster_cats_data[TIMEPOINT_KEY].append(timepoint)
                    cluster_cats_data[self.CLUSTER_CATEGORY_KEY].append(cat)
                    cluster_cats_data[self.SIZE_RATIO_KEY].append(sizeFrac)
        return PDPipelineData(pd.DataFrame.from_dict(data=cluster_cats_data),
                              input_data.trange()[input_data.trange() % self.output_tstep == 0])

    def can_parallelize(self):
        return True

    def get_output_data_type(self):
        return PipelineDataType.PIPELINE_DATATYPE_DATAFRAME


# class SmartClassifyClusters(ClassifyClusters):
#     """
#     A "smart" version of ClassifyClusters, which incorporates trajectory data
#     I wrote like half of this and then gave up
#     """
#     def get_output_data_type(self):
#         return PipelineDataType.PIPELINE_DATATYPE_DATAFRAME
#
#     load_cached_files = load_cached_pd_data
#
#     def exec(self,
#              graph_data: GraphPipelineData,
#              traj_data: PDPipelineData) -> PDPipelineData:
#         traj = traj_data.get()
#
#         shared_timepoints = set(traj_data.trange()).intersection(graph_data.trange())
#
#         cluster_cats_data = {
#             TIMEPOINT_KEY: [],
#             self.CLUSTER_CATEGORY_KEY: [],
#             self.SIZE_RATIO_KEY: []
#         }
#
#         # loop timepoints in input graph data
#         for timepoint in shared_timepoints:
#             graphs = graph_data.get()[timepoint]
#
#             time_data = traj.loc[traj[TIMEPOINT_KEY] == timepoint]
#
#             # loop cluster graphs at this timepoint
#             for g in graphs:
#
#                 g_particle_types = {node:
#                     time_data.loc[time_data[LoadParticlesTraj.PARTICLE_IDX_KEY] == node][LoadParticlesTraj.P_TYPE_ID_KEY][0]
#                     for node in g
#                 }
#                 cat, sizeFrac = self.target.compare(g)
#
#                 # assign stuff
#                 cluster_cats_data[TIMEPOINT_KEY].append(timepoint)
#                 cluster_cats_data[self.CLUSTER_CATEGORY_KEY].append(cat)
#                 cluster_cats_data[self.SIZE_RATIO_KEY].append(sizeFrac)
#
#         return PDPipelineData(pd.DataFrame.from_dict(data=cluster_cats_data), graph_data.trange())


YIELD_KEY = "yield"


class ComputeClusterYield(AnalysisPipelineStep):
    cutoff: float
    overreach: bool
    target: YieldAnalysisTarget

    def __init__(self,
                 name: str,
                 cutoff: float,
                 overreach: bool,
                 input_tstep: Union[int, None] = None,
                 output_tstep: Union[int, None] = None):
        super().__init__(name, input_tstep, output_tstep)
        self.cutoff = cutoff
        self.overreach = overreach

    load_cached_files = load_cached_pd_data

    def exec(self, cluster_categories: PDPipelineData) -> PDPipelineData:
        """
        returns a pandas DataFrame where each row corresponds to a timepoint
        the resulting dataframe will be indexed by timepoint
        """
        # filter off-target graphs
        data: pd.DataFrame = cluster_categories.get()[
            cluster_categories.get()[ClassifyClusters.CLUSTER_CATEGORY_KEY] != ClusterCategory.SMALLER_NOT_SUB]
        # filter too-small graphs
        data = data[data[ClassifyClusters.SIZE_RATIO_KEY] >= self.cutoff]
        if not self.overreach:
            # filter clusters that are larger than the largest clusters
            data = data[data[ClassifyClusters.CLUSTER_CATEGORY_KEY] != ClusterCategory.OVER]
        else:  # not something I'm currently using by may be useful later
            # max cluster yield should be 1.0
            data[ClassifyClusters.SIZE_RATIO_KEY] = data[ClassifyClusters.SIZE_RATIO_KEY].apply(np.ceil)
        # discard cluster categories column
        data.drop(ClassifyClusters.CLUSTER_CATEGORY_KEY, axis=1)
        # group by timepoint, average, reset index
        data = data.groupby(TIMEPOINT_KEY).sum().reset_index()
        # rename column
        data = data.rename(mapper={ClassifyClusters.SIZE_RATIO_KEY: YIELD_KEY}, axis="columns")
        # data = data.set_index([TIMEPOINT_KEY])
        data = data.loc[data[TIMEPOINT_KEY] % self.output_tstep == 0]
        missing_timepoints = cluster_categories.missing_timepoints(data[TIMEPOINT_KEY].unique().data)
        data = data.append(pd.DataFrame.from_dict({
            TIMEPOINT_KEY: missing_timepoints,
            YIELD_KEY: 0
        }), ignore_index=True)
        return PDPipelineData(data,
                              cluster_categories.trange()[cluster_categories.trange() % self.output_tstep == 0])

    def get_output_data_type(self):
        """
        Returns:
            the datatype produced by this pipeline step (here, PipelineDataType.PIPELINE_DATATYPE_DATAFRAME)
        """
        return PipelineDataType.PIPELINE_DATATYPE_DATAFRAME

    def can_parallelize(self):
        """
        Returns:
            True
        """
        return True


class ComputeClusterSizeData(AnalysisPipelineStep):
    """
    computes the minimum cluster size, maximum cluster size, average cluster size,
    and total number of particles in clusters
    """

    MEAN_KEY = "size_mean"
    MEDIAN_KEY = "size_median"
    MIN_KEY = "size_min"
    MAX_KEY = "size_max"
    STDEV_KEY = "size_stdev"

    def __init__(self,
                 name: str,
                 input_tstep: Union[int, None] = None,
                 output_tstep: Union[int, None] = None,
                 minsize=0):
        super().__init__(name, input_tstep, output_tstep)
        self.minsize = minsize

    load_cached_files = load_cached_pd_data


    def exec(self, input_graphs: GraphPipelineData) -> PDPipelineData:

        cluster_size_data = {
            TIMEPOINT_KEY: [],
            self.MIN_KEY: [],
            self.MAX_KEY: [],
            self.MEAN_KEY: [],
            self.MEDIAN_KEY: [],
            self.STDEV_KEY: []
        }
        # loop timepoints in input graph data
        for timepoint in input_graphs.get():
            if timepoint % self.output_tstep == 0:
                graph_sizes = [len(g) for g in input_graphs[timepoint] if len(g) >= self.minsize]
                cluster_size_data[TIMEPOINT_KEY] = timepoint
                cluster_size_data[self.MIN_KEY].append(min(graph_sizes))
                cluster_size_data[self.MAX_KEY].append(max(graph_sizes))
                cluster_size_data[self.MEDIAN_KEY].append(np.median(np.array(graph_sizes)))
                cluster_size_data[self.MEDIAN_KEY].append(sum(graph_sizes) / len(graph_sizes))
                cluster_size_data[self.STDEV_KEY].append(np.std(np.array(graph_sizes)))
        return PDPipelineData(pd.DataFrame.from_dict(data=cluster_size_data), input_graphs.trange())

    def get_output_data_type(self) -> PipelineDataType:
        """
        Returns:
            PipelineDataType.PIPELINE_DATATYPE_DATAFRAME
        """
        return PipelineDataType.PIPELINE_DATATYPE_DATAFRAME

    def can_parallelize(self):
        """
        Returns:
            True
        """
        return True


class ComputeSpecGroupClusterYield(AggregateAnalysisPipelineStep):
    """
    Simulation step which computes yields over multiple simulations
    This Step was specifically designed to compute average, min, max,
    stdev over multiple duplicates

    The step uses simple statistical methods and does NOT attempt to line
    up formation curves or any other thing I just thought of
    """

    def __init__(self,
                 name: str,
                 aggregate_over: EnsembleParameter,
                 input_tstep: Union[int, None] = None,
                 output_tstep: Union[int, None] = None):
        """
        Constructor
        """
        super().__init__(name, input_tstep, output_tstep, tuple(aggregate_over))

    MIN_KEY = "yield_min"
    MAX_KEY = "yield_max"
    AVG_KEY = "yield_avg"
    STD_KEY = "yield_std"

    load_cached_files = load_cached_pd_data

    def exec(self, yield_data: PDPipelineData) -> PDPipelineData:
        """
        yield_data should be a pd.DataFrame with columns name YIELD_KEY and TIMEPOINT_KEY
        returns a pd.DataFrame where each row is a timepoint, with columns for min, max,
        mean average, and standard deviation of the yield at that timepoint
        the resulting dataframe will be indexed by timepoint
        """

        gb = yield_data.get().groupby(TIMEPOINT_KEY)
        data = pd.DataFrame(
            index=pd.RangeIndex(
                start=yield_data[TIMEPOINT_KEY].min(),
                stop=yield_data[TIMEPOINT_KEY].max(),
                step=self.output_tstep),
            columns=[self.MIN_KEY, self.MAX_KEY, self.AVG_KEY, self.STD_KEY])
        data.update(gb.min().rename({YIELD_KEY: self.MIN_KEY}))
        data.update(gb.max().rename({YIELD_KEY: self.MAX_KEY}))
        data.update(gb.mean().rename({YIELD_KEY: self.AVG_KEY}))
        data.update(gb.std().rename({YIELD_KEY: self.STD_KEY}))

        return PDPipelineData(data, yield_data.trange())

    def can_parallelize(self):
        """
        Returns:
            True
        """
        return True

    def get_output_data_type(self):
        """
        Returns:
            PipelineDataType.PIPELINE_DATATYPE_DATAFRAME
        """
        return PipelineDataType.PIPELINE_DATATYPE_DATAFRAME
