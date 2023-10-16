import json
import pickle
import re
from pathlib import Path
from typing import Union

import networkx as nx
import numpy as np
import oxDNA_analysis_tools.distance
import pandas as pd
from oxDNA_analysis_tools.UTILS.RyeReader import get_confs, describe, inbox
from oxDNA_analysis_tools.UTILS.data_structures import Configuration

from pypatchy.patchy.simulation_specification import PatchySimulation

from .ensemble_parameter import EnsembleParameter
from .simulation_ensemble import PatchySimulationEnsemble
from ..analpipe.analysis_pipeline_step import AnalysisPipelineStep, AggregateAnalysisPipelineStep, AnalysisPipelineHead, \
    PipelineDataType, PipelineData
from .patchy_sim_observable import PatchySimObservable
from pypatchy.analpipe.yield_analysis_target import ClusterCategory

from pypatchy.analpipe.yield_analysis_target import YieldAnalysisTarget

from ..analpipe.analysis_data import PDPipelineData, ObjectPipelineData, TIMEPOINT_KEY, load_cached_pd_data, \
    load_cached_object_data, RawPipelineData, MissingCommonDataError

import drawsvg as draw

from ..polycubeutil.polycubesRule import PolycubesRule
from ..polycubeutil.structure import PolycubeStructure
from ..util import get_input_dir


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

    Should produce a Data

    Note that I have not yet tested this class so it will likely collapse in spectacular fashion if used
    """

    normalize_coords: bool

    def __init__(self,
                 name,
                 normalize_coords=True,
                 input_tstep: Union[int, None] = 1,
                 output_tstep: Union[int, None] = None):
                 # traj_file_regex=r"trajectory.dat",
                 # first_conf_file_name="init.conf"):
        """
        Constructor for traj read step
        I strongly advise initializing with:
        input_tstep = ensemble.get_input_file_param("print_conf_interval")
        traj_file_regex = ensemble.get_input_file_param("trajectory_file")
        first_conf_file_name = ensemble.get_input_file_param("conf_file")
        """
        super().__init__(name, input_tstep, output_tstep)
        self.normalize_coords = normalize_coords
        # really hate that 0th conf isn't included in trajectory
        # self.trajfile = traj_file_regex
        # self.first_conf = first_conf_file_name

    load_cached_files = load_cached_object_data

    def exec(self,
             ensemble: PatchySimulationEnsemble,
             sim: PatchySimulation,
             *args: Path) -> PipelineData:
        top_file = ensemble.folder_path(sim) / ensemble.sim_get_param(sim, "topology")
        # this will go up in smoke very badly if it comes into contact with staged assembly
        with open(top_file) as f:
            f.readline()  # skip line
            particle_type_ids = [int(i) for i in f.readline().split()]
        # load first conf (not incl in traj for some reason)
        top_file_path = ensemble.paramfile(sim, "topology")
        first_conf_file_path = ensemble.paramfile(sim, "conf_file")
        traj_file_path = ensemble.paramfile(sim, "trajectory_file")
        top_info, init_conf_info = describe(str(top_file), str(first_conf_file_path))
        firstconf = get_confs(traj_info=init_conf_info,
                              top_info=top_info,
                              start_conf=0,
                              n_confs=1)[0]
        confdict = {0: (firstconf, particle_type_ids)}
        # load trajectory confs
        top_info, traj_info = describe(
            str(top_file_path),
            str(traj_file_path)
        )
        confs = get_confs(
            traj_info=traj_info,
            top_info=top_info,
            start_conf=0,
            n_confs=traj_info.nconfs
        )

        # def norm_coord(coord: float, increment: float) -> float:
        #     while coord < increment:
        #         coord += increment
        #     return coord % increment
        # norm_coord_vectorize = np.vectorize(norm_coord)

        for conf in confs:
            if conf.time % self.output_tstep == 0:
                if self.normalize_coords:
                    conf = inbox(conf, True)
                # assert (conf.positions < conf.box[np.newaxis, :]).all()
                confdict[conf.time] = (conf, particle_type_ids)
        return RawPipelineData(confdict)


    def draw(self) -> tuple[tuple[int, int], draw.Group]:
        (w, y), g = super().draw()
        g.append(draw.Rectangle(0, y, w, 40, stroke="black", stroke_width=1, fill="tan"))
        g.append(draw.Text("TODO: write description!", font_size=7, x=1, y=y+7))
        y += 40
        return (w, y), g


    def get_data_in_filenames(self) -> list[str]:
        return []

    def get_output_data_type(self):
        """
        Returns:
            PipelineDataType.PIPELINE_DATATYPE_DATAFRAME
        """
        return PipelineDataType.PIPELINE_DATATYPE_RAWDATA


class BlobsFromClusters(AnalysisPipelineHead):
    """
    Analysis operation that reads in the text file produced by the observable PLCluster
    and produces a list of lists of lists of particle types
    """

    def get_data_in_filenames(self) -> list[str]:
        return [self.source_observable.file_name]

    def load_cached_files(self, f: Path) -> PipelineData:
        assert f.is_file()
        with f.open("rb") as datafile:
            return pickle.load(datafile)

    def exec(self, din: Path) -> PipelineData:
        clusters_lists = {}
        stepcounter = 0
        with open(din, "r") as f:
            # iterate lines in the graph file
            for line in f:
                # skip timepoints that aren't multiples of the specified timestep
                if stepcounter % self.output_tstep == 0:

                    time_clusters = []
                    # Match all the patterns that look like (numbers)
                    for match in re.findall(r'\(([^)]+)\)', line):
                        # Split the string by space and convert each element to an integer
                        inner_list = list(map(int, match.split()))
                        time_clusters.append(inner_list)
                    clusters_lists[stepcounter] = []
                stepcounter += self.source_observable.print_every
        return ObjectPipelineData(clusters_lists)

    def get_output_data_type(self) -> PipelineDataType:
        return PipelineDataType.PIPELINE_DATATYPE_OBJECTS

    source_observable: PatchySimObservable

    def __init__(self,
                 name: str,
                 source: PatchySimObservable,
                 output_tstep: Union[int, None] = None):
        super().__init__(name, int(source.print_every), output_tstep)
        self.source_observable = source


class GraphsToStructures(AnalysisPipelineHead):
    """
    Returns a list of Structure objects from an observable
    """
    def get_data_in_filenames(self) -> list[str]:
        pass

    def load_cached_files(self, f: Path) -> PipelineData:
        pass

    def exec(self, *args: Union[PipelineData, AnalysisPipelineStep]) -> PipelineData:
        pass

    def get_output_data_type(self) -> PipelineDataType:
        pass


class GraphsFromClusterTxt(AnalysisPipelineHead):
    """
    Analysis operation that reads in the text file produced by the observable
    PLClusterTopology and outputs a dict where they keys are timepoints and the
    values are the list of cluster graphs at each of those timepoints
    """

    source_observable: PatchySimObservable

    def __init__(self,
                 name: str,
                 source: PatchySimObservable,
                 output_tstep: Union[int, None] = None):
        super().__init__(name, int(source.print_every), output_tstep)
        self.source_observable = source

    load_cached_files = load_cached_object_data

    def exec(self, _, __, din: Path) -> ObjectPipelineData:
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
        return ObjectPipelineData(graphs)

    def get_data_in_filenames(self):
        return [self.source_observable.file_name]

    def get_output_data_type(self):
        return PipelineDataType.PIPELINE_DATATYPE_GRAPH

    def draw(self) -> tuple[tuple[int, int], draw.Group]:
        (w, y), g = super().draw()
        g.append(draw.Rectangle(0, y, w, 40, stroke="black", stroke_width=1, fill="tan"))
        g.append(draw.Text(f"Source Observable: {self.source_observable.file_name}", font_size=7, x=1, y=y + 7))  # TODO: more info?
        y += 10
        g.append(draw.Text("TODO: write description!", font_size=7, x=1, y=y+7))
        y += 28
        return (w, y), g


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

    target: YieldAnalysisTarget

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

    def exec(self, input_data: ObjectPipelineData) -> PDPipelineData:
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

    def draw(self) -> tuple[tuple[int, int], draw.Group]:
        (w, y), g = super().draw()
        g.append(draw.Rectangle(0, y, w, 40, stroke="black", stroke_width=1, fill="tan"))
        g.append(draw.Text(f"Target topology: {self.target.name}", font_size=7, x=1, y=y+7))
        y += 12
        g.append(draw.Text("This step classifies cluster graphs as 'match', 'smaller subset',\n"
                           "'smaller not subset', or 'non-match'. The step uses igraph's \n"
                           "`get_subisomorphisms_vf2` function. The step produces a dataframe \n"
                           "where each row is a cluster with columns for category, size ratio, \n"
                           "and timepoint.", font_size=7, x=1, y=y+7))
        y += 28
        return (w, y), g

class ClassifyPolycubeClusters(AnalysisPipelineStep):
    """
    Modified version of ClassifyClusters that takes Polycube structure into account
    Compares graphs of clusters to a specified target graph, and
    produces a Pandas DataFrame of results
    each row in the dataframe corresponds to a cluster graph at a timepoint
    The dataframe has four columns:
        an integer index
        timepoint (int)
        size ratio (size of graph / size of target)
        category (see ClusterCategory enum at the top of this file)
    """

    target: YieldAnalysisTarget
    target_polycube: PolycubeStructure
    graphedgelen: float
    graphedgetolerence: float

    def __init__(self,
                 name: str,
                 target_name: str,
                 expected_edge_length: float = 1,
                 edge_distance_tolerance: float = 0.1,
                 input_tstep: Union[int, None] = None,
                 output_tstep: Union[int, None] = None):
        super().__init__(name, input_tstep, output_tstep)
        with (get_input_dir() / "targets" / (target_name + ".json")).open("r") as f:
            target_data = json.load(f)
            rule = PolycubesRule(rule_json=target_data["cube_types"])
            self.target_polycube = PolycubeStructure(rule=rule, structure=target_data["cubes"])
            self.target = YieldAnalysisTarget(target_name, self.target_polycube.graph_undirected())
            self.graphedgelen = expected_edge_length
            self.graphedgetolerence = edge_distance_tolerance

    CLUSTER_CATEGORY_KEY = "clustercategory"
    SIZE_RATIO_KEY = "sizeratio"
    CLUSTER_EDGE_LEN_AVG_KEY = "avgedgelen"
    CLUSTER_EDGE_LEN_STD_KEY = "stdedgelen"
    CLUSTER_NUM_DROPPED_EDGES_KEY = "numdroppededges"

    load_cached_files = load_cached_pd_data

    def exec(self, input_data_1: ObjectPipelineData, input_data_2: ObjectPipelineData) -> PDPipelineData:
        # use data class types to identify inputs
        if isinstance(input_data_2, RawPipelineData):
            graph_input_data = input_data_1
            traj_data = input_data_2
        else:
            graph_input_data = input_data_2
            traj_data = input_data_1
        cluster_cats_data = {
            TIMEPOINT_KEY: [],
            self.CLUSTER_CATEGORY_KEY: [],
            self.SIZE_RATIO_KEY: [],
            self.CLUSTER_EDGE_LEN_AVG_KEY: [],
            self.CLUSTER_EDGE_LEN_STD_KEY: [],
            self.CLUSTER_NUM_DROPPED_EDGES_KEY: []
        }
        polycube_type_ids = [cube.get_type() for cube in self.target_polycube.cubeList]
        polycube_type_map = {
            type_id: polycube_type_ids.count(type_id)
            for type_id in set(polycube_type_ids)
        }
        # loop timepoints in input graph data
        shared_timepoints = np.intersect1d(graph_input_data.trange(), traj_data.trange())
        if not len(shared_timepoints):
            raise MissingCommonDataError(graph_input_data, traj_data)

        for timepoint in shared_timepoints:
            # check output tstep
            if timepoint % self.output_tstep == 0:
                # grab conf and top data at this timepoint (only really need top until i rope in SVD superimposer)

                assert timepoint in traj_data.trange()
                conf, top = traj_data.get()[timepoint]

                # loop cluster graphs at this timepoint
                for g in graph_input_data.get()[timepoint]:
                    g2, edge_lens = self.engage_filter(g, conf)
                    if len(g2.edges) > 0:

                        # get particle ids for graph nodes
                        particle_ids = [top[n] for n in g.nodes]

                        # get counts for particle ids
                        particle_type_counts = {
                            typeid: particle_ids.count(typeid) for typeid in set(particle_ids)
                        }

                        # test that all particle types in the structure are in the polycube,
                        # and are contained the same or fewer number of times
                        if all(
                            [type_id in polycube_type_map and particle_type_counts[type_id] <= polycube_type_map[type_id]
                             for type_id in particle_type_counts]
                        ):
                            avg_edge_len = np.mean(edge_lens)
                            edge_len_std = np.std(edge_lens)
                            # only then do the comparison
                            cat, sizeFrac = self.target.compare(g2)

                            # assign stuff
                            cluster_cats_data[TIMEPOINT_KEY].append(timepoint)
                            cluster_cats_data[self.CLUSTER_CATEGORY_KEY].append(cat)
                            cluster_cats_data[self.SIZE_RATIO_KEY].append(sizeFrac)
                            cluster_cats_data[self.CLUSTER_EDGE_LEN_AVG_KEY].append(avg_edge_len)
                            cluster_cats_data[self.CLUSTER_EDGE_LEN_STD_KEY].append(edge_len_std)
                            cluster_cats_data[self.CLUSTER_NUM_DROPPED_EDGES_KEY].append(len(g.edges) - len(g2.edges))

        return PDPipelineData(pd.DataFrame.from_dict(data=cluster_cats_data),
                              graph_input_data.trange()[graph_input_data.trange() % self.output_tstep == 0])

    def engage_filter(self, g: nx.Graph, conf: Configuration) -> tuple[nx.Graph, list[float]]:
        # let's not damage the existing graph
        g2 = g.copy()
        edge_lens = []
        for p1, p2 in g.edges:
            distance = np.linalg.norm(
                conf.positions[p1, :] -
                    conf.positions[p2, :])
            if self.graphedgelen > 0 and abs(distance - self.graphedgelen) > self.graphedgetolerence:
                g2.remove_edge(p1, p2)
                continue
            edge_lens.append(distance)
        isolate_nodes = [*nx.isolates(g2)]
        g2.remove_nodes_from(isolate_nodes)
        return g2, edge_lens

    def can_parallelize(self):
        return True

    def get_output_data_type(self):
        return PipelineDataType.PIPELINE_DATATYPE_DATAFRAME

    def draw(self) -> tuple[tuple[int, int], draw.Group]:
        (w, y), g = super().draw()
        g.append(draw.Rectangle(0, y, w, 40, stroke="black", stroke_width=1, fill="tan"))
        g.append(draw.Text(f"Target topology: {self.target.name}", font_size=7, x=1, y=y+7))
        y += 12
        g.append(draw.Text("This step classifies cluster graphs as 'match', 'smaller subset',\n"
                           "'smaller not subset', or 'non-match'. The step uses igraph's \n"
                           "`get_subisomorphisms_vf2` function.\n "
                           "The function oeperates by TODO EXPLAIN\n"
                           "The step produces a dataframe \n"
                           "where each row is a cluster with columns for category, size ratio, \n"
                           "average node distance, stdev of node distances, and timepoint.", font_size=7, x=1, y=y+7))
        y += 28
        return (w, y), g

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
        data = data.groupby(TIMEPOINT_KEY).sum(numeric_only=True).reset_index()
        # rename column
        data = data.rename(mapper={ClassifyClusters.SIZE_RATIO_KEY: YIELD_KEY}, axis="columns")
        # data = data.set_index([TIMEPOINT_KEY])
        data = data.loc[data[TIMEPOINT_KEY] % self.output_tstep == 0]
        missing_timepoints = cluster_categories.missing_timepoints(data[TIMEPOINT_KEY].unique().data)
        data = pd.concat([data, pd.DataFrame.from_dict({
            TIMEPOINT_KEY: missing_timepoints,
            YIELD_KEY: 0
        })], ignore_index=True)
        return PDPipelineData(data,
                              cluster_categories.trange()[cluster_categories.trange() % self.output_tstep == 0])

    def get_output_data_type(self):
        """
        Returns:
            the datatype produced by this pipeline step (here, PipelineDataType.PIPELINE_DATATYPE_DATAFRAME)
        """
        return PipelineDataType.PIPELINE_DATATYPE_DATAFRAME

    def draw(self) -> tuple[tuple[int, int], draw.Group]:
        (w, y), g = super().draw()
        g.append(draw.Rectangle(0, y, w, 40, stroke="black", stroke_width=1, fill="tan"))
        g.append(draw.Text(f"Cutoff: {self.cutoff}", font_size=7, x=1, y=y+7))
        y += 10
        g.append(draw.Text(f"Overreach: {self.overreach}", font_size=7, x=1, y=y+7))
        y += 10
        g.append(draw.Text("TODO: write description!", font_size=7, x=1, y=y+7))
        y += 16
        return (w, y), g


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


    def exec(self, input_graphs: ObjectPipelineData) -> PDPipelineData:

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

    def draw(self) -> tuple[tuple[int, int], draw.Group]:
        (w, y), g = super().draw()
        g.append(draw.Rectangle(0, y, w, 40, stroke="black", stroke_width=1, fill="tan"))
        g.append(draw.Text(f"Min Cluster Size: {self.minsize}", font_size=7, x=1, y=y+7))
        y += 10
        g.append(draw.Text("TODO: write description!", font_size=7, x=1, y=y+7))
        y += 30
        return (w, y), g


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
                start=yield_data.get()[TIMEPOINT_KEY].min(),
                stop=yield_data.get()[TIMEPOINT_KEY].max(),
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
