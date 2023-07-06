import re

import networkx as nx
import numpy as np
import pandas as pd

from .analysis_pipeline_step import *
from .patchy_sim_observable import PatchySimObservable
from .yield_analysis_target import ClusterCategory

from .yield_analysis_target import YieldAnalysisTarget

TIMEPOINT_KEY = "timepoint"


# all classes in this document should extend AnalysisPipelineStep

class GraphsFromClusterTxt(AnalysisPipelineHead):
    """
    Analysis operation that reads in the text file produced by the observable
    PLPatchyTopology and outputs a dict where they keys are timepoints and the
    values are the list of cluster graphs at each of those timepoints
    """

    def can_parallelize(self):
        return True

    source_observable: PatchySimObservable

    def __init__(self,
                 name: str,
                 source: PatchySimObservable,
                 output_tstep: Union[int, None] = None):
        super().__init__(name, int(source.print_every), output_tstep)
        self.source_observable = source

    def load_cached_files(self, p: Path) -> nx.Graph:
        assert p.exists()
        with open(p, "rb") as f:
            return pickle.load(f)

    def data_matches_trange(self, data: dict[int: list[nx.Graph]], trange: range) -> bool:
        return all(t in data for t in trange)

    def exec(self, din: Path) -> dict[int: list[nx.Graph]]:
        graphs = {}
        stepcounter = 0
        with open(din, "r") as f:
            # iterate lines in the graph file
            for line in f:
                # skip timepoints that aren't multiples of the specified timestep
                if stepcounter % self.input_tstep == 0:

                    clusterGraphs = []
                    # regex for a single cluster
                    clusters = re.finditer('\[.+?\]', line)

                    # iter regex matches
                    for cluster in clusters:
                        G = nx.Graph()
                        # iter entries within cluster
                        # entries are in format "[source-particle] -> ([space-seperated-list-of-connected-particles])
                        matches = re.finditer(
                            '(\d+) -> \(((?:\d+ ?)+)\)', cluster.group()
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
        return graphs

    def get_input_data_type(self) -> PipelineDataType:
        return PipelineDataType.PIPELINE_DATATYPE_OBSERVABLE

    def get_data_in_filenames(self):
        return [self.source_observable.name]

    def get_output_data_type(self) -> PipelineDataType:
        return PipelineDataType.PIPELINE_DATATYPE_GRAPH

    def get_py_steps_slurm(self,
                           data_sources: tuple[Path],
                           cache_file: Path):
        return "from pypatchy.patchy.analysis_lib import GraphsFromClusterTxt\n" \
               f"obs = PatchySimObservable(**{self.source_observable.to_dict()})\n" \
               f"step = GraphsFromClusterTxt(0, {self.input_tstep}, {self.output_tstep}, obs)\n" \
               f"data = step.exec(Path(\"{data_sources[0]}\")\n)" \
               f"step.cache_data(data, {str(cache_file)})\n"


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

    def can_parallelize(self):
        return True

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

    def load_cached_files(self, p: Path) -> pd.DataFrame:
        with open(p, "r") as f:
            return pd.read_csv(f)

    def data_matches_trange(self, data: PipelineData, trange: range) -> bool:
        return (data["tstep"] == np.array(trange)).all()

    def exec(self, input_graphs: dict[int: list[nx.Graph]]) -> pd.DataFrame:
        cluster_cats_data = {
            TIMEPOINT_KEY: [],
            self.CLUSTER_CATEGORY_KEY: [],
            self.SIZE_RATIO_KEY: []
        }
        # loop timepoints in input graph data
        for timepoint in input_graphs:
            # loop cluster graphs at this timepoint
            for g in input_graphs[timepoint]:
                cat, sizeFrac = self.target.compare(g)

                # assign stuff
                cluster_cats_data[TIMEPOINT_KEY].append(timepoint)
                cluster_cats_data[self.CLUSTER_CATEGORY_KEY].append(cat)
                cluster_cats_data[self.SIZE_RATIO_KEY].append(sizeFrac)
        return pd.DataFrame.from_dict(data=cluster_cats_data)

    def get_input_data_type(self) -> PipelineDataType:
        return PipelineDataType.PIPELINE_DATATYPE_GRAPH

    def get_output_data_type(self) -> PipelineDataType:
        return PipelineDataType.PIPELINE_DATATYPE_DATAFRAME

    def get_py_steps_slurm(self, data_sources: tuple[Path], cache_file: Path) -> str:
        return "import networkx as nx\n" \
               "from pypatchy.patchy.analysis_lib import ClassifyClusters\n" \
               f"target = YieldAnalysisTarget({self.target.name})\n" \
               f"step = ClassifyClusters(0,{self.input_tstep},{self.output_tstep},(),{self.target_name},target_graph)\n" \
               f"data = step.exec(Path(\"{data_sources[0]}\")\n)" \
               f"step.cache_data(data, {str(cache_file)})\n"


YIELD_KEY = "yield"


class ComputeClusterYield(AnalysisPipelineStep):
    def can_parallelize(self):
        return True


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

    def load_cached_files(self, p: Path) -> pd.DataFrame:
        assert p.exists()
        with open(p, "r") as f:
            return pd.read_csv(f)  # TODO: more params probably

    def data_matches_trange(self, data: pd.DataFrame, trange: range) -> bool:
        return (data["tstep"] == np.array(trange)).all()

    def exec(self, cluster_categories: pd.DataFrame) -> pd.DataFrame:
        """
        returns a pandas DataFrame where each row corresponds to a timepoint
        the resulting dataframe will be indexed by timepoint
        """
        # create a 'default' version of the yield dataframe to serve for timepoints with no clusters

        # filter off-target graphs
        data: pd.DataFrame = cluster_categories[
            cluster_categories[ClassifyClusters.CLUSTER_CATEGORY_KEY != ClusterCategory.SMALLER_NOT_SUB]]
        # filter too-small graphs
        data = data[data[ClassifyClusters.SIZE_RATIO_KEY] >= self.cutoff]
        if not self.overreach:
            # filter clusters that are larger than the largest clusters
            data = data[data[ClassifyClusters.CLUSTER_CATEGORY_KEY] != ClusterCategory.OVER]
        else:  # not something I'm currently using by may be useful later
            # max cluster yield should be 1.0
            data[ClassifyClusters.SIZE_RATIO_KEY] = data[ClassifyClusters.SIZE_RATIO_KEY].apply(np.ceil)
        # discard cluster categories column
        data.drop(ClassifyClusters.CLUSTER_CATEGORY_KEY)
        # group by timepoint, average, reset index
        data = data.groupby(TIMEPOINT_KEY).sum().reset_index()
        # rename column
        data = data.rename(mapper={ClassifyClusters.SIZE_RATIO_KEY: YIELD_KEY})
        data = data.set_index([TIMEPOINT_KEY])
        data = data.loc[data[TIMEPOINT_KEY] % self.output_tstep == 0]
        return data

    def get_input_data_type(self) -> PipelineDataType:
        return PipelineDataType.PIPELINE_DATATYPE_DATAFRAME

    def get_output_data_type(self) -> PipelineDataType:
        return PipelineDataType.PIPELINE_DATATYPE_DATAFRAME


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

    def load_cached_files(self, p: Path) -> pd.DataFrame:
        assert p.exists()
        with open(p, "r") as f:
            return pd.read_csv(f)

    def data_matches_trange(self, data: pd.DataFrame, trange: range) -> bool:
        return (data[TIMEPOINT_KEY] == np.array(trange)).all()

    def exec(self, input_graphs: dict[int: list[nx.Graph]]) -> pd.DataFrame:

        cluster_size_data = {
            TIMEPOINT_KEY: [],
            self.MIN_KEY: [],
            self.MAX_KEY: [],
            self.MEAN_KEY: [],
            self.MEDIAN_KEY: [],
            self.STDEV_KEY: []
        }
        # loop timepoints in input graph data
        for timepoint in input_graphs:
            if timepoint % self.output_tstep == 0:
                graph_sizes = [len(g) for g in input_graphs[timepoint] if len(g) >= self.minsize]
                cluster_size_data[TIMEPOINT_KEY] = timepoint
                cluster_size_data[self.MIN_KEY].append(min(graph_sizes))
                cluster_size_data[self.MAX_KEY].append(max(graph_sizes))
                cluster_size_data[self.MEDIAN_KEY].append(np.median(np.array(graph_sizes)))
                cluster_size_data[self.MEDIAN_KEY].append(sum(graph_sizes) / len(graph_sizes))
                cluster_size_data[self.STDEV_KEY].append(np.std(np.array(graph_sizes)))
        return pd.DataFrame.from_dict(data=cluster_size_data)

    def get_input_data_type(self) -> PipelineDataType:
        return PipelineDataType.PIPELINE_DATATYPE_DATAFRAME

    def get_output_data_type(self) -> PipelineDataType:
        return PipelineDataType.PIPELINE_DATATYPE_DATAFRAME

    def can_parallelize(self):
        return False


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
        super().__init__(name, input_tstep, output_tstep, tuple(aggregate_over))

    MIN_KEY = "yield_min"
    MAX_KEY = "yield_max"
    AVG_KEY = "yield_avg"
    STD_KEY = "yield_std"

    def load_cached_files(self, p: Path) -> pd.DataFrame:
        with open(p, "r") as f:
            return pd.read_csv(f)

    def data_matches_trange(self, data: pd.DataFrame, trange: range) -> bool:
        return (data[TIMEPOINT_KEY] == np.array(trange)).all()

    def exec(self, yield_data: pd.DataFrame) -> pd.DataFrame:
        """
        yield_data should be a pd.DataFrame with columns name YIELD_KEY and TIMEPOINT_KEY
        returns a pd.DataFrame where each row is a timepoint, with columns for min, max,
        mean average, and standard deviation of the yield at that timepoint
        the resulting dataframe will be indexed by timepoint
        """

        gb = yield_data.groupby(TIMEPOINT_KEY)
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

        return data

    def get_input_data_type(self) -> PipelineDataType:
        return PipelineDataType.PIPELINE_DATATYPE_DATAFRAME

    def get_output_data_type(self) -> PipelineDataType:
        return PipelineDataType.PIPELINE_DATATYPE_DATAFRAME

    def get_py_steps_slurm(self,
                           data_sources: tuple[Path],
                           cache_file: Path):
        return "from pypatchy.patchy.analysis_lib import ComputeSpecGroupClusterYield\n" \
               f"step = ComputeSpecGroupClusterYield(0, {self.input_tstep}, {self.output_tstep},(),{self.cutoff},{self.overreach},{self.target_name})\n" \
               f"in_data = [pd.read_csv(fpath) for fpath in {data_sources}]\n" \
               "in_data = pd.concat(in_data, ignore_index=True)\n" \
               "data = step.exec(in_data)\n)" \
               f"step.cache_data(data, {str(cache_file)})\n"

    def can_parallelize(self):
        return True
