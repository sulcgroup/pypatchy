"""
This module contains a bunch of classes that extend AnalysisPipelineStep, for use
in the pypatchy.analysis system
"""

# python builtin imports
from __future__ import annotations
import json
import re
import warnings
from functools import cached_property
from typing import TextIO

# third party imports
import networkx as nx
import igraph as ig

# oat/oxpy imports
from oxDNA_analysis_tools.UTILS.RyeReader import get_confs, describe, inbox
from oxDNA_analysis_tools.UTILS.data_structures import Configuration

# ipy_oxDNA imports
from ipy_oxdna.utils.observable import Observable

# pypatchy imports
from .pl.patchyio import PLBaseWriter
from .pl.plparticle import PLPatchyParticle
from .pl.plparticleset import PLParticleSet

from .stage import Stage
from ..analysis.analysis_pipeline_step import *
from .patchy_sim_observable import PatchySimObservable

from ..analysis.analysis_data import *

from ..polycubeutil.polycube_structure import PolycubeStructure
from ..structure import TypedStructure, Structure, read_topology


# this file contains classes that are useful in analysis, but aren't required by other PyPatchy modules
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
    """

    normalize_coords: bool

    def __init__(self,
                 name: str,
                 normalize_coords=True,
                 input_tstep: Union[int, None] = 1,
                 output_tstep: Union[int, None] = None):
        """
        Constructor for traj read step
        I strongly advise initializing with:
        input_tstep = ensemble.get_input_file_param("print_conf_interval")
        :param name: name of step
        :param normalize_coords: whether to "normalize" coordinates (inbox the confs)
        :param input_tstep: input timepoint interval
        :param output_tstep: output timepoint interval
        """
        super().__init__(name, input_tstep, output_tstep)
        self.normalize_coords = normalize_coords
        # really hate that 0th conf isn't included in trajectory
        # self.trajfile = traj_file_regex
        # self.first_conf = first_conf_file_name

    load_cached_files = load_cached_object_data
    DATA_IN_FILENAMES = ["topology", "conf_file", "trajectory_file"]

    def __call__(self,
             ensemble: Any, #PatchySimulationEnselble but we can't use the actual type hint because of circular imports
             sim: PatchySimulation,
             stages: list[Stage],
             *args: list[Path]) -> RawPipelineData:
        # load first conf (not incl in traj for some reason)
        # iter stages
        staged_data = []
        for stage, top_file, first_conf_file_path, traj_file_path in zip(stages, *args):
            ensemble.writer.set_directory(traj_file_path.parent)
            top = ensemble.writer.read_top(top_file)
            top_info, init_conf_info = describe(str(top_file), str(first_conf_file_path))

            # the initial conf isn't part of the traj file so in stage 0 we need to add it ourselves
            # doing this for other stages will cause problems
            if stage.is_first():
                firstconf = get_confs(traj_info=init_conf_info,
                                      top_info=top_info,
                                      start_conf=0,
                                      n_confs=1)[0]
                confdict = {0: (firstconf, top.get_particles_types())}
            else:
                confdict = {}
            # load trajectory confs
            top_info, traj_info = describe(
                str(top_file),
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
                    confdict[conf.time] = (conf, top.get_particles_types())
            staged_data.append(RawPipelineData(confdict))
        data = sum(staged_data, start=RawPipelineData({}))
        return RawPipelineData(data.data)

    def draw(self) -> tuple[tuple[int, int], draw.Group]:
        (w, y), g = super().draw()
        g.append(draw.Rectangle(0, y, w, 40, stroke="black", stroke_width=1, fill="tan"))
        g.append(draw.Text("TODO: write description!", font_size=7, x=1, y=y + 7))
        y += 40
        return (w, y), g

    def get_data_in_filenames(self) -> list[str]:
        return self.DATA_IN_FILENAMES

    def get_output_data_type(self):
        """
        :return: PipelineDataType.PIPELINE_DATATYPE_DATAFRAME
        """
        return PipelineDataType.PIPELINE_DATATYPE_RAWDATA


class LoadEnergies(AnalysisPipelineHead):
    """
    Important to note that the timepoints logged in the energy file are in TIME UNITS while the
    `print_energy_every` param in the input file is in STEPS, no, I do not know why
    """
    POTENTIAL_ENERGY_KEY = "pe"
    KINETIC_ENERGY_KEY = "ke"
    TOTAL_ENERGY_KEY = "te"

    # bad things happen if we try to recompute. i could try to solve or i could Not

    def __init__(self, step_name: str, **kwargs):
        kwargs["step_name"] = step_name

        if "input_tstep" not in kwargs:
            kwargs["input_tstep"] = 1
        else:
            raise Exception("Input tsteps for energy are hardcoded and cannot be set manually")
        super(LoadEnergies, self).__init__(**kwargs)
        self.force_recompute = True

    def get_data_in_filenames(self) -> list[str]:
        return ["energy_file"]

    load_cached_files = load_cached_pd_data

    def __call__(self, e, sim, stages: list[Stage], energy_files: list[Path]) -> PDPipelineData:
        # todo: catch empty csv problem, raise custom exception
        stages_data = [pd.read_csv(stage_energy_file, sep="\s+", header=None) for stage_energy_file in energy_files]
        df = pd.concat(stages_data)
        dt = e.sim_get_param(sim, "dt")
        # set columns names
        df.columns = [TIMEPOINT_KEY, self.POTENTIAL_ENERGY_KEY, self.KINETIC_ENERGY_KEY, self.TOTAL_ENERGY_KEY]
        df[TIMEPOINT_KEY] = df[TIMEPOINT_KEY] / dt  # convert to timepoints
        print_energy_every = e.sim_get_param(sim, "print_energy_every")
        sim_tstep = print_energy_every / dt
        # todo: if the step length isn't a multple of the energy print interval, this will fail
        # if not np.array_equal(df[TIMEPOINT_KEY].values, np.arange(0,
        #                                                           df[TIMEPOINT_KEY].values.max()+1,
        #                                                           sim_tstep) ):
        #     raise Exception(f"Mismatch between timepoint set found in energy files(s)" + ",".join(['`'+str(ef)+'`' for ef in energy_files])
        #                     + f" and expected timepoints based on simulation parameters. ")
        # filter to output tstep
        df = df[np.mod(df[TIMEPOINT_KEY].astype(int), int(self.output_tstep)) == 0]
        return PDPipelineData(df, df[TIMEPOINT_KEY].values)

    def get_output_data_type(self) -> PipelineDataType:
        return PipelineDataType.PIPELINE_DATATYPE_DATAFRAME


class BlobsFromClusters(AnalysisPipelineHead):
    """
    Analysis operation that reads in the text file produced by the observable ?????
    and produces a list of lists of lists of particle types
    """

    def get_data_in_filenames(self) -> list[str]:
        return [self.source_observable.file_name]

    def load_cached_files(self, f: Path) -> PipelineData:
        assert f.is_file()
        with f.open("rb") as datafile:
            return pickle.load(datafile)

    def __call__(self, din: Path) -> PipelineData:
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


class GraphsFromPatchyBonds(AnalysisPipelineHead):
    source_observable: Observable

    def get_output_data_type(self) -> PipelineDataType:
        return PipelineDataType.PIPELINE_DATATYPE_GRAPH

    def __init__(self,
                 name: str,
                 source: Observable,
                 output_tstep: Union[int, None] = None):
        super().__init__(name, int(source.print_every), output_tstep)
        self.source_observable = source

    def get_data_in_filenames(self) -> list[str]:
        return [self.source_observable.file_name]

    def load_cached_files(self, f: Path) -> PipelineData:
        assert f.is_file()
        with f.open("rb") as datafile:
            return pickle.load(datafile)

    def __call__(self, _, __, stages: list[Stage], graphs_files: list[Path]) -> ObjectPipelineData:
        """
        __call__utes analysis step
        """
        graphs = {}
        # iter stages
        for stage, graph_file in zip(stages, graphs_files):
            # first read stage topology
            stage.getctxt().writer.set_directory(stage.getctxt().folder_path(stage.spec(), stage))
            top: PLBaseWriter.PatchyTopology = stage.getctxt().writer.read_top(
                stage.getctxt().sim_stage_get_param(stage.spec(),
                                                    stage,
                                                    "topology"))
            particle_types: PLParticleSet = stage.getctxt().sim_get_particles_set(stage.spec())

            # open stage graph file
            with graph_file.open("r") as f:
                # find stage start time
                while True:  # AAARRRGHHH
                    line = f.readline()
                    if not line:
                        # if line is empty, we are done read file
                        break
                    assert line.startswith("# step"), f"Expected header line beginning with `# step`, got line `{line}`"
                    _, _, step_count, _, n_particles = line.split()
                    step_count = int(step_count)
                    n_particles = int(n_particles)
                    # assert n_particles == stage.num_particles(), "Mismatch between number of particles in stage vs cluster data file."

                    if step_count % self.output_tstep == 0:
                        graphs[step_count] = self.parse_graph(f, n_particles, particle_types, top)

                    else:
                        f.seek(2 * n_particles + 1,
                               1)  # if we are not reading    this timepoint, jump ahead to the next one
        return ObjectPipelineData(graphs)

    def parse_graph(self, f: TextIO, n_particles: int, particle_types: PLParticleSet,
                    top: PLBaseWriter.PatchyTopology) -> nx.MultiDiGraph:
        """
        parses a timepoint of a PatchyBonds input file
        """
        # skip timepoints that aren't multiples of the specified timestep
        G = nx.MultiDiGraph()
        G.add_nodes_from(range(n_particles))
        for p_idx in range(n_particles):
            particle_type: PLPatchyParticle = particle_types.particle(top.get_particles_types()[p_idx])
            G.nodes[p_idx]["particle_type"] = particle_type
            line = f.readline()
            # counts of number of bonds formed by each patch
            occupancies = [int(i) for i in line.split()]
            assert len(occupancies) == particle_type.num_patches(), \
                "Mismatch between number of patches specificed in particle set and clusters file"
            line = f.readline()
            # ids of particles bound to the particle
            # IT INDEXES FROM 1 FOR SOME REASON!?!?!?!?
            ids = [int(i) - 1 for i in line.split()]
            assert len(ids) == sum(occupancies), "Mismatch between bond count and total occupancies!"
            i = 0
            # skip particles w/ 0 bonds
            if sum(occupancies) > 0:
                # convert bond data to graph edges
                for patch, n_bonds in zip(particle_type.patches(), occupancies):
                    for bond in range(n_bonds):
                        assert -1 < ids[i] < n_particles
                        G.add_edge(p_idx, ids[i], patch=patch)
                        i += 1
                assert i == len(ids)
        f.readline()
        # check graph
        for u, v, k in G.edges(keys=True):
            if not G.has_edge(v, u):
                warnings.warn(f"Missing reverse edge ({v}, {u}) for edge ({u}, {v})!")
        return G


class GraphsFromClusterTxt(AnalysisPipelineHead):
    """
    Analysis operation that reads in the text file produced by the observable
    PLClusterTopology and outputs a dict where they keys are timepoints and the
    values are the list of cluster graphs at each of those timepoints
    """

    source_observable: Observable

    def __init__(self,
                 name: str,
                 source: Observable,
                 output_tstep: Union[int, None] = None):
        """
        constructs a graphs from clusters.txt node
        """
        super().__init__(name, int(source.print_every), output_tstep)
        if len(source) == 1:
            assert (*source.get_cols(),)[0].get_type_name() == "PLClusterTopology", \
                f"Observable column has incorrect type! Type should be PLClusterTopology, is {(*source.get_cols(),)[0].get_type_name()}"
        elif len(source) == 2:
            assert (*source.get_cols(),)[1].get_type_name() == "PLClusterTopology", \
                f"Observable column has incorrect type! Type should be PLClusterTopology, is {(*source.get_cols(),)[1].get_type_name()}"
            assert (*source.get_cols(),)[0].get_type_name() == "step", \
                f"Observable column has incorrect type! Type should be step, is {(*source.get_cols(),)[0].get_type_name()}"
        self.source_observable = source


    load_cached_files = load_cached_object_data

    def __call__(self, _, __, stages: list[Stage], graphs_files: list[Path]) -> ObjectPipelineData:
        """
        reads the output of the PLClusterTopology from a text file and converts
        to networkx graphs
        todo: a better way to do at least some of this
        :param _: skip param to match expected signature
        :param __: skip param to match expected signature
        :param stages: list of stages
        :param graphs_files: list of graph files
        """
        graphs: dict[int, list[nx.Graph]] = {}
        # iter stages
        for stage, graph_file in zip(stages, graphs_files):
            # open stage graph file
            with open(graph_file, "r") as f:
                # find stage start time
                # observable does not print at t=0
                # introducing new option: two coulumn output where the first column ist the step
                if len(self.source_observable) == 1:
                    stepcounter = stage.start_time() + self.source_observable.print_every
                # iterate lines in the graph file
                # todo context mgr
                w = stage.getctxt().writer
                w.set_directory(stage.sim_dir)
                # topology in the simulation sense (list of particles w/ typing info)
                sim_top = w.read_top(stage.getctxt().sim_get_stage_top(stage.spec(), stage))
                for line in f:
                    # get timepoint; we will autogenerate them if no timepoints are in observable output
                    if len(self.source_observable) == 2:
                        # split first token from line
                        stepcounter, line = line.split(maxsplit=1)
                        stepcounter = int(stepcounter)
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
                            # add particle types
                            for particle_id in G.nodes:
                                G.nodes[particle_id]["particle_type"] = sim_top.get_particle_type(particle_id)
                            clusterGraphs.append(G)
                        graphs[stepcounter] = clusterGraphs
                    # increment the stepcounter; we'll be hard-setting it at next loop if the timepoint is in the file
                    stepcounter += self.source_observable.print_every
        # idk if this is a thing with all observables or just PLClusterTopology but
        # it doesn't output anything at t=0
        # so if we're starting at the first timestep and don't have a t=0 datapoint,
        # add an empty one
        # TODO: my god please anything else
        if self.source_observable.print_every in graphs and 0 not in graphs:
            graphs[0] = []
        return ObjectPipelineData(graphs)

    def get_data_in_filenames(self):
        return [self.source_observable.file_name]

    def get_output_data_type(self):
        return PipelineDataType.PIPELINE_DATATYPE_GRAPH

    def draw(self) -> tuple[tuple[int, int], draw.Group]:
        (w, y), g = super().draw()
        g.append(draw.Rectangle(0, y, w, 40, stroke="black", stroke_width=1, fill="tan"))
        g.append(draw.Text(f"Source Observable: {self.source_observable.file_name}", font_size=7, x=1,
                           y=y + 7))  # TODO: more info?
        y += 10
        g.append(draw.Text("TODO: write description!", font_size=7, x=1, y=y + 7))
        y += 28
        return (w, y), g

class MatchGraphToCrystal(AnalysisPipelineStep):
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

    target: TypedStructure

    def __init__(self,
                 name: str,
                 target: TypedStructure,
                 input_tstep: Union[int, None] = None,
                 output_tstep: Union[int, None] = None):
        super().__init__(name, input_tstep, output_tstep)
        self.target = target

    CLUSTER_CATEGORY_KEY = "clustercategory"
    SIZE_RATIO_KEY = "sizeratio"

    load_cached_files = load_cached_pd_data

    def __call__(self, input_data: ObjectPipelineData) -> PDPipelineData:
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
                timepoint_data: nx.Graph = input_data.get()[timepoint]
                for cluster_nodes in nx.connected_components(nx.to_undirected(timepoint_data)):
                    g = timepoint_data.subgraph(cluster_nodes)
                    cat, sizeFrac = self.target.compare(g)

                    # assign stuff
                    cluster_cats_data[TIMEPOINT_KEY].append(timepoint)
                    cluster_cats_data[self.CLUSTER_CATEGORY_KEY].append(cat)
                    cluster_cats_data[self.SIZE_RATIO_KEY].append(sizeFrac)
        return PDPipelineData(pd.DataFrame.from_dict(data=cluster_cats_data),
                              input_data.trange()[input_data.trange() % self.output_tstep == 0])

    def get_output_data_type(self):
        return PipelineDataType.PIPELINE_DATATYPE_DATAFRAME

    def draw(self) -> tuple[tuple[int, int], draw.Group]:
        (w, y), g = super().draw()
        g.append(draw.Rectangle(0, y, w, 40, stroke="black", stroke_width=1, fill="tan"))
        g.append(draw.Text(f"Target topology: {self.target.name}", font_size=7, x=1, y=y + 7))
        y += 12
        g.append(draw.Text("This step classifies cluster graphs as 'match', 'smaller subset',\n"
                           "'smaller not subset', or 'non-match'. The step uses igraph's \n"
                           "`get_subisomorphisms_vf2` function. The step produces a dataframe \n"
                           "where each row is a cluster with columns for category, size ratio, \n"
                           "and timepoint.", font_size=7, x=1, y=y + 7))
        y += 28
        return (w, y), g

class ClusterCategory(Enum):
    """
    NEVER actually use one of these objects directly!
    Otherwise comparisons will fail because Pandas is very badly coded!
    [censored swearing profusely]
    """
    OVER = 0
    SMALLER_NOT_SUB = 1
    SUBSET = 2
    MATCH = 3

def graphShape(shapePath):
    with open(shapePath, 'r') as f:
        data = f.read()
    solveSpec = json.loads(data)
    G = nx.Graph()
    for i, _, j, _ in solveSpec['bindings']:
        G.add_edge(i, j)
    return G


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

    target: Structure

    def __init__(self,
                 name: str,
                 target: Union[nx.Graph, Structure],
                 input_tstep: Union[int, None] = None,
                 output_tstep: Union[int, None] = None):
        """
        :param name: name of step
        :param target: target graph or structure to compare clusters to
        :param input_tstep: input timepoint interval
        :param output_tstep: output timepoint interval
        """
        super().__init__(name, input_tstep, output_tstep)
        if isinstance(target, Path) or  isinstance(target, str):
            raise Exception("String paths no longer supported")
        if not isinstance(target, Structure):
            self.target = Structure(target)
        else:
            self.target = target

    CLUSTER_CATEGORY_KEY = "clustercategory"
    SIZE_RATIO_KEY = "sizeratio"

    load_cached_files = load_cached_pd_data

    @cached_property
    def graph_ig(self) -> ig.Graph:
        return ig.Graph.from_networkx(self.target.graph.to_undirected())

    def compare(self, g: nx.Graph) -> tuple[ClusterCategory, float]:
        """
        Compares a cluster graph to the analysis target and returns a classification of the cluster and its yield
        todo does this need to exist

        :param g: a graph

        :return: a tuple where the first element is the category of the cluster, and the seonc element is a float representation of the yield
        """
        # compute size fraction
        sizeFrac = len(g) / len(self.target)
        if sizeFrac > 1:
            return ClusterCategory.OVER, sizeFrac
        # check if g is a subgraph of the target graph
        g_ig = ig.Graph.from_networkx(g.to_undirected())
        if len(self.graph_ig.get_subisomorphisms_vf2(g_ig)) > 0:
            if sizeFrac == 1:
                cat = ClusterCategory.MATCH
            else:
                cat = ClusterCategory.SUBSET
        elif sizeFrac <= 1:
            cat = ClusterCategory.SMALLER_NOT_SUB
        return cat, sizeFrac



    def __call__(self, input_data: ObjectPipelineData) -> PDPipelineData:
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
                    cat, sizeFrac = self.compare(g)

                    # assign stuff
                    cluster_cats_data[TIMEPOINT_KEY].append(timepoint)
                    cluster_cats_data[self.CLUSTER_CATEGORY_KEY].append(cat)
                    cluster_cats_data[self.SIZE_RATIO_KEY].append(sizeFrac)
        return PDPipelineData(pd.DataFrame.from_dict(data=cluster_cats_data),
                              input_data.trange()[input_data.trange() % self.output_tstep == 0])

    def get_output_data_type(self):
        return PipelineDataType.PIPELINE_DATATYPE_DATAFRAME

    def draw(self) -> tuple[tuple[int, int], draw.Group]:
        (w, y), g = super().draw()
        g.append(draw.Rectangle(0, y, w, 40, stroke="black", stroke_width=1, fill="tan"))
        g.append(draw.Text(f"Target topology: {self.target.name}", font_size=7, x=1, y=y + 7))
        y += 12
        g.append(draw.Text("This step classifies cluster graphs as 'match', 'smaller subset',\n"
                           "'smaller not subset', or 'non-match'. The step uses igraph's \n"
                           "`get_subisomorphisms_vf2` function. The step produces a dataframe \n"
                           "where each row is a cluster with columns for category, size ratio, \n"
                           "and timepoint.", font_size=7, x=1, y=y + 7))
        y += 28
        return (w, y), g


class ClassifyPolycubeClusters(ClassifyClusters):
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

    target: PolycubeStructure
    graphedgelen: float
    graphedgetolerence: float

    def __init__(self,
                 name: str,
                 target_name: PolycubeStructure,
                 expected_edge_length: float = 1,
                 edge_distance_tolerance: float = 0.1,
                 input_tstep: Union[int, None] = None,
                 output_tstep: Union[int, None] = None):
        """

        """
        super().__init__(name, target_name, input_tstep, output_tstep)

        self.graphedgelen = expected_edge_length
        self.graphedgetolerence = edge_distance_tolerance

    CLUSTER_CATEGORY_KEY = "clustercategory"
    SIZE_RATIO_KEY = "sizeratio"
    CLUSTER_EDGE_LEN_AVG_KEY = "avgedgelen"
    CLUSTER_EDGE_LEN_STD_KEY = "stdedgelen"
    CLUSTER_NUM_DROPPED_EDGES_KEY = "numdroppededges"

    load_cached_files = load_cached_pd_data

    def __call__(self, input_data_1: ObjectPipelineData, input_data_2: ObjectPipelineData) -> PDPipelineData:
        """
        __call__utes the step
        """
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
            # self.CLUSTER_EDGE_LEN_AVG_KEY: [],
            # self.CLUSTER_EDGE_LEN_STD_KEY: [],
            # self.CLUSTER_NUM_DROPPED_EDGES_KEY: []
        }
        # polycube_type_ids = [cube.get_type() for cube in self.target_polycube.particles()]
        # polycube_type_map = {
        #     type_id: polycube_type_ids.count(type_id)
        #     for type_id in set(polycube_type_ids)
        # }
        # loop timepoints in input graph data
        shared_timepoints = np.intersect1d(graph_input_data.trange(), traj_data.trange())
        if not len(shared_timepoints):
            raise MissingCommonDataError(graph_input_data, traj_data)

        def compare_particle_types(g1: ig.Graph, g2: ig.Graph, v1:int, v2:int):
            return g1.vs[v1]["type"] == g2.vs[v2]["particle_type"]

        for timepoint in shared_timepoints:
            # check output tstep
            if timepoint % self.output_tstep == 0:
                # grab conf and top data at this timepoint (only really need top until i rope in SVD superimposer)

                assert timepoint in traj_data.trange()

                # loop cluster graphs at this timepoint
                for g in graph_input_data.get()[timepoint]:
                    # if the graph isn't just "a particle":
                    igg: ig.Graph = ig.Graph.from_networkx(g).as_directed()
                    if len(igg.vs) > 2:
                        size_frac = len(igg.vs) / len(self.target)
                        if size_frac <= 1:
                            try:
                                subiso  = self.target.to_igraph().get_subisomorphisms_vf2(
                                    igg,
                                    node_compat_fn=compare_particle_types
                                )[0]
                                if size_frac == 1.:
                                    cat = ClusterCategory.MATCH
                                else:
                                    cat = ClusterCategory.SUBSET
                            except IndexError as e:
                                cat = ClusterCategory.SMALLER_NOT_SUB
                        else:
                            cat = ClusterCategory.OVER

                    # # OLD
                    # g2, edge_lens = self.engage_filter(g, conf)
                    # if len(g2.edges) > 0:
                    #     # get particle ids for graph nodes
                    #     particle_ids = [top[n] for n in g.nodes]
                    #
                    #     # get counts for particle ids
                    #     particle_type_counts = {
                    #         typeid: particle_ids.count(typeid) for typeid in set(particle_ids)
                    #     }
                    #
                    #     # test that all particle types in the structure are in the polycube,
                    #     # and are contained the same or fewer number of times
                    #     if all(
                    #             [type_id in polycube_type_map and particle_type_counts[type_id] <= polycube_type_map[
                    #                 type_id]
                    #              for type_id in particle_type_counts]
                    #     ):
                    #     avg_edge_len = np.mean(edge_lens)
                    #     edge_len_std = np.std(edge_lens)
                        # only then do the comparison
                        # cat, sizeFrac = self.target.compare(g2)

                        # assign stuff
                        cluster_cats_data[TIMEPOINT_KEY].append(timepoint)
                        cluster_cats_data[self.CLUSTER_CATEGORY_KEY].append(cat)
                        cluster_cats_data[self.SIZE_RATIO_KEY].append(size_frac)
                        # cluster_cats_data[self.CLUSTER_EDGE_LEN_AVG_KEY].append(avg_edge_len)
                        # cluster_cats_data[self.CLUSTER_EDGE_LEN_STD_KEY].append(edge_len_std)
                        # cluster_cats_data[self.CLUSTER_NUM_DROPPED_EDGES_KEY].append(len(g.edges) - len(g2.edges))

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
        g.append(draw.Text(f"Target topology: {self.target.name}", font_size=7, x=1, y=y + 7))
        y += 12
        g.append(draw.Text("This step classifies cluster graphs as 'match', 'smaller subset',\n"
                           "'smaller not subset', or 'non-match'. The step uses igraph's \n"
                           "`get_subisomorphisms_vf2` function.\n "
                           "The function oeperates by TODO EXPLAIN\n"
                           "The step produces a dataframe \n"
                           "where each row is a cluster with columns for category, size ratio, \n"
                           "average node distance, stdev of node distances, and timepoint.", font_size=7, x=1, y=y + 7))
        y += 28
        return (w, y), g

YIELD_KEY = "yield"

class ComputeClusterYield(AnalysisPipelineStep):
    """
    Computes the yield of cluster data produced by a step like ClassifyPolycubeClusters or ClassifyClusters
    Yield is defined as the number of particles in clusters that are at least a certain size
    Option whether to include clusters that are larger than the target structure
    """
    cutoff: float
    overreach: bool

    def __init__(self,
                 name: str,
                 cutoff: float,
                 overreach: bool = False,
                 input_tstep: Union[int, None] = None,
                 output_tstep: Union[int, None] = None):
        super().__init__(name, input_tstep, output_tstep)
        self.cutoff = cutoff
        self.overreach = overreach

    load_cached_files = load_cached_pd_data

    def __call__(self, cluster_categories: PDPipelineData) -> PDPipelineData:
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
        g.append(draw.Text(f"Cutoff: {self.cutoff}", font_size=7, x=1, y=y + 7))
        y += 10
        g.append(draw.Text(f"Overreach: {self.overreach}", font_size=7, x=1, y=y + 7))
        y += 10
        g.append(draw.Text("TODO: write description!", font_size=7, x=1, y=y + 7))
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

    def __call__(self, input_graphs: ObjectPipelineData) -> PDPipelineData:

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
                graph_sizes = [len(g) for g in input_graphs.get()[timepoint] if len(g) >= self.minsize]
                cluster_size_data[TIMEPOINT_KEY].append(timepoint)
                if not len(graph_sizes):
                    cluster_size_data[self.MIN_KEY].append(0)
                    cluster_size_data[self.MAX_KEY].append(0)
                    cluster_size_data[self.MEDIAN_KEY].append(0)
                    cluster_size_data[self.MEAN_KEY].append(0)
                    cluster_size_data[self.STDEV_KEY].append(0)
                else:
                    cluster_size_data[self.MIN_KEY].append(min(graph_sizes))
                    cluster_size_data[self.MAX_KEY].append(max(graph_sizes))
                    cluster_size_data[self.MEDIAN_KEY].append(np.median(np.array(graph_sizes)))
                    cluster_size_data[self.MEAN_KEY].append(sum(graph_sizes) / len(graph_sizes))
                    cluster_size_data[self.STDEV_KEY].append(np.std(np.array(graph_sizes)))
        return PDPipelineData(pd.DataFrame.from_dict(data=cluster_size_data), input_graphs.trange())

    def get_output_data_type(self) -> PipelineDataType:
        """
        :return: PipelineDataType.PIPELINE_DATATYPE_DATAFRAME
        """
        return PipelineDataType.PIPELINE_DATATYPE_DATAFRAME

    def draw(self) -> tuple[tuple[int, int], draw.Group]:
        (w, y), g = super().draw()
        g.append(draw.Rectangle(0, y, w, 40, stroke="black", stroke_width=1, fill="tan"))
        g.append(draw.Text(f"Min Cluster Size: {self.minsize}", font_size=7, x=1, y=y + 7))
        y += 10
        g.append(draw.Text("TODO: write description!", font_size=7, x=1, y=y + 7))
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

    def __call__(self, yield_data: PDPipelineData) -> PDPipelineData:
        """
        :param yield_data: should be a pd.DataFrame with columns name YIELD_KEY and TIMEPOINT_KEY
        :return: a pd.DataFrame where each row is a timepoint, with columns for min, max,
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
