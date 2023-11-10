from __future__ import annotations

import sys
import json
import matplotlib.pyplot as plt

from pypatchy.design.pathway import Joint, Pathway
from pypatchy.vis_util import get_particle_color
from ..structure import *
from ..polycubeutil.polycube_structure import PolycubeStructure


import itertools
from collections import OrderedDict

from typing import Union, Iterable, Generator

import networkx as nx
import numpy as np


class DesignPathType:
    """
    A design path is a directed graph which contains the same particle types in the same order

    It does not nessecarily include the same set of Joints!
    """

    # list of type indices of particles types in the path - ORDER MATTERS!!!
    _type_path: list[int]
    # dict which maps each generic form of the Pathway to specific Pathways
    # the generic pathway differs from the type_path
    # maintain order of elements in order to make sure it lines up with _joint_class_map
    _instances: OrderedDict[Pathway, list[Pathway]]
    # indexes in self._generic_path of nodes that have the same joint across all paths
    # in other words, places in the path where the same particle is joined to the same other two particles by the
    # same patches
    # _shared_generic_joint_idxs: list[list[set[int]]]

    # flag to refresh a lot of analytics, to avoid too much computation
    _flag_need_refresh: bool

    # array of ints where each element is an index in joint_classes
    _joint_class_map: np.array
    _joint_classes: list[tuple[int, int, int]]
    _score: float  # computed from something or other

    def __init__(self, structure: YieldAllosteryDesigner, *args: Pathway):
        """
        Constructor for a design path. Reqires a minimum of one Pathway
        Args:
            *args:_type_paths
        """
        type_specimen = args[0]

        self._type_path = [structure.particle_type(p.particle()) for p in type_specimen]
        self._instances = OrderedDict()
        # self._shared_generic_joint_idxs = list()
        self._joint_class_map = np.empty(shape=())
        self._joint_classes = []
        self._flag_need_refresh = True
        for path_instance in args:
            self.add_instance(structure, path_instance)

    def triplets(self) -> zip:
        # pairwise('ABCDEFG') --> ABC BCD CDE DEF EFG
        a, b, c = itertools.tee(self._type_path, 3)
        next(b, None)
        next(c, None)
        next(c, None)
        return zip(a, b, c)

    def num_generic_pathways(self) -> int:
        """
        Returns: the number of generic forms of the type pathway
        """
        return len(self._instances)

    def num_pathways(self) -> int:
        return sum([len(pathways) for pathways in self._instances.values()])

    def set_refresh_flag(self):
        self._flag_need_refresh = True

    def clear_refresh_flag(self):
        self._flag_need_refresh = False

    def has_refresh_flag(self) -> bool:
        return self._flag_need_refresh

    def get_instances(self, key: Pathway) -> list[Pathway]:
        return self._instances[key]

    def add_instance(self, structure: YieldAllosteryDesigner, path_instance: Pathway):
        # use "very generic" cycle which uses cube types and colors rather than
        key = Pathway(*[structure.as_very_generic(b) for b in path_instance])
        found = False
        for inst_key in self._instances:
            if inst_key.cycle_equiv(key):
                self._instances[inst_key].append(path_instance)
                found = True
                break
        if not found:
            self._instances[key] = [path_instance]
        # if key not in self._instances:
        #     self._instances[key] = []
        # self._instances[key].append(path_instance)
        self.set_refresh_flag()

    def flat_instances(self) -> Iterable[Pathway, None, None]:
        yield from itertools.chain.from_iterable(self._instances.values())

    def type_keys(self) -> Iterable[Pathway, None, None]:
        yield from self._instances.keys()

    def refresh_numerics(self):
        """
        WARNING: TO FUTURE JOSH
        DO NOT USE ACCESSOR METHODS FOR JOINT_CLASS_MAP OR JOINT_CLASSES IN THIS METHOD
        DOING SO WILL CAUSE. ISSUES

        Returns:

        """
        # clear joint classes
        self._joint_class_map = np.full(fill_value=-1, shape=(self.path_len(), self.num_generic_pathways()))
        self._joint_classes = []

        # iterate path types
        for j, pathway in enumerate(self.type_keys()):
            # iter joints in type path
            for i, joint in enumerate(pathway):
                found_joint_class = False
                for jcidx, existing_jc in enumerate(self._joint_classes):
                    if joint.inner() == existing_jc:
                        self._joint_class_map[i, j] = jcidx
                        found_joint_class = True
                        break
                if not found_joint_class:
                    self._joint_class_map[i, j] = len(self._joint_classes)
                    self._joint_classes.append(joint.inner())

        # compute favorability factor
        slices = self.sorted_slices()
        self._score = (self.rate_slice(*slices[0]) if len(slices) > 0 else 0) / self.path_len()

        self.clear_refresh_flag()
        # probably possible to write a better version of this algorithm but idc
        # old_shared_idxs = self.get_shared_generic_joint_idxs()
        # list each behavior in the different instance type keys
        # generic_joint_types = [[path_type[i] for path_type in self._instances] for i, _ in enumerate(self._type_path)]
        # # find only those joints that are identical in all path types
        # self._shared_generic_joint_idxs = [
        #     list() for _ in generic_joint_types
        # ]
        # # loop through generic joints in sequence
        # for joint_idx, generic_joint_list in enumerate(generic_joint_types):
        #     for path_idx, joint in enumerate(generic_joint_list):
        #         # identify joint type
        #         # loop sets we've already created
        #         found_joint_match = False
        #         for generic_joint_set in self._shared_generic_joint_idxs[joint_idx]:
        #             # take 0th element, all should be functionally the same
        #             gjoint = generic_joint_list[list(generic_joint_set)[0]]
        #             # if joint equivelant
        #             if gjoint.inner_equal(joint):
        #                 found_joint_match = True
        #                 generic_joint_set.add(joint_idx)
        #                 break
        #         if not found_joint_match:
        #             # new set of generic joints
        #             self._shared_generic_joint_idxs[joint_idx].append({joint_idx})
        # # self._shared_generic_joint_idxs = {
        # #     idx for idx, generic_joint_list in enumerate(generic_joint_types)
        # #     if all([b == generic_joint_list[0] for b in generic_joint_list])
        # # }
        # return old_shared_idxs == self.get_shared_generic_joint_idxs()

    def get_shared_generic_joint_idxs(self) -> set[int]:
        return {i for i in range(self.path_len()) if len(np.unique(self.joint_map()[i, :])) == 1}
        # return {*self._shared_generic_joint_idxs}

    def joint_classes(self) -> list[tuple[int, int, int]]:
        if self.has_refresh_flag():
            self.refresh_numerics()
        return self._joint_classes

    def joint_map(self) -> np.ndarray:
        if self.has_refresh_flag():
            self.refresh_numerics()
        return self._joint_class_map

    def matches(self, context: YieldAllosteryDesigner, other: Pathway) -> Union[Pathway, None]:
        """
        Checks if this cycle passed matches the type path
        The cycle is said to "match" if there exists a shift (not reflection!) that will make the cube types line up properly
        Args:
            context:
            other:

        Returns:

        """
        assert other.is_cycle()
        # if the paths are different lengths there is no way they will match
        if self.path_len() != len(other):
            return None
        for shift, _ in enumerate(self._type_path):
            shifted_cycle = other << shift
            if all([context.particle_type(shifted_cycle[idx].particle()) == ptype
                    for idx, ptype in enumerate(self.type_path())]):
                return shifted_cycle
        return None
        # return any([
        #     all([
        #             context.particle_type((other << i)[idx].particle()) == ptype for idx, ptype in enumerate(self._type_path)
        #     ]) for i, _ in enumerate(self._type_path)
        # ])
        # return self.path_len() == len(other) and all([
        #     context.particle_type(other[idx].particle()) == ptype for idx, ptype in enumerate(self._type_path)
        # ])

    def path_len(self) -> int:
        return len(self._type_path)

    def generic_paths(self) -> list[Pathway]:
        """
        Returns a list of generic paths that are components of this design path type
        """
        return list(self._instances.keys())

    def type_path(self) -> list[int]:
        return self._type_path

    def __contains__(self, item: Union[Pathway]) -> bool:
        if isinstance(item, Pathway):
            # i'm like 50/50 on this one
            return any([item.cycle_equiv(p) for p in self._instances])
        else:
            raise TypeError()

    def type_contains(self, other: DesignPathType) -> bool:
        """

        """
        if len(self.type_path()) < len(other.type_path()):
            return False
        # iter possible slices of self.type_path
        for i in range(len(self.type_path()) - len(other.type_path())):
            subpath = self.type_path()[i:i + len(other.type_path())]
            if subpath == other.type_path():
                return True
        return False

    def any_generic_path_contains(self, other: DesignPathType) -> bool:
        for gpath, other_g_path in itertools.product([self.generic_paths(), other.generic_paths()]):
            if other_g_path in gpath:
                return True
        return False

    def describe(self) -> str:
        sz = ','.join([str(x) if i not in self.get_shared_generic_joint_idxs() else '*'+str(x)+'*' for i, x in enumerate(self._type_path)])
        reprstr = f"Path with x {self.num_generic_pathways()}, len {self.path_len()}: {sz}\n"
        reprstr += "\n".join([f"\t{i} (x {len(self._instances[generic_path])}): {repr(generic_path)}" for i, generic_path in enumerate(self._instances) ])
        reprstr += "\nJoint Map"
        reprstr += "\n" + str(self.joint_map().T)
        reprstr += f"\nFavorability Factor: {self.fav_factor()}"
        return reprstr

    def fav_factor(self) -> float:
        """
        computes the design's favorability factor
        TODO: BETTER NAME
        """
        if self.has_refresh_flag():
            self.refresh_numerics()
        return self._score
        # return 1 / self.path_len()
        # return self.num_generic_pathways() * len(self.get_shared_generic_joint_idxs()) / self.path_len()

    # def find_forks(self) -> Iterable[int, None, None]:
    #     """
    #     Locates positions in the design path where the same particle instance joins something to different particle
    #     instances of the same type
    #     Returns:
    #
    #     """
    #     # iter joint idxs
    #     for idx in range(self.path_len()):
    #         p0 = list(self.generic_paths())[0][idx].particle()
    #         p1 = list(self.flat_instances())[0][idx].next_neighbor()
    #         if all([pathway[idx].particle() == p0 for pathway in self.generic_paths()]):
    #             if not all([pathway[idx].next_neighbor() == p1 for pathway in self.flat_instances()]):
    #                 yield idx

    # def find_origin_point(self):
    #     pass

    def slices(self, gtypeidx: int) -> Generator[tuple[int, int], None, None]:
        """
        Iterates through positions on the path searching for Slice Points
        a pair of points on a cycle are slice points if they subdivide the cycle
        into two paths such that no joint and its reverse both appear on the same path
        Args:
            the generic type to work with

        Returns:

        """
        cycle = self.generic_paths()[gtypeidx]
        for x, _ in enumerate(cycle):
            for y, _ in enumerate(cycle[x:]):
                # slice subcycles
                p1 = cycle[x:y:1]
                if x != y:
                    p2 = cycle[y:x:1]
                else:
                    p2 = Pathway()
                # check if slices are valid for slice pts
                assert len(p1) + len(p2) == len(cycle)
                p1_ok = not any([~j in p1 for j in p1])
                p2_ok = not any([~j in p2 for j in p2])
                if p1_ok and p2_ok:
                    yield x, y

    def rate_slice(self, gpathidx: int, x: int, y: int):
        """
        The rating of a slice is the sum of the squares of the lengths of a subpaths formed by the slice points
        Best possible slice is len(path)^2, worst is len(path)^2 / 2

        Args:
            gpathidx: generic path index
            x: one slice point
            y: another slice point

        Returns:

        """
        p1 = self.generic_paths()[gpathidx][x:y:1]
        p2 = self.generic_paths()[gpathidx][y:x:-1]
        return len(p1) ** 2 + len(p2) ** 2
        # return (x - y) % len(self.generic_paths()[gpathidx]) ** 2 + (y - x) % len(self.generic_paths()[gpathidx])

    def sorted_slices(self) -> list[tuple[int, int, int]]:
        """
        Returns a list of ways to slice the design path, sorted by how good it do slice
        """
        slices = itertools.chain.from_iterable([[(i, *s) for s in self.slices(i)] for i, _ in enumerate(self.generic_paths())])
        slices = [i for i in sorted(slices, key=lambda x: self.rate_slice(*x), reverse=True)]
        return slices


    # def divisions(self) -> Iterable[tuple[int, int], None, None]:
    #     """
    #
    #     Returns:
    #         A generator which yields pairs of coordinates within the design path which divide the path
    #         TODO TODO TODO
    #     """
    #     pass


class YieldAllosteryDesigner(PolycubeStructure):
    """
    Class to design allosteric rules that improve yield. hopefully
    this class should be immutable-ish!!! structure component at least. do not mess with structure
    outside __init__!!! pls.
    """

    # ?
    ct_allo_vars: list[set]
    # ???
    vjoints: dict[int, set[Joint]]
    # list of design paths
    _design_paths: Union[list[DesignPathType], None]
    # design path graph tree thingy
    _design_path_tree: Union[nx.DiGraph, None]

    def __init__(self, r, structure):
        super(YieldAllosteryDesigner, self).__init__(rule=r, structure=structure)
        self.ct_allo_vars = [set() for _ in r.particles()]
        self.vjoints = dict()
        self.joints = set()
        # loop nodes
        for n in self.graph.nodes:
            self.vjoints[n] = set()
            adj_edges = self.graph.out_edges(n)
            adj_nodes = [v for u, v in adj_edges]
            ncube = self.cubeList[n]
            for n1, n3 in itertools.permutations(adj_nodes, 2):
                # grab cube objects for n1, n3
                n1cube = self.cubeList[n1]
                n3cube = self.cubeList[n3]
                # grab diridxs of in, out edges. could be important if we ever want to
                # diversify this algorityhm out of cubes
                d1in = self.graph.get_edge_data(n1, n)["dirIdx"]
                d1out = self.graph.get_edge_data(n, n1)["dirIdx"]
                d3in = self.graph.get_edge_data(n3, n)["dirIdx"]
                d3out = self.graph.get_edge_data(n, n3)["dirIdx"]
                edge_in = (ncube.get_patch(d1out).get_id(), n1cube.get_patch(d1in).get_id())
                edge_out = (ncube.get_patch(d3out).get_id(), n3cube.get_patch(d3in).get_id())
                assert edge_out[0] != edge_in[0] # no particle should have two patches w/ same idx
                j = Joint(n,
                          n1,
                          n3,  # TODO: DO PATCHES IN POLYCUBESTRUCTURES HAVE UNIQUE IDS OR TYPE IDS
                          edge_in,
                          edge_out
                          )
                self.joints.add(j)
                self.vjoints[n].add(j)
        self._design_paths = None

    def get_joint(self, n1, n2, n3) -> Union[Joint, None]:
        """

        Args:
            n1: node preceeding n2
            n2: n2
            n3: node succeeding n2

        Returns:
            A Joint linking n1 to n2 and n2 to n3, or None if no such joint exists

        """
        if n2 not in self.vjoints:
            return None
        options = self.vjoints[n2]
        myjoint = [x for x in options if x.unique_tuple() == (n1, n2, n3)]
        if len(myjoint) == 1:
            return myjoint[0]
        else:
            return None

    def as_generic(self, b: Joint):
        """
        Args:
            b: a Joint object

        Returns: the "generic" form of the joint object, where particle UIDs are replaced by particle type indices

        """
        return Joint(
            self.particle_type(b.particle()),
            self.particle_type(b.prev_neighbor()),
            self.particle_type(b.next_neighbor()),
            b.prev_edge(),
            b.next_edge()
        )

    def as_very_generic(self, b: Joint) -> Joint:
        """
        todo: TORSION! cross product or something? idk
        Args:
            b: a Joint object

        Returns: the "very generic" form of the joint object, where particle UIDs are replaced by
         particle type indices and patch IDs are replaced by colors

        """
        return Joint(
            self.particle_type(b.particle()),
            self.particle_type(b.prev_neighbor()),
            self.particle_type(b.next_neighbor()),
            (
                self.rule.patch(b.particle_prev_patch()).color(),
                self.rule.patch(b.prev_edge()[1]).color()
            ),
            (
                self.rule.patch(b.particle_next_patch()).color(),
                self.rule.patch(b.next_edge()[1]).color()
            )
        )

    # def align_cycles_generic(self, c1: Pathway, c2:Pathway) ->bool:
    #     assert c1.is_cycle() and c2.is_cycle()


    # def subbehavior(self, b: Joint):
    #     """
    #     todo: figure out what i'm trying to do here and then do it
    #     Args:
    #         b:
    #
    #     Returns:
    #
    #     """
    #     type_idx = self.particle_type(b.particle())
    #     p_in = self.rule.patch(b.particle_prev_patch())
    #     p_out = self.rule.patch(b.particle_next_patch())
    #
    #     return None

    def cycles_by_size(self, filter_repeat_nodes=True) -> dict[int: list[int]]:
        """
        Code partially written by ChatGPT
        Construct a list of unique cycles in the graph,
        where each second-level list is composed of cycles of the same size,
        and each cycle visits each node at most once.

        Returns:
        dict: dictionary where keys are cycle sizes and values are lists of unique cycles
        """

        # Use simple_cycles to get all cycles in the graph
        all_cycles = list(nx.simple_cycles(self.graph))

        # Filter out cycles with fewer than 3 nodes
        # (since the graph is undirected, any edge is also a "cycle" so filter those out)
        cycles = [cycle for cycle in all_cycles if len(cycle) > 2]
        if filter_repeat_nodes:
            # filter and cycles that visit any node more than once
            cycles = [cycle for cycle in cycles if len(cycle) == len(set(cycle))]

        # Remove duplicates from the cycles (ignoring the order of nodes)
        unique_cycles = list(set(frozenset(cycle) for cycle in cycles))

        # Initialize a default dictionary to store cycles by size
        unique_cycles_by_size = defaultdict(list)

        # Iterate over the cycles
        for cycle in unique_cycles:
            # The size of the cycle is its length
            size = len(cycle)

            # Append the cycle to the list of cycles of the same size
            unique_cycles_by_size[size].append(self.graph.subgraph(list(cycle)))

        return unique_cycles_by_size

    def add_allostery_badly(self) -> Iterable[PolycubesRule, None, None]:
        """
        Adds allosteric controls
        """
        # iterate design paths, from the one with the best design size to the one with the worse design size
        design_paths = self.design_paths()
        types_processed = set()
        for dp in sorted(design_paths, key=lambda x: x.fav_factor(), reverse=True):
            self.describe_design_path(dp)
            for type_idx in dp.get_shared_generic_joint_idxs():
                ctid = dp.type_path()[type_idx]
                if ctid in types_processed:
                    continue
                else:
                    types_processed.add(ctid)
                    ct = self.rule.particle(ctid)
                    j: Joint = dp.generic_paths()[0][type_idx]
                    # safeguard: don't add allostery to patches that already start on
                    # TODO: potentially, modify this to allow for existing allsotery to be modified
                    if ct.count_start_on_patches() == 1:
                        continue

                    # get patch that will activate this node's cube
                    # if the origin patch doesn't already have a state variable, make one
                    origin_patch = self.rule.patch(j.particle_prev_patch())
                    if not origin_patch.state_var():
                        # add a state variable to the current node
                        origin_state = ct.add_state_var()
                        origin_patch.set_state_var(origin_state)
                        self.ct_allo_vars[ct.type_id()].add(origin_state)

                    # add an activator
                    target_patch = self.rule.patch(j.particle_next_patch())
                    # target patch should not already be allosterically controlled
                    if target_patch.activation_var():
                        continue
                    new_activation_state = ct.add_state_var()
                    target_patch.set_activation_var(new_activation_state)

                    # make all cycle control states that this type has required for activating the connection
                    ct.add_effect(DynamicEffect(self.ct_allo_vars[ct.type_id()], new_activation_state))

                    yield copy.deepcopy(self.rule)

        # cycles = self.cycles_by_size(False)
        # if len(cycles) == 0:
        #     print("Graph has no cycles. exiting...")
        #     return
        # for cycle_size, cycle_list in sorted(cycles.items()):
        #     # i believe this was a rejected name for dykes on bikes
        #     # ok but seriously: check for cycles that contain the same cube type
        #     # and same connection faces on types
        #     # in the same pattern. we should be able to design allostery for the cycles
        #     for homocycles in self.homologous_cycles(cycle_list):
        #
        #         # get list of nodes that will be used to construct allostery for this group of homologous cycles
        #         nodes_to_design = self.get_design_path(homocycles)
        #
        #         types_processed = set()
        #
        #         for prev_node, node, next_node in triplets(nodes_to_design):
        #             ct = self.cubeList[node].get_type()
        #
        #             # safeguard: don't add allostery to patches that already start on
        #             # TODO: potentially, modify this to allow for existing allsotery to be modified
        #             if ct.count_start_on_patches() == 1:
        #                 continue
        #             if not ct.type_id() in types_processed:
        #                 types_processed.add(ct.type_id())
        #
        #                 # get patch that will activate this node's cube
        #                 origin_patch = ct.patch(self.get_arrow_local_diridx(node, prev_node))
        #                 # if the origin patch doesn't already have a state variable, make one
        #                 if not origin_patch.state_var():
        #                     # add a state variable to the current node
        #                     origin_state = ct.add_state_var()
        #                     origin_patch.set_state_var(origin_state)
        #                     self.ct_allo_vars[ct.type_id()].add(origin_state)
        #
        #                 # add an activator
        #                 target_patch = ct.patch(self.get_arrow_local_diridx(node, next_node))
        #                 # target patch should not already be allosterically controlled
        #                 if target_patch.activation_var():
        #                     continue
        #                 new_activation_state = ct.add_state_var()
        #                 target_patch.set_activation_var(new_activation_state)
        #
        #                 # make all cycle control states that this type has required for activating the connection
        #                 ct.add_effect(DynamicEffect(self.ct_allo_vars[ct.type_id()], new_activation_state))
        #
        #                 yield copy.deepcopy(self.rule)

                # # starting from common nodes, add allostery to particles, where
                # # the patch closer to the common nodes activates the one farther
                #
                # cycle = homocycles[0]  # at this point the homologous cycles are functionally indistinguishable
                #
                # # create a set for nodes we've already done in this step
                #
                # # don't add allostery to a cube type more than once in the same cycle
                # types_processed = {self.cubeList[start_node].get_type().getID()}
                #
                # allo_nodes_this_step = {start_node}
                # # this while loop is a time bomb
                # while len(allo_nodes_this_step) < cycle_size:
                #     next_head_nodes = []
                #
                #     # loop head nodes
                #     for current_node in cycle_nodes_to_process:
                #         if current_node not in allo_nodes_this_step:
                #             # move to next node
                #             # advance the head to a node which is in the cycle that is not in allo_nodes_this_step,
                #             # and find indexes in RULE_ORDER of faces on cube type that are responsible
                #             # for joining our new head_node to the previous and next nodes in the cycle
                #             face_conn_prev, current_node, face_conn_next = self.next_node_in_cycle(current_node,
                #                                                                                    cycle,
                #                                                                                    allo_nodes_this_step)
                #
                #             ct: PolycubeRuleCubeType
                #             ct = self.cubeList[current_node].get_type()
                #             if not ct.getID() in types_processed:
                #                 types_processed.add(ct.getID())
                #                 # add a state variable to the current node
                #                 new_state = ct.add_state_var()
                #                 self.ct_allo_vars[ct.getID()].append(new_state)
                #
                #                 # add an activator
                #                 new_activation_state = ct.add_state_var()
                #
                #                 # make all cycle control states that this type
                #                 # has required for activating the connection
                #                 ct.add_effect(DynamicEffect(self.ct_allo_vars[ct.getID()], new_activation_state))
                #
                #             # add to set of nodes we've processed
                #             allo_nodes_this_step.add(current_node)
                #             # add our new head node to the list of head nodes for the next step
                #             next_head_nodes.add(current_node)
                #     cycle_nodes_to_process = next_head_nodes

    def describe_design_path(self, dp: DesignPathType):
        print(dp.describe())
        layout = nx.spring_layout(self.graph)
        xs = max([len(dp.get_instances(p)) for p in dp.generic_paths()])
        ys = 1 + len(dp.generic_paths())
        fig, axs = plt.subplots(ncols=xs, nrows=ys, figsize=(4 * xs, 4 * ys))
        fig.set_label(f"Path Hash {hash(frozenset(dp.type_path()))}")
        self.draw_structure_graph(axs[0, 0], layout)
        # erase dostractomg empty plots
        for x in range(1, xs):
            axs[0, x].set_visible(False)
        # make color map dict
        ptypemap = {particle_id: get_particle_color(self.particle_type(particle_id)) for particle_id in self.graph.nodes}
        for i, p_generic in enumerate(dp.generic_paths()):
            p_instances = dp.get_instances(p_generic)
            for j, p in enumerate(p_instances):
                G = p.path_graph()
                nx.draw(G, layout, ax=axs[i + 1, j], with_labels=True, connectionstyle='arc3,rad=0.1', node_color=[
                    ptypemap[particle_id] for particle_id in G.nodes
                ])

                # Create edge labels for each edge including the 'patch' attribute
                edge_labels = nx.get_edge_attributes(G, 'patch')

                # Draw edge labels, manually adjusting the position if necessary
                for edge in G.edges():
                    edge_label = {edge: edge_labels[edge]}

                    # Adjust the label position based on the edge direction
                    if edge[::-1] in G.edges():  # Check if the reverse edge exists
                        # Shift the edge label position to avoid overlap
                        label_pos = {edge: [(layout[edge[0]][0] + layout[edge[1]][0]) / 2,
                                            (layout[edge[0]][1] + layout[edge[1]][1]) / 2 + 0.1]}  # Adjust 0.1 or as needed
                    else:
                        label_pos = {edge: [(layout[edge[0]][0] + layout[edge[1]][0]) / 2,
                                            (layout[edge[0]][1] + layout[edge[1]][1]) / 2]}

                    nx.draw_networkx_edge_labels(G, ax=axs[i + 1, j], pos=layout, edge_labels=edge_label)

                # Show the plot
                # plt.show()
                # nx.draw(p.graph(), axs[j, i+1], pos=layout, node_color=ptypemap)
            for xtra in range(len(p_instances), xs):
                axs[ i+1, xtra].set_visible(False)
        # fig.tight_layout()
        plt.show()
    # def pathway(self, *args: int):
    #     for i, j, k in triplets(args):
    #         e1 = (i, j)
    #         e2 = (j, k)

    def evans_cycles(self,
                     current_node: int,
                     cycle: Union[Pathway, None] = None,
                     previous_node: Union[int, None] = None) -> Generator[Pathway, None, None]:
        """
        ChatGPT wrote this code
        Returns "josh cycles" (todo: figure out real term), which visit each edge at most once
        but can visit nodes unlimited times

        this algorithm is incredibly dangerous

        Args:
            current_node: the current node
            cycle: the cycle being generated by this
            previous_node: the previous node
        Returns:

        """
        # loop neighbors of current node
        for neighbor in self.graph.neighbors(current_node):
            if neighbor == previous_node:
                continue
            edge = (current_node, neighbor)
            # if self, neighbor isn't in our current edge set
            # also check if cycle is still empty (begin case)
            if cycle is None or not cycle.in_main(edge):
                # a joint requires 3 nodes, so can't start populating our generator until the
                # third node
                # handle origin case - cycle is none
                if previous_node is None:
                    assert cycle is None
                    # recursion!
                    yield from self.evans_cycles(neighbor, cycle, current_node)
                else:
                    if neighbor == previous_node:
                        continue
                    new_joint = self.get_joint(previous_node, current_node, neighbor)
                    # if we're on node 2 of our cycle
                    if cycle is None:
                        # create cycle
                        new_cycle = Pathway(new_joint)
                        # yield from functon
                        yield from self.evans_cycles(neighbor, new_cycle, current_node)
                    # gotta keep default case seperate from origin case
                    else:
                        # if neighbor is already in cycle
                        if neighbor == cycle[0].particle() and cycle[0].prev_neighbor() == current_node:
                            # Close the cycle
                            # cycle_start = cycle.index(neighbor)
                            new_cycle = cycle + self.get_joint(previous_node, current_node, neighbor)
                            assert new_cycle.is_cycle()
                            yield new_cycle
                        else:
                            yield from self.evans_cycles(neighbor, cycle + new_joint, current_node)

    def all_unique_evans_cycles(self):
        # create set of unique cycle ids to aid in hashing
        unique_cycles = set()
        for i in range(self.num_vertices()):
            for cycle in self.evans_cycles(i):
                if cycle.cycle_id() not in unique_cycles:
                    # emplace cycle
                    unique_cycles.add(cycle.cycle_id())
                    # yield

                    yield cycle

    def compute_design_paths(self):
        """
        Computes list of design paths for this structure
        this method should only have to be called once
        don't want to put it in the constructor since it may be time-consuming
        """
        # list is NOT indexed! instead it's sorted by design path size
        self._design_paths: list[DesignPathType] = []
        # loop evans cycles
        for cycle in self.all_unique_evans_cycles():
            # loop all design paths
            found_path = False
            for dpath in self._design_paths:
                # if the design path contains this cycle
                cycle_shifted = dpath.matches(self, cycle)
                if cycle_shifted:
                    found_path = True
                    dpath.add_instance(self, cycle_shifted)
            # if no existing design path, add it
            if not found_path:
                self._design_paths.append(DesignPathType(self, cycle))
                # not enough complexity to bother with a bisect sort
                self._design_paths = sorted(self._design_paths, key=lambda d: d.path_len(), reverse=True)

        # construct graph with nodes that are design path types
        self._design_path_tree = nx.DiGraph()
        all_type_keys = list(itertools.chain.from_iterable([dp.type_keys() for dp in self._design_paths]))
        for i, key in enumerate(all_type_keys):
            self._design_path_tree.add_node(key, dp_id=chr(i+65))
        # add edges showing relations between design paths
        for key1, key2 in itertools.product(all_type_keys, all_type_keys):
            if key1 == key2:
                continue
            if key1 in key2:
                self._design_path_tree.add_edge(key2, key1)

    def design_paths(self) -> list[DesignPathType]:
        """
        Returns: a list of design path types, sorted largest->smallest
        """
        if self._design_paths is None:
            self.compute_design_paths()
        return self._design_paths

    # def iter_design_paths(self, n: Union[int, None],
    #                       visited_edges: Union[list[tuple[int, int]], None] = None,
    #                       veset: Union[set[tuple[int, int]], None] = None) -> Generator[DesignPath, None, None]:
    #     """
    #     Recursive depth-first search for design paths
    #     Warning: this could potentially eat a LOT of memory
    #     Should generally return longer paths first although this is not guarenteed
    #
    #     Args:
    #         n: a node to start (or continue) from
    #         visited_edges: an ordered list of edges which have been visited
    #         veset: a set of edges which have been visited, non-ordered but optimized for quick access
    #
    #     Returns:
    #
    #     """
    #
    #     if visited_edges is None:
    #         visited_edges = []
    #         veset = set()
    #
    #     if n is None:
    #         n = self.graph.nodes[0]
    #
    #     for n_dest in self.graph.out_edges()[n]:
    #         # non-directional check
    #         if (n, n_dest) not in veset and (n_dest, n) not in veset:
    #             # append edge to list
    #             extension_list = [*visited_edges, (n, n_dest)]
    #             extension_set = veset.copy()
    #             extension_set.add((n, n_dest))
    #             # recurse, w/ new path and destination node
    #             for p in self.iter_design_paths(n_dest, extension_list, veset):
    #                 yield p

    def get_graphs_center(self, node_list):
        return sum([self.cubeList[n].get_position() for n in node_list]) / len(node_list)

    # def get_design_path(self, cycles):
    #
    #     # vector of the center of the homologous cycles
    #     # compute valid paths. can take cycles[0] as our graph because cycles are homologous
    #     all_paths = all_unique_paths(cycles[0])
    #     valid_paths = [p for p in all_paths if self.is_valid_design_path(p)]
    #
    #     # compute overlap of homologous cycles
    #     cycles_nodes_overlap = get_nodes_overlap(cycles)
    #
    #     # if overlap exists, require design paths to start from an overlap node
    #     if len(cycles_nodes_overlap) > 0:
    #         valid_paths = [p for p in valid_paths if p[0] in cycles_nodes_overlap]
    #
    #     assert len(valid_paths) > 0
    #
    #     # sort paths from longest to shortest
    #     best_paths = longest_paths(valid_paths)
    #
    #     # compute centerpoint
    #     centerpoint = self.get_graphs_center([*itertools.chain.from_iterable(cycles)])
    #
    #     shortest_distance = math.inf
    #     best_path = []
    #
    #     # find path which is closest to the center point
    #     for p in best_paths:
    #         distance = np.linalg.norm(centerpoint - self.cubeList[p[0]].get_position())
    #         if distance < shortest_distance:
    #             shortest_distance = distance
    #             best_path = p
    #     return best_path

    def is_valid_design_path(self, p):
        """
        Parameters:
            p (list of ints) an ordered list of node IDs representing a path
        Returns:
            true if p is a valid design path, false otherwise
        """

        behavior_set: set[tuple[int, int, int]] = set()
        # iterate through triplets in the design path
        for prev_node, curr_node, next_node in triplets(p):
            # the cube type at the origin of the design path can't occur anywhere else in the path
            if self.cubeList[curr_node].get_type().type_id() == self.cubeList[p[0]].get_type().type_id():
                return False
            curr_prev_edge = self.get_arrow_local_diridx(curr_node, prev_node)
            curr_next_edge = self.get_arrow_local_diridx(curr_node, next_node)
            # "back" and "front" here are meant not in a physical sense but in the sense of the cycle
            # synonyms to "next" and "prev" kinda
            # get cube type so we use the same patch ids for different cube instances of the same type
            back_patch = self.cubeList[curr_node].get_type().patch(curr_prev_edge)
            front_patch = self.cubeList[curr_node].get_type().patch(curr_next_edge)
            # cube behavior is
            behavior = (back_patch.get_id(),
                        self.cubeList[curr_node].get_type().type_id(),
                        front_patch.get_id())

            # if the path passes through the same behavior in reverse, it's not valid
            if reversed(behavior) in behavior_set:
                return False
            else:
                behavior_set.add(behavior)
        # if len(p) was less than 2, nothing happened in that for-loop because a 2-length path has no triples
        return len(p) > 2

    def summarize(self):
        self.compute_design_paths()
        fig, ax = plt.subplots(figsize=(14, 8))
        label_mapping = nx.get_node_attributes(self._design_path_tree, "dp_id")

        # Make sure to invert the label mapping if necessary.
        # In the legend, we want to show the label (the long name) and map it to the id (the short name).
        inverted_label_mapping = {v: k for k, v in label_mapping.items()}

        # Draw the graph using the labels from label_mapping for the node labels
        nx.draw(self._design_path_tree, ax=ax, with_labels=True, labels=label_mapping)

        # Create legend handles manually
        legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markersize=15, markerfacecolor='skyblue',
                                     label=f'{dp_id}: {inverted_label_mapping[dp_id]}')  # Here we use the inverted mapping
                          for dp_id in label_mapping.values()]  # The values are the actual ids used for the labels

        # Add the legend to the plot
        ax.legend(handles=legend_handles, title="Design Paths", bbox_to_anchor=(0.25, 1), loc='upper left')

        # This will ensure that the legend and the graph are not cut off
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        for dp in self.design_paths():
            self.describe_design_path(dp)


def triplets(iterable: typing.Iterable):
    # pairwise('ABCDEFG') --> ABC BCD CDE DEF EFG
    a, b, c = itertools.tee(iterable, 3)
    next(b, None)
    next(c, None)
    next(c, None)
    return zip(a, b, c)


def longest_paths(paths):
    best = []
    best_length = 0
    for p in paths:
        if len(p) == best_length:
            best.append(p)
        elif len(p) > best_length:
            best = [p]
            best_length = len(p)

    return best


def generate_all_paths(graph, start, path=None):
    """
    Generate all possible paths in the graph that start at a given node.

    Parameters:
    graph (networkx.classes.digraph.DiGraph): The graph
    start: The starting node

    Returns:
    generator: A generator that yields all possible paths
    """
    if path is None:
        path = [start]

    yield path

    for neighbor in graph.neighbors(start):
        if neighbor not in path:
            yield from generate_all_paths(graph, neighbor, path + [neighbor])


def all_unique_paths(graph):
    """
    Generate all unique paths in the graph.

    Parameters:
    graph (networkx.classes.digraph.DiGraph): The graph

    Returns:
    list: A list of all unique paths
    """
    paths = []
    for node in graph.nodes:
        paths.extend(generate_all_paths(graph, node))

    return paths


if __name__ == "__main__":
    jsonfile = sys.argv[1]
    with open(jsonfile, 'r') as f:
        j = json.load(f)
    rule = PolycubesRule(rule_json=j["cube_types"])
    designer = YieldAllosteryDesigner(rule, j["cubes"])
    for allosteric_rule in designer.add_allostery_badly():
        print(allosteric_rule)
