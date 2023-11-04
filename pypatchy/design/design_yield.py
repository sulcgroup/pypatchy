from __future__ import annotations

import sys
import json
from collections import OrderedDict
from typing import Iterable

from pypatchy.structure import *


class Joint:
    """
    This class is designed to be immutable! Please respect!
    """
    # particle id for self. whether this is a unique id or a type id depends on the context
    _particle: int
    # particle id for previous neighbor. whether this is a unique id or a type id depends on the context
    _prev_neighbor: int
    # particle id for next neighbor. whether this is a unique id or a type id depends on the context
    _next_neighbor: int
    # tuple where the first value is the patch ID on self that connects to _prev_neighbor
    # and the second value is the patch ID on _prev_neigbor that connects to self
    _prev_top_edge: tuple[int, int]
    # tuple where the first value is the patch ID on self that connects to _next_neighbor
    # and the second value is the patch ID on _next_neigbor that connects to self
    _next_top_edge: tuple[int, int]

    def __init__(self,
                 particleid: int,
                 prevneighbor: int,
                 nextneighbor: int,
                 prevtopedge: tuple[int, int],
                 nexttopedge: tuple[int, int]):
        self._particle = particleid
        # assert prevneighbor != nextneighbor, "U turns are not allowed!"
        self._prev_neighbor = prevneighbor
        self._next_neighbor = nextneighbor
        # assert prevtopedge[1] != nexttopedge[0], "Invalid patch layout! Cannot have two patches with same ID"
        self._prev_top_edge = prevtopedge
        self._next_top_edge = nexttopedge

    def unique_tuple(self) -> tuple[int, int, int]:
        """
        WARNING: DON'T USE THIS FUNCTION FOR GENERIC JOINTS unless you want it to do what it's actually going to do
        in which case go ahead
        Returns:
            a simplified version of this behavior that - if ids are unique - will nonetheless act as a unique identifier within this structure

        """
        return (
            self._prev_neighbor, self._particle, self._next_neighbor
        )

    def particle(self) -> int:
        return self._particle

    def next_neighbor(self) -> int:
        return self._next_neighbor

    def prev_neighbor(self) -> int:
        return self._prev_neighbor

    def prev_edge(self) -> tuple[int, int]:
        return self._prev_top_edge

    def next_edge(self) -> tuple[int, int]:
        return self._next_top_edge

    def particle_prev_patch(self) -> int:
        return self.prev_edge()[0]

    def particle_next_patch(self) -> int:
        return self.next_edge()[0]

    def inner(self) -> tuple[int, int, int]:
        return self.particle_prev_patch(), self.particle(), self.particle_next_patch()

    def inner_equal(self, joint: Joint) -> bool:
        return self.inner() == joint.inner()

    def __hash__(self):
        """
        This can act as a unique identifier for a behavior within a structure or alternateively
        can be used to compare two behaviors

        Returns:
            an int that acts as a unique identifier for this behavior
        """
        return hash((
            self.unique_tuple(),
            self._prev_top_edge,
            self._next_top_edge
        ))

    def __invert__(self) -> Joint:
        """

        Returns:
            a behavior that is this behavior but in reverse

        """
        return Joint(self._particle,
                     self._next_neighbor, self._prev_neighbor,
                     self._next_top_edge, self._prev_top_edge)

    def __repr__(self):
        return f"({self.prev_neighbor()}){self.prev_edge()[1]}->{self.prev_edge()[0]}({self.particle()}){self.next_edge()[0]}->{self.next_edge()[1]}({self.next_neighbor()})"

class Pathway:
    """
    immutable.
    represents a path through a structure as a list of Joints
    mostly a wrapper for a tuple of behaviors
    this seemed like a good idea at some point
    """
    __inner: tuple[Joint]
    __cycle_id: int

    def __init__(self, *args: Joint):
        self.__inner = args
        # unique integer identifier that should be the same for all cycles containing the same set of joints
        # todo: prove that no cycle exists which contains the same set of joints but in a different order?
        self.__cycle_id = hash(frozenset(self.__inner))

    def __len__(self):
        return len(self.__inner)

    def __iter__(self):
        return iter(self.__inner)

    def __reversed__(self):
        reverse_path = (~b for b in reversed(self.__inner))
        return Pathway(*reverse_path)

    def __hash__(self) -> int:
        return self.__cycle_id

    def __getitem__(self, item: Union[int, slice]) -> Union[Joint, Pathway]:
        if isinstance(item, int):
            if item < 0:
                return self.__inner[len(self) + item]
            else:
                return self.__inner[item]
        elif isinstance(item, slice):
            # slicing linear pathways is easy!
            if not self.is_cycle():
                return Pathway(*self.__inner[item])
            # slicing cyclic pathways is... not easy
            else:
                # make slice explicit
                if item.start is None:
                    item = slice(0, item.stop, item.step)
                elif item.start < 0:
                    item = slice(len(self) + item.start, item.stop, item.step)
                if item.stop is None:
                    item = slice(item.start, len(self), item.step)
                elif item.stop < 0:
                    item = slice(item.start, len(self) + item.stop, item.step)
                if item.step is None:
                    item = slice(item.start, item.stop, 1)
                assert item.step
                # base cases: complete slice 0:0:1 and 0:0:-1
                if item.start == item.stop:
                    sliced = self
                elif item.start < item.stop:
                    sliced = Pathway(*self.__inner[item])
                # if start is after stop and step is positive
                elif item.start >= item.stop:
                    # lshift the path so that the start point lines up with the path origin then return a slice from the shifted path
                    sliced = (self << item.start)[:(item.stop - item.start):abs(item.step)]
                else:
                    assert (item.start < item.stop) == (item.step > 0), "mismatch!!!!"
                    sliced = Pathway(*self.__inner[item])
                if item.step < 0:
                    sliced = reversed(sliced)
                return sliced

        else:
            raise Exception(f"Invalid indexer {type(item)}")

    def __repr__(self):
        if len(self) == 0:
            return "Empty pathway"
        elif len(self) == 1:
            return f"[{repr(self[0])}]"
        else:
            return "[" + repr(self[0]) + "".join(repr(x)[repr(x).find(f"({x.particle()})") + len(f"({x.particle()}")+1:] for x in self) + "]"
        # return repr(self.__inner)

    def index(self, item):
        if isinstance(item, Joint):
            return self.__inner.index(item)
        elif isinstance(item, int):
            return [j.particle() for j in self].index(item)
        else:
            raise ValueError(f"Invalid indexer {item}")

    def in_main(self, item: Union[int, tuple]):
        if isinstance(item, tuple):
            return any([j1.particle() in item and j2.particle() in item for j1, j2 in zip(self, self[1:])])
        elif isinstance(item, int):
            return any([j.particle() == item for j in self])

    def __contains__(self, item):
        """
        WILL RETURN TRUE FOR PARTICLES / EDGES IN ENDMEMBERS!!!
        Args:
            item:

        Returns:

        """
        if isinstance(item, Joint):
            try:
                self.index(item)
                return True
            except ValueError:
                return False
        elif isinstance(item, int):
            return any([j.particle() == item for j in self]) or self[0].prev_neighbor() == item or self[-1].next_neighbor() == item
        elif isinstance(item, tuple):
            # edge checker
            assert len(item) == 2
            return any([sum([j.prev_neighbor() in item, j.next_neighbor() in item, j.particle() in item]) == 2 for j in self])
        else:
            raise TypeError(f"Invalid arg to Joint::__contains__! Arg type: {type(item)}")

    def is_cycle(self) -> bool:
        return self[0].prev_neighbor() == self[-1].particle() and self[-1].next_neighbor() == self[0].particle()

    def cycle_id(self) -> int:
        return self.__cycle_id

    def cycle_equiv(self, other: Pathway) -> bool:
        """

        given that both self and other are cycles,

        Args:
            other: another cycle

        Returns:
            true if there is a way to shift this cycle so that it's equal to other. false otherwise.
        """
        assert self.is_cycle() and other.is_cycle()
        return self.cycle_id() == other.cycle_id()
        # return len(self) == len(other) and any([self >> i == other for i in range(len(self))])



    def __add__(self, other: Joint):
        """
        Returns this pathway with a joint added to the end
        Args:
            other:

        Returns:

        """
        assert other is not None
        assert other.particle() == self[-1].next_neighbor()
        assert other.prev_neighbor() == self[-1].particle()
        return Pathway(*self.__inner, other)

    def __lshift__(self, n: int) -> Pathway:
        """
        Shifts the cycle n iterations to the left
        Args:
            n: number of items to shift

        Returns:
            self shifted n items to the left

        """
        assert self.is_cycle()
        if n == 0:
            return self
        shifted = Pathway(*self[-n:], *self[:-n])
        assert len(shifted) == len(self)
        return shifted

    def __rshift__(self, n: int) -> Pathway:
        """
        Shifts the cycle n iterations to the right
        Args:
            n: number of spaces to shift item to the right

        Returns:
            self, shifted n items to the right
        """
        assert self.is_cycle()
        if n == 0:
            return self
        shifted = Pathway(*self[n:], *self[:n])
        assert len(shifted) == len(self)
        return shifted

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
    _score: float # computed from something or other

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
        self._joint_class_map = np.full(fill_value=-1, shape=(self.path_len(), self.num_generic_pathways()))
        self._joint_classes = []

        for j, pathway in enumerate(self.type_keys()):
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

    def generic_paths(self) -> list:
        return list(self._instances.keys())

    def type_path(self) -> list[int]:
        return self._type_path

    def __contains__(self, item: Union[Pathway]) -> bool:
        if isinstance(item, Pathway):
            # i'm like 50/50 on this one
            return any([item.cycle_equiv(p) for p in self._instances])
        else:
            raise TypeError()

    def __repr__(self) -> str:
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

    def find_forks(self) -> Iterable[int, None, None]:
        """
        Locates positions in the design path where the same particle instance joins something to different particle
        instances of the same type
        Returns:

        """
        # iter joint idxs
        for idx in range(self.path_len()):
            p0 = list(self.generic_paths())[0][idx].particle()
            p1 = list(self.flat_instances())[0][idx].next_neighbor()
            if all([pathway[idx].particle() == p0 for pathway in self.generic_paths()]):
                if not all([pathway[idx].next_neighbor() == p1 for pathway in self.flat_instances()]):
                    yield idx

    def find_origin_point(self):
        pass

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

    def divisions(self) -> Iterable[tuple[int, int], None, None]:
        """

        Returns:
            A generator which yields pairs of coordinates within the design path which divide the path
            TODO TODO TODO
        """
        pass


class YieldAllosteryDesigner(PolycubeStructure):
    ct_allo_vars: list[set]
    vjoints: dict[int, set[Joint]]

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

    def align_cycles_generic(self, c1: Pathway, c2:Pathway) ->bool:
        assert c1.is_cycle() and c2.is_cycle()


    def subbehavior(self, b: Joint):
        """
        todo: figure out what i'm trying to do here and then do it
        Args:
            b:

        Returns:

        """
        type_idx = self.particle_type(b.particle())
        p_in = self.rule.patch(b.particle_prev_patch())
        p_out = self.rule.patch(b.particle_next_patch())

        return None

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
        design_paths = self.list_design_paths()
        types_processed = set()
        for dp in sorted(design_paths, key=lambda x: x.fav_factor(), reverse=True):
            print(dp)
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

    def pathway(self, *args: int):
        for i, j, k in triplets(args):
            e1 = (i, j)
            e2 = (j, k)

    def josh_cycles(self,
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
                    yield from self.josh_cycles(neighbor, cycle, current_node)
                else:
                    if neighbor == previous_node:
                        continue
                    new_joint = self.get_joint(previous_node, current_node, neighbor)
                    # if we're on node 2 of our cycle
                    if cycle is None:
                        # create cycle
                        new_cycle = Pathway(new_joint)
                        # yield from functon
                        yield from self.josh_cycles(neighbor, new_cycle, current_node)
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
                            yield from self.josh_cycles(neighbor, cycle + new_joint, current_node)

    def all_unique_evans_cycles(self):
        # create set of unique cycle ids to aid in hashing
        unique_cycles = set()
        for i in range(self.num_vertices()):
            for cycle in self.josh_cycles(i):
                if cycle.cycle_id() not in unique_cycles:
                    # emplace cycle
                    unique_cycles.add(cycle.cycle_id())
                    # yield

                    yield cycle

    def list_design_paths(self) -> list[DesignPathType]:
        """
        Returns: a list of design path types, sorted largest->smallest
        """
        # list is NOT indexed! instead it's sorted by design path size
        design_paths: list[DesignPathType] = []
        # loop evans cycles
        for cycle in self.all_unique_evans_cycles():
            # loop all design paths
            found_path = False
            for dpath in design_paths:
                # if the design path contains this cycle
                cycle_shifted = dpath.matches(self, cycle)
                if cycle_shifted:
                    found_path = True
                    dpath.add_instance(self, cycle_shifted)
            # if no existing design path, add it
            if not found_path:
                design_paths.append(DesignPathType(self, cycle))
                # not enough complexity to bother with a bisect sort
                design_paths = sorted(design_paths, key=lambda d: d.path_len(), reverse=True)

        return design_paths

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
