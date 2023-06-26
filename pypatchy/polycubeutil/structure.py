from __future__ import annotations

import itertools
from typing import TypeVar

from scipy.spatial.transform import Rotation

import networkx as nx
from collections import defaultdict

from ..util import getRotations, enumerateRotations

from polycubesRule import *

TCommonComponent = TypeVar("TCommonComponent", bound="CommonComponent")
TStructuralHomomorphism = TypeVar("TStructuralHomomorphism", bound="StructuralHomomorphism")
TStructure = TypeVar("TStructure", bound="Structure")


def get_nodes_overlap(homocycles: list[list[int]]) -> set[int]:
    """
    Get the list of nodes that are common to all cycles in the list.

    Parameters:
    homocycles (list): The list of homocycles.

    Returns:
    list: List of common nodes.
    """
    # Convert each cycle to a set of nodes
    sets_of_nodes = [set(cycle) for cycle in homocycles]

    # Find the intersection of all sets of nodes
    common_nodes = set.intersection(*sets_of_nodes)

    return common_nodes


class Structure:
    """
    A structure is defined in
    https://paper.dropbox.com/doc/Computational-Design-of-Allostery-ddFV7iLkKaua1tnDZOQFu#:uid=923787386309510691921072&h2=Empirical-Definition-of-a-Comp
    so go read that
    """

    def __init__(self, **kwargs):
        self.graph: nx.DiGraph = nx.DiGraph()

        if "bindings" in kwargs:
            for n1, d1, n2, d2 in kwargs["bindings"]:
                if n1 not in self.graph:
                    self.graph.add_node(n1)
                if n2 not in self.graph:
                    self.graph.add_node(n2)
                self.graph.add_edge(n1, n2, dirIdx=d1)
                self.graph.add_edge(n2, n1, dirIdx=d2)
        if "graph" in kwargs:
            self.graph = kwargs["graph"]

        # if bindings aren't provided, the object is initialized empty
        # (useful for calls from subconstructor, see below)

    def is_common_component(self, structures: Union[list, tuple]) -> Union[TCommonComponent, bool]:
        """
        Checks if self is a common component (see doc: "Computational Design of Allostery")
        of the provided structures
        Parameters:
             structures (list) a list of Structure objects

        Returns:
            A tuple where the first element is True if self is a common component of all provided
            structures, and the second element is a list of StructuralHomomorphism of (self->st)
             for st in the provided list of structures
        """

        if not nx.components.is_strongly_connected(self.graph):
            return False

        homomorphisms = [self.homomorphism(s) for s in structures]
        if all(homomorphisms):
            return CommonComponent(self, structures, homomorphisms)
        else:
            return False

    def cycles_by_size(self) -> dict[int: list[int]]:
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

        # Filter out cycles with fewer than 3 nodes and cycles that visit any node more than once
        cycles = [cycle for cycle in all_cycles if len(cycle) > 2 and len(cycle) == len(set(cycle))]

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

    def homomorphism(self, structure: TStructure) -> Union[bool, TStructuralHomomorphism]:
        """
        Constructs the graph injective homomorphism of self.graph -> structure.graph
        Parameters:
            structure (Structure) a structure object
        Return:
            a StructuralHomomorphism object if a homomorphism between the two graphs is possible
            else False
        """

        # TODO: TEST

        assert len(structure) >= len(self)

        # for now, brute-force this
        for rmapidx, rotation_mapping in enumerateRotations().items():
            # Ben would say rotation_mapping is a symmetry group of a cube or something
            for node_list_target in itertools.permutations(structure.graph.nodes, r=len(self.graph.nodes)):
                # node lists aren't nessecarily indexed from 0
                # sometimes they are but it's not a safe assumption
                node_list_src = list(self.graph.nodes)

                node_list_mapping = {n1: n2 for n1, n2 in zip(node_list_src, node_list_target)}
                reverse_mapping = {n2: n1 for n1, n2 in zip(node_list_src, node_list_target)}

                homomorphism_is_valid = True

                # list of source node IDs

                # loop pairs of nodes in this mapping
                for n1, n2 in zip(node_list_src, node_list_target):
                    # loop outgoing edges for this node
                    if len(self.graph.out_edges(n1)) != len(structure.graph.out_edges(n2)):
                        homomorphism_is_valid = False
                    else:
                        # for any e1 in self.graph.out_edges(n1), an e2 exists
                        # in structure.graph.out_edges(n2) such that
                        # rotation_mapping[e1["dirIdx"]] == e2["dirIdx"]
                        for u1, v1, d1 in self.graph.out_edges.data("dirIdx"):
                            if u1 != n1:
                                continue
                            # drop first element of tuple b/c it equals n1
                            found_homologous_edge = False
                            for u2, v2, d2 in structure.graph.out_edges.data("dirIdx"):
                                if u2 != n2:
                                    continue
                                # drop first element of tuple b/c it equals n2

                                # edges match if the direction indexes map onto each other
                                rot_map_valid = rotation_mapping[d1] == d2
                                # and the destination nodes also map onto each other
                                edges_same_node = node_list_mapping[v1] == v2
                                found_homologous_edge |= rot_map_valid and edges_same_node
                                if found_homologous_edge:
                                    break
                            # if no homologous edge to e1 exists, homomorphism is invalid
                            if not found_homologous_edge:
                                homomorphism_is_valid = False
                                break
                    # if we've invalidated the homomorphism
                    if not homomorphism_is_valid:
                        break

                if homomorphism_is_valid:
                    return StructuralHomomorphism(self,
                                                  structure,
                                                  rmapidx,
                                                  node_list_mapping,
                                                  reverse_mapping)
        return False

    def edge_exists(self, v: int, delta: int) -> bool:
        return len([d for a, b, d in self.graph.out_edges(v, "dirIdx") if d == delta])

    def __contains__(self, item: Union[int]) -> bool:
        if isinstance(item, int):
            # item is assumed to be a node index
            return item in list(self.graph.nodes)

    def __len__(self) -> int:
        return len(self.graph.nodes)


class CommonComponent(Structure):
    def __init__(self, component, full_structures: list, homomorphisms: list):
        self.graph = component.graph
        self.full_structures = full_structures
        self.homomorphisms = homomorphisms

    def is_superimposition_point(self, v: int) -> bool:
        """
        See defn. of a superimposition point in the doc "Computational Design of Allostery"
        """
        # get the obvious one out of the way
        if v not in list(self.graph.nodes):
            return False
        for delta, d in enumerate(RULE_ORDER):
            if not self.edge_exists(v, delta):
                continue
            if len([i for i in range(self.nstructures()) if self.homomorphism_contains(i, v, delta)]) > 1:
                return False

    def is_crucial_point(self, v: int) -> bool:
        """
        See defn. of a crucial point in the doc "Computational Design of Allostery"
        """
        # get the obvious one out of the way
        if v not in list(self.graph.nodes):
            return False
        # loop structures that this is a component of
        for s, f in zip(self.full_structures, self.homomorphisms):
            # loop pairs of nodes in s
            for u1, u2 in itertools.combinations(list(s.graph.nodes), 2):
                # skip instances where either u1 or u2 are in the common component
                if f.rmap_location(u1) in self or f.rmap_location(u2) in self:
                    continue
                # loop all possible simple paths from u1 to u2
                for p in nx.algorithms.all_simple_paths(s.graph, u1, u2):
                    # loop nodes in path
                    for n in p:
                        # if node n in the path is in the common component but
                        # isn't the node (v) that we're testing for cruciality,
                        # node v is not a crucial point
                        if f.rmap_location(n) in self and f.rmap_location(n) != v:
                            return False
        return True

    def is_pivot_point(self, v: int) -> bool:
        """
        See defn. of a pivot point in the doc "Computational Design of Allostery"
        """
        if not self.is_crucial_point(v) or not self.is_superimposition_point(v):
            return False
        for n in list(self.graph.nodes):
            if n != v and self.is_crucial_point(n):
                return False
        return True

    def is_macguffin(self) -> int:
        """
        Tests if this common component is a macguffin,
        Returns:
            -1 if this is not a macguffin, otherwise the pivot point
        """
        for v in list(self.graph.nodes):
            if self.is_pivot_point(v):
                return v
        else:
            return -1

    def nstructures(self) -> int:
        return len(self.full_structures)

    def homomorphism_contains(self, i: int, v: int, delta: int) -> bool:
        assert i < self.nstructures()
        return self.full_structures[i].edge_exists(self.homomorphisms[i].map_location(v),
                                                   self.homomorphisms[i].map_direction(delta))


def get_common_components(*args: Structure) -> list[CommonComponent]:
    """
    Parameters:
         args: Structure objects
    """
    s0 = args[0]
    common_components = []
    for n in range(2, len(s0) + 1):
        for nodes in itertools.combinations(s0.graph, n):
            subgraph = s0.graph.subgraph(nodes)
            if nx.algorithms.components.is_strongly_connected(subgraph):
                component = Structure(graph=subgraph)
                component = component.is_common_component(args)
                if component:
                    # TODO: check for redundency? somehow
                    common_components.append(component)
    return common_components


class StructuralHomomorphism:
    def __init__(self,
                 source_structure: Structure,
                 target_structure: Structure,
                 rotation_mapping_idx: int,
                 location_mapping: dict,
                 reverse_location_mapping: dict):
        self.source = source_structure
        self.target = target_structure
        self._rmapidx = rotation_mapping_idx
        self.lmap = location_mapping
        self.rlmap = reverse_location_mapping

    def map_location(self, i: int) -> int:
        assert i in self.lmap
        return self.lmap[i]

    def map_direction(self, d: Union[int, np.ndarray]) -> int:
        if isinstance(d, np.ndarray):
            d = diridx(d)
        assert d > -1
        assert d < len(RULE_ORDER)
        return enumerateRotations()[self._rmapidx][d]

    # def reverse_map_direction(self, d):
    #     if isinstance(d, int):
    #         d = RULE_ORDER[d]
    #     return enumerateRotations()[self._rmapidx] ....


TPolycubesStructureCube = TypeVar("TPolycubesStructureCube", bound="PolycubesStructureCube")


class PolycubeStructure(Structure):
    # mypy type specs
    rule: PolycubesRule
    cubeList: list
    cubeMap: dict

    def __init__(self, rule, structure):
        super(PolycubeStructure, self).__init__()

        # load rule
        self.rule = rule

        # needed for solving later. assume starting from no allostery

        # read structure
        self.cubeMap = {}
        # nodes in the graph are cube uids
        # edges attrs are FACE INDEXES IN RULE_ORDER, IN THE CUBE TYPE
        self.cubeList = []
        for cube_idx, cube in enumerate(structure):
            # extract cube position and rotation
            cube_position = from_xyz(cube["position"])
            # rotation quaternion wxyz
            cr = cube["rotation"]
            rot_quaternion = np.array((
                cr['x'],
                cr['y'],
                cr['z'],
                cr['w']
            ))
            cube_type = self.rule.particle(cube["type"])
            # emplace cube in map
            cubeObj = PolycubesStructureCube(cube_idx,
                                             cube_position,
                                             rot_quaternion,
                                             cube_type,
                                             cube["state"])
            self.cubeMap[cube_position.tobytes()] = cubeObj
            self.cubeList.append(cubeObj)
            self.graph.add_node(cube_idx, cube=cubeObj)

        # loop cube pairs (brute-forcing topology)
        for cube1, cube2 in itertools.combinations(self.cubeList, 2):
            # if cubes are adjacent
            if np.linalg.norm(cube1.get_position() - cube2.get_position()) == 1:
                d1 = cube2.get_position() - cube1.get_position()
                d2 = d1 * -1
                # if both cubes have patches on connecting faces and the patch colors match
                if cube1.has_patch(d1) and cube2.has_patch(d2):
                    p1 = cube1.get_patch(d1)
                    p2 = cube2.get_patch(d2)
                    if p1.color() == -p2.color():
                        align1 = cube1.get_rotation().apply(p1.alignDir()).round()
                        align2 = cube2.get_rotation().apply(p2.alignDir()).round()
                        if (align1 == align2).all():
                            if cube1.state(p1.state_var()) and cube2.state(p2.state_var()):
                                # add edge in graph
                                self.graph.add_edge(cube1.get_id(), cube2.get_id(), dirIdx=diridx(d1))
                                self.graph.add_edge(cube2.get_id(), cube1.get_id(), dirIdx=diridx(d2))

    def homologous_cycles(self, cycle_list: list[list[int]]) -> list:
        """
        Warning: ChatGPT produced this
        Group cycles in the list that contain the same cube type
        and same connection faces on types in the same pattern.

        Parameters:
        cycle_list (list): The list of cycles.

        Returns:
        list: List of homologous cycles.
        """
        # Initialize a dictionary to store cycles by their pattern
        cycles_by_pattern = defaultdict(list)

        # Iterate over the cycles
        for cycle in cycle_list:
            # Convert each cycle to a tuple of node types (cube types)
            pattern = tuple(sorted(self.cubeList[node].get_type().getID() for node in cycle))

            # Append the cycle to the list of cycles of the same pattern
            cycles_by_pattern[pattern].append(cycle)

        # Return the cycles grouped by their pattern
        return list(cycles_by_pattern.values())

    def next_node_in_cycle(self, n: int, cycle: list[int], processed_nodes: list[int]) -> tuple[int, int, int]:
        """
        Find the next node in the cycle that is not in the set of processed nodes.

        Parameters:
        head_node: The current node, represented as an int.
        cycle: The cycle, represented as a list of nodes (ints).
        processed_nodes: The set of nodes already processed, as a list of ints.

        Returns:
        tuple: The face connection to the previous node, the next node, and the face connection to the next node.
        """

        # The faces connecting the current node to the previous and next nodes
        head_neighbors = [n for n in self.graph.neighbors(n) if n in cycle]
        assert len(head_neighbors) == 2
        next_node = [n for n in head_neighbors if n not in processed_nodes][0]
        prev_node = [n for n in head_neighbors if n in processed_nodes][0]

        return (self.get_arrow_local_diridx(n, prev_node),
                next_node,
                self.get_arrow_local_diridx(n, next_node))

    def cubeAtPosition(self, v) -> TPolycubesStructureCube:
        return self.cubeMap[v.tobytes()]

    def get_arrow_diridx(self, node: int, adj: int) -> int:
        return self.graph.get_edge_data(node, adj)["dirIdx"]

    def get_arrow_local_diridx(self, n: int, adj: int) -> int:
        return self.cubeList[n].typedir(self.get_arrow_diridx(n, adj))


class PolycubesStructureCube:
    def __init__(self,
                 uid: int,
                 cube_position: np.ndarray,
                 cube_rotation: Union[np.ndarray, int],
                 cube_type: PolycubeRuleCubeType,
                 state: list[bool] = [True]):
        self._uid = uid
        self._pos = cube_position
        if isinstance(cube_rotation, np.ndarray) and len(cube_rotation) == 4:
            self._rot = Rotation.from_quat(cube_rotation)
        elif isinstance(cube_rotation, int):
            self._rot = Rotation.from_matrix(getRotations()[cube_rotation])
        else:
            assert False, "Rotation matrices or whatever not supported yet."
        self._type = cube_type
        self._state = state

    def get_id(self) -> int:
        return self._uid

    def get_position(self) -> np.ndarray:
        return self._pos

    def get_rotation(self) -> Rotation:
        return self._rot

    def get_type(self) -> PolycubeRuleCubeType:
        return self._type

    def typedir(self, direction: Union[int, np.ndarray]) -> np.ndarray:
        """
        Converts the global-space direction into a local-space direction
        """
        if isinstance(direction, int):  # if the arguement is provided as an index in RULE_ORDER
            direction = RULE_ORDER[direction]
        return self.get_rotation().inv().apply(direction).round()

    def has_patch(self, direction: Union[int, np.ndarray]) -> bool:
        return self.get_type().has_patch(self.typedir(direction))

    def get_patch(self, direction: Union[int, np.ndarray]) -> PolycubesPatch:
        return self.get_type().patch(self.typedir(direction))

    def state(self, i=None):
        if i is None:
            return self._state
        else:
            assert abs(i) < len(self._state)
            if i < 0:
                return not self._state[-i]
            else:
                return self._state[i]
