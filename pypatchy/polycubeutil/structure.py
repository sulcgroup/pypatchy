from __future__ import annotations

import itertools
from typing import TypeVar, Generator

from scipy.spatial.transform import Rotation

import networkx as nx
import numpy as np
from collections import defaultdict

from ..util import getRotations, enumerateRotations, from_xyz

from pypatchy.polycubeutil.polycubesRule import *

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
    graph: nx.DiGraph
    bindings_list: set[tuple[int, int, int, int]]

    def __init__(self, **kwargs):
        self.graph: nx.DiGraph = nx.DiGraph()
        self.bindings_list = set()
        if "bindings" in kwargs:
            for n1, d1, n2, d2 in kwargs["bindings"]:
                if n1 not in self.graph:
                    self.graph.add_node(n1)
                if n2 not in self.graph:
                    self.graph.add_node(n2)
                self.graph.add_edge(n1, n2, dirIdx=d1)
                self.graph.add_edge(n2, n1, dirIdx=d2)
                self.bindings_list.add((n1, d1, n2, d2))
        if "graph" in kwargs:
            # TODO: test!
            self.graph = kwargs["graph"]
            handled_set = set()
            # iter edges
            for u, v in self.graph.edges:
                if (v,u) not in handled_set:
                    u_diridx = self.graph.get_edge_data(u, v)["dirIdx"]
                    assert (v, u) in self.graph.edges
                    v_diridx = self.graph.get_edge_data(v, u)["dirIdx"]
                    self.bindings_list.add((
                            u, u_diridx,
                            v, v_diridx
                        ))
                    handled_set.add((u, v)) # only add each edge once

        # if bindings aren't provided, the object is initialized empty
        # (useful for calls from subconstructor, see below)

    def vertices(self) -> list[int]:
        return [int(n) for n in self.graph.nodes]

    def substructures(self) -> Generator[Structure]:
        # iterate possible sets of nodes in this graph
        # a subgraph of 1 node isn't a graph for our purposes; a subgraph of all nodes is self
        for n in range(2, len(self)):
            for subset in itertools.combinations(self.vertices(), n):
                # grab subgraph
                subgraph = self.graph.subgraph(subset)
                # check if subgraph si connected
                if nx.algorithms.components.is_strongly_connected(subgraph):
                    yield self.substructure(subset)

    def substructure(self, nodes: tuple[int]) -> Structure:
        """
        Returns:
             a Structre object that's a substructure of this
        """
        assert nx.algorithms.components.is_strongly_connected(self.graph.subgraph(nodes))
        lmap = dict()
        counter = 0
        # remap node indeces in self to new graph
        for n in self.vertices():
            if n in nodes:
                lmap[n] = counter
                counter += 1
        return Structure(bindings=[
            (lmap[u], du, lmap[v], dv) for u, du, v, dv in self.bindings_list if u in lmap and v in lmap
        ])

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

    def homomorphism(self, structure: Structure) -> Union[bool, StructuralHomomorphism]:
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
        return len([d for a, b, d in self.graph.out_edges(v, "dirIdx") if d == delta]) > 0

    def is_connected(self):
        return nx.is_weakly_connected(self.graph)

    def is_multifarious(self):
        return not self.is_connected()

    def matrix(self) -> np.ndarray:
        """
        Returns the structure as a N x 3 matrix where cols are x,y,z coordinates
        """
        # TODO: compute this on init? idk
        assert self.is_connected(), "Please don't try to get a matrix for a non connected structure. " \
                                    "It's not invalid I just hate it."
        # start with zeroes
        processed_coords = {0}
        cubes_coords = {0: np.array([0, 0, 0])}
        # loop until we've processed all coords
        while len(processed_coords) < len(self):
            for v1, d1, v2, d2 in self.bindings_list:
                # if this binding connects a cube which has been processed
                if (v1 in processed_coords and v2 not in processed_coords) or (v2 in processed_coords and v1 not in processed_coords):
                    if v1 in processed_coords:
                        origin = v1
                        destination = v2
                        direction = d1
                    else:
                        origin = v2
                        destination = v1
                        direction = d2
                    cubes_coords[destination] = cubes_coords[origin] + RULE_ORDER[direction]
                    processed_coords.add(destination)
        a = np.array([*cubes_coords.values()])
        assert a.shape == (len(self), 3)
        return a

    def __contains__(self, item: Union[int]) -> bool:
        if isinstance(item, int):
            # item is assumed to be a node index
            return item in list(self.graph.nodes)

    def __len__(self) -> int:
        return len(self.graph.nodes)

    def __str__(self) -> str:
        return f"Structure with {len(self.vertices())} particles and {len(self.bindings_list)} connections"


class StructuralHomomorphism:
    def __init__(self,
                 source_structure: Structure,
                 target_structure: Structure,
                 rotation_mapping_idx: int,
                 location_mapping: dict,
                 reverse_location_mapping: Union[dict, None] = None):
        self.source = source_structure
        self.target = target_structure
        self._rmapidx = rotation_mapping_idx
        self.lmap = location_mapping
        if reverse_location_mapping is not None:
            self.rlmap = reverse_location_mapping
        else:
            self.rlmap = {
                location_mapping[k]: k for k in location_mapping
            }

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


def identity_homomorphism(s: Structure) -> StructuralHomomorphism:
    """
    Returns the identity homomorphism of the provided structure, which maps the structure onto
    itself
    """
    return StructuralHomomorphism(s, s, 0, {i: i for i in s.graph.nodes})

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
            pattern = tuple(sorted(self.cubeList[node].get_type().type_id() for node in cycle))

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

    def cubeAtPosition(self, v) -> PolycubesStructureCube:
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
