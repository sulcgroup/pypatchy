from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generator, Iterable

import networkx as nx
from Bio.SVDSuperimposer import SVDSuperimposer

from pypatchy.polycubeutil.polycubesRule import *
from pypatchy.util import enumerateRotations


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
    graph: nx.MultiDiGraph
    # TODO: deprecated bindings_list and just use graph instead since it contains the same data
    bindings_list: set[tuple[int, int, int, int]]

    def __init__(self, **kwargs):
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
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
                if (v, u) not in handled_set:
                    u_diridx = self.graph.get_edge_data(u, v)["dirIdx"]
                    assert (v, u) in self.graph.edges
                    v_diridx = self.graph.get_edge_data(v, u)["dirIdx"]
                    self.bindings_list.add((
                        u, u_diridx,
                        v, v_diridx
                    ))
                    handled_set.add((u, v))  # only add each edge once

        # if bindings aren't provided, the object is initialized empty
        # (useful for calls from subconstructor, see below)

    def vertices(self) -> list[int]:
        return [int(n) for n in self.graph.nodes]

    def get_empties(self) -> list[tuple[int, int]]:
        """
        calcEmptyFromTop has... issues
        """
        empties = []
        # return calcEmptyFromTop(self.bindings_list)
        for vert_id in self.vertices():
            for di, _ in enumerate(RULE_ORDER):
                if not self.bi_edge_exists(vert_id, di):
                    empties.append((vert_id, di))
        return empties

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

    # def cycles_by_size(self, filter_repeat_nodes=True) -> dict[int: list[int]]:
    #     """
    #     Code partially written by ChatGPT
    #     Construct a list of unique cycles in the graph,
    #     where each second-level list is composed of cycles of the same size,
    #     and each cycle visits each node at most once.
    #
    #     Returns:
    #     dict: dictionary where keys are cycle sizes and values are lists of unique cycles
    #     """
    #
    #     # Use simple_cycles to get all cycles in the graph
    #     all_cycles = list(nx.simple_cycles(self.graph))
    #
    #     # Filter out cycles with fewer than 3 nodes
    #     # (since the graph is undirected, any edge is also a "cycle" so filter those out)
    #     cycles = [cycle for cycle in all_cycles if len(cycle) > 2]
    #     if filter_repeat_nodes:
    #         # filter and cycles that visit any node more than once
    #         cycles = [cycle for cycle in cycles if len(cycle) == len(set(cycle))]
    #
    #     # Remove duplicates from the cycles (ignoring the order of nodes)
    #     unique_cycles = list(set(frozenset(cycle) for cycle in cycles))
    #
    #     # Initialize a default dictionary to store cycles by size
    #     unique_cycles_by_size = defaultdict(list)
    #
    #     # Iterate over the cycles
    #     for cycle in unique_cycles:
    #         # The size of the cycle is its length
    #         size = len(cycle)
    #
    #         # Append the cycle to the list of cycles of the same size
    #         unique_cycles_by_size[size].append(self.graph.subgraph(list(cycle)))
    #
    #     return unique_cycles_by_size

    def homomorphism(self, structure: Structure) -> Union[bool, StructuralHomomorphism]:
        hms = self.homomorphisms(structure)
        return next(hms, False)

    def homomorphisms(self, structure: Structure) -> Generator[bool, StructuralHomomorphism]:
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
                    yield StructuralHomomorphism(self,
                                                 structure,
                                                 rmapidx,
                                                 node_list_mapping,
                                                 reverse_mapping)

    def edge_exists(self, v: int, delta: int) -> bool:
        """
        returns true if the digraph contains an edge out from position v with direction delta
        """
        return len([d for _, _, d in self.graph.out_edges(v, "dirIdx") if d == delta]) > 0

    def bi_edge_exists(self, v: int, delta: int) -> bool:
        return self.edge_exists(v, delta) or any([
            d == rdir(delta) for _, _, d in self.graph.in_edges(v, "dirIdx")
        ])

    def positions_connected(self, v1: int, v2: int) -> bool:
        """

        """
        return (v1, v2) in self.graph.to_undirected().edges

    def is_connected(self) -> bool:
        return nx.is_weakly_connected(self.graph)

    def is_multifarious(self) -> bool:
        return not self.is_connected()

    def num_vertices(self) -> int:
        return len(self.graph.nodes)

    def matrix(self) -> np.ndarray:
        """
        Returns the structure as a N x 3 matrix where cols are x,y,z coordinates
        Strictly speaking this should be a method for `FiniteLatticeStructure` but
        it's kept here for backwards compatibility purposes
        """
        # TODO: compute this on init? idk
        assert self.is_connected(), "Please don't try to get a matrix for a non connected structure. " \
                                    "It's not invalid I just hate it."
        # start with zeroes
        processed_coords = {0}
        cubes_coords = {0: np.array([0, 0, 0])}
        # loop until we've processed all coords
        loopcount = 0
        while len(processed_coords) < len(self):
            for v1, d1, v2, d2 in self.bindings_list:
                # if this binding connects a cube which has been processed
                if (v1 in processed_coords and v2 not in processed_coords) or (
                        v2 in processed_coords and v1 not in processed_coords):
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
            loopcount += 1
            assert loopcount < 1e7
        a = np.array([position for _, position in sorted(cubes_coords.items(), key=lambda x: x[0])])
        # a = np.array([*cubes_coords.values()])
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


class FiniteLatticeStructure(Structure):
    __cube_positions: np.ndarray
    __cube_index_map: dict[bytes, int]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__cube_positions = self.matrix()
        self.__cube_index_map = dict()
        for i in range(self.num_vertices()):
            self.__cube_index_map[self.cube(i).tobytes()] = i

    def cube(self, idx: int) -> np.ndarray:
        return self.__cube_positions[idx, :]

    def cube_idx(self, coords: np.ndarray) -> int:
        return self.__cube_index_map[coords.tobytes()]

    def positions_connected(self, c1: Union[np.ndarray, int], c2: Union[np.ndarray, int]) -> bool:
        """
        override of positions_connected that can convert position vectors (np arrays) to idxs
        """
        if isinstance(c1, np.ndarray):
            c1 = self.cube_idx(c1)
        if isinstance(c2, np.ndarray):
            c2 = self.cube_idx(c2)
        return Structure.positions_connected(self, c1, c2)


class StructuralHomomorphism:
    # structure that this homomorphism maps from
    source: Structure
    # structure that this homomorphism maps onto
    target: Structure

    # index in enumerateRotations of the rotation mapping in this sturctural homomorphism
    _rmapidx: int

    # map that maps location indexes from source onto target
    lmap: dict[int, int]
    # map that maps location indexes from targets onto source
    rlmap: dict[int, int]

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
        """Maps a location from the origin onto the destination"""
        assert i in self.lmap
        return self.lmap[i]

    def rmap_location(self, i: int) -> int:
        """Maps a location from the destination onto the origin"""

        assert i in self.rlmap
        return self.rlmap[i]

    def map_direction(self, d: Union[int, np.ndarray]) -> int:
        if isinstance(d, np.ndarray):
            d = diridx(d)
        assert d > -1
        assert d < len(RULE_ORDER)
        return enumerateRotations()[self._rmapidx][d]

    def rmap_direction(self, d: Union[int, np.ndarray]) -> int:
        if isinstance(d, np.ndarray):
            d = diridx(d)
        assert d > -1
        assert d < len(RULE_ORDER)
        for k, v in enumerateRotations()[self._rmapidx].items():
            if v == d:
                return k
        raise Exception("Good god what did you DO???")

    def __len__(self):
        return len(self.lmap)

    def as_transform(self) -> tuple[np.ndarray, np.ndarray]:
        # align matrices
        mat = self.source.matrix()
        targ = self.target.matrix()
        src_coords = mat.copy()
        targ_coords = targ.copy()
        for i, (k, v) in enumerate(self.lmap.items()):
            src_coords[i, :] = mat[sorted(self.source.graph.nodes).index(k), :]
            targ_coords[i, :] = targ[sorted(self.target.graph.nodes).index(v), :]


        # a few ways to proceed from here, most of which i hate
        # going with the "reinvent wheel" method
        svd = SVDSuperimposer()
        svd.set(src_coords, targ_coords)
        svd.run()
        assert svd.get_rms() < 1e-8, "No good transformation found!!!!"
        r, t = svd.get_rotran()
        return r.round(), t.round()

    def contains_edge(self, v: int, delta: int) -> bool:
        """
        Given a node and edge in the target, returns True if the source has a corresponding out-edge
        from that node. false otherwise
        """
        return self.source.edge_exists(self.rmap_location(v),
                                       self.rmap_direction(delta))

    def target_contains_edge(self, v: int, delta: int) -> bool:
        return self.target.edge_exists(self.map_location(v),
                                       self.map_direction(delta))


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


class TypedStructure(Structure, ABC):
    """
    Class representing a structure formed out of particles with defined types
    """

    @abstractmethod
    def particle_type(self, particle_id: int) -> int:
        pass

    @abstractmethod
    def get_particle_types(self) -> dict[int, int]:
        pass


def calcEmptyFromTop(top: Iterable[tuple[int, int, int, int]]) -> list[tuple[int, int]]:
    """

    """
    ids = set(i for i, _, _, _ in top).union(set(j for _, _, j, _ in top))
    patches = set(((i, dPi) for i, dPi, _, _ in top)).union(((j, dPj) for _, _, j, dPj in top))

    empty = []
    for i in ids:
        for dPi in range(6):
            if not (i, dPi) in patches:
                empty.append((i, dPi))
    return empty
