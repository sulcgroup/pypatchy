from __future__ import annotations

import copy
import functools
import itertools
import json
from abc import ABC, abstractmethod
from typing import Generator, Iterable, Callable, Container, Collection

import networkx as nx
import scipy as sp
from Bio.SVDSuperimposer import SVDSuperimposer
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import igraph as ig

from .polycubeutil.polycubesRule import *
from .util import enumerateRotations, rotidx, get_input_dir

Binding = tuple[int, int, int, int]


def flip(b: Binding) -> Binding:
    return (b[2], b[3], b[0], b[1])


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

    # graph
    graph: nx.MultiDiGraph

    # TODO: deprecated bindings_list and just use graph instead since it contains the same data
    # bindings_list: set[tuple[int, int, int, int]]

    def __init__(self, **kwargs):
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        # self.bindings_list = set()
        if "bindings" in kwargs:
            for n1, d1, n2, d2 in kwargs["bindings"]:
                if n1 not in self.graph:
                    self.graph.add_node(n1)
                if n2 not in self.graph:
                    self.graph.add_node(n2)
                self.graph.add_edge(n1, n2, dirIdx=d1)
                self.graph.add_edge(n2, n1, dirIdx=d2)
                # self.bindings_list.add((n1, d1, n2, d2))
            assert len(self.graph.edges) == 2 * len(kwargs["bindings"])
        # if we have passed a graph object
        if "graph" in kwargs:
            # TODO: test!
            assert isinstance(kwargs["graph"], nx.MultiDiGraph)
            self.graph = kwargs["graph"]
            handled_set = set()
            # iter edges
            for u, v, k in self.graph.edges:
                if (v, u, k) not in handled_set:
                    u_diridx = self.graph.get_edge_data(u, v, k)["dirIdx"]
                    assert (v, u) in self.graph.edges
                    v_diridx = self.graph.get_edge_data(v, u, k)["dirIdx"]
                    self.bindings_list.add((
                        u, u_diridx,
                        v, v_diridx
                    ))
                    handled_set.add((u, v, k))  # only add each edge once

        # if bindings aren't provided, the object is initialized empty
        # (useful for calls from subconstructor, see below)

    def to_json(self) -> list[list[int]]:
        return [
            [u, du, v, dv] for u, du, v, dv in self.bindings_list
        ]

    @property
    @functools.cache
    def bindings_list(self) -> set[tuple[int, int, int, int]]:
        """
        Written with the aid of chatGPT

        Computes the set of unique bindings (edges with directions) from the current graph.
        Each binding is a tuple: (u, du, v, dv) where:
        - u, v are node IDs
        - du is the direction index of edge u -> v
        - dv is the direction index of edge v -> u (should be reciprocal)
        """
        bindings = set()
        seen = set()

        for u, v, k, data in self.graph.edges(keys=True, data=True):
            if (v, u, k) in seen:
                continue  # already added this binding from the other direction
            du = data["dirIdx"]
            dv = self.graph.get_edge_data(v, u)[k]["dirIdx"]
            bindings.add((u, du, v, dv))
            seen.add((u, v, k))

        assert len(self.graph.edges) == 2 * len(bindings)
        return bindings

    def vertices(self) -> list[int]:
        return [int(n) for n in self.graph.nodes]

    def filter_edges(self, f: Callable[[Structure, int, int, int, int], bool]) -> list[tuple[int, int, int]]:
        """
        returns a list containing all edges in the structure that satisfy the function passed
        """
        edges: list[tuple[int, int, int]] = []
        for u, v, k in self.graph.edges:
            dirIdx = self.graph.edges[u, v, k]["dirIdx"]
            if f(self, u, v, k, dirIdx):
                edges.append((u, v, k))
        return edges

    def rotate(self, rotation_map: dict[int, int]) -> StructuralHomomorphism:
        """
        rotates the structure by applying the rotation map to the structure
        returns a structural homomorphism representing the rotation operation
        """
        structure_rotated = Structure(bindings=[(u, rotation_map[du], v, rotation_map[dv])
                                                for u, du, v, dv in self.bindings_list])
        return StructuralHomomorphism(self,
                                      structure_rotated,
                                      rotation_mapping_idx=rotidx(rotation_map),
                                      # use identity lmap
                                      location_mapping={i: i for i in self.vertices()}
                                      )

    def num_dimensions(self) -> int:
        return 3  # todo: dynamic?

    def dimension_sizes(self) -> tuple[int, ...]:
        """
        returns a list of ints that is the max length along each dimension
        most useful for periodic structures
        """
        assert self.is_connected(), "Cannot compute dimension sizes of a multifarious structure"
        sizes = [0 for _ in range(self.num_dimensions())]
        for d in range(self.num_dimensions()):
            slices = self.substructure(self.vertices())

            def dimension_filter(s, u, v, k, dirIdx) -> bool:
                return dirIdx != 2 * d

            edges_to_remove: list[tuple[int, int, int]] = slices.filter_edges(dimension_filter)
            for e in edges_to_remove:
                slices.graph.remove_edge(*e)
            sizes[d] = max([len(c) for c in nx.simple_cycles(slices.graph)])

        return tuple(sizes)

    def get_empties(self) -> list[tuple[int, int]]:
        """
        calcEmptyFromTop has... issues
        todo: make this versitile for non-cubic-lattice structures
        """
        empties = []
        # return calcEmptyFromTop(self.bindings_list)
        for vert_id in self.vertices():
            for di, _ in enumerate(RULE_ORDER):
                if not self.bi_edge_exists(vert_id, di):
                    empties.append((vert_id, di))
        return empties

    def draw(self, pos=None):
        """
        Draws the MultiDiGraph with matplotlib.
        Parameters:
            graph (nx.MultiDiGraph): The graph to draw.
            pos (dict, optional): Dictionary defining the layout of nodes.
                                  If None, the spring layout will be used.
        """

        # If no position is provided, use spring layout
        if pos is None:
            pos = nx.spring_layout(self.graph)

        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, node_size=700, node_color='lightblue')

        # Draw edges, distinguishing direction with arrows
        nx.draw_networkx_edges(self.graph, pos, edgelist=self.graph.edges, edge_color='black', arrows=True,
                               connectionstyle='arc3,rad=0.2')

        # Add edge labels for direction (dirIdx)
        edge_labels = {(u, v, k): d['dirIdx'] for u, v, k, d in self.graph.edges(keys=True, data=True)}
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)

        # Add node labels
        nx.draw_networkx_labels(self.graph, pos, font_size=12, font_color='black')

        # Display the graph
        plt.show()

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

    def substructure(self, nodes: Iterable[int]) -> Structure:
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

    def homomorphism(self, structure: Structure) -> Union[bool, StructuralHomomorphism]:
        hms = self.homomorphisms(structure)
        return next(hms, False)

    def homomorphisms(self, structure: Structure) -> Generator[StructuralHomomorphism]:
        """
        Constructs the graph injective homomorphism of self.graph -> structure.graph
        Uses igraph's isomorphisms_vf2 for efficient mapping.
        """

        # Only proceed if structure is at least as large as self
        if len(structure) >= len(self):
            # Find all subgraph isomorphisms from g1 to g2
            for rmapidx, rotation_mapping in enumerateRotations().items():
                def edgecompare(g1: ig.Graph, g2: ig.Graph, edge_1: int, edge_2: int) -> bool:
                    e1 = g1.es[edge_1]
                    e2 = g2.es[edge_2]
                    return rotation_mapping[e1["dirIdx"]] == e2["dirIdx"]
                # Optionally, could use edge/vertex attributes for more constraints
                for mapping in self._candidate_homomorphisms(structure, edgecompare):
                    # mapping: dict from g1 vertex idx to g2 vertex idx
                    node_list_src = list(self.graph.nodes)
                    node_list_target = list(structure.graph.nodes)
                    # igraph uses 0-based indices, so map back to node labels
                    node_list_mapping = {node_list_src[i]: node_list_target[mapping[i]] for i in range(len(mapping))}
                    reverse_mapping = {v: k for k, v in node_list_mapping.items()}
                    yield StructuralHomomorphism(self, structure, rmapidx, node_list_mapping, reverse_mapping)

        else:
            raise TypeError("Cannot find homomorphism of a larger structure onto a smaller structure")

    def _candidate_homomorphisms(self, structure: Structure, edgecompare: Callable) -> Generator[list[int]]:
        """
        wrapper for get_subisomorphisms_vf2 that can be overwritten in subclasses to add type constraints
        """
        yield from structure.to_igraph().get_subisomorphisms_vf2(self.to_igraph(),
                                                                 edge_compat_fn=edgecompare)

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
        tests if an edge exists between the two positions
        i could probably use "edge_exists" or smthre
        """
        return (v1, v2) in self.graph.to_undirected().edges

    def is_connected(self) -> bool:
        return nx.is_weakly_connected(self.graph)

    def is_multifarious(self) -> bool:
        return not self.is_connected()

    def iter_components(self) -> Generator[Structure, None, None]:
        for cc in nx.connected_components(self.graph.to_undirected()):
            yield Structure(graph=self.graph.subgraph(cc))

    def num_vertices(self) -> int:
        return len(self.graph.nodes)

    def splice(self, other: Structure,
               splice_points: Collection[tuple[Binding, Binding]]) -> Structure:
        """
        Written with the aid of chatGPT

        Splices another structure into this one at specified binding points.

        Parameters:
            other: another Structure instance to be spliced in
            splice_points: list of tuples of bindings
                Each tuple is (binding_from_self, binding_from_other)
                These bindings are removed from each respective structure and replaced with new
                cross-structure connections that link the two structures.

        Returns:
            A new Structure with the two original structures connected at the splice points.
        """
        # check that bindings are properly categorized initially
        for b in list(zip(*splice_points))[0]:
            assert b in self.bindings_list or flip(b) in self.bindings_list
        for b in list(zip(*splice_points))[1]:
            assert b in other.bindings_list or flip(b) in other.bindings_list

        # Step 1: Remap `other` to avoid conflicting vertex IDs with `self`
        hom = other.remap_vert_ids(self.vertices())

        # Step 2: Combine all bindings from both structures
        combined_bindings = list(self.bindings_list) + list(hom.target.bindings_list)

        # Step 3: For each splice point, remove old bindings and add cross-structure connections
        for b1, b2 in splice_points:
            # Remove both directions of b1 (from self)
            if b1 in combined_bindings:
                combined_bindings.remove(b1)
            # check transpose of bindings
            elif flip(b1) in combined_bindings:
                combined_bindings.remove(flip(b1))
            else:
                raise Exception()

            # Remove both directions of b2 (from other — needs remapping first)
            remapped_b2 = tuple(hom.lmap[i] if j % 2 == 0 else i for j, i in enumerate(b2))  # remap u and v only
            if remapped_b2 in combined_bindings:
                combined_bindings.remove(remapped_b2)
                # transpose
            elif (flip(b2)) in combined_bindings:
                combined_bindings.remove(flip(b2))
            else:
                raise Exception()

            # Add new cross-structure edges: self's u -> remapped other's u
            assert rdir(b1[1]) == remapped_b2[3]
            assert rdir(b2[1]) == remapped_b2[3]
            combined_bindings.append((b1[0], b1[1], remapped_b2[2], remapped_b2[3]))
            combined_bindings.append((remapped_b2[0], remapped_b2[1], b1[2], b1[3]))

        # Step 4: return a new Structure
        s = Structure(bindings=combined_bindings)
        assert len(s) == len(self) + len(other)
        assert len(s.graph.edges) == len(self.graph.edges) + len(other.graph.edges)
        return s

    def slice(self, axis: int) -> Structure:
        """
        returns: a structure (not connected) which only includes edges orthogonal to axis
        """
        return Structure(bindings=[(u, du, v, dv) for u, du, v, dv in self.bindings_list
                                   if du not in (axis, rdir(axis)) and dv not in (axis, rdir(axis))])

    def remap_vert_ids(self, verts_to_avoid: Iterable[int]) -> StructuralHomomorphism:
        """
        written by chatGPT
        returns a mapping which maps self onto a copy with verts renamed to avoid uid conflicts
        """
        if not isinstance(verts_to_avoid, frozenset):
            verts_to_avoid = frozenset(verts_to_avoid)

        old_ids = sorted(self.vertices())
        new_ids = []
        next_id = 0

        while len(new_ids) < len(old_ids):
            if next_id not in verts_to_avoid:
                new_ids.append(next_id)
            next_id += 1

        remap = dict(zip(old_ids, new_ids))

        new_bindings = [
            (remap[u], du, remap[v], dv) for u, du, v, dv in self.bindings_list
        ]

        remapped_structure = Structure(bindings=new_bindings)

        # no rotation so use identity rotation
        return StructuralHomomorphism(self, remapped_structure, 0, remap)

    def matrix(self) -> np.ndarray:
        """
        Returns the structure as a N x 3 matrix where cols are x,y,z coordinates
        Strictly speaking this should be a method for `FiniteLatticeStructure` but
        it's kept here for backwards compatibility purposes
        """
        cubes_coords = self._get_cubes_coords()
        a = np.array([position for _, position in sorted(cubes_coords.items(), key=lambda x: x[0])])
        # a = np.array([*cubes_coords.values()])
        assert a.shape == (len(self), 3)
        return a

    def neighbor(self, vert: int, direction: int) -> Union[None, int]:
        if not self.edge_exists(vert, direction):
            return None
        # todo: this line is 100% wrong
        return [e for e in self.graph.edges[vert] if e["diridx"] == direction][0][1]

    def _get_cubes_coords(self) -> dict[int, np.ndarray]:
        """
        component method of Structure::matrix() (see above)
        extractracted here for use in subclass overrides of matrix()
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
        return cubes_coords

    def __contains__(self, item: Union[int, Binding]) -> bool:
        if isinstance(item, int):
            # item is assumed to be a node index
            return item in list(self.graph.nodes)
        elif isinstance(item, tuple):
            return item in self.bindings_list or flip(item) in self.bindings_list
        else:
            raise Exception()

    def __len__(self) -> int:
        return len(self.graph.nodes)

    def __str__(self) -> str:
        return f"Structure with {len(self.vertices())} particles and {len(self.bindings_list)} connections"

    def replace_edges(self, bindings_from: list[Binding], bindings_to: list[Binding]):
        """
        in place, replace edges of `bindings_from` with `bindings_to
        """
        # fill in
        pass

    def is_crystal(self) -> bool:
        """
        Written with the aid of chatGPT

        A structure is considered crystalline if there are no cycles formed
        using only edges that go along one axis (e.g. ±x, ±y, or ±z).
        This ensures translational periodicity in all directions.
        """
        for axis in range(3):  # 3 axes: x, y, z
            dirs = {2 * axis, 2 * axis + 1}

            # Create a substructure with only edges in this axis's directions
            sub = Structure(bindings=[
                (u, du, v, dv)
                for u, du, v, dv in self.bindings_list
                if du in dirs and dv in dirs and rdir(du) == dv
            ])

            # Check for cycles in this substructure
            try:
                _ = next(nx.simple_cycles(sub.graph))
                return True
            except StopIteration:
                continue

        return False  # No directional loops → structure is crystalline

    @functools.cache
    def to_igraph(self):
        """
        Converts the structure graph (as stored internally as a networkx DiGraph) to an igraph Graph
        Stores 'uid' attribute as the original NetworkX node label.
        """

        g_nx = self.graph

        # Map original node labels to 0-based indices for igraph
        label_to_index = {label: idx for idx, label in enumerate(g_nx.nodes)}
        index_to_label = list(g_nx.nodes)

        # Convert edges (drop multiedge keys)
        edge_list = [(label_to_index[u], label_to_index[v]) for u, v, _ in g_nx.edges]

        # Build igraph
        ig_g = ig.Graph(directed=True)
        ig_g.add_vertices(len(index_to_label))
        ig_g.vs['uid'] = index_to_label
        ig_g.add_edges(edge_list)

        # assign edge directions
        for e in ig_g.es:
            e["dirIdx"] = g_nx.get_edge_data(index_to_label[e.source], index_to_label[e.target])[0]["dirIdx"]

        # --- Perform checks on igraph ---
        for v in ig_g.vs:
            out_deg = ig_g.degree(v.index, mode="OUT")
            if out_deg < 1 or out_deg > 6:
                raise ValueError(f"Vertex {v.index} (uid={v['uid']}) has {out_deg} out-edges, must be 1–6.")

        # Check reciprocal edges
        for e in ig_g.es:
            src, tgt = e.source, e.target
            if not ig_g.are_connected(tgt, src):
                uid_src = ig_g.vs[src]['uid']
                uid_tgt = ig_g.vs[tgt]['uid']
                raise ValueError(f"Missing reciprocal edge: ({uid_src}, {uid_tgt}) -> ({uid_tgt}, {uid_src})")

        return ig_g

    def is_bindings_euclidian(self) -> bool:
        return all([
            abs(du - dv) == 1 # direction index within 1 of each other = opposites,
            # accordiing to rule_order
            for _, du, _, dv in self.bindings_list
        ])


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

class StructuralMapping:
    """
    Written with the aid of chatGPT

    A base class for structural mappings that only map vertex locations
    between a source and a target structure.
    """

    # structure that this homomorphism maps from
    source: Structure
    # structure that this homomorphism maps onto
    target: Structure
    # map that maps location indexes from source onto target
    lmap: dict[int, int]
    # map that maps location indexes from targets onto source
    rlmap: dict[int, int]

    def __init__(self,
                 source_structure: Structure,
                 target_structure: Structure,
                 location_mapping: dict[int, int],
                 reverse_location_mapping: Union[dict[int, int], None] = None):
        self.source = source_structure
        self.target = target_structure
        self.lmap = location_mapping
        self.rlmap = (reverse_location_mapping if reverse_location_mapping is not None
                      else {v: k for k, v in location_mapping.items()})

    def map_location(self, i: int) -> int:
        """Map a location index from source to target"""
        assert i in self.lmap, f"Source location {i} not in location map"
        return self.lmap[i]

    def rmap_location(self, i: int) -> int:
        """Map a location index from target to source"""
        assert i in self.rlmap, f"Target location {i} not in reverse location map"
        return self.rlmap[i]

    def __len__(self) -> int:
        return len(self.lmap)

    def as_pairs(self) -> list[tuple[int, int]]:
        """Return the mapping as a list of (source, target) index pairs"""
        return list(self.lmap.items())

    def __getitem__(self, item: int) -> int:
        return self.lmap[item]

    def __contains__(self, item: int) -> bool:
        return item in self.lmap


class StructuralHomomorphism(StructuralMapping):
    # index in enumerateRotations of the rotation mapping in this sturctural homomorphism
    _rmapidx: int

    def __init__(self, source_structure: Structure, target_structure: Structure
                 , rotation_mapping_idx: int, location_mapping: dict[int:int], reverse_location_mapping=None):
        super().__init__(source_structure, target_structure, location_mapping, reverse_location_mapping)
        self._rmapidx = rotation_mapping_idx

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

    def apply_to(self, bindings: Iterable[Binding]) -> list[Binding]:
        return [
            (self.map_location(u), self.map_direction(du), self.map_location(v), self.map_direction(dv))
            for u, du, v, dv in bindings
        ]

class Tiling:
    source_structure: Structure
    tiled_structure: Structure
    periodic_bindings: list[Binding]

    # mapping of vertex IDs on source_structure to vertex mapping on target_structure
    vertex_mapping: dict[int: set[int]]

    def __init__(self,
                 source_structure: Structure,
                 periodic_bindings: list[Binding],
                 dimensional_tiling_counts: tuple[int, ...],):
        self.source_structure = source_structure

        assert len(dimensional_tiling_counts) <= 3, "Dimensions greater than 4 not supported yet"

        assert source_structure.is_bindings_euclidian(), "Cannot tile non-euclidian structures!"
        assert source_structure.is_crystal(), "Cannot tile non-crystal structures!"
        self.tiled_structure = copy.deepcopy(source_structure)
        # we will need to alter this within __tile_single_dimension
        self.periodic_bindings = [tuple(b) for b in periodic_bindings]

        # loop dimensions
        for dimension, dimensional_count in enumerate(dimensional_tiling_counts):
            if dimensional_count < 2: continue
            dimensional_bindings = [
                # can assume/skip dv
                (u, du, v, dv) for u, du, v, dv in self.periodic_bindings
                if du in [2 * dimension, 2 * dimension + 1]
            ]
            assert len(dimensional_bindings) > 0
            new_bindings = self.__tile_single_dimension(copy.deepcopy(self.tiled_structure),
                                                        dimensional_count,
                                                        dimensional_bindings)
        # do we need an assert here?
        assert all([self.tiled_structure.graph.degree[v]/2 <= 6 for v in self.tiled_structure.graph.nodes]),\
            f"Some nodes have non-allowed degrees have degrees {[(v, self.tiled_structure.graph.degree[v]/2) for v in self.tiled_structure.graph.nodes if self.tiled_structure.graph.degree[v] > 6]}"
        assert all(b in self.tiled_structure.bindings_list or flip(b) in self.tiled_structure.bindings_list
                   for b in self.periodic_bindings)
        assert self.tiled_structure.is_crystal()
        assert self.tiled_structure.is_bindings_euclidian()

    def __tile_single_dimension(self,
                                monomer: Structure,
                                number_of_tilings: int,
                                dimensional_bindings: list[Binding]) -> list[Binding]:
        """
        internel method to use in constructor
        """
        n_bindings_start = len(self.periodic_bindings)
        assert number_of_tilings >= 2
        if number_of_tilings == 2:
            mapping = monomer.remap_vert_ids(monomer.vertices())
            # rotation mapping is identity
            crystal_bindings_others = [
                (mapping.map_location(u), du, mapping.map_location(v), dv)
                for (u, du, v, dv) in dimensional_bindings
            ]
            binding_remap = [
                (b1, b2) if b1[1] == b2[1] else (b1, flip(b2))
                for b1, b2 in  zip(dimensional_bindings, crystal_bindings_others)
            ]
            tiled_structure = monomer.splice(mapping.target, binding_remap)
            # extras = [
            #     (u, du, v, dv) for
            #     ((u, du, _, _), (_, _, v, dv)) in zip(dimensional_bindings, crystal_bindings_others)
            # ]
            extras = [
                (u, du, mapping.map_location(v), dv)
                for (u, du, v, dv) in dimensional_bindings
            ]
            assert all(du == rdir(dv) for (_, du, _, dv) in extras)
            to_pop = dimensional_bindings

        else:
            # recurse
            # construct tiling of n-1 structures, if we do this enough we will get to n = 2
            # "sub tile" not "subtle"
            subtile_periodic_bindings = self.__tile_single_dimension(monomer,
                                                                     number_of_tilings - 1,
                                                                     dimensional_bindings)
            assert all([b in self.tiled_structure for b in subtile_periodic_bindings])
            to_pop = subtile_periodic_bindings
            # make sure we haven't somehow conjured new bindings into existance
            assert len(subtile_periodic_bindings) == len(dimensional_bindings)

            assert set(self.source_structure.vertices()) <= set(monomer.vertices())
            mapping = monomer.remap_vert_ids(self.tiled_structure.vertices())
            # mapping = self.tiled_structure.remap_vert_ids(monomer.vertices())
            # construct map of vertex IDs from
            crystal_bindings_other = mapping.apply_to(dimensional_bindings)

            # crystal_bindings = [
            #     (u, rdir(du), v, rdir(dv)) for u, du, v, dv
            #     in mapping.apply_to(dimensional_bindings)
            # ]

            # align faces
            binding_remap = [
                (b1, b2) if b1[1] == b2[1] else (b1, flip(b2))
                for b1, b2 in zip(crystal_bindings_other, subtile_periodic_bindings)
            ]

            # binding_remap = [
            #     ((u1, du1, v2, dv2), (u2, du2, v1, dv1))
            #     for ((u1, du1, v1, dv1), (u2, du2, v2, dv2)) in binding_remap
            # ]

            # make sure bindings are lined up
            # binding_remap = list(zip(crystal_bindings_other, subtile_periodic_bindings))
            tiled_structure = mapping.target.splice(self.tiled_structure, binding_remap)
            assert tiled_structure.is_bindings_euclidian()
            # extras = [(v, dv, u, du) for u, du, v, dv in crystal_bindings_other]
            # flip directionality
            # extras = [(u, rdir(du), v, rdir(dv))
            #           for u, du, v, dv in crystal_bindings_other]
            extras = [
                (mapping.lmap[u1], du1, v2, dv2)
                for ((u1, du1, v1, dv1), (u2, du2, v2, dv2)) in zip(dimensional_bindings, subtile_periodic_bindings)
            ]
            assert all(du == rdir(dv) for (_, du, _, dv) in extras)

        # update bindings other than this one
        # pb = copy.deepcopy(self.periodic_bindings) # don't work on main list
        self.periodic_bindings.extend(mapping.apply_to(
            filter(
                lambda b: b not in dimensional_bindings and b not in dimensional_bindings and b[0] in mapping and b[2] in mapping,
                self.periodic_bindings
            )
        ))

        # for (u, du, v, dv) in pb:
        #     if (u, du, v, dv) not in dimensional_bindings and (v, dv, u, du) not in dimensional_bindings:
        #         if u in mapping.lmap and v in mapping.lmap:
        #             assert rdir(du) == dv
        #             self.periodic_bindings.append((mapping.lmap[u], du, mapping.lmap[v], dv))
        self.periodic_bindings.extend(extras)
        for b in to_pop:
            self.periodic_bindings.remove(b)

        # check binding counts
        assert len(tiled_structure.bindings_list) == number_of_tilings * len(monomer.bindings_list)
        assert number_of_tilings*(n_bindings_start - len(dimensional_bindings)) + len(dimensional_bindings) == len(self.periodic_bindings)

        self.tiled_structure = tiled_structure
        assert all(extra in tiled_structure for extra in extras)
        return extras

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
        """
        :return: the type of the particle with the provided id
        """
        pass

    @abstractmethod
    def get_particle_types(self) -> dict[int, int]:
        """
        :return: the map of location identifiers to particle types
        """
        pass

    @abstractmethod
    def num_particle_types(self) -> int:
        """
        :return: the number of particle types
        """
        pass

    def _candidate_homomorphisms(self, structure: TypedStructure, edgecompare: Callable) -> Generator[list[int]]:
        """
        wrapper for get_subisomorphisms_vf2 that can be overwritten in subclasses to add type constraints
        """
        def vertcompare(g1, g2, v1, v2):
            return structure.particle_type(g1.vs[v1]['uid']) == self.particle_type(g2.vs[v2]['uid'])
        yield from structure.to_igraph().get_subisomorphisms_vf2(self.to_igraph(),
                                                                 node_compat_fn=vertcompare,
                                                                 edge_compat_fn=edgecompare)

    def is_symmetric(self, axis: int) -> bool:
        """
        determines if structure has rotational symmetry on specified axis
        todo: impl in c++
        :param axis: a number in RULE_ORDER representing an axis, positive and negative being functionally identical
        """
        assert -1 < axis < len(RULE_ORDER)
        # step 1: isolate edges that aren't along the specified axis
        layer_loops: list[tuple[int, int, int, int]] = [(u, du, v, dv) for u, du, v, dv in self.bindings_list
                                                        if
                                                        du not in (axis, rdir(axis)) and dv not in (axis, rdir(axis))]
        layers = TypedStructure(bindings=layer_loops, types=self.get_particle_types())
        # handle layers sperately
        for layer in layers.iter_components():
            for rot in range(4):  # todo: smarter rotation code
                rotation = getRotationMap(axis, rot)
                layer = layer.rotate(rotation)
            # help
        return True

    def draw(self, pos=None):
        """
        Draws the MultiDiGraph with matplotlib.
        :param: graph (nx.MultiDiGraph): The graph to draw.
        :param: pos (dict, optional): Dictionary defining the layout of nodes. If None, the spring layout will be used.
        """

        # Get a list of available colors from matplotlib
        available_colors = list(mcolors.TABLEAU_COLORS.values())

        if len(set(self.get_particle_types().values())) > len(available_colors):
            raise ValueError(f"Too many particle types for available colors! Max is {len(available_colors)}.")

        # Generate a color map assigning a color to each particle type
        color_map = {i: available_colors[i] for i in set(self.get_particle_types().values())}

        # If no position is provided, use spring layout
        if pos is None:
            pos = nx.spring_layout(self.graph)

        # Draw nodes
        nx.draw_networkx_nodes(self.graph,
                               pos,
                               node_size=700,
                               node_color=[color_map[self.particle_type(v)] for v in self.graph.nodes])

        # Draw edges, distinguishing direction with arrows
        nx.draw_networkx_edges(self.graph, pos, edgelist=self.graph.edges, edge_color='black', arrows=True,
                               connectionstyle='arc3,rad=0.2')

        # Add edge labels for direction (dirIdx)
        edge_labels = {(u, v, k): d['dirIdx'] for u, v, k, d in self.graph.edges(keys=True, data=True)}
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)

        # Add node labels
        nx.draw_networkx_labels(self.graph, pos, font_size=12, font_color='black')

        # Display the graph
        plt.show()

    def matrix_etc(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        returns the result of Structure::matrix + other stuff

        :returns: tuple where first element is an array of positions, second is an array of uids, third is an array of particle type ids
        """

        cubes_coords: dict[int, np.ndarray] = self._get_cubes_coords()
        data: list[tuple[np.ndarray, int, int]] = sorted([
            (position, uid, self.particle_type(uid))
            for (uid, position) in cubes_coords.items()
        ], key=lambda item: item[0][0])
        a, uids, ptypes = zip(*data)
        a = np.stack(a)
        # a = np.array([*cubes_coords.values()])
        assert a.shape == (self.num_vertices(), 3)
        return a, np.array(uids), np.array(ptypes)

    @functools.cache
    def to_igraph(self):
        """
        Convert a NetworkX DiGraph to an igraph Graph (directed) with 'type' as a vertex attribute.
        """
        ig_g = super().to_igraph()
        types = [self.particle_type(ig_g.vs['uid'][v]) for v in ig_g.vs.indices]
        ig_g.vs['type'] = types
        return ig_g

class GenericTypedStructure(TypedStructure):
    """
    minimal implementation of TypedStructure
    """

    # keys are vertex (location) IDs, variables are particle type IDs
    particle_types: dict[int, int]

    def __init__(self, **kwargs):
        super(GenericTypedStructure, self).__init__(**kwargs)
        if "types" in kwargs:
            self.particle_types = kwargs["types"]
        else:
            self.particle_types = dict()
        if "fill_type" in kwargs:
            for v in self.vertices():
                if v not in self.particle_types:
                    self.particle_types[v] = kwargs["fill_type"]
        assert all([
            n in self.particle_types for n in self.vertices()
        ]), "Missing types for some particles!"

    def particle_type(self, particle_id: int) -> int:
        return self.particle_types[particle_id]

    def set_particle_types(self, new_types: dict[int, int]):
        assert all(key in self.particle_types for key in new_types)
        self.particle_types.update(new_types)

    def remap_vert_ids(self, verts_to_avoid: Iterable[int]) -> StructuralHomomorphism:
        """
        written by chatGPT
        Returns a StructuralHomomorphism mapping this structure to a new one with
        vertex IDs remapped to avoid collisions. Also updates particle types.
        """
        if not isinstance(verts_to_avoid, frozenset):
            verts_to_avoid = frozenset(verts_to_avoid)

        old_ids = sorted(self.vertices())
        new_ids = []
        next_id = 0

        while len(new_ids) < len(old_ids):
            if next_id not in verts_to_avoid:
                new_ids.append(next_id)
            next_id += 1

        remap = dict(zip(old_ids, new_ids))

        # Remap bindings
        new_bindings = [
            (remap[u], du, remap[v], dv) for u, du, v, dv in self.bindings_list
        ]

        # Remap particle types
        new_types = {remap[k]: v for k, v in self.particle_types.items()}

        assert len(new_bindings) == len(self.bindings_list)

        remapped_structure = GenericTypedStructure(bindings=new_bindings, types=new_types)

        return StructuralHomomorphism(self, remapped_structure, 0, remap)

    def get_particle_types(self) -> dict[int, int]:
        return self.particle_types

    def num_particle_types(self) -> int:
        return len(set(self.particle_types.values()))

    def substructures(self) -> Generator[GenericTypedStructure]:
        for sub in super(GenericTypedStructure, self).substructures():
            yield GenericTypedStructure(graph=sub.graph, types={
                k: v for k, v in self.particle_types.items() if k in sub.vertices()
            })

    def substructure(self, nodes: tuple[int]) -> GenericTypedStructure:
        sub = super(GenericTypedStructure, self).substructure(nodes)
        sub = GenericTypedStructure(graph=sub.graph, types={
            k: v for k, v in self.particle_types.items() if k in sub.vertices()
        })
        return sub

    def iter_components(self) -> Generator[Structure, None, None]:
        for cc in nx.connected_components(self.graph.to_undirected()):
            yield GenericTypedStructure(graph=self.graph.subgraph(cc),
                                        types={v: t for v, t in self.particle_types.items() if v in cc})

    def __str__(self) -> str:
        return f"{self.bindings_list}\nParticle Typing: {self.get_particle_types()}"

    def splice(self, other: GenericTypedStructure,
               splice_points: list[tuple[Binding, Binding]]) -> GenericTypedStructure:
        """
        Written with the aid of chatGPT

        Splices another GenericTypedStructure into this one by calling the base class splice method
        and then merging the particle type annotations appropriately.

        Parameters:
            other: the GenericTypedStructure to splice into self
            splice_points: a list of tuples (binding_from_self, binding_from_other)
                specifying the connections at which to splice the structures.

        Returns:
            A new GenericTypedStructure instance that contains the spliced structure with combined
            bindings and merged particle types.
        """
        # Step 1: Remap the other structure so that its vertex IDs do not clash with those of self.
        hom = other.remap_vert_ids(self.vertices())
        # Create the remapped version of other (hom.target) and also capture the mapping hom.lmap.

        # Step 2: Call the superclass splice method to get a new Structure with combined bindings.
        spliced_structure = super().splice(other, splice_points)
        # At this point, spliced_structure is an instance of Structure.

        # Step 3: Merge particle types.
        # For self, the types remain unchanged.
        # For other, we need to update the keys (vertex IDs) using the remapping hom.lmap.
        remapped_types = {hom.lmap[k]: v for k, v in other.particle_types.items()}
        combined_types = {**self.particle_types, **remapped_types}

        # Step 4: Construct and return a new GenericTypedStructure using the spliced bindings and
        # the merged particle types.
        return GenericTypedStructure(bindings=list(spliced_structure.bindings_list), types=combined_types)

    def rotate(self, rotation_map: dict[int, int]) -> StructuralHomomorphism:
        pass

    def slice(self, axis: int) -> Structure:
        """
        returns: a structure (not connected) which only includes edges orthogonal to axis
        """
        return GenericTypedStructure(bindings=[(u, du, v, dv) for u, du, v, dv in self.bindings_list
                                               if du not in (axis, rdir(axis)) and dv not in (axis, rdir(axis))],
                                     types=self.get_particle_types())


def read_topology(top: Union[str, dict, Path], pt_idxs: set[int] = set()) -> GenericTypedStructure:
    # if string, assume this is a file path
    if isinstance(top, str):
        if top.startswith("~"):
            top = Path(top).expanduser()
        elif not top.startswith("/"):
            top = get_input_dir() / "topologies" / Path(top)
        else:
            top = Path(top)
    with top.open("r") as f:
        top = json.load(f)
    assert isinstance(top, dict), "Invalid topology"
    bindings = copy.deepcopy(top["bindings"])
    if "extraConnections" in top:
        bindings += top["extraConnections"]

    type_id_map = dict()
    for pt1, _, pt2, _ in bindings:
        for pt in [pt1, pt2]:
            if pt in pt_idxs:
                type_id_map[pt] = next(itertools.filterfalse(pt_idxs.__contains__, itertools.count(1)))
    bindings = [
        [
            type_id_map[pt1] if pt1 in type_id_map else pt1,
            dir1,
            type_id_map[pt2] if pt2 in type_id_map else pt2,
            dir2
        ]
        for pt1, dir1, pt2, dir2 in bindings
    ]

    nps = {
        int(k): top["nanoparticles"][k] + 1 for k in top["nanoparticles"]
    } if "nanoparticles" in top else {}
    assert len(set(nps.values())) <= 1, "god please no"
    return GenericTypedStructure(bindings=bindings, types=nps, fill_type=0)


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

