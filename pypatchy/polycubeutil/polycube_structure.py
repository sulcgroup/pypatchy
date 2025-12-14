"""
Module containing polycube structure
"""

from __future__ import annotations

import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Union, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from oxDNA_analysis_tools.UTILS.data_structures import Configuration
from pypatchy.structure import GenericTypedStructure

from ipy_oxdna.utils.util import process_path

from ..cell_lists import CellLists

from .cube import PolycubesStructureCube
from ..vis_util import get_particle_color

from ..patchy_base_particle import PatchyBaseParticle, BaseParticleSet
from ..scene import Scene
from .polycubesRule import PolycubesRule, diridx, PolycubesPatch, RULE_ORDER

from ..structure import TypedStructure, Structure, Binding, StructuralHomomorphism
from ..util import from_xyz, get_input_dir


# todo: add cell lists? probably overkill
class PolycubeStructure(TypedStructure, Scene):
    """
    A structure formed from connected cubes on a 3D integer grid
    """
    # mypy type specs
    rule: PolycubesRule
    cubeMap: dict[bytes, PolycubesStructureCube]
    particle_uid_map: dict[int, PolycubesStructureCube]

    def __init__(self, **kwargs):
        """
        Malliable constructor for a polycube structure. Accepts a bunch of different keyword arguement options
        :param rule: polycube rule for the structure
        :param cube_types: polycube rule, as a json object
        :param structure:
        :param cubes:
        """
        # TODO: refactor this constructor to be normal constructor, have make_polycube() method external
        super(PolycubeStructure, self).__init__()
        CellLists.__init__(self)
        self.particle_uid_map = dict()

        # load rule
        if "rule" in kwargs:
            if isinstance(kwargs["rule"], PolycubesRule):
                self.rule = kwargs["rule"]
            elif isinstance(kwargs["rule"], dict):
                self.rule = PolycubesRule(rule_json=kwargs["rule"])
            elif isinstance(kwargs["rule"], str):
                self.rule = PolycubesRule(rule_str=kwargs["rule"])
            else:
                # default: empty rule
                self.rule = PolycubesRule()
        elif "cube_types" in kwargs:
            self.rule = PolycubesRule(rule_json=kwargs["cube_types"])
        else:
            self.rule = PolycubesRule()
        # read structure
        self.cubeMap = {}
        # nodes in the graph are cube uids
        # edges attrs are FACE INDEXES IN RULE_ORDER, IN THE CUBE TYPE
        self._particles = []
        if "structure" in kwargs:
            # structure passed as nested dict, as read from json
            for cube_idx, cube in enumerate(kwargs["structure"]):
                # extract cube position and rotation
                cube_position = from_xyz(cube["position"])
                # rotation quaternion wxyz
                cr = cube["rotation"]
                if isinstance(cr, dict):
                    rot_quaternion = np.array((
                        cr['x'],
                        cr['y'],
                        cr['z'],
                        cr['w']
                    ))
                cube_type = self.rule.particle(cube["type"])
                # type checking for cube state (ive been burned before)
                assert all([isinstance(sval, bool) for sval in cube["state"]]), f"Cube {cube_idx} has invalid state {str(cube['state'])}"
                if not cube["state"][0]:
                    print(f"Warning! Cube {cube_idx} has tautology state variable s[0] set to false, which seems "
                          f"bad but maybe you're doing something clever.")
                # emplace cube in map
                cubeObj = PolycubesStructureCube(cube_idx,
                                                 cube_position,
                                                 rot_quaternion,
                                                 cube_type,
                                                 cube["state"])
                self.cubeMap[cube_position.tobytes()] = cubeObj
                self._particles.append(cubeObj)
                self.graph.add_node(cube_idx, cube=cubeObj)
        elif "cubes" in kwargs:
            # structure passed as list of cube objects
            for cube_uid, cube in enumerate(kwargs["cubes"]):
                if isinstance(cube, dict):
                    if isinstance(cube["type"], int):
                        cube_type = self.rule.particle(cube["type"])
                    cube = PolycubesStructureCube(**cube, uid=cube_uid, cube_type=cube_type)
                assert cube.get_type() in [ct.type_id() for ct in self.rule.particles()], "Cube type not found in rule!"
                self._particles.append(cube)
                self.cubeMap[cube.position().tobytes()] = cube
                self.graph.add_node(cube.get_uid(), cube=cube)
        else:
            # empty structure
            pass

        # find topology (method courtesy of chatGPT)
        # construct a spacial hash of cubes
        spatial_hash = create_spatial_hash(self._particles)
        for cell, cubes in spatial_hash.items():
            # iter cubes
            for cube1 in cubes:
                # iter cells neighboring cube
                for neighbor_cell in get_neighboring_cells(cell):
                    for cube2 in spatial_hash.get(neighbor_cell, []):
                        if cube1 == cube2:
                            continue  # Skip self-comparison
                        if np.linalg.norm(cube1.position() - cube2.position()) == 1:
                            d1 = cube2.position() - cube1.position()
                            d2 = d1 * -1
                            if cube1.has_patch(d1) and cube2.has_patch(d2):
                                p1 = cube1.patch(d1)
                                p2 = cube2.patch(d2)
                                if p1.color() == -p2.color():
                                    if p1.has_torsion():
                                        assert p2.has_torsion(), "One patch has torsion but the other doesn't!"
                                        align1 = p1.alignDir().round()
                                        align2 = p2.alignDir().round()
                                        align_good = (align1 == align2).all()
                                    else: align_good=True
                                    if align_good:
                                        if not self.graph.has_edge(cube1.get_uid(), cube2.get_uid()):
                                            if cube1.state(p1.state_var()) and cube2.state(p2.state_var()):
                                                self.graph.add_edge(cube1.get_uid(), cube2.get_uid(), dirIdx=diridx(d1))
                                                self.graph.add_edge(cube2.get_uid(), cube1.get_uid(), dirIdx=diridx(d2))
                                                self.bindings_list.add((
                                                    cube1.get_uid(), diridx(d1),
                                                    cube2.get_uid(), diridx(d2)
                                                ))

                                                assert self.graph.out_degree[cube1.get_uid()] <= 6, "Added too many out edges to a cube!"
                                                assert self.graph.out_degree[cube2.get_uid()] <= 6, "Added too many out edges to a cube!"
        for cube in self.particles():
            self.particle_uid_map[cube.get_uid()] = cube

    def num_cubes_of_type(self, ctidx: int) -> int:
        return sum([1 for cube in self._particles if cube.get_cube_type().type_id() == ctidx])

    def get_cube(self, uid: int) -> PolycubesStructureCube:
        # todo: this can ansolutely be a map like cmon
        # assert -1 < uid < len(self.cubeList)
        for cube in self._particles:
            if cube.get_uid() == uid:
                return cube
        raise Exception(f"No cube with ID {uid}")

    def cube_type_counts(self) -> list[int]:
        return [self.num_cubes_of_type(i) for i, _ in enumerate(self.rule.particles())]

    def homologous_cycles(self, cycle_list: list[list[int]]) -> list:
        """
        Warning: ChatGPT produced this
        Group cycles in the list that contain the same cube type
        and same connection faces on types in the same pattern.

        :param cycle_list: The list of cycles.

        :return: List of homologous cycles.
        """
        # Initialize a dictionary to store cycles by their patters
        cycles_by_pattern = defaultdict(list)

        # Iterate over the cycles
        for cycle in cycle_list:
            # Convert each cycle to a tuple of node types (cube types)
            pattern = tuple(sorted(self._particles[node].get_type().type_id() for node in cycle))

            # Append the cycle to the list of cycles of the same pattern
            cycles_by_pattern[pattern].append(cycle)

        # Return the cycles grouped by their pattern
        return list(cycles_by_pattern.values())

    def next_node_in_cycle(self, n: int, cycle: list[int], processed_nodes: list[int]) -> tuple[int, int, int]:
        """
        Find the next node in the cycle that is not in the set of processed nodes.

        :param head_node: The current node, represented as an int.
        :param cycle: The cycle, represented as a list of nodes (ints).
        :param processed_nodes: The set of nodes already processed, as a list of ints.

        :param return: The face connection to the previous node, the next node, and the face connection to the next node.
        """

        # The faces connecting the current node to the previous and next nodes
        head_neighbors = [n for n in self.graph.neighbors(n) if n in cycle]
        assert len(head_neighbors) == 2
        next_node = [n for n in head_neighbors if n not in processed_nodes][0]
        prev_node = [n for n in head_neighbors if n in processed_nodes][0]

        return (self.get_arrow_local_diridx(n, prev_node),
                next_node,
                self.get_arrow_local_diridx(n, next_node))

    def cubeAtPosition(self, v: np.ndarray) -> PolycubesStructureCube:
        return self.cubeMap[v.astype(int).tobytes()]

    def has_cube_at(self, v: np.ndarray) -> bool:
        """
        tests if the polycube has a cube at a given position in space
        :param v: spacial position to check
        :return: true if the polycube has a cube at position v, false oterwise
        """
        return v.astype(int).tobytes() in self.cubeMap

    def get_arrow_diridx(self, node: int, adj: int) -> int:
        return self.graph.get_edge_data(node, adj)["dirIdx"]

    def get_arrow_local_diridx(self, n: int, adj: int) -> int:
        return self._particles[n].typedir(self.get_arrow_diridx(n, adj))

    def particle_type(self, particle_id: int) -> int:
        return self.particle_uid_map[particle_id].get_type()

    def graph_undirected(self) -> nx.Graph:
        return self.graph.to_undirected()

    # def homomorphism(self, structure: PolycubeStructure) -> Union[bool, StructuralHomomorphism]:
    #     """
    #     Constructs the graph injective homomorphism of self.graph -> structure.graph, taking cube types into account
    #     Parameters:
    #         structure (Structure) a structure object
    #     Return:
    #         a StructuralHomomorphism object if a homomorphism between the two graphs is possible
    #         else False
    #     """
    #
    #     # TODO: TEST
    #
    #     if len(structure) < len(self):
    #         return False
    #     if not all([a >= b for a, b in zip(structure.cube_type_counts(), self.cube_type_counts())]):
    #         return False
    #     # if not all([self.num_cubes_of_type(ctself.type_id()) == structure.num_cubes_of_type(ctother.type_id())
    #     #             for ctself, ctother in zip(self.rule.particles(), structure.rule.particles())]):
    #     #     return False
    #
    #     # for better efficiency, group nodes by cube type
    #
    #     nodes_sets = [list() for _ in structure.rule.particles()]
    #     for cube in structure.cubeList:
    #         nodes_sets[cube.get_cube_type().type_id()].append(cube.get_id())
    #
    #     node_permutations = [itertools.permutations(nodes, r=self.num_cubes_of_type(ctidx))
    #                          for ctidx, nodes in enumerate(nodes_sets)]
    #
    #     node_list_src = list(itertools.chain.from_iterable([
    #         [
    #             cube.get_id() for cube in self.cubeList if cube.get_cube_type().type_id() == ct.type_id()
    #         ]
    #         for ct in self.rule.particles()
    #     ]))
    #     # for now, brute-force this
    #     for rmapidx, rotation_mapping in enumerateRotations().items():
    #         # Ben would say rotation_mapping is a symmetry group of a cube or something
    #         for nodeperms in itertools.product(node_permutations):
    #             node_list_target = list(itertools.chain.from_iterable(nodeperms))
    #             # node lists aren't nessecarily indexed from 0
    #             # sometimes they are but it's not a safe assumption
    #
    #             node_list_mapping = {n1: n2 for n1, n2 in zip(node_list_src, node_list_target)}
    #             reverse_mapping = {n2: n1 for n1, n2 in zip(node_list_src, node_list_target)}
    #
    #             homomorphism_is_valid = True
    #
    #             # list of source node IDs
    #
    #             # loop pairs of nodes in this mapping
    #             for n1, n2 in zip(node_list_src, node_list_target):
    #                 # loop outgoing edges for this node
    #                 if len(self.graph.out_edges(n1)) != len(structure.graph.out_edges(n2)):
    #                     homomorphism_is_valid = False
    #                 else:
    #                     # for any e1 in self.graph.out_edges(n1), an e2 exists
    #                     # in structure.graph.out_edges(n2) such that
    #                     # rotation_mapping[e1["dirIdx"]] == e2["dirIdx"]
    #                     for u1, v1, d1 in self.graph.out_edges.data("dirIdx"):
    #                         if u1 != n1:
    #                             continue
    #                         # drop first element of tuple b/c it equals n1
    #                         found_homologous_edge = False
    #                         for u2, v2, d2 in structure.graph.out_edges.data("dirIdx"):
    #                             if u2 != n2:
    #                                 continue
    #                             # drop first element of tuple b/c it equals n2
    #
    #                             # edges match if the direction indexes map onto each other
    #                             rot_map_valid = rotation_mapping[d1] == d2
    #                             # and the destination nodes also map onto each other
    #                             edges_same_node = node_list_mapping[v1] == v2
    #                             found_homologous_edge |= rot_map_valid and edges_same_node
    #                             if found_homologous_edge:
    #                                 break
    #                         # if no homologous edge to e1 exists, homomorphism is invalid
    #                         if not found_homologous_edge:
    #                             homomorphism_is_valid = False
    #                             break
    #                 # if we've invalidated the homomorphism
    #                 if not homomorphism_is_valid:
    #                     break
    #
    #             if homomorphism_is_valid:
    #                 return StructuralHomomorphism(self,
    #                                               structure,
    #                                               rmapidx,
    #                                               node_list_mapping,
    #                                               reverse_mapping)
    #     return False

    def transform(self, rot: Optional[np.ndarray]=None, tran: Optional[np.ndarray]=None) -> PolycubeStructure:
        """
        perform a spacial transform on a polycube structure
        :return rot: a 3x3 rotation matrix
        :return tran: a 3x translation vector
        :return: a copy of the structure, with the spacial transform applied
        """
        assert rot is not None or tran is not None, "Must provide at least one of rot or tran"
        # assert transformation.shape == (4, 4), "Wrong shape for transformation"
        if rot is None:
            rot = np.identity(3)
        assert rot.shape == (3, 3)
        if tran is None:
            tran = np.zeros(shape=(3,))
        assert tran.shape[0] == 3
        # todo: check transform is legal
        transformed_structure = copy.deepcopy(self)
        transformed_structure.cubeMap = {}
        for cube in transformed_structure._particles:
            cube.set_position(cube.position() @ rot + tran)
            transformed_structure.cubeMap[cube.position().tobytes()] = cube

        return transformed_structure

    def save_polycube(self, fp: Union[Path, str]):
        """
        saves a polycube as a JSON file which can be read by load_polycube or into the web app
        :param fp: a file path at which to save the json file
        """
        fp = process_path(fp, get_input_dir() / "scenes")
        with fp.open("w") as f:
            json.dump(self.to_json(), f)

    def to_json(self) -> dict:
        """
        converts the polycube structure into a form that can be json-serialized
        :return: the polycube as a JSON object (aka a dict with only json-serializable values)
        """
        return {
            'cube_types': self.rule.toJSON(),
            'cubes': [
                cube.toJSON() for cube in self.particles()
            ]
        }

    def substructure(self, nodes: tuple[int]) -> Structure:
        """
        Returns:
             a Structre object that's a substructure of this
        """
        assert nx.algorithms.components.is_strongly_connected(self.graph.subgraph(nodes))
        return PolycubeStructure(cubes=[c for i, c in enumerate(self._particles) if c.get_uid() in nodes], rule=self.rule)

    def num_particle_types(self) -> int:
        return self.rule.num_particle_types()

    def particle_types(self) -> BaseParticleSet:
        return self.rule

    def get_conf(self) -> Configuration:
        pass

    def matrix(self) -> np.ndarray:
        """
        MUCH faster than the base-class Structure method!
        """
        return np.stack([cube.position() for cube in self._particles])

    def draw_structure_graph(self, ax: plt.Axes, layout: Union[None, dict] = None):
        if layout is None:
            layout = nx.spring_layout(self.graph)
        ptypemap = [get_particle_color(self.particle_type(j)) for j in self.graph.nodes]
        nx.draw(self.graph, ax=ax, with_labels=True, node_color=ptypemap, pos=layout)

    def set_particle_types(self, ptypes: BaseParticleSet):
        self.rule = ptypes

    def particles_bound(self, p1: PatchyBaseParticle, p2: PatchyBaseParticle) -> bool:
        return self.graph.has_edge(p1.get_uid(), p2.get_uid())

    def patches_bound(self,
                      particle1: PolycubesStructureCube,
                      p1: PolycubesPatch,
                      particle2: PolycubesStructureCube,
                      p2: PolycubesPatch) -> bool:
        if p1.color() + p2.color() != 0:
            return False
        if (particle2.position() != particle1.position() + p1.direction()).any():
            return False
        if (particle1.position() != particle2.position() + p2.direction()).any():
            return False
        if ((p1.alignDir() - p2.alignDir()) > 1e-6).any():
            return False
        return True

    def num_connections(self):
        """
        Returns: the number of cube-cube connections in this structure
        """
        return len(self.graph.edges) // 2  # divide by 2 b/c graph is bidirerctional

    def get_particle_types(self) -> dict[int, int]:
        return {
            p.get_uid(): p.get_type() for p in self.particles()
        }


    def centroid(self):
        return np.round(super().centroid())

    def rotate(self, rotation_map: dict[int, int]) -> StructuralHomomorphism:
        raise Exception("Not implemented yet")

    def splice(self, other: Structure, splice_points: list[tuple[Binding,
                                                                 Binding]]) -> Structure:
        raise Exception("Not valid for this subclass")

    def to_typed_structure(self, type_map: Optional[dict[int: int]]) -> TypedStructure:
        if type_map is None:
            type_map = {ct.type_id(): ct.type_id() for ct in self.rule}
        return GenericTypedStructure(
            bindings=self.bindings_list,
            types={p.get_uid(): type_map[p.get_type()] for p in self.particles()},
        )

    def add_particle(self, new_cube: PolycubesStructureCube):
        self._particles.append(new_cube)
        self.cubeMap[new_cube.position().astype(int).tobytes()] = new_cube
        self.graph.add_node(new_cube.get_uid(), cube=new_cube)
        self.update_graph_for(new_cube.get_uid())

    def update_graph_for(self, p_uid: int):
        """
        update the internal graph structure for a given particle
        """
        cube = self.get_cube(p_uid)
        n_cubes_start = self.num_particles()
        for dir_idx, direction in enumerate(RULE_ORDER):
            if self.has_cube_at(cube.position() + direction):
                other_cube = self.cubeAtPosition(cube.position() + direction)
                has_edge = self.particles_bound(self.get_cube(p_uid), other_cube)
                try:
                    is_bound = self.patches_bound(self.get_cube(p_uid),
                                                  self.get_cube(p_uid).patch(direction),
                                                  other_cube,
                                                  other_cube.patch(-direction)
                    )
                except IndexError as e:
                    is_bound = False
                if has_edge != is_bound:
                    if not is_bound:
                        self.graph.remove_edge(p_uid, dir_idx)
                    else:
                        self.graph.add_edge(p_uid, other_cube.get_uid(), dir_idx)
        assert self.num_particles() == n_cubes_start, \
            "update_graph_for modified number of particles! (should not do this)"

    def __iadd__(self, other: PolycubeStructure):
        assert isinstance(other, PolycubeStructure), "Can only add PolycubeStructure to PolycubeStructure"
        assert self.rule == other.rule, "Can only add PolycubeStructures with the same rule"
        # offset uids
        uid_offset = len(self._particles)
        for cube in other._particles:
            new_cube = copy.deepcopy(cube)
            new_cube.set_uid(new_cube.get_uid() + uid_offset)
            self.add_particle(new_cube)
            self.update_graph_for(new_cube.get_uid())

        return self

def load_polycube(file_path: Union[Path, str]) -> PolycubeStructure:
    """
    Loads a polycube from a json file.
    :param file_path: a path to a json file. If the path is implicitly relative, it will be prepended with `~/.pypatchy/input`
    :return: the PolycubeStructure stored in the file
    """
    file_path = process_path(file_path, get_input_dir())
    with file_path.open("r") as f:
        data = json.load(f)
    rule = PolycubesRule(rule_json=data["cube_types"])
    return PolycubeStructure(rule=rule, structure=data["cubes"])



# these next two methods were written by chatGPT to help make topologies faster
def create_spatial_hash(particles):
    hash_map = defaultdict(list)
    for cube in particles:
        # Directly use position as the key since positions are integers
        key = tuple(cube.position())
        hash_map[key].append(cube)
    return hash_map


def get_neighboring_cells(cell: tuple) -> tuple:
    """Generate keys for neighboring cells (including the cell itself)."""
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                yield cell[0] + dx, cell[1] + dy, cell[2] + dz
