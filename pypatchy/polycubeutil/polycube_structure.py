from __future__ import annotations

import copy
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from oxDNA_analysis_tools.UTILS.data_structures import Configuration
from scipy.spatial.transform import Rotation

from pypatchy.vis_util import get_particle_color

from pypatchy.patchy_base_particle import PatchyBaseParticle, BaseParticleSet
from pypatchy.scene import Scene
from pypatchy.polycubeutil.polycubesRule import PolycubesRule, diridx, PolycubeRuleCubeType, RULE_ORDER, PolycubesPatch

from pypatchy.structure import TypedStructure, Structure
from pypatchy.util import from_xyz, getRotations, get_input_dir


class PolycubeStructure(TypedStructure, Scene):
    # mypy type specs
    rule: PolycubesRule
    cubeMap: dict[bytes, PolycubesStructureCube]

    def __init__(self, **kwargs):
        super(PolycubeStructure, self).__init__()

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
                self._particles.append(cubeObj)
                self.graph.add_node(cube_idx, cube=cubeObj)
        elif "cubes" in kwargs:
            # structure passed as list of cube objects
            for cube in kwargs["cubes"]:
                if isinstance(cube, dict):
                    cube = PolycubesStructureCube()
                assert cube.get_type() in [ct.type_id() for ct in self.rule.particles()], "Cube type not found in rule!"
                self._particles.append(cube)
                self.cubeMap[cube.position().tobytes()] = cube
                self.graph.add_node(cube.get_id(), cube=cube)
        else:
            # empty structure
            pass

        # loop cube pairs (brute-forcing topology)
        for cube1, cube2 in itertools.combinations(self._particles, 2):
            # if cubes are adjacent
            if np.linalg.norm(cube1.position() - cube2.position()) == 1:
                d1 = cube2.position() - cube1.position()
                d2 = d1 * -1
                # if both cubes have patches on connecting faces and the patch colors match
                if cube1.has_patch(d1) and cube2.has_patch(d2):
                    p1 = cube1.patch(d1)
                    p2 = cube2.patch(d2)
                    if p1.color() == -p2.color():
                        align1 = cube1.rotation().apply(p1.alignDir()).round()
                        align2 = cube2.rotation().apply(p2.alignDir()).round()
                        if (align1 == align2).all():
                            if cube1.state(p1.state_var()) and cube2.state(p2.state_var()):
                                # add edge in graph
                                self.graph.add_edge(cube1.get_id(), cube2.get_id(), dirIdx=diridx(d1))
                                self.graph.add_edge(cube2.get_id(), cube1.get_id(), dirIdx=diridx(d2))
                                self.bindings_list.add((
                                    cube1.get_id(), diridx(d1),
                                    cube2.get_id(), diridx(d2)
                                ))

    def num_cubes_of_type(self, ctidx: int) -> int:
        return sum([1 for cube in self._particles if cube.get_cube_type().type_id() == ctidx])

    def get_cube(self, uid: int) -> PolycubesStructureCube:
        # assert -1 < uid < len(self.cubeList)
        for cube in self._particles:
            if cube.get_id() == uid:
                return cube
        raise Exception(f"No cube with ID {uid}")

    def cube_type_counts(self) -> list[int]:
        return [self.num_cubes_of_type(i) for i, _ in enumerate(self.rule.particles())]

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
        return self._particles[n].typedir(self.get_arrow_diridx(n, adj))

    def particle_type(self, particle_id: int) -> int:
        return self._particles[particle_id].get_type()

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

    def transform(self, rot: np.ndarray, tran: np.ndarray) -> PolycubeStructure:
        # assert transformation.shape == (4, 4), "Wrong shape for transformation"
        assert rot.shape == (3, 3)
        assert tran.shape[0] == 3
        transformed_structure = copy.deepcopy(self)
        transformed_structure.cubeMap = {}
        for cube in transformed_structure._particles:
            cube.set_position(cube.position() @ rot + tran)
            transformed_structure.cubeMap[cube.position().tobytes()] = cube

        return transformed_structure

    def substructure(self, nodes: tuple[int]) -> Structure:
        """
        Returns:
             a Structre object that's a substructure of this
        """
        assert nx.algorithms.components.is_strongly_connected(self.graph.subgraph(nodes))
        return PolycubeStructure(cubes=[c for i, c in enumerate(self._particles) if i in nodes], rule=self.rule)

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
        return self.graph.has_edge(p1.get_id(), p2.get_id())

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
        return len(self.graph.edges) / 2  # divide by 2 b/c graph is bidirerctional


class PolycubesStructureCube(PatchyBaseParticle):
    _type_cube: PolycubeRuleCubeType

    def __init__(self,
                 uid: int,
                 cube_position: np.ndarray,
                 cube_rotation: Union[np.ndarray, int],
                 cube_type: PolycubeRuleCubeType,
                 state: list[bool] = [True]):
        """
        Parameters:
            uid (int): a unique identifier for this cube
            cube_position (np.ndarray): the position of the particle, as a 3-length integer vector
            cube_rotation (np.ndarray, int): a quaternion or integer represtation of cube rotation
        """
        super(PolycubesStructureCube, self).__init__(uid, cube_type.type_id(), cube_position)
        if isinstance(cube_rotation, np.ndarray) and len(cube_rotation) == 4:
            # if rotation hsa been passed as a quaternion
            self._rot = Rotation.from_quat(cube_rotation)
        elif isinstance(cube_rotation, int):
            # if rotation is a an integer, representing an index in rotation enumeration
            self._rot = Rotation.from_matrix(getRotations()[cube_rotation])
        else:
            raise TypeError("Rotation matrices or whatever not supported yet.")
        self._state = state
        self._type_cube = cube_type

    def get_cube_type(self) -> PolycubeRuleCubeType:
        return self._type_cube

    def rotation(self) -> Rotation:
        return self._rot

    def rot_mat(self) -> np.ndarray:
        return self.rotation().as_matrix()

    def rotate(self, rotation: Rotation):
        """
        todo: more param options
        """
        self._rot = self._rot * rotation

    def typedir(self, direction: Union[int, np.ndarray]) -> np.ndarray:
        """
        Converts the global-space direction into a local-space direction
        """
        if isinstance(direction, int):  # if the arguement is provided as an index in RULE_ORDER
            direction = RULE_ORDER[direction]
        return self.rotation().inv().apply(direction).round()

    def has_patch(self, direction: Union[int, np.ndarray]) -> bool:
        return self.get_cube_type().has_patch(self.typedir(direction))

    def patch(self, direction: Union[int, np.ndarray]) -> PolycubesPatch:
        return self.get_cube_type().patch(self.typedir(direction))

    def num_patches(self) -> int:
        return self.get_cube_type().num_patches()

    def state(self, i=None):
        if i is None:
            return self._state
        else:
            assert abs(i) < len(self._state)
            if i < 0:
                return not self._state[-i]
            else:
                return self._state[i]

    def patches(self) -> list[PolycubesPatch]:
        """
        Returns the patches on this polycube, rotated correctly
        """
        return [p.rotate(self.rot_mat()) for p in self.get_cube_type().patches()]


def load_polycube(file_path: Union[Path, str]) -> PolycubeStructure:
    if isinstance(file_path, str):
        file_path = Path(file_path)
    if not file_path.is_absolute():
        file_path = get_input_dir() / file_path
    with file_path.open("r") as f:
        data = json.load(f)
        rule = PolycubesRule(rule_json=data["cube_types"])
        return PolycubeStructure(rule=rule, structure=data["cubes"])
