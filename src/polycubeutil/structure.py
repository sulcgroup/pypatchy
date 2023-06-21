import itertools
from scipy.spatial.transform import Rotation

from polycubesRule import *

import networkx as nx
from collections import defaultdict


def get_nodes_overlap(homocycles):
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

class PolycubeStructure:
    # mypy type specs
    rule: PolycubesRule
    cubeList: list
    cubeMap: dict
    graph: nx.DiGraph

    def __init__(self, rule, structure):
        # load rule
        self.rule = rule

        # needed for solving later. assume starting from no allostery

        # read structure
        self.cubeMap = {}
        # nodes in the graph are cube uids
        # edges attrs are FACE INDEXES IN RULE_ORDER, IN THE CUBE TYPE
        self.graph = nx.DiGraph()
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
            cubeObj = PolycubesStructureCube(cube_idx, cube_position, rot_quaternion, cube_type)
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
                    if cube1.get_patch(d1).color() == -cube2.get_patch(d2).color():
                        # TODO: check if patches are active?
                        # add edge in graph
                        ct_dir_idx_1 = diridx(cube1.typedir(d1))
                        ct_dir_idx_2 = diridx(cube2.typedir(d2))
                        self.graph.add_edge(cube1.get_id(), cube2.get_id(), dirIdx=ct_dir_idx_1)
                        self.graph.add_edge(cube2.get_id(), cube1.get_id(), dirIdx=ct_dir_idx_2)

    def homologous_cycles(self, cycle_list):
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

    def next_node_in_cycle(self, n, cycle, processed_nodes):
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

        return (self.graph.get_edge_data(n, prev_node)["dirIdx"],
                next_node,
                self.graph.get_edge_data(n, next_node)["dirIdx"])

    import networkx as nx
    from collections import defaultdict

    import networkx as nx
    from collections import defaultdict

    def cycles_by_size(self):
        """
        Code partially written by ChatGPT
        Construct a list of unique cycles in the graph,
        where each second-level list is composed of cycles of the same size,
        and each cycle visits each node at most once.

        Parameters:
        graph (networkx.classes.graph.Graph): The graph

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

    def cubeAtPosition(self, v):
        return self.cubeMap[v.tobytes()]

    def get_arrow_diridx(self, node, adj):
        return self.graph.get_edge_data(node, adj)["dirIdx"]


class PolycubesStructureCube:
    def __init__(self, uid, cube_position, cube_rotation, cube_type):
        self._uid = uid
        self._pos = cube_position
        if isinstance(cube_rotation, np.ndarray) and len(cube_rotation) == 4:
            self._rot = Rotation.from_quat(cube_rotation)
        elif isinstance(cube_rotation, int):
            self._rot = cube_rotation
        else:
            assert False, "Rotation matrices or whatever not supported yet."
        self._type = cube_type

    def get_id(self):
        return self._uid

    def get_position(self):
        return self._pos

    def get_rotation(self):
        return self._rot

    def get_type(self):
        return self._type

    def typedir(self, direction):
        """
        Converts the global-space direction into a local-space direction
        """
        if isinstance(direction, int):  # if the arguement is provided as an index in RULE_ORDER
            direction = RULE_ORDER[direction]
        return self.get_rotation().inv().apply(direction).round()

    def has_patch(self, direction):
        return self.get_type().has_patch(self.typedir(direction))

    def get_patch(self, direction):
        return self.get_type().patch(self.typedir(direction))
