import itertools
import sys
import json
import networkx as nx

from polycubesRule import *

import networkx as nx
from collections import defaultdict


def cycles_by_size(graph):
    """
    Fair warning: this code written by chatGPT
    Construct a list of lists of cycles in the graph,
    where each second-level list is composed of cycles of the same size

    Parameters:
    graph (networkx.classes.graph.Graph): The graph

    Returns:
    dict: dictionary where keys are cycle sizes and values are lists of cycles
    """
    # Use simple_cycles to get all cycles in the graph
    cycles = list(nx.simple_cycles(graph))

    # Initialize a default dictionary to store cycles by size
    cycles_by_size = defaultdict(list)

    # Iterate over the cycles
    for cycle in cycles:
        # The size of the cycle is its length
        size = len(cycle)

        # Append the cycle to the list of cycles of the same size
        cycles_by_size[size].append(cycle)

    return cycles_by_size


class AllosteryDesigner:
    def __init__(self, rule, structure):
        # TODO: load structure from
        self.rule = rule
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
            cube_rotation = np.array([cube["rotation"][k] for k in cube["rotation"]])
            cube_type = self.rule.particle(cube["type"])
            # emplace cube in map
            cubeObj = PolycubesStructureCube(cube_idx, cube_position, cube_rotation, cube_type)
            self.cubeMap[cube_position.tobytes()] = cubeObj
            self.cubeList.append(cubeObj)
            self.graph.addnode(cube_idx, {'cube': cubeObj})

        # loop cube pairs (brute-forcing topology)
        for cube1, cube2 in itertools.combinations(self.cubeList, 2):
            # if cubes are adjacent
            if (cube1.position() - cube2.position()).norm() == 1:
                d1 = cube2.position() - cube1.position()
                d2 = d1 * -1
                # if both cubes have patches on connecting faces and the patch colors match
                if cube1.has_patch(d1) and cube2.has_patch(d2) and \
                        cube1.get_patch(d1).color() == cube2.get_patch(d2).color():
                    # TODO: check if patches are active?
                    # add edge in graph
                    self.graph.add_edge(cube1.get_id(), cube2.get_id())

    def add_allostery(self):
        """
        Adds allosteric controls one step at a time
        """
        cycles = cycles_by_size(self.graph)
        for cycle_size in cycles:
            cycle_list = cycles[cycle_size]
            # i believe this was a rejected name for dykes on bikes
            # ok but seriously: check for cycles that contain the same cube type
            # and same connection faces on types
            # in the same pattern. we should be able to design allostery for the cycles
            for homocycles in homologous_cycles(cycle_list):

                # get list of nodes with same cubes (same uid)
                common_nodes = get_nodes_overlap(homocycles)

                # starting from common nodes, add allostery to particles, where
                # the patch closer to the common nodes activates the one farther

                cycle = homocycles[0]  # at this point the homologous cycles are functionally indistinguishable

                # create a set for nodes we've already done in this step

                allo_nodes_this_step = set()
                # "head" nodes are the nodes we're currently processing
                head_nodes = common_nodes
                # this while loop is a time bomb
                while len(allo_nodes_this_step) < cycle_size:
                    next_head_nodes = []

                    # loop head nodes
                    for head_node in head_nodes:
                        if head_node not in allo_nodes_this_step:
                            # move to next node
                            # advance the head to a node which is in the cycle that is not in allo_nodes_this_step,
                            # and find indexes in RULE_ORDER of faces on cube type that are responsible
                            # for joining our new head_node to the previous and next nodes in the cycle
                            face_conn_prev, head_node, face_conn_next = next_node_in_cycle(head_node,
                                                                                           cycle,
                                                                                           allo_nodes_this_step)
                            head_node.get_type().add_allostery
                            # add to set of nodes we've processed
                            allo_nodes_this_step.add(head_node)
                            # add our new head node to the list of head nodes for the next step
                            next_head_nodes.add(head_node)
                    head_nodes = next_head_nodes

    def cubeAtPosition(self, v):
        return self.cubeMap[v.tobytes()]


class PolycubesStructureCube:
    def __init__(self, uid, cube_position, cube_rotation, cube_type):
        self._uid = uid
        self._pos = cube_position
        self._rot = cube_rotation
        self._type = cube_type

    def get_id(self):
        return self._uid

    def get_position(self):
        return self._pos

    def get_rotation(self):
        return self._rot

    def get_type(self):
        return self._type


if __name__ == "__main__":
    jsonfile = sys.argv[1]
    with open(jsonfile, 'r') as f:
        j = json.load(f)
        rule = PolycubesRule(jsonfile["cube_types"])
        AllosteryDesigner(rule, json["cubes"])
