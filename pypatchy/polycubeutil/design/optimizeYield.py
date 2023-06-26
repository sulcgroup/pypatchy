import copy
import itertools
import sys
import json
from ..structure import *


class YieldAllosteryDesigner(PolycubeStructure):
    def __init__(self, r, structure):
        super(YieldAllosteryDesigner, self).__init__(r, structure)
        self.ct_allo_vars = [set() for _ in r.particles()]

    def add_allostery(self):
        """
        Adds allosteric controls one step at a time
        """
        cycles = self.cycles_by_size()
        if len(cycles) == 0:
            print("Graph has no cycles. exiting...")
            return
        for cycle_size, cycle_list in sorted(cycles.items()):
            # i believe this was a rejected name for dykes on bikes
            # ok but seriously: check for cycles that contain the same cube type
            # and same connection faces on types
            # in the same pattern. we should be able to design allostery for the cycles
            for homocycles in self.homologous_cycles(cycle_list):

                # get list of nodes that will be used to construct allostery for this group of homologous cycles
                nodes_to_design = self.get_design_path(homocycles)

                types_processed = set()

                for prev_node, node, next_node in triplets(nodes_to_design):
                    ct = self.cubeList[node].get_type()

                    # safeguard: don't add allostery to patches that already start on
                    # TODO: potentially, modify this to allow for existing allsotery to be modified
                    if ct.count_start_on_patches() == 1:
                        continue
                    if not ct.getID() in types_processed:
                        types_processed.add(ct.getID())

                        # get patch that will activate this node's cube
                        origin_patch = ct.patch(self.get_arrow_local_diridx(node, prev_node))
                        # if the origin patch doesn't already have a state variable, make one
                        if not origin_patch.state_var():
                            # add a state variable to the current node
                            origin_state = ct.add_state_var()
                            origin_patch.set_state_var(origin_state)
                            self.ct_allo_vars[ct.getID()].add(origin_state)

                        # add an activator
                        target_patch = ct.patch(self.get_arrow_local_diridx(node, next_node))
                        # target patch should not already be allosterically controlled
                        if target_patch.activation_var():
                            continue
                        new_activation_state = ct.add_state_var()
                        target_patch.set_activation_var(new_activation_state)

                        # make all cycle control states that this type has required for activating the connection
                        ct.add_effect(DynamicEffect(self.ct_allo_vars[ct.getID()], new_activation_state))

                        yield copy.deepcopy(self.rule)

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

    def get_graphs_center(self, node_list):
        return sum([self.cubeList[n].get_position() for n in node_list]) / len(node_list)

    def get_design_path(self, cycles):

        # vector of the center of the homologous cycles
        # compute valid paths. can take cycles[0] as our graph because cycles are homologous
        all_paths = all_unique_paths(cycles[0])
        valid_paths = [p for p in all_paths if self.is_valid_design_path(p)]

        # compute overlap of homologous cycles
        cycles_nodes_overlap = get_nodes_overlap(cycles)

        # if overlap exists, require design paths to start from an overlap node
        if len(cycles_nodes_overlap) > 0:
            valid_paths = [p for p in valid_paths if p[0] in cycles_nodes_overlap]

        assert len(valid_paths) > 0

        # sort paths from longest to shortest
        best_paths = longest_paths(valid_paths)

        # compute centerpoint
        centerpoint = self.get_graphs_center([*itertools.chain.from_iterable(cycles)])

        shortest_distance = math.inf
        best_path = []

        # find path which is closest to the center point
        for p in best_paths:
            distance = np.linalg.norm(centerpoint - self.cubeList[p[0]].get_position())
            if distance < shortest_distance:
                shortest_distance = distance
                best_path = p
        return best_path

    def is_valid_design_path(self, p):
        """
        Parameters:
            p (list of ints) an ordered list of node IDs representing a path
        Returns:
            true if p is a valid design path, false otherwise
        """

        behavior_set = set()
        for prev_node, curr_node, next_node in triplets(p):
            # the cube type at the origin of the design path can't occur anywhere else in the path
            if self.cubeList[curr_node].get_type().getID() == self.cubeList[p[0]].get_type().getID():
                return False
            curr_prev_edge = self.get_arrow_local_diridx(curr_node, prev_node)
            curr_next_edge = self.get_arrow_local_diridx(curr_node, next_node)
            # "back" and "front" here are meant not in a physical sense but in the sense of the cycle
            # synonyms to "next" and "prev" kinda
            back_patch_id = self.cubeList[curr_node].get_type().patch(curr_prev_edge)
            front_patch_id = self.cubeList[curr_node].get_type().patch(curr_next_edge)

            behavior = (back_patch_id,
                        self.cubeList[curr_node].get_type().getID(),
                        front_patch_id)

            if behavior in behavior_set:
                return False
            else:
                behavior_set.add(behavior)
        return len(p) > 2


def triplets(iterable):
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
    for allosteric_rule in designer.add_allostery():
        print(allosteric_rule)
