from __future__ import annotations

import itertools
from typing import Union

import matplotlib.pyplot as plt
import networkx as nx

from pypatchy.vis_util import get_particle_color
from pypatchy.polycubeutil.polycubesRule import RULE_ORDER
from pypatchy.structure import Structure, StructuralHomomorphism, TypedStructure


class CommonComponent(Structure):
    full_structures: list[Structure]
    # structural homomorphisms map node indices from each of the structures in `full_structures`
    # onto the common component
    homomorphisms: list[StructuralHomomorphism]

    def __init__(self,
                 component: Structure,
                 full_structures: list[Structure],
                 homomorphisms: list[StructuralHomomorphism]):
        super().__init__(graph=component.graph)
        # check outputs
        assert len(full_structures) == len(homomorphisms), "Mismatch beteween counts of provided structures and " \
                                                           "structureal homomorposms!"
        for s, h in zip(full_structures, homomorphisms):
            assert all([n in h.rlmap for n in self.graph.nodes]), "Not all nodes in common component are targeted by" \
                                                                  "homomorphism!"
            assert all([n in s.graph.nodes for n in h.lmap]), "Not all nodes in structures "
        self.full_structures = full_structures
        self.homomorphisms = homomorphisms

    def show(self) -> plt.Figure:
        fig, axs = plt.subplots(nrows=2, ncols=len(self.full_structures))
        # Increase the bottom margin to make space for the text below the subplots
        plt.subplots_adjust(bottom=0.2)

        axs[0, 0].set_title("Common Component")
        nx.draw(self.graph, ax=axs[0, 0], with_labels=True)

        for i, (s, f) in enumerate(zip(self.full_structures, self.homomorphisms)):
            if i > 0:
                axs[0, i].set_visible(False)  # Hide extra axes in the top row if more than one structure
            draw_params = {
                "ax": axs[1, i],
                "with_labels": True
            }
            if isinstance(s, TypedStructure):
                ptypemap = [get_particle_color(s.particle_type(j)) for j in s.graph.nodes]
                draw_params["node_color"] = ptypemap

            nx.draw(s.graph, **draw_params)
            axs[1, i].set_title(f"Structure {i}")

            # Prepare the mapping text
            mappingtxt = "\n".join([f"{a} -> {b}" for a, b in f.lmap.items()])

            # Get the bounding box of the subplot in figure coordinates
            bbox = axs[1, i].get_position()
            text_x = bbox.x0 + 0.5 * bbox.width  # Horizontal center of the bbox
            text_y = bbox.y0 - 0.1  # Just below the bbox; adjust as needed

            # Place the text below the axes using figure coordinates
            fig.text(text_x, text_y, "Mapping of structure onto\ncommon component:\n" + mappingtxt,
                     ha="center", va="top", transform=fig.transFigure, fontsize=9)

        # Use tight_layout to automatically adjust subplot params
        fig.set_figheight(fig.get_figheight() + (len(self) + 2)*75 / fig.dpi)

        plt.show()

        return fig

    def is_superimposition_point(self, v: int) -> bool:
        """
        See defn. of a superimposition point in the doc "Computational Design of Allostery"

        Parameters:
            v: a point in the common component
        """
        # get the obvious one out of the way
        if v not in list(self.graph.nodes):
            # if point v isn't even in the graph it won't be a superimposition point
            return False
        # loop rule order directions
        for delta, _ in enumerate(RULE_ORDER):
            # if there common component has an out-edge from v with this rule order position, ignore
            if self.edge_exists(v, delta):
                continue
            # loop structures that compose common component.
            # if more than one structure has out edge delta in this homomorpism, return false
            if len([i for i in range(self.nstructures()) if self.homomorphism_contains(i, v, delta)]) > 1:
                return False
        return True

    def is_crucial_point(self, v: int) -> bool:
        """
        See defn. of a crucial point in the doc "Computational Design of Allostery"
        Parameters:
            v: a point in the common component
        """
        # get the obvious one out of the way
        if v not in list(self.graph.nodes):
            return False
        # loop structures that this is a component of
        for istructure, (s, f) in enumerate(zip(self.full_structures, self.homomorphisms)):
            # loop pairs of nodes in s
            for u1, u2 in itertools.combinations(list(s.graph.nodes), 2):
                # note: u1 and v1 are node IDs in the structure not in the common component
                # so if we need to compare to v we need to use the f.lmap
                # skip instances where either u1 or u2 are in the common component
                if u1 in f.lmap or u2 in f.lmap:
                    continue
                # loop all possible simple paths from u1 t   o u2
                for p in nx.algorithms.all_simple_paths(s.graph, u1, u2):
                    # loop nodes in path
                    for n in p:
                        # if node n in the path is in the common component but
                        # isn't the node (v) that we're testing for cruciality,
                        # node v is not a crucial point
                        if n in f.lmap and f.lmap[n] != v:
                            return False
        # check for deltaA, deltaB....
        # for i in range(self.nstructures()):
        #     if not any(self.homomorphism_contains(i, v, delta) for delta, _ in enumerate(RULE_ORDER)):
        #         return False

        return True

    def is_pivot_point(self, v: int) -> bool:
        """
        See defn. of a pivot point in the doc "Computational Design of Allostery"
        Parameters:
            v: a node identifier in the COMMON COMPONENT
        Returns:
            True if v is a pivot point, false otherwise
        """
        # check if vertex is a crucial point and is a superimposition point
        crucial_pt = self.is_crucial_point(v)
        sup_pt = self.is_superimposition_point(v)
        if not crucial_pt or not sup_pt:
            return False
        # check if any other vertices in this Common Component are crucial points
        for n in list(self.graph.nodes):
            if n != v and self.is_crucial_point(n):
                return False
        return True

    def is_macoco(self) -> Union[int, bool]:
        """
        Tests if this common component is a macoco,
        Returns:
            False if this is not a macguffin, otherwise the pivot point
        """
        for v in list(self.graph.nodes):
            if self.is_pivot_point(v):
                return v
        else:
            return False

    def nstructures(self) -> int:
        """
        Returns:
            the number of structures that this object is a common component of
        """
        return len(self.full_structures)

    def node_in_structure(self, n: int, s: int) -> Union[int, bool]:
        """
        Given a node in the common component, finds the corresponding node in a structure if one exists
        Parameters:
            n: integer identifier for a noed in the common component
            s: index of full structure to check
        Returns:
            the identifier for node n in the structure with index s, if such a node exists. otherwise False
        """
        assert -1 < s < self.nstructures(), f"Value {s} out of bounds!"
        assert n in self.graph.nodes, f"No node in common component with id {n}"
        f = self.homomorphisms[s]

        # if node n is not in the inverse of the homomorphism (maps cc node ids onto structure node ids)
        if n not in f.rlmap:
            return False
        else:
            return f.rlmap[n]

    def homomorphism_contains(self, i: int, v: int, delta: int) -> bool:
        assert i < self.nstructures()
        return self.homomorphisms[i].contains_edge(v, delta)
        # return self.full_structures[i].edge_exists(self.homomorphisms[i].rmap_location(v),
        #                                            self.homomorphisms[i].rmap_direction(delta))

    def disjoint_n_edges(self, i: int) -> int:
        return len(self.full_structures[i].graph.edges) - len(self.graph.edges)

    def disjoint_n_nodes(self, i: int) -> int:
        return len(self.full_structures[i].graph.nodes) - len(self.graph.nodes)

    def to_solve_spec(self) -> dict:
        return {
            "bindings": list(self.bindings_list),
            "nDims": 3,
            "torsion": True,
            "stopAtFirst": True
        }


def is_common_component(candidate: Structure, structures: Union[list, tuple]) -> Union[CommonComponent, None]:
    """
    Checks if candidate is a common component (see doc: "Computational Design of Allostery")
    of the provided structures
    Parameters:
        candidate: a Structure object
        structures (list) a list of Structure objects

    Returns:
        A tuple where the first element is True if self is a common component of all provided
        structures, and the second element is a list of StructuralHomomorphism of (self->st)
         for st in the provided list of structures
    """

    if not nx.components.is_strongly_connected(candidate.graph):
        return None

    # identifies a homomorphism for each structure onto this
    homomorphisms = [candidate.homomorphism(s) for s in structures]
    # if homomorphisms exist for all structures, this is common component
    if all(homomorphisms):
        return CommonComponent(candidate, structures, homomorphisms)
    else:
        return None


def get_common_components(*args: Structure) -> list[CommonComponent]:
    """
    Parameters:
         args: Structure objects
    """
    assert len(args) >= 2
    # start with an arbitrary structure
    s0 = args[0]
    common_components = []
    # loop possible sizes for components of s0
    for n in range(2, len(s0) + 1):
        # loop combinations of nodes in s0
        for nodes in itertools.combinations(s0.graph, n):
            # grab subgraph
            subgraph = s0.graph.subgraph(nodes)
            # check if subgraph si connected
            if nx.algorithms.components.is_strongly_connected(subgraph):
                # grab Structure object
                component = Structure(graph=subgraph)
                # check if this is a common component
                component = is_common_component(component, args)
                if component is not None:
                    # TODO: check for redundency? somehow
                    common_components.append(component)
    return common_components
