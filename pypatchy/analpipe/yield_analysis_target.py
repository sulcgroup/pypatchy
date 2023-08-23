import json
from enum import Enum
from typing import Union

import networkx as nx
import igraph as ig
from networkx.algorithms import isomorphism

from pypatchy.util import get_input_dir


class ClusterCategory(Enum):
    """
    NEVER actually use one of these objects directly!
    Otherwise comparisons will fail because Pandas is very badly coded!
    [censored swearing profusely]
    """
    OVER = 0
    SMALLER_NOT_SUB = 1
    SUBSET = 2
    MATCH = 3


class YieldAnalysisTarget:
    """
    very simple wrapper class for yield target
    The yield analysis target is a particle cluster represented as a graph.
    The yield of a cluster is the number of nodes in the largest possible subgraph of the target which
    is isomorphic to the cluster graph
    """
    name: str
    graph: nx.Graph

    def __init__(self, name: str, graph: Union[nx.Graph, None] = None):
        """
        Constructor
        """
        self.name = name
        if not graph:
            graph = graphShape(get_input_dir() / "targets" / f"{name}.json")
        self.graph = graph

    def __len__(self) -> int:
        """
        Returns:
            the length of the graph representation of this analysis target
        """
        return len(self.graph)

    def compare(self, g: nx.Graph) -> tuple[int, float]:
        """
        Compares a cluster graph to the analysis target and returns a classification of the cluster and its yield

        Args:
            g: a graph

        Returns:
             a tuple where the first element is the category of the cluster, and the seonc element is a float representation of the yield
        """
        # compute size fraction
        sizeFrac = len(g) / len(self)
        # check if g is a subgraph of the target graph
        if len(ig.Graph.from_networkx(self.graph).get_subisomorphisms_vf2(ig.Graph.from_networkx(g))) > 0:
            if sizeFrac == 1:
                cat = ClusterCategory.MATCH.value
            else:
                cat = ClusterCategory.SUBSET.value
        else:
            if sizeFrac < 1:
                cat = ClusterCategory.SMALLER_NOT_SUB.value
            else:
                cat = ClusterCategory.OVER.value
        return cat, sizeFrac


def graphShape(shapePath):
    with open(shapePath, 'r') as f:
        data = f.read()
    solveSpec = json.loads(data)
    G = nx.Graph()
    for i, _, j, _ in solveSpec['bindings']:
        G.add_edge(i, j)
    return G
