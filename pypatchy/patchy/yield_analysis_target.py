import json
from enum import Enum
from typing import Union

import networkx as nx
from networkx.algorithms import isomorphism

from pypatchy.util import get_input_dir


class ClusterCategory(Enum):
    OVER = 0
    SMALLER_NOT_SUB = 1
    SUBSET = 2
    MATCH = 3


class YieldAnalysisTarget:
    # very simple wrapper class for yield target
    name: str
    graph: nx.Graph

    def __init__(self, name: str, graph: Union[nx.Graph, None]=None):
        self.name = name
        if not graph:
            graph = graphShape(get_input_dir() / "targets" / f"{name}.json")
        self.graph = graph

    def __len__(self) -> int:
        return len(self.graph)

    def compare(self, g: nx.Graph) -> tuple[ClusterCategory, float]:
        # compute size fraction
        sizeFrac = len(g) / len(self)
        # check if g is a subgraph of the target graph
        if isomorphism.GraphMatcher(nx.line_graph(self.graph),
                                    nx.line_graph(g)
                                    ).subgraph_is_isomorphic():
            if sizeFrac == 1:
                cat = ClusterCategory.MATCH
            else:
                cat = ClusterCategory.SUBSET
        else:
            if sizeFrac < 1:
                cat = ClusterCategory.SMALLER_NOT_SUB
            else:
                cat = ClusterCategory.OVER
        return cat, sizeFrac


def graphShape(shapePath):
    with open(shapePath, 'r') as f:
        data = f.read()
    solveSpec = json.loads(data)
    G = nx.Graph()
    for i, _, j, _ in solveSpec['bindings']:
        G.add_edge(i, j)
    return G
