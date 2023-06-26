from __future__ import annotations

import re
from typing import Union

import networkx as nx
from networkx.algorithms import isomorphism
import json
import pickle
from pathlib import Path
from enum import Enum


# commenting out because it turns out it's not actually helpful
# import cugraph as cnx

class ClusterCategory(Enum):
    OVER = 0
    SMALLER_NOT_SUB = 1
    SUBSET = 2
    MATCH = 3


def graphShape(shapePath):
    with open(shapePath, 'r') as f:
        data = f.read()
    solveSpec = json.loads(data)
    G = nx.Graph()
    for i, _, j, _ in solveSpec['bindings']:
        G.add_edge(i, j)
    return G


def graphsFromClusters(line: str) -> list[nx.Graph]:
    clusterGraphs = []
    clusters = re.finditer('\[.+?\]', line)

    for cluster in clusters:
        G = nx.Graph()
        matches = re.finditer(
            '(\d+) -> \(((?:\d+ ?)+)\)', cluster.group()
        )
        for m in matches:
            source = m.group(1)
            for dest in m.group(2).split(' '):
                G.add_edge(int(source), int(dest))
        clusterGraphs.append(G)
    return clusterGraphs


def getGraphOverlap(g1: nx.Graph,
                    g2: nx.Graph,
                    cutoff=1.0,
                    include_overreach=False) -> float:
    sizeFrac = len(g2) / len(g1)
    if not include_overreach or sizeFrac <= 1:
        return sizeFrac if 1 >= sizeFrac >= cutoff and isomorphism.GraphMatcher(
            nx.line_graph(g1), nx.line_graph(g2)
        ).subgraph_is_isomorphic() else 0
    else:
        isomorphic_subgraphs = list(
            isomorphism.GraphMatcher(nx.line_graph(g2), nx.line_graph(g1)).subgraph_isomorphisms_iter())
        return len(isomorphic_subgraphs)


def getClusterYield(line: str,
                    refGraph: nx.Graph,
                    cutoff: float,
                    overreach: bool):
    return sum(getGraphOverlap(refGraph, g, cutoff, overreach) for g in graphsFromClusters(line))


def readClusters(clustersPath: str,
                 shapePath,
                 cutoff: float,
                 nSamples=float("inf")) -> list[float]:
    refGraph = graphShape(shapePath)
    with open(clustersPath) as f:
        lines = [line for line in f]
        nLines = len(lines)
        nSamples = min(nSamples, nLines)
        sampleEvery = round(nLines / nSamples)
        clusters = [getClusterYield(line, refGraph, cutoff) for i, line in enumerate(lines) if i % sampleEvery == 0]
    return clusters


def getVal(path: Union[Path, str],
           key: str) -> float:
    with open(path, 'r') as f:
        for line in f:
            if key in line:
                return float(line.split('=')[-1])


def analyse(clusterPath: str,
            shapeDir,
            cutoff: float,
            nSamplePoints: int,
            clusterPrintEvery=2e6):
    clusterPath = str(Path(clusterPath).absolute())
    if 'duplicate' in clusterPath:
        try:
            shape, duplStr, potential, tempStr, clusterFile = clusterPath.split('/')[-5:]
            duplicate = float(duplStr.strip('duplicate_'))
        except:
            duplStr, shape, potential, tempStr, clusterFile = clusterPath.split('/')[-5:]
            duplicate = float(duplStr.strip('duplicate_'))
    else:
        shape, potential, tempStr, clusterFile = clusterPath.split('/')[-4:]
        duplicate = 0

    temp = float(tempStr.strip('T_'))

    t = shape.rsplit('_', 1)

    if t[-1] == 'full' or t[-1] == 'inter':
        shapeType = t[-1]
        shape = t[0]
    else:
        shapeType = 'minimal'

    maxTimeStep = getVal(Path(clusterPath).parent.absolute() / "last_conf.dat", 't = ')
    dt = getVal(Path(clusterPath).parent.absolute() / "input", 'dt = ')

    clusters = readClusters(
        clusterPath,
        shapeDir + '/{}.json'.format(shape),
        cutoff,
        nSamplePoints
    )

    timeFactor = dt * maxTimeStep / len(clusters)

    data = []
    maxYield = 0
    for t, clusterYield in enumerate(clusters):
        maxYield = max(maxYield, clusterYield)
        data.append({
            'shape': shape,
            'type': shapeType,
            'temp': temp,
            'potential': potential,
            'duplicate': duplicate,
            'yield': clusterYield,
            'time': t * timeFactor
        })
    print("{} {} {} T={} - Max yield: {}".format(shape, potential, shapeType, temp, maxYield))

    with open(Path(clusterPath).parent.absolute() / "clusters.pickle", 'wb') as f:
        pickle.dump(data, f)

    return data


# if __name__ == '__main__':
#     if len(sys.argv) != 5:
#         print("Incorrect number of arguments (need 4 not {}):".format(len(sys.argv) - 1))
#         print(sys.argv[0] + " clusterPath shapeDir cutoff nSamplePoints")
#     else:
#         analyse(sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), clusterPrintEvery=2e6)
