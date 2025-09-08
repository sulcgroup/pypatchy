from __future__ import annotations

import copy
import itertools
from argparse import ArgumentError
from typing import Union, Generator

import networkx as nx
import numpy as np


def make_int_mat_for_colors(maxColor: int):
    return InteractionMatrix([((-c, c), 1.) for c in range(1, maxColor+1)])

class InteractionMatrix:
    """
    interaction matrix class. wraps dict w/ helper methods
    currently this is only used by patchy but it will soon be incorporated into crystal designs
    """
    __interactions: dict[tuple[int, int], float]
    # todo: cache min & max color?

    def __init__(self, interactions: Union[dict[tuple[int, int], float], list[tuple[tuple[int, int], float]]] = dict()):
        self.__interactions = {
            ((c1, c2) if c2 > c1 else (c2, c1)): val
            for (c1, c2), val in (interactions.items() if type(interactions) == dict else interactions)
        }

    def intmap(self) -> dict[tuple[int,int],float]:
        return {
            **copy.deepcopy(self.__interactions),
            **{
                (c2, c1): self[(c1, c2)] for c1, c2 in self.__interactions
            }
        }

    def __getitem__(self, item: tuple[int, int]) -> float:
        c1, c2 = item
        if item not in self:
            return 0.
        elif c1 > c2:
            return self[(c2, c1)]
        else:
            return self.__interactions[item]

    def __setitem__(self, key:tuple[int,int], strength: float):
        c1, c2 = key
        assert isinstance(c1, int)
        assert isinstance(c2, int)

        if c1 > c2:
            self[(c2, c1)] = strength
        else:
            self.__interactions[(c1, c2)] = strength

    def __contains__(self, item: tuple[int, int]):
        if isinstance(item, tuple):
            c1, c2 = item
            assert type(c1) == int and type(c2) == int
            if c1 > c2:
                return (c2, c1) in self
            else:
                return (c1, c2) in self.__interactions
        elif isinstance(item, int):
            for _ in self.get_interacting_colors(item):
                # if get_interacting_colors yields any items, return True
                return True
            return False
        raise ArgumentError("Illegal arguement type")


    def num_interactions(self) -> int:
        return len(self.__interactions)

    # def is_one_to_one(self) -> bool:
    #     """
    #     an interaction matrix is one-to-one if and only if all patch interactions that have a strength
    #     value are between colors that add to zero
    #     """
    #     return all([c1 + c2 == 0 for (c1, c2), strength in self.__interactions.items() if strength])
    def is_one_to_one(self) -> bool:
        """
        correction: an interaction matrix is one-to-one iff every connected component of the interaction
        graph has two or fewer nodes
        """
        return all([
            len(cc) <= 2
            for cc in nx.connected_components(self.graph())
        ])

    def num_colors(self) -> int:
        """
        returns num colors, positive + negative
        """
        return len(set(itertools.chain.from_iterable(self.__interactions.keys())))

    def to_array(self) -> np.ndarray:
        """
        converts interaction matrix object to 2d array where values are interaction strengths and x,y coords
        are patch type IDs (or colors or whatever)
        """
        max_index = self.num_colors()
        # Create an array with dimensions based on the maximum index
        array = np.zeros((max_index + 1, max_index + 1))  # +1 because indices are zero-based
        for (i, j), value in self.__interactions.items():
            array[i, j] = array[j, i] = value

        return array

    def self_interacts(self, color: int) -> bool:
        """
        checks if a color can interact with itself according to this interaction matrix
        """
        return self[(color, color)] != 0.

    def __iter__(self):
        yield from self.__interactions.items()

    def get_interacting_colors(self, c:int)-> Generator[int, None, None]:
        for (c1,c2), s in self:
            if not s: continue
            if c1 == c:
                yield c2
            elif c2 == c:
                yield c1

    def graph(self) -> nx.Graph:
        """
        expresses the interaction matrix as a graph, where each color is a node
        and interacting colors are connected by edges.
        """
        G = nx.Graph()
        for (c1, c2), strength in self:
            G.add_edge(c1, c2, strength=strength)
        return G