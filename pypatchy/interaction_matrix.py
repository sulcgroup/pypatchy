from __future__ import annotations

import math
from typing import Union

import numpy as np

from pypatchy.patchy.pl.plpatch import PLPatch


def make_interaction_matrix(patches: list[PLPatch]) -> InteractionMatrix:
    intmatrix = dict()
    # we do want these to contain both (i,j) and (j,i)
    for p1 in patches:
        for p2 in patches:
            if p1.color() == -p2.color():
                intmatrix[(p1.color(), p2.color())] = math.sqrt(p1.strength() * p2.strength())
    return InteractionMatrix(interactions=intmatrix)


class InteractionMatrix:
    """
    interaction matrix class. wraps dict w/ helper methods
    currently this is only used by patchy but it will soon be incorporated into crystal designs
    """
    __interactions: dict[tuple[int, int], float]
    # todo: cache min & max color?

    def __init__(self, interactions: Union[dict[tuple[int, int], float], list[tuple[int, int], float]] = dict()):
        self.__interactions = {
            ((c1, c2) if c2 > c1 else (c2, c1)): val
            for (c1, c2), val in (interactions.items() if type(interactions) == dict else interactions)
        }

    def __getitem__(self, item: tuple[int, int]) -> float:
        c1, c2 = item
        if c1 > c2:
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
        c1, c2 = item
        assert type(c1) == item and type(c2) == int
        if c1 > c2:
            return (c2, c1) in self
        else:
            return (c1, c2) in self.__interactions

    def num_interactions(self) -> int:
        return len(self.__interactions)

    def is_one_to_one(self) -> bool:
        """
        an interaction matrix is one-to-one if and only if all patch interactions that have a strength
        value are between colors that add to zero
        """
        return all([c1 + c2 == 0 for (c1, c2), strength in self.__interactions.items() if strength])

    def num_colors(self) -> int:
        """
        returns num colors, positive + negative
        """

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
