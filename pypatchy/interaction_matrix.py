from __future__ import annotations

import copy
import itertools
from argparse import ArgumentError
from collections.abc import Generator

import networkx as nx
import numpy as np


def make_int_mat_for_colors(maxColor: int):
    return InteractionMatrix([((-c, c), 1.) for c in range(1, maxColor+1)])


def make_torsional_int_mat_for_colors(maxColor: int) -> TorsionalInteractionMatrix:
    """
    Builds a TorsionalInteractionMatrix that reproduces the standard polycubes convention: colors
    bind 1-to-1 with their negation, only at relative angle 0 (i.e. matching alignDirs exactly -
    the same condition as the old binary "torsional" patch behavior).
    """
    return TorsionalInteractionMatrix([((-c, c, 0), 1.) for c in range(1, maxColor + 1)])

class InteractionMatrix:
    """
    interaction matrix class. wraps dict w/ helper methods
    currently this is only used by patchy but it will soon be incorporated into crystal designs
    """
    __interactions: dict[tuple[int, int], float]
    # todo: cache min & max color?

    def __init__(self, interactions: dict[tuple[int, int], float] | list[tuple[tuple[int, int], float]] = dict()):
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
        return all(len(cc) <= 2
            for cc in nx.connected_components(self.graph()))

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


class TorsionalInteractionMatrix(InteractionMatrix):
    """
    An interaction matrix that adds torsion as a third dimension: entries are keyed by
    (color1, color2, angle) rather than just (color1, color2), where angle is a discrete relative
    rotation index (0-3, in steps of 90 degrees - see polycubeutil.polycubesRule.relative_align_rot_num)
    describing the twist between the two patches' alignment (a2/a3) vectors around their shared bond
    axis.

    This lets a color pair's binding depend on relative patch orientation with more granularity than
    a plain InteractionMatrix-based scheme paired with a binary torsional ("alignDirs must match
    exactly")/non-torsional ("angle ignored") per-patch flag: e.g. a color pair can be set to bind
    only at specific angles, or with different strengths at different angles.

    A color pair with no orientation dependence (the old "non-torsional" case) should be given the
    same strength at all four angles; see make_torsional_int_mat_for_colors for the old
    "torsional, angle must be 0" case.

    NOTE: this class only overrides the parts of InteractionMatrix's interface that still make sense
    with a 3-key store. to_array()/intmap() are not supported and will raise AttributeError if
    called, since they operate on the (unused, in this subclass) 2-key store.
    """
    __interactions: dict[tuple[int, int, int], float]

    def __init__(self, interactions: dict[tuple[int, int, int], float] | list[tuple[tuple[int, int, int], float]] = dict()):
        self.__interactions = {
            self._normalize_key(key): val
            for key, val in (interactions.items() if type(interactions) == dict else interactions)
        }

    @staticmethod
    def _normalize_key(key: tuple[int, int, int]) -> tuple[int, int, int]:
        c1, c2, angle = key
        return (c1, c2, angle) if c1 <= c2 else (c2, c1, angle)

    def __getitem__(self, item: tuple[int, int, int]) -> float:
        return self.__interactions.get(self._normalize_key(item), 0.)

    def __setitem__(self, key: tuple[int, int, int], strength: float):
        c1, c2, angle = key
        assert isinstance(c1, int)
        assert isinstance(c2, int)
        assert isinstance(angle, int) and 0 <= angle < 4, f"Angle index must be an int in [0,4), got {angle}"
        self.__interactions[self._normalize_key(key)] = strength

    def __contains__(self, item: tuple[int, int, int] | tuple[int, int] | int) -> bool:
        if isinstance(item, tuple) and len(item) == 3:
            return self._normalize_key(item) in self.__interactions
        elif isinstance(item, tuple) and len(item) == 2:
            return self.colors_interact(*item)
        elif isinstance(item, int):
            for _ in self.get_interacting_colors(item):
                return True
            return False
        raise ArgumentError("Illegal arguement type")

    def num_interactions(self) -> int:
        return len(self.__interactions)

    def best_strength(self, c1: int, c2: int) -> float:
        """
        The strongest interaction strength between this color pair, across all four angles. Use
        this (rather than indexing by a specific angle) when one or both patches involved don't
        have torsion, since then the relative angle can't be computed/doesn't matter.
        """
        return max((self[(c1, c2, angle)] for angle in range(4)), default=0.)

    def colors_interact(self, c1: int, c2: int) -> bool:
        """
        True if this color pair binds at any angle. See best_strength.
        """
        return self.best_strength(c1, c2) != 0.

    def self_interacts(self, color: int) -> bool:
        return self.colors_interact(color, color)

    def __iter__(self):
        yield from self.__interactions.items()

    def get_interacting_colors(self, c: int) -> Generator[int, None, None]:
        seen = set()
        for (c1, c2, angle), s in self:
            if not s:
                continue
            other = c2 if c1 == c else c1 if c2 == c else None
            if other is not None and other not in seen:
                seen.add(other)
                yield other

    def num_colors(self) -> int:
        return len(set(itertools.chain.from_iterable((c1, c2) for c1, c2, _ in self.__interactions)))

    def graph(self) -> nx.Graph:
        """
        Collapses the angle dimension: two colors are connected iff they bind at any angle, with
        edge attr 'strength' set to the strongest binding across angles. Mainly useful for
        is_one_to_one()/connectivity-style checks inherited from InteractionMatrix.
        """
        G = nx.Graph()
        for (c1, c2, angle), strength in self:
            if not strength:
                continue
            if G.has_edge(c1, c2):
                G[c1][c2]["strength"] = max(G[c1][c2]["strength"], strength)
            else:
                G.add_edge(c1, c2, strength=strength)
        return G