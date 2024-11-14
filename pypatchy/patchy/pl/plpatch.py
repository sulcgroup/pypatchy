from __future__ import annotations

import math
from typing import Union

import networkx as nx
import numpy as np

from ...interaction_matrix import InteractionMatrix, make_int_mat_for_colors
from ...patchy_base_particle import BasePatchType


class PLPatch(BasePatchType):
    _type: int
    _strength: float

    def __init__(self,
                 type_id: Union[None, int] = None,
                 color: Union[None, int] = None,
                 relposition: Union[None, np.ndarray] = None,
                 a1: Union[None, np.ndarray] = None,
                 a2: Union[None, np.ndarray] = None,
                 strength: float = 1.0):
        super().__init__(type_id, color)
        self._key_points = [
            relposition,
            a1,
            a2
        ]
        self._type = type_id
        self._strength = strength

    def type_id(self) -> int:
        return self._type

    # TODO: make sure this isn't a horrible mistake!!!
    def get_id(self) -> int:
        return self.type_id()

    def set_type_id(self, new_val: int):
        self._type = new_val

    def strength(self) -> float:
        return self._strength

    def set_strength(self, new_val: float):
        self._strength = new_val

    def position(self) -> np.ndarray:
        return self._key_points[0]

    def set_position(self, newposition: np.ndarray):
        self._key_points[0] = newposition

    def colornum(self) -> int:
        return self.color()

    def a1(self) -> np.ndarray:
        return self._key_points[1]

    def set_a1(self, new_a1: np.ndarray):
        self._key_points[1] = new_a1

    def a2(self) -> np.ndarray:
        return self._key_points[2]

    def set_a2(self, new_a2: np.ndarray):
        self._key_points[2] = new_a2

    def a3(self) -> np.ndarray:
        return np.cross(self.a1(), self.a2())

    def get_abs_position(self, r) -> np.ndarray:
        return r + self._position

    def save_to_string(self, extras={}) -> str:
        # print self._type,self._type,self._color,1.0,self._position,self._a1,self._a2

        outs = f'patch_{self.type_id()} = ' + '{\n ' \
                                              f'\tid = {self.type_id()}\n' \
                                              f'\tcolor = {self.color()}\n' \
                                              f'\tstrength = {self.strength()}\n' \
                                              f'\tposition = {np.array2string(self.position(), separator=",")[1:-1]}\n' \
                                              f'\ta1 = {np.array2string(self.a1(), separator=",")[1:-1]}\n'
        if self.a2() is not None:  # tolerate missing a2s
            outs += f'\ta2 = {np.array2string(self.a2(), separator=",")[1:-1]}\n'
        else:
            # make shit up
            outs += f'\ta2 = {np.array2string(np.array([0, 0, 0]), separator=",")[1:-1]}\n'
        outs += "\n".join([f"t\t{key} = {extras[key]}" for key in extras])
        outs += "\n}\n"
        return outs

    def init_from_dps_file(self, fname: str, line_number: int):
        handle = open(fname)
        line = handle.readlines()[line_number]
        positions = [float(x) for x in line.strip().split()]
        self._key_points[0] = np.array(positions)

    def init_from_string(self, patch_data: dict[str, str]):
        for key, val in patch_data.items():
            if key == "id":
                try:
                    self._type = int(val)
                except ValueError:
                    self._type = int(val.split('_')[1])
            if key == "color":
                self._color = int(val)
            elif key == "a1":
                x, y, z = [float(g) for g in val.split(',')]
                self.set_a1(np.array([x, y, z]))
            elif key == "a2":
                x, y, z = [float(g) for g in val.split(',')]
                self.set_a2(np.array([x, y, z]))
            elif key == "position":
                x, y, z = [float(g) for g in val.split(',')]
                self.set_position(np.array([x, y, z]))
            elif key == "strength":
                self.set_strength(float(val))

    def can_bind(self, other: BasePatchType) -> bool:
        if abs(self.color()) > 20:
            return self.color() == -other.color()
        else:
            return self.color() == other.color()

    def __str__(self) -> str:
        return f"Patch type {self.get_id()} with color {self.color()} and strength {self.strength()} in position {self.position()}"

    def has_torsion(self):
        return self.num_key_points() == 2

    def __eq__(self, other: PLPatch) -> bool:
        # check IDs
        if self.get_id() != other.get_id():
            return False
        # check colors
        if self.color() != other.color():
            return False
        # check torsion (just for fun)
        if self.has_torsion() != other.has_torsion():
            return False
        if not np.allclose(self.a1(), other.a1()):
            return False
        if not np.allclose(self.a3(), other.a3()):
            return False
        if not np.allclose(self.position(), other.position()):
            return False
        if not self.strength() == other.strength():
            return False
        return True

    def __ne__(self, other: PLPatch):
        return not (self == other)


def assign_colors_one_to_one(interaction_matrix: InteractionMatrix, patches: list[PLPatch]) -> tuple[InteractionMatrix, list[int]]:
    """
    given an interaction matrix tuned to patch IDs and a list of patches indexed by IDs, constructs
    an assignment a color to each patch and an 1:1 opposite-type color interaction matrix
    todo: make this a PLParticleSet class method?
    Returns:
        a tuple where the first element is an interaction matrix for the new color set, and the second element is
        a list of colors to assign to the patches
    """
    num_patches = interaction_matrix.num_colors()
    assert all([patch.get_id() == i for i, patch in enumerate(patches)])
    assert num_patches == len(patches), f"Size of patch interaction matrix {num_patches} is not compatible " \
                                        f"with length of patches array {len(patches)}!"

    # colors = np.zeros(num_patches, dtype=int)  # Color 0 means uncolored
    patch_colors: list[int] = [None for _ in patches]
    # depth-first search which assigns colors
    def dfs(patch_id: int, color: int):
        if patch_colors[patch_id] != 0 and patch_colors[patch_id] is not None:
            return patch_colors[patch_id] == color
        patch_colors[patch_id] = color
        for neighbor in range(num_patches):
            if interaction_matrix[patch_id, neighbor] > 0:
                if not dfs(neighbor, -color):
                    return False
        return True

    for patch in range(num_patches):
        if (patch_colors[patch] == 0 or patch_colors[patch] is None) and not dfs(patch, 1):
            raise ValueError("Cannot assign colors such that interacting patches add to zero")
    return make_int_mat_for_colors(max(patch_colors)), patch_colors

def assign_colors(interaction_matrix: InteractionMatrix, patches: list[PLPatch]) -> tuple[InteractionMatrix, list[int]]:
    """
    assigns colors, somehow
    the maximum solution for this problem is to give each patch a unique color, but we would ideally
    like to avoid this
    taking suggestions for algorithms
    """
    # first we try 1:1 color assignments
    try:
        return assign_colors_one_to_one(interaction_matrix, patches)
    except ValueError as e:
        if not str(e).startswith("Cannot"):
            raise e
    # dump approach: litersally no thought
    return interaction_matrix, list(range(len(patches)))

    # TODO: SMART VERSION
    # first we get a graph repr of the interaction matrix
    # G: nx.Graph = interaction_matrix.graph()
    # # each connected component  in the graph is an orthogonal interaction behavior
    # for cc in nx.connected_components(G):
    #     # if connected component is one node
    #     if len(cc.nodes) == 1:
    #         # should always be one edge, connecting to itself
    #         if len(cc.edges) == 1:
    #
    #         else:
    #             # presumably no edges, color does nothing, idk what to actually do here




    # finally, return the interaction matrix for our new color set, along with a list of colors for each patch

def make_interaction_matrix(patches: list[PLPatch]) -> InteractionMatrix:
    intmatrix = dict()
    # we do want these to contain both (i,j) and (j,i)
    for p1 in patches:
        for p2 in patches:
            if p1.color() == -p2.color():
                intmatrix[(p1.color(), p2.color())] = math.sqrt(p1.strength() * p2.strength())
    return InteractionMatrix(interactions=intmatrix)
