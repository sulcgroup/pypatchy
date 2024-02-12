from __future__ import annotations

from typing import Union

import numpy as np

from pypatchy.patchy_base_particle import BasePatchType


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
