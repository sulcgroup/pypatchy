from __future__ import annotations

from typing import Union

import numpy as np
from scipy.spatial.transform import Rotation

from pypatchy.patchy_base_particle import PatchyBaseParticle
from pypatchy.polycubeutil.polycubesRule import PolycubeRuleCubeType, RULE_ORDER, PolycubesPatch
from pypatchy.util import getRotations, to_xyz


class PolycubesStructureCube(PatchyBaseParticle):
    _type_cube: PolycubeRuleCubeType
    _rot: Rotation
    _state: list[bool]

    def __init__(self,
                 uid: int,
                 cube_position: np.ndarray,
                 cube_rotation: Union[np.ndarray, int],
                 cube_type: PolycubeRuleCubeType,
                 state: list[bool] = [True]):
        """
        Parameters:
            uid (int): a unique identifier for this cube
            cube_position (np.ndarray): the position of the particle, as a 3-length integer vector
            cube_rotation (np.ndarray, int): a quaternion or integer represtation of cube rotation
        """
        super(PolycubesStructureCube, self).__init__(uid, cube_type.type_id(), cube_position)
        if isinstance(cube_rotation, np.ndarray) and len(cube_rotation) == 4:
            # if rotation hsa been passed as a quaternion
            self._rot = Rotation.from_quat(cube_rotation)
        elif isinstance(cube_rotation, int):
            # if rotation is a an integer, representing an index in rotation enumeration
            self._rot = Rotation.from_matrix(getRotations()[cube_rotation])
        else:
            raise TypeError("Rotation matrices or whatever not supported yet.")
        self._state = state
        self._type_cube = cube_type

    def get_cube_type(self) -> PolycubeRuleCubeType:
        return self._type_cube

    def rotation(self) -> Rotation:
        return self._rot

    def rot_mat(self) -> np.ndarray:
        return self.rotation().as_matrix()

    def rotate(self, rotation: Rotation):
        """
        todo: more param options
        """
        self._rot = self._rot * rotation

    def typedir(self, direction: Union[int, np.ndarray]) -> np.ndarray:
        """
        Converts the global-space direction into a local-space direction
        """
        if isinstance(direction, int):  # if the arguement is provided as an index in RULE_ORDER
            direction = RULE_ORDER[direction]
        return self.rotation().inv().apply(direction).round()

    def has_patch(self, direction: Union[int, np.ndarray]) -> bool:
        return self.get_cube_type().has_patch(self.typedir(direction))

    def patch(self, direction: Union[int, np.ndarray]) -> PolycubesPatch:
        return self.get_cube_type().patch(self.typedir(direction))

    def num_patches(self) -> int:
        return self.get_cube_type().num_patches()

    def state(self, i=None):
        if i is None:
            return self._state
        else:
            assert abs(i) < len(self._state)
            if i < 0:
                return not self._state[-i]
            else:
                return self._state[i]

    def patches(self) -> list[PolycubesPatch]:
        """
        Returns the patches on this polycube, rotated correctly
        """
        return [p.rotate(self.rot_mat()) for p in self.get_cube_type().patches()]

    def toJSON(self) -> dict:
        return {
            "position": to_xyz(self.position()),
            "rotation": {
                k: float(v) for k, v in zip(
                    ["x", "y", "z", "w"], # as_quat uses xyzw format
                    self.rotation().as_quat())
            },
            "state": [True, *([False] * 12)],  # ignore for now TODO come back to
            "type": self.get_cube_type().type_id()
        }
