import copy
import json
from dataclasses import dataclass
from enum import Enum

import numpy as np

from pypatchy.patchy.plpatchy import PLPSimulation, PLPatchyParticle
from pypatchy.polycubeutil.structure import PolycubeStructure
from pypatchy.util import get_input_dir


class Stage:
    _stage_num: int
    _time: int
    _particles_to_add: list[int]
    _add_method: str

    def __init__(self, stagenum: int, t: int, particles: list[int],
                 box_size: np.ndarray = np.array((0, 0, 0)),
                 add_method: str = "RANDOM"):
        self._stage_num = stagenum
        self._time = t
        self._particles_to_add = particles
        self._box_size = box_size
        self._add_method = add_method

    def idx(self) -> int:
        return self._stage_num

    def box_size(self) -> np.ndarray:
        return self._box_size

    def set_box_size(self, box_size: np.ndarray):
        self._box_size = box_size

    def particles_to_add(self) -> list[int]:
        return self._particles_to_add

    def get_time(self) -> int:
        return self._time

    def apply(self, scene: PLPSimulation):
        if self._add_method == "RANDOM":
            particles = [copy.deepcopy(scene.particle_types()[i]) for i in self._particles_to_add]
            scene.add_particle_rand_positions(particles)
        elif "=" in self._add_method:
            mode, src = self._add_method.split("=")
            if mode == "from_conf":
                raise Exception("If you're seeing this, this feature hasn't been implemented yet")
            elif mode == "from_polycubes":
                with open(get_input_dir() / src, "r") as f:
                    pc = PolycubeStructure(json.load(f))

