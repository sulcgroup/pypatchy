import copy
import json
from dataclasses import dataclass
from enum import Enum
from typing import Union

import numpy as np

from pypatchy.patchy.plpatchy import PLPSimulation, PLPatchyParticle
from pypatchy.polycubeutil.structure import PolycubeStructure
from pypatchy.util import get_input_dir, append_to_file_name


class Stage:
    # the index (starting from 0) of this stage
    _stage_num: int
    # the name of this stage
    _stage_name: str
    # the step (time) that this stage starts
    _stage_start_time: int
    # a list of PARTICLE TYPE IDS of particles to add
    # the length of this list should be the number of particles to add in
    # this stage and each item is a TYPE ID of a particle type to add.
    _particles_to_add: list[int]
    _add_method: str
    _stage_vars: dict

    def __init__(self,
                 stagenum: int,
                 stagename: str,
                 t: int,
                 particles: list[Union[int, str]],
                 box_size: np.ndarray = np.array((0, 0, 0)),
                 add_method: str = "RANDOM",
                 stage_vars: dict = {},
                 tlen: int = 0,
                 tend: int = 0
                 ):
        assert tlen or tend, "Specify stage length with "
        self._stage_num = stagenum
        self._stage_name = stagename
        self._stage_start_time = t
        self._particles_to_add = particles
        self._box_size = box_size
        self._add_method = add_method
        self._stage_vars = stage_vars
        if tlen:
            self._stage_time_length = tlen
        else:
            self._stage_time_length = tend - t

    def idx(self) -> int:
        return self._stage_num

    def name(self) -> str:
        return self._stage_name

    def box_size(self) -> np.ndarray:
        return self._box_size

    def set_box_size(self, box_size: np.ndarray):
        self._box_size = box_size

    def particles_to_add(self) -> list[int]:
        return self._particles_to_add

    def start_time(self) -> int:
        return self._stage_start_time

    def time_length(self) -> int:
        return self._stage_time_length

    def end_time(self) -> int:
        return self.start_time() + self.time_length()

    def apply(self, scene: PLPSimulation):
        scene.set_box_size(self.box_size())
        assert all(self.box_size()), "Box size hasn't been set!!!"
        if self._add_method == "RANDOM":
            particles = [copy.deepcopy(scene.particle_types().particle(i)) for i in self._particles_to_add]
            scene.add_particle_rand_positions(particles, overlap_min_dist=1)
        elif "=" in self._add_method:
            mode, src = self._add_method.split("=")
            if mode == "from_conf":
                raise Exception("If you're seeing this, this feature hasn't been implemented yet")
            elif mode == "from_polycubes":
                with open(get_input_dir() / src, "r") as f:
                    pc = PolycubeStructure(json.load(f))

    def adjfn(self, file_name: str) -> str:
        if self.idx() > 0:
            return append_to_file_name(file_name, self.name())
        else:
            return file_name

