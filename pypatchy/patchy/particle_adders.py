"""
Collected dataclasses to add particles to patchy simulations
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator

from pypatchy.patchy.pl.plscene import PLPSimulation
from pypatchy.polycubeutil.polycube_structure import PolycubeStructure, load_polycube


class StageParticleAdder:
    """
    Class to store inforation about how particles are added during staged assembly
    todo: make ABC with particle counts here
    """


@dataclass
class RandParticleAdder(StageParticleAdder):
    density: float = field()


class FromPolycubeAdder(StageParticleAdder):
    """
    class that adds particles to
    """
    @dataclass
    class AddablePolycube:
        # path to polycube file to add
        polycube_file_path: PolycubeStructure = field()
        n_copies: int = field(default=1)
        patch_distance_multiplier: float = field(default=1.)

        def __post_init__(self):
            if isinstance(self.polycube_file_path, str):
                polycube_path = Path(self.polycube_file_path).expanduser()
            else:
                polycube_path = self.polycube_file_path
            self.polycube_file_path = load_polycube(polycube_path)

    polycubes: list[AddablePolycube]
    density: float

    def __init__(self, *args: dict, **kwargs):
        assert len(args) == 1 and len(kwargs) == 1
        self.polycubes = [self.AddablePolycube(**arg) for arg in args]
        self.density = kwargs["density"]

    def get_particle_counts(self) -> list[int]:
        return list(itertools.chain.from_iterable([
            itertools.chain.from_iterable([
                [ct.type_id()] * pc.polycube_file_path.num_cubes_of_type(ct.type_id()) * pc.n_copies
                for ct in pc.polycube_file_path.particle_types()
            ])
            for pc in self.polycubes
        ]))

    def iter_polycubes(self) -> Generator[FromPolycubeAdder.AddablePolycube]:
        for pc in self.polycubes:
            for _ in range(pc.n_copies):
                yield pc

# TODO: write this one!
class FromConfAdder(StageParticleAdder):
    miniconfs: list[PLPSimulation]

    def get_particle_counts(self) -> list[int]:
        return list(itertools.chain.from_iterable([
            [
                p.type_id()
                for p in conf.particle_types()
            ]
            for conf in self.miniconfs
        ]))