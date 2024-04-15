"""
Collected dataclasses to add particles to patchy simulations
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pypatchy.polycubeutil.polycube_structure import PolycubeStructure, load_polycube


class StageParticleAdder:
    """
    Class to store information about how particles are added during staged assembly
    """


@dataclass
class RandParticleAdder(StageParticleAdder):
    density: float = field()


class FromPolycubeAdder(StageParticleAdder):
    @dataclass
    class AddablePolycube:
        # path to polycube file to add
        polycube_file_path: PolycubeStructure = field()
        n_copies: int = field(default=1)

        def __post_init__(self):
            if isinstance(self.polycube_file_path, str):
                polycube_path = Path(self.polycube_file_path).expanduser()
            else:
                polycube_path = self.polycube_file_path
            self.polycube_file_path = load_polycube(polycube_path)

    polycubes: list[AddablePolycube]
    density: float

    def __init__(self, *args: dict, **kwargs):
        assert len(args) == 1 and len(kwargs) == 0
        self.polycubes = [self.AddablePolycube(**arg) for arg in args]
        self.density = kwargs["density"]


# TODO: write this one!
class FromConfAdder(StageParticleAdder):
    miniconfs: Any
