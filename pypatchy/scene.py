from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from typing import Iterable, Generator

from oxDNA_analysis_tools.UTILS.data_structures import Configuration

from pypatchy.patchy_base_particle import PatchyBaseParticle, BaseParticleSet, BasePatchType


class Scene(ABC):
    _particles: list[PatchyBaseParticle]

    def __init__(self):
        self._particles = []

    def particles(self) -> list[PatchyBaseParticle]:
        """
        Returns a list of individual particles in this scene
        """
        return self._particles

    def num_particles(self) -> int:
        return len(self._particles)

    def particle_type_counts(self) -> dict[int, int]:
        counts = {p.type_id(): 0 for p in self.particle_types().particles()}
        for particle in self.particles():
            counts[particle.get_type()] += 1
        return counts

    def add_particle(self, p: PatchyBaseParticle):
        self._particles.append(p)

    def add_particles(self, particles: Iterable[PatchyBaseParticle]):
        self._particles.extend(particles)

    def get_particle(self, pidx: int) -> PatchyBaseParticle:
        assert self._particles[pidx].get_id() == pidx
        return self._particles[pidx]

    @abstractmethod
    def num_particle_types(self) -> int:
        pass

    @abstractmethod
    def particle_types(self) -> BaseParticleSet:
        pass

    @abstractmethod
    def set_particle_types(self, ptypes: BaseParticleSet):
        pass

    @abstractmethod
    def get_conf(self) -> Configuration:
        pass

    @abstractmethod
    def particles_bound(self, p1: PatchyBaseParticle, p2: PatchyBaseParticle) -> bool:
        pass

    def iter_bound_particles(self) -> Generator[tuple[PatchyBaseParticle, PatchyBaseParticle]]:
        for p1, p2 in itertools.combinations(self.particles(), 2):
            if self.particles_bound(p1, p2):
                yield p1, p2

    def iter_binding_patches(self, particle1: PatchyBaseParticle, particle2: PatchyBaseParticle) -> Generator[tuple[BasePatchType, BasePatchType], None, None]:
        for p1, p2 in itertools.product(particle1.patches(), particle2.patches()):
            if self.patches_bound(particle1, p1, particle2, p2):
                yield p1, p2

    @abstractmethod
    def patches_bound(self,
                      particle1: PatchyBaseParticle,
                      p1: BasePatchType,
                      particle2: PatchyBaseParticle,
                      p2: BasePatchType) -> bool:
        pass
