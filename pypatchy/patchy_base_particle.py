from __future__ import annotations

import abc
import itertools
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Union, Any, Iterable, Generator

import numpy as np
from oxDNA_analysis_tools.UTILS.data_structures import Configuration, TopInfo


class PatchyBaseParticleType(ABC):
    """
    Lowest possible level base patchy particle
    """
    _type_id: int

    _patches: list[BasePatchType]

    def __init__(self, uid: int, patches: list[BasePatchType]):
        """
        Constructs particle
        """
        self._type_id = uid
        self._patches = patches

    def type_id(self) -> int:
        """
        Returns:
            the particle's type id
        """
        return self._type_id

    def set_id(self, new_id: int):
        """
        Sets particle id
        """
        self._type_id = new_id

    @abstractmethod
    def name(self) -> str:
        pass

    def num_patches(self) -> int:
        """
        Returns:
            number of patches on the particle
        """
        return len(self._patches)

    def patches(self) -> list[Any]:
        """
        Returns:
            list of particle patches
        """
        return self._patches

    def patch(self, patch_idx: Union[int, np.ndarray]) -> Any:
        """
        Accessor for patch

        Args:
            patch_idx: int index of patch, or 3-length vector for patch position

        Returns:
            patch object specified by `patch_index`
        """
        if isinstance(patch_idx, int):
            assert -1 < patch_idx < self.num_patches(), "Index out of bounds"
            return self._patches[patch_idx]
        else:
            assert isinstance(patch_idx, np.ndarray), "Index is not an int or an np array"
            assert any([np.linalg.norm(p.position() - patch_idx) < 1e-10 for p in self._patches])
            return [p for p in self._patches if np.linalg.norm(p.position() - patch_idx) < 1e-10][0]

    @abstractmethod
    def radius(self, normal: np.ndarray = np.zeros(shape=(3,))) -> float:
        """
        Returns the radius of the particle along the provided normal from the particle centroid
        """
        pass


class BasePatchType(ABC):
    """
    Lowest possible level base patch
    Should patch ID be handled here? leaning no
    """
    _uid: int
    # position is deliberately not defined here
    # points in 3-space that are important for this particle
    # this var is made for easily working with additional points such as position, a1, a2, etc.
    _key_points: list[np.ndarray]
    _color: Any  # leave room for fwd-compatibility with specific sequences, etc.

    def __init__(self, uid: int, color: Any):
        self._uid = uid
        self._color = color
        # skip defining _key_points here and do it in superconstructor

    def get_id(self) -> int:
        return self._uid

    def set_id(self, new_id: int):
        """
        Sets patch ID.
        Use with caution! Problems will arise if the patch's ID
        doesn't match its index in a PolycubesRule object!!!
        """
        self._uid = new_id

    def add_key_point(self, pt: np.ndarray) -> int:
        self._key_points.append(pt)
        return len(self._key_points) - 1

    def get_key_point(self, idx: int) -> np.ndarray:
        return self._key_points[idx]

    def num_key_points(self) -> int:
        return len(self._key_points)

    @abstractmethod
    def position(self) -> np.ndarray:
        pass  # neeed to make this method abstract for compatibility with different patch types

    def color(self) -> Any:
        return self._color

    @abstractmethod
    def colornum(self) -> int:
        pass

    def set_color(self, value: int):
        self._color = value

    @abstractmethod
    def can_bind(self, other: BasePatchType):
        pass


class BaseParticleSet:
    """
    Base class for patches to inherit from
    """
    _particle_types: list
    _patch_types: list

    def __init__(self, particles: Union[list[PatchyBaseParticleType, None]] = None):
        if particles is None:
            self._particle_types = []
        else:
            self._particle_types = particles
        self._patch_types = []

    def particles(self):
        return self._particle_types

    def particle(self, idx: Union[int, str]):
        if isinstance(idx, int):
            assert -1 < idx < self.num_particle_types()
            return self._particle_types[idx]
        else:
            for particle in self.particles():
                if particle.name() == idx:
                    return particle
            raise Exception(f"No such particle {idx}")

    def num_particle_types(self):
        return len(self._particle_types)

    def patches(self):
        return self._patch_types

    def patch(self, i: int) -> Any:
        return self._patch_types[i]

    def num_patches(self):
        return len(self._patch_types)

    def add_particles(self, particles: Iterable[PatchyBaseParticleType]):
        for particle in particles:
            self.add_particle(particle)

    def add_particle(self, particle: PatchyBaseParticleType):
        # if particle.type_id() is not None and particle.type_id() != -1:
        #     assert particle.type_id() == self.num_particle_types()
        # else:
        particle.set_id(self.num_particle_types())
        self._particle_types.append(particle)
        for patch in particle.patches():
            if patch not in self.patches():
                if patch.get_id() != self.num_patches():
                    pp = deepcopy(patch)
                    pp.set_id(self.num_patches())
                    self.add_patch(pp)
                else:
                    self.add_patch(patch)

    def add_patches(self, patches):
        self._patch_types.extend(patches)

    def add_patch(self, patch):
        self._patch_types.append(patch)

    def __len__(self):
        return len(self.particles())


class PatchyBaseParticle(ABC):
    """Abstract base class for particle instances"""
    _uid: int
    _type_id: int
    _position: np.array

    def __init__(self, uid: int, type_id: int, position: np.ndarray):
        self._uid = uid
        self._type_id = type_id
        self._position = position

    def get_type(self) -> int:
        """
        Returns id number of particle's type
        """
        return self._type_id

    def get_id(self) -> int:
        return self._uid

    @abstractmethod
    def rotation(self) -> Any:
        """
        return type depends on implementation
        """
        pass

    def position(self) -> np.ndarray:
        return self._position

    def set_position(self, newval: np.ndarray):
        """
        suggest not using directly; Scene objects should implement their own
        set_position methods with parameter validity checking
        """
        self._position = newval

    @abstractmethod
    def patches(self):
        """
        Accessor for particle patches. implementation-dependant.
        """
        pass

    @abstractmethod
    def patch(self, idx: int) -> BasePatchType:
        """
        Accessor for particle patch. Inpl-dependant.
        Will generally somehow redirect to PatchyBaseParticleType.patch
        Classes which extend both PatchyBaseParticleType and PatchyBaseParticle can skip manual implementation
        The default (pydoc) method signature here takes an int indexer but other options are permitted
        when overriding
        """
        pass

    @abstractmethod
    def num_patches(self) -> int:
        """
        Similar conceptually to PatchyBaseParticle::patch
        (see above)
        """
        pass

    def translate(self, translation_vector: np.ndarray):
        self._position += translation_vector

    @abstractmethod
    def rotate(self, rotation: Any):
        pass


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

    @abstractmethod
    def num_particle_types(self) -> int:
        pass

    @abstractmethod
    def particle_types(self) -> BaseParticleSet:
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
