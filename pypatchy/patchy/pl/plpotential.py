from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np

from .plparticle import PLPatchyParticle
from .plpatch import PLPatch


# TODO: forces
class PLPotential(ABC):
    """
    abstract base class for interaction potential between two patchy particles
    """

    __rmax: float  # max interaction radius
    __rmax_sqr: float  # cache this

    def __init__(self, rmax: float):
        self.__rmax = rmax
        self.__rmax_sqr = rmax ** 2

    def rmax(self) -> float:
        return self.__rmax

    def rmax_sqr(self) -> float:
        return self.__rmax_sqr

    @abstractmethod
    def energy(self,
               box: np.ndarray,  # for periodic boundry conditions
               p1: PLPatchyParticle,
               p2: PLPatchyParticle,
               ) -> float:
        pass


class PLExclVolPotential(PLPotential):
    # eqn. 9 in https://pubs.acs.org/doi/10.1021/acsnano.2c09677
    # TODO: auto-set quadratic smoothing params to make sure piecewise potential is differentiable

    __rstar: float  # cutoff point for quadratic smoothing of lj potential
    __b: float  # quadratic smoothing param
    __epsilon: float

    def __init__(self, rmax: float, rstar: float, b: float, epsilon: float = 2):
        super().__init__(rmax)
        self.__rstar = rstar
        self.__b = b
        self.__epsilon = epsilon

    def rstar(self) -> float:
        return self.__rstar

    def epsilon(self) -> float:
        return self.__epsilon

    def b(self) -> float:
        return self.__b

    def energy(self,
               box: np.ndarray,  # for periodic boundry conditions
               p1: PLPatchyParticle,
               p2: PLPatchyParticle) -> float:
        """
        Compute energy of excluded volume potential, using a smoothed
        lennard-jones
        Should usually if not always return a positive value, since excl. vol. is energetically
        unfavored (duh)
        """
        tot_radius = p1.radius() + p2.radius()
        # if r > rmax, no interaction
        e = 0.

        p1pos = p1.position()
        p2pos = p2.position()
        r_squared = periodic_dist_sqrd(box, p1.position(), p2.position())
        # if r is greater than max interaction distance, no energy
        # typically this is the sum of the radii
        if r_squared > (self.rmax() + tot_radius) ** 2:
            return e
        # if r is less than the quadratic smoothing cutoff
        sigma = p1.radius() + p2.radius()
        # (sigma^2 / r^2) ^ 3 = (sigma / r) ^ 6
        lj_partial = (sigma ** 2 / r_squared) ** 3
        if r_squared < self.rstar() ** 2:
            # no idea what "rrc" means
            rrc = math.sqrt(r_squared) - self.rstar()
            # amalgam of code
            e = self.epsilon() * self.b() / (sigma ** 2) * rrc ** 2
        # normal lj, which basically means IT'S OVER 9000!!!!!
        else:
            # compute distance between surfaces
            # lennard jones potential
            e = 2 * self.epsilon() * (lj_partial ** 2 - lj_partial)

        return e


class PLPatchyPotential(PLPotential):
    # non-torsional potential from https://pubs.acs.org/doi/10.1021/acsnano.2c09677
    __alpha: float  # patchy alpha potential
    __alpha_sqr: float

    def __init__(self, rmax: float, alpha: float):
        super().__init__(rmax)
        self.__alpha = alpha
        self.__alpha_sqr = alpha ** 2

    def alpha(self) -> float:
        return self.__alpha

    def alpha_sqr(self) -> float:
        return self.__alpha_sqr

    def energy(self,
               box: np.ndarray,  # for periodic boundry conditions

               p1: PLPatchyParticle,
               p2: PLPatchyParticle) -> float:
        e = 0.
        distsqr = periodic_dist_sqrd(box, p1.position(), p2.position())
        if distsqr > (self.rmax() + p1.radius() + p2.radius()) ** 2:
            return e
        # TODO: could optimize more if i cared to
        for p1patch, p2patch in zip(p1.patches(), p2.patches()):
            e += self.energy_2_patch(box, p1, p1patch, p2, p2patch)
        return e

    def energy_2_patch(self,
                       box: np.ndarray,  # for periodic boundry conditions
                       particle1: PLPatchyParticle, patch1: PLPatch,
                       particle2: PLPatchyParticle, patch2: PLPatch) -> float:
        # there's DEFINATELY a way to simplify this
        patch1_pos: np.ndarray = particle1.patch_position(patch1)
        patch2_pos: np.ndarray = particle2.patch_position(patch2)
        dist = patch1_pos - patch2_pos
        rsqr = dist @ dist.T
        if rsqr > self.rmax_sqr():
            return 0
        # check if patches are complimentary
        if patch1.can_bind(patch1):
            # check binding geometry
            # (warning: sus)
            # todo: verify that this behavior is correct!
            e = -1.001 ** ((-rsqr / self.alpha_sqr()) ** 5)  # plus a constant, but constant = 0 i think?
            return e
        return 0


def periodic_dist_sqrd(box: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Computes the squared distance between p1 and p2 with periodic boundaries specified by box.
    If a dimension in `box` is zero, that dimension is treated as non-periodic.
    """
    delta = p1 - p2  # Initial difference vector between the points
    for i in range(len(box)):
        if box[i] > 0:  # Check if the dimension is periodic
            # Apply PBC adjustment only if the dimension is periodic
            delta[i] -= box[i] * np.round(delta[i] / box[i])

    # Calculate and return the squared distance considering the adjusted differences
    dist_sqrd = np.dot(delta, delta)
    return dist_sqrd


class PLPTorsionalPatchyPotential(PLPatchyPotential):
    # torsional potential from https://pubs.acs.org/doi/10.1021/acsnano.2c09677
    def energy_2_patch(self,
                       box: np.ndarray,  # for periodic boundry conditions
                       particle1: PLPatchyParticle, patch1: PLPatch,
                       particle2: PLPatchyParticle, patch2: PLPatch) -> float:
        # begin with the non-torsional potential
        e = super().energy_2_patch(box, particle1, patch1, particle2, patch2)
        # TODO

