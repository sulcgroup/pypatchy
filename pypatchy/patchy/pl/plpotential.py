from __future__ import annotations

import itertools
import math
from abc import ABC, abstractmethod
from typing import Union

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

"""
Lorenzo's excluded-volume potential
TODO: find paper this was based on!
"""
class PLLRExclVolPotential(PLPotential):

    __particle_radius: float = 0.5 # TODO: make settable!
    # repulsive radial cutoff???
    # TODO: make this dependant on particle radii!!!
    __rep_rcut: float = 2 ** (1./6.)
    # the above, squared
    __rep_sqr_rcut: float = __rep_rcut ** 2
    __spherical_attraction_strength: float
    __spherical_E_cut: float
    __sqr_spherical_rcut: float
    __epsilon: float

    def __init__(self,
                 rmax: float,
                 epsilon: float = 1.,
                 spherical_attr_strength=0.,
                 spherical_E_cut=0.,
                 sqr_spherical_rcut=0.):
        super().__init__(rmax)
        self.__epsilon = epsilon
        self.__spherical_attraction_strength = spherical_attr_strength
        self.__spherical_E_cut = spherical_E_cut
        self.__sqr_spherical_rcut = sqr_spherical_rcut

    def rep_sqr_rcut(self, particle_radius_sum: float):
        assert particle_radius_sum == 1. # todo: custom radius
        return self.__rep_sqr_rcut

    def epsilon(self) -> float:
        return self.__epsilon


    """
    direct copy of DetailedPatchySwapInteraction::_spherical_patchy_two_body in oxDNA
    """
    def energy(self,
               box: np.ndarray,  # for periodic boundry conditions
               p1: PLPatchyParticle,
               p2: PLPatchyParticle) -> float:
        sqr_r = periodic_dist_sqrd(box, p1.position(), p2.position())
        energy = 0.

        # if distance between particle centers is greater than the maximum,
        # it's
        if sqr_r > self.rmax_sqr():
            return 0.

        rep_sqr_rcut = self.rep_sqr_rcut(p1.radius() + p2.radius())
        if sqr_r < rep_sqr_rcut or (sqr_r < self.__sqr_spherical_rcut and self.__spherical_attraction_strength > 0.0):
            ir2 = 1.0 / sqr_r
            # lennard-jones part? partial?
            # lj_part evaluates to (sigma / r) ** 6 from the LJ 12-6
            lj_part = ir2 * ir2 * ir2
            if sqr_r < rep_sqr_rcut:
                # i have not been using spherical attraction, idk what it does
                spherical_part = self.__spherical_attraction_strength + self.__spherical_E_cut
                # Lorenzo's version has epsilon hardcoded to 1.0, so it's a bit hard to tell if
                # the spherical part belongs inside or outside the parentheses
                energy = self.epsilon() * (4 * (lj_part ** 2 - lj_part) + 1.0 - spherical_part)
            else:
                if sqr_r < self.__sqr_spherical_rcut and self.__spherical_attraction_strength > 0.0:
                    energy = 4 * self.__spherical_attraction_strength * (lj_part ** 2 - lj_part) - self._spherical_E_cut

        return energy

class PLLRPatchyBond:
    pass
class PLLRPatchyPotential(PLPotential):
    # it's extremly unclear what this is
    __sigma_ss: float
    __rcut_ss: float
    # square of patchy interaction distance cutoff
    # two patches with a square-distance greater than this cannot interact
    __sqr_patch_rcut: float
    __interaction_matrix: dict[tuple[int, int], float]

    # patchy interaction params
    __A_part: float
    __B_part: float

    __no_three_body: bool
    # used for three-body interaction
    __lambda: float

    def __init__(self, rmax: float, interaction_matrix: dict[tuple[int, int], float],
                 sigma_ss: float = 0.4):
        super().__init__(rmax)
        self.__interaction_matrix = interaction_matrix
        self.__sigma_ss = sigma_ss
        self.__rcut_ss = 1.5 * sigma_ss
        self.__sqr_patch_rcut = self.__rcut_ss ** 2
        # no idea what these next few lines mean, i lifted them directly from DetailedPatchySwapInteraction::init()
        B_ss = 1. / (1. + 4. * (1. - self.__rcut_ss / self.__sigma_ss) ** 2)
        # a_part evaluates to 2 * e**2 for some reason
        self.__A_part = -1. / (B_ss - 1.) / math.exp(1. / (1. - self.__rcut_ss / self.__sigma_ss))
        self.__B_part = B_ss * self.__sigma_ss ** 4

        self.__no_three_body = False

    def rcut_ss(self):
        return self.__rcut_ss

    def sigma_ss(self):
        return self.__sigma_ss

    def A_part(self) -> float:
        return self.__A_part

    def B_part(self) -> float:
        return self.__B_part;

    def sqr_patch_rcut(self):
        return self.__sqr_patch_rcut

    def is_three_body(self):
        return not self.__no_three_body

    def color_pair_interaction(self, c1: int, c2: int):
        if c1 > c2:
            return self.color_pair_interaction(c2, c1)
        else:
            return self.__interaction_matrix[(c1, c2)]

    def energy(self,
               box: np.ndarray,  # for periodic boundry conditions
               p: PLPatchyParticle,
               q: PLPatchyParticle) -> float:
        computed_r = periodic_dist_sqrt_vec(box, p.position(), q.position())
        sqr_r = np.dot(computed_r, computed_r)
        if sqr_r > self.rmax_sqr():
            return 0.0

        # TODO: compute patch pairs more efficiently, like i did in the FR version?
        energy = 0.0
        for (p_patch, q_patch) in itertools.product(p.patches(), q.patches()):
            epsilon = self.color_pair_interaction(p_patch.color(), q_patch.color())

            # interaction energy calculates much faster so check it first
            if epsilon != 0.0:
                r_patch_sqr = periodic_dist_sqrd(box,
                                                 p.patch_position(p_patch),
                                                 q.patch_position(q_patch))
                if r_patch_sqr < self.sqr_patch_rcut():

                    # compute actual distance between patches
                    r_p = r_patch_sqr ** 0.5
                    # compute
                    exp_part = math.exp(self.sigma_ss() / (r_p - self.rcut_ss()))
                    tmp_energy = epsilon * self.A_part() * exp_part * (self.B_part() / r_patch_sqr ** 2 - 1.0)

                    energy += tmp_energy

                    tb_energy = epsilon if r_p < self.sigma_ss() else -tmp_energy

                    # p_bond = PLLRPatchyBond(q, r_p, p_patch, q_patch, tb_energy)
                    # q_bond = PLLRPatchyBond(p, r_p, q_patch, p_patch, tb_energy)
                    #
                    # if self.is_three_body():
                    #     energy += self._three_body(p, p_bond)
                    #     energy += self._three_body(q, q_bond)

        return energy

    @staticmethod
    def make_interaction_matrix(patches: list[PLPatch]) -> dict[tuple[int, int], float]:
        intmatrix = dict()
        # we do want these to contain both (i,j) and (j,i)
        for p1 in patches:
            for p2 in patches:
                if p1.color() == -p2.color():
                    intmatrix[(p1.color(), p2.color())] = math.sqrt(p1.strength() * p2.strength())
        return intmatrix


"""
Flavio's excluded volume potential
eqn. 9 in https://pubs.acs.org/doi/10.1021/acsnano.2c09677
"""
class PLFRExclVolPotential(PLPotential):
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


"""
non-torsional potential from https://pubs.acs.org/doi/10.1021/acsnano.2c09677
"""
class PLFRPatchyPotential(PLPotential):
    #
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
        # this is harder than i initially thought
        # step one:
        patchy_rsqrds = np.zeros(shape=[p1.num_patches(), p2.num_patches()])

        for (i, p1patch), (j, p2patch) in itertools.product(enumerate(p1.patches()),
                                                            enumerate(p2.patches())):
            patch1_pos: np.ndarray = p1.patch_position(p1patch)
            patch2_pos: np.ndarray = p2.patch_position(p2patch)
            dist = patch1_pos - patch2_pos
            patchy_rsqrds[i, j] = dist @ dist.T

        # this next bit is from chatGPT so it may not be well optimized
        # Get indices of the patch pairs with minimum distances
        patch_pairs = []
        used_p1_patches = np.zeros(p1.num_patches(), dtype=bool)
        used_p2_patches = np.zeros(p2.num_patches(), dtype=bool)

        sorted_indices = np.argsort(patchy_rsqrds, axis=None)  # Flatten and sort distances
        rsqrds = np.ravel(patchy_rsqrds)
        for idx in sorted_indices:
            i, j = np.unravel_index(idx, patchy_rsqrds.shape)
            if not used_p1_patches[i] and not used_p2_patches[j]:
                patch_pairs.append((i, j, rsqrds[idx]))
                used_p1_patches[i] = True
                used_p2_patches[j] = True
                if np.all(used_p1_patches) or np.all(used_p2_patches):
                    break  # Stop if all patches are used
        assert len(patch_pairs) == min(p1.num_patches(), p2.num_patches())
        for i, j, rsqrd in patch_pairs:
            e += self.fr_energy_2_patch(box, p1, p1.patch(i), p2, p2.patch(j))
        return e

    """
    flavio energy two patch point potential evangelion
    """
    def fr_energy_2_patch(self,
                          box: np.ndarray,  # for periodic boundry conditions
                          particle1: PLPatchyParticle, patch1: PLPatch,
                          particle2: PLPatchyParticle, patch2: PLPatch,
                          rsqr: Union[None, float] = None) -> float:
        if rsqr is None:
            # there's DEFINATELY a way to simplify this
            patch1_pos: np.ndarray = particle1.patch_position(patch1)
            patch2_pos: np.ndarray = particle2.patch_position(patch2)
            dist = patch1_pos - patch2_pos
            rsqr = dist @ dist.T
        if rsqr > self.rmax_sqr():
            return 0
        # check if patches are complimentary
        if patch1.can_bind(patch2):
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
    delta = periodic_dist_sqrt_vec(box, p1, p2)
    dist_sqrd = np.dot(delta, delta)
    return dist_sqrd

"""
computes the displacement vector between two particles, accounting for periodic boundry
conditions
"""
def periodic_dist_sqrt_vec(box: np.ndarray, p1: np.ndarray, p2: np.ndarray):
    delta = p1 - p2  # Initial difference vector between the points
    for i in range(len(box)):
        if box[i] > 0:  # Check if the dimension is periodic
            # Apply PBC adjustment only if the dimension is periodic
            delta[i] -= box[i] * np.round(delta[i] / box[i])
    return delta

class PLFRTorsionalPatchyPotential(PLFRPatchyPotential):
    # torsional potential from https://pubs.acs.org/doi/10.1021/acsnano.2c09677
    def fr_energy_2_patch(self,
                          box: np.ndarray,  # for periodic boundry conditions
                          particle1: PLPatchyParticle, patch1: PLPatch,
                          particle2: PLPatchyParticle, patch2: PLPatch,
                          rsqrd: Union[None, float] = None) -> float:
        # begin with the non-torsional potential
        e = super().fr_energy_2_patch(box, particle1, patch1, particle2, patch2)
        # TODO

