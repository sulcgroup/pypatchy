from __future__ import annotations

import itertools
import math
from abc import ABC, abstractmethod
from typing import Union

import numpy as np

from .plparticle import PLPatchyParticle
from .plpatch import PLPatch
from ...interaction_matrix import InteractionMatrix


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

class PLPatchyPotential(PLPotential, ABC):
    """
    Abstract base class lass for interaction potential between two patches.
    """

    # PLACEHOLDER: TODO
    _multipatch_behavior: int = 0

    @abstractmethod
    def energy_2_patch(self,
                       box: np.ndarray,
                       particle1: PLPatchyParticle,
                       patch1: PLPatch,
                       particle2: PLPatchyParticle,
                       patch2: PLPatch) -> float:
        pass

class PLLRExclVolPotential(PLPotential):
    """
    Lorenzo's excluded-volume potential
    from 
    """
    __particle_radius: float  # TODO: make settable!
    # repulsive radial cutoff???
    # TODO: make this dependant on particle radii!!!
    __rep_rcut: float
    # the above, squared
    __rep_sqr_rcut: float
    __spherical_attraction_strength: float
    __spherical_E_cut: float
    __sqr_spherical_rcut: float
    __epsilon: float

    def __init__(self,
                 rmax: float,
                 epsilon: float = 1.,
                 spherical_attr_strength=0.,
                 spherical_E_cut=0.,
                 sqr_spherical_rcut=0.,
                 particle_radius=0.5):
        super().__init__(rmax)
        self.__epsilon = epsilon
        self.__spherical_attraction_strength = spherical_attr_strength
        self.__spherical_E_cut = spherical_E_cut
        self.__sqr_spherical_rcut = sqr_spherical_rcut
        self.__particle_radius = particle_radius
        # for default particle radius=0.5, the repulsive r cutoff = 2 ** (1./6.)
        self.__rep_rcut = 2 * particle_radius * 2 ** (1. / 6.)
        self.__rep_sqr_rcut = self.__rep_rcut ** 2

    def rep_sqr_rcut(self, particle_radius_sum: float):
        assert particle_radius_sum == 2 * self.__particle_radius
        # todo: custom radius
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


class PLLRPatchyPotential(PLPatchyPotential, InteractionMatrix):
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

    # used for three-body interaction
    __lambda: float
    
    def __init__(self,
                 rmax: float,
                 interaction_matrix: Union[dict[tuple[int, int], float], InteractionMatrix],
                 sigma_ss: float = 0.4):
        """
        nighmare constructor
        """
        InteractionMatrix.__init__(self, interaction_matrix if type(interaction_matrix) == dict else interaction_matrix.intmap())
        PLPotential.__init__(self, rmax)
        self.__sigma_ss = sigma_ss
        self.__rcut_ss = 1.5 * sigma_ss
        self.__sqr_patch_rcut = self.__rcut_ss ** 2
        # no idea what these next few lines mean, i lifted them directly from DetailedPatchySwapInteraction::init()
        B_ss = 1. / (1. + 4. * (1. - self.__rcut_ss / self.__sigma_ss) ** 2)
        # a_part evaluates to 2 * e**2 for some reason
        self.__A_part = -1. / (B_ss - 1.) / math.exp(1. / (1. - self.__rcut_ss / self.__sigma_ss))
        self.__B_part = B_ss * self.__sigma_ss ** 4

        self._multipatch_behavior = 0

    def rcut_ss(self):
        return self.__rcut_ss

    def sigma_ss(self):
        return self.__sigma_ss

    def A_part(self) -> float:
        return self.__A_part

    def B_part(self) -> float:
        return self.__B_part

    def sqr_patch_rcut(self):
        return self.__sqr_patch_rcut

    def is_three_body(self):
        return not self.__no_three_body

    def energy(self,
               box: np.ndarray,  # for periodic boundry conditions
               p: PLPatchyParticle,
               q: PLPatchyParticle) -> float:
        computed_r = periodic_dist_sqrt_vec(box, p.position(), q.position())
        sqr_r = np.dot(computed_r, computed_r)
        if sqr_r > self.rmax_sqr():
            return 0.0

        assert self._multipatch_behavior ==0, "you have hallucinated a feature again"

        # TODO: compute patch pairs more efficiently, like i did in the FR version?
        energy = 0.0
        for (p_patch, q_patch) in itertools.product(p.patches(), q.patches()):
            pqenergy = self.energy_2_patch(box, p, p_patch, q, q_patch)
            # TODO: SCREAMS: NO MULTIPATCH
            energy += pqenergy

        return energy

    def energy_2_patch(self,
                       box: np.ndarray,
                       particle1: PLPatchyParticle,
                       patch1: PLPatch,
                       particle2: PLPatchyParticle,
                       patch2: PLPatch) -> float:

        epsilon = -self[patch1.color(), patch2.color()]

        # interaction energy calculates much faster so check it first
        if epsilon != 0.0:
            if epsilon > 0: print(
                "It's not entirely unphyiscal to have repulsive-default itneractions but IMHO this is probably a mistake")
            r_patch_sqr = periodic_dist_sqrd(box,
                                             particle1.patch_position(patch1),
                                             particle2.patch_position(patch2))
            if r_patch_sqr < self.sqr_patch_rcut():
                # compute actual distance between patches
                r_p = r_patch_sqr ** 0.5
                # compute
                exp_part = math.exp(self.sigma_ss() / (r_p - self.rcut_ss()))
                return epsilon * self.A_part() * exp_part * (self.B_part() / r_patch_sqr ** 2 - 1.0)
        return 0.


class PLFRExclVolPotential(PLPotential):
    """
    Flavio's excluded volume potential
    eqn. 9 in https://pubs.acs.org/doi/10.1021/acsnano.2c09677
    """
    # TODO: auto-set quadratic smoothing params to make sure piecewise potential is differentiable

    __rstar: float  # cutoff point for quadratic smoothing of lj potential
    __b: float  # quadratic smoothing param
    __epsilon: float

    # NOTE: defautls are r* = 2 * sphere radius * 0.9053 , b = 677.505671539 / sphere radius, rmax = 0.99888 * 2 * sphere radius
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

        r_squared = periodic_dist_sqrd(box, p1.position(), p2.position())
        # if r is greater than max interaction distance, no energy
        # typically this is the sum of the radii
        if r_squared > self.rmax() ** 2:
            return e
        # if r is less than the quadratic smoothing cutoff
        sigma = p1.radius() + p2.radius()
        # (sigma^2 / r^2) ^ 3 = (sigma / r) ^ 6
        # if r^2 > (r*)^2, use quadratic smoothing
        if r_squared > (self.rstar() ** 2):
            # no idea what "rrc" means
            rrc = math.sqrt(r_squared) - self.rmax()

            # amalgam of code
            e = self.epsilon() * self.b() * rrc ** 2
        # normal lj, which basically means IT'S OVER 9000!!!!!
        else:
            lj_partial = (sigma ** 2 / r_squared) ** 3
            # compute distance between surfaces
            # lennard jones potential
            e = 4 * self.epsilon() * (lj_partial ** 2 - lj_partial)

        return e


class PLFRPatchyPotential(PLPatchyPotential):
    """
    non-torsional potential from https://pubs.acs.org/doi/10.1021/acsnano.2c09677
    todo: incorporate plinteractionmatrix
    """
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
        if distsqr > self.rmax_sqr():
            return e
        # this is harder than i initially thought
        # step one:
        # construct empty array which stores distances between pairs of patches
        patchy_rsqrds = np.zeros(shape=[p1.num_patches(), p2.num_patches()])

        # for each possible patch pairing on the particles
        for (i, p1patch), (j, p2patch) in itertools.product(enumerate(p1.patches()),
                                                            enumerate(p2.patches())):
            # compute position of patch 1
            patch1_pos: np.ndarray = p1.patch_position(p1patch)
            patch2_pos: np.ndarray = p2.patch_position(p2patch)
            # compute position of patch 2
            patchy_rsqrds[i,j] = periodic_dist_sqrd(box, patch1_pos, patch2_pos)

        # this next bit is from chatGPT so it may not be well optimized
        # or correct!
        # Get indices of the patch pairs with minimum distances
        patch_pairs = []
        # arr to keep track of which patches are "claimed"
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
            e += self.energy_2_patch(box, p1, p1.patch(i), p2, p2.patch(j), rsqrd)
        # for p1patch, p2patch in itertools.product(p1.patches(),
        #                                                     p2.patches()):
        #     e += self.energy_2_patch(box, p1, p1patch, p2, p2patch)
        return e

    def energy_2_patch(self,
                          box: np.ndarray,  # for periodic boundry conditions
                          particle1: PLPatchyParticle, patch1: PLPatch,
                          particle2: PLPatchyParticle, patch2: PLPatch,
                          rsqr: Union[None, float] = None) -> float:
        """
        flavio energy two patch point potential evangelion
        """
        if rsqr is None:
            # there's DEFINATELY a way to simplify this
            patch1_pos: np.ndarray = particle1.patch_position(patch1)
            patch2_pos: np.ndarray = particle2.patch_position(patch2)
            dist = patch1_pos - patch2_pos
            rsqr = dist @ dist.T
        if rsqr > self.rmax_sqr():
            return 0
        # check if patches are complimentary
        # TODO: replace with an interaction matrix?
        if patch1.can_bind(patch2):
            # check binding geometry
            e = -1.001 * math.exp(-0.5 * (rsqr / self.alpha_sqr()) ** 5)
            # plus a constant, but constant = 0 i think?
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


def periodic_dist_sqrt_vec(box: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """
    computes the displacement vector between two particles, accounting for periodic boundry
    conditions
    """
    delta = p1 - p2  # Initial difference vector between the points
    for i in range(len(box)):
        if box[i] > 0:  # Check if the dimension is periodic
            # Apply PBC adjustment only if the dimension is periodic
            delta[i] -= box[i] * np.round(delta[i] / box[i])
    return delta


def narrow_type(nt: int) -> dict[str, float]:
    """
    values from https://doi.org/10.1021/acsnano.2c09677, fig 9
    """
    if nt == 0:
        return {
            "width": 2.345750,
            "delta": 0.7,
            "delta_c": 3.105590,
            "a": 0.46,
            "b": 0.133855
        }
    elif nt == 1:
        return {
            "width": 0.954736,
            "delta": 0.2555,
            "delta_c": 1.304631,
            "a": 3,
            "b": 0.730694
        }
    elif nt == 2:
        return {
            "width": 0.656996,
            "delta": 0.2555,
            "delta_c": 0.782779,
            "a": 5,
            "b": 2.42282
        }
    elif nt == 3:
        return {
            "width": 0.396613,
            "delta": 0.17555,
            "delta_c": 0.438183,
            "a": 13,
            "b": 8.689492
        }
    elif nt == 4:
        return {
            "width": 0.336622,
            "delta": 0.17555,
            "delta_c": 9.322741,
            "a": 17.65,
            "b": 21.0506
        }
    else:
        raise Exception(f"No narrow type {nt}")


class PLFRTorsionalPatchyPotential(PLFRPatchyPotential):
    # for explaination of torsional params, see https://doi.org/10.1021/acsnano.2c09677, fig 9
    __theta_0: float = 0

    __width: float
    __delta: float
    __delta_c: float
    __a: float
    __b: float

    def __init__(self, rmax: float, alpha: float, **kwargs):
        """

        """
        super().__init__(rmax=rmax, alpha=alpha)
        if "theta_0" in kwargs:
            self.__theta_0 = kwargs["theta_0"]
        # if we pass a narrow type number, use narrow-type params
        if "narrow_type" in kwargs:
            kwargs.update(narrow_type(kwargs["narrow_type"]))
        self.__width = kwargs["width"]
        # todo: calculate dynamically from width? or just replace w/ non piecewise function
        self.__delta = kwargs["delta"]
        self.__delta_c = kwargs["delta_c"]
        self.__a = kwargs["a"]
        self.__b = kwargs["b"]

    # torsional potential from https://pubs.acs.org/doi/10.1021/acsnano.2c09677
    def energy_2_patch(self,
                          box: np.ndarray,  # for periodic boundry conditions
                          particle1: PLPatchyParticle, patch1: PLPatch,
                          particle2: PLPatchyParticle, patch2: PLPatch,
                          rsqrd: Union[None, float] = None) -> float:
        # begin with the non-torsional potential
        e = super().energy_2_patch(box, particle1, patch1, particle2, patch2, rsqrd)
        # don't bother with multipliers for zero energy
        if e < -1e-5:
            # compute distance between patches
            # there's DEFINATELY a way to simplify this
            patch1_pos: np.ndarray = particle1.patch_position(patch1)
            patch2_pos: np.ndarray = particle2.patch_position(patch2)
            # distance vector = particle 1 -> particle 2
            dist: np.ndarray = periodic_dist_sqrt_vec(box,
                                                      particle1.position(),
                                                      particle2.position())
            rsqr: float = dist @ dist.T
            dist_norm = dist / math.sqrt(rsqr)
            # https://pubs.acs.org/doi/10.1021/acsnano.2c09677 equation 2

            # compute angle between patches
            # angle from particle 1 to patch 1
            # compute dot product of particle 1 -> particle 2 . particle 1 -> patch 1
            n = -dist_norm @ (patch1.a1() @ particle1.rotmatrix())
            n = max(-1, min(1, n))
            va = self.v_angmod(math.acos(n))
            if va == 0:
                return 0.
            n = dist_norm @ (patch2.a1() @ particle2.rotmatrix())
            n = max(-1, min(1, n))
            vb = self.v_angmod(math.acos(n))
            if vb == 0:
                return 0.
            # get angle between a3 vectors
            n = (patch1.a2() @ particle1.rotmatrix()) @ (patch2.a2() @ particle2.rotmatrix())
            n = max(-1, min(1, n))
            vt = self.v_angmod(math.acos(n))
            return e * vt * va * vb
        else:
            return e

    def v_angmod(self, theta: float) -> float:
        """
        computes angle-modifier for patchy interaction
        in the paper and the c++ implementation the equation (eqns. 4, 5, 6) is restructured as a differentiable
        piecewise function but for our purposes right now this is unnnessecary

        except i don't actually know the correct equation
        """

        theta -= self.__theta_0
        theta = abs(theta)

        # outer limit of acceptable angles is deltaC
        if theta < self.__delta_c:
            if theta > self.__delta:
                # if we're outside the delta range, use the normal quadratic
                val = self.__b * (self.__delta_c - theta) ** 2
            else:
                # if we're within delta range, use the inverted quadratic,
                val = 1. - self.__a * theta ** 2
            return val
        else:
            return 0.
