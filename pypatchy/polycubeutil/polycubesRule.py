from __future__ import annotations

import copy
import itertools
import math
import typing
from typing import Union

import drawsvg
import networkx as nx
import numpy as np

from .effect import Effect, StringConditionalEffect, DynamicEffect, EnvironmentalEffect, EFFECT_CLASSES
from ..util import rotAroundAxis, selectColor, powerset
import re

from pypatchy.util import rotation_matrix, from_xyz, to_xyz, getSignedAngle
from pypatchy.patchy_base_particle import PatchyBaseParticleType, BasePatchType, BaseParticleSet

import drawsvg as draw

FACE_NAMES = ("left", "right", "bottom", "top", "back", "front")
RULE_ORDER = (
    np.array((-1, 0, 0)),
    np.array((1, 0, 0)),
    np.array((0, -1, 0)),
    np.array((0, 1, 0)),
    np.array((0, 0, -1)),
    np.array((0, 0, 1))
)

FACE_SVG_COORDS = [
    (0, 2), #left
    (2, 2), #right
    (1, 0), #bottom
    (1, 2), #top
    (1, 3), #back
    (1, 1) #front
]

FACE_SQUARE_W = 64
FACE_FIG_PAD = 12

# old diridx - bad/buggy
# def diridx(a):
#     return np.all(np.array(RULE_ORDER) == np.array(list(a))[np.newaxis, :], axis=1).nonzero()[0][0]

# new diridx - written by chatgpt
def diridx(a: np.ndarray) -> int:
    for idx, rule in enumerate(RULE_ORDER):
        if np.array_equal(a, rule):
            return idx
    assert False, f"{a} is not in RULE_ORDER"


def rdir(d: int) -> int:
    if d % 2 == 0:
        return (d + 1) % 6
    else:
        return (d - 1) % 6


def get_orientation(face_idx: int,
                    ori_idx: int) -> np.ndarray:
    zero_rot_idx = (face_idx + 2) % len(RULE_ORDER)  # feels like it should be +4 but trust me bro
    zero_rotation = RULE_ORDER[zero_rot_idx]
    rot_mat = rotation_matrix(RULE_ORDER[face_idx], ori_idx * math.pi / 2)
    return (rot_mat @ zero_rotation).round()


# TODO: inherit from a klossar-type patch class and/or PL Patch
class PolycubesPatch(BasePatchType):
    """
    A patch on a patchy particle.
    """

    def __init__(self,
                 uid: Union[int, None],
                 color: int,
                 direction: Union[int, np.ndarray],
                 orientation: Union[int, np.ndarray],
                 stateVar: int = 0,
                 activationVar: int = 0):
        super().__init__(uid, color)
        self._key_points = [direction if isinstance(direction, np.ndarray) else RULE_ORDER[direction],
                            orientation if isinstance(orientation, np.ndarray) else (RULE_ORDER[orientation])]
        self._stateVar = stateVar
        self._activationVar = activationVar

    def set_state_var(self, newVal: int):
        self._stateVar = newVal

    def set_activation_var(self, newVal: int):
        self._activationVar = newVal

    def dirIdx(self) -> int:
        return diridx(self.direction())

    def position(self) -> np.ndarray:
        return self.direction() / 2  # assume radius is 0.5

    def colornum(self) -> int:
        return self.color()

    def direction(self) -> np.ndarray:
        return self._key_points[0]

    def alignDir(self) -> np.ndarray:
        return self._key_points[1]

    def set_align(self, new_align: np.ndarray):
        self._key_points[1] = new_align

    def get_align_rot_num(self) -> int:
        # grab face direction
        face_dir = self.direction()
        # grab align direction
        face_align_dir = self.alignDir()
        # default alignment for face
        face_align_zero = RULE_ORDER[(self.dirIdx() + 2) % 6]
        # compute angle
        angleFromTo = getSignedAngle(face_align_zero, face_align_dir, face_dir)
        # angle to enumeration
        align_num = int(4 * angleFromTo / (math.pi * 2))
        if align_num < 0:
            align_num += 4
        return align_num

    def set_align_rot(self, new_align: int):
        """
        See above method get_align_rot_num
        Parameters:
            :param new_align an integer from 0 to 4 specifiying a rotation from the default patch orientation
        """
        face_dir = self.direction()
        face_align_zero = RULE_ORDER[(self.dirIdx() + 2) % 6]
        rot_radians = new_align * math.pi * 2 / 4
        face_new_align_dir = rotAroundAxis(face_align_zero, face_dir, rot_radians).round()
        self.set_align(face_new_align_dir)

    def state_var(self) -> int:
        return self._stateVar

    def activation_var(self) -> int:
        return self._activationVar

    def rotate(self, rotation: np.ndarray) -> PolycubesPatch:
        """
        Rotates and returns copy of patch
        """
        p = PolycubesPatch(self.get_id(),
                           self.color(),
                           np.matmul(rotation, self.direction()).round(),
                           np.matmul(rotation, self.alignDir()).round(),
                           self.state_var(),
                           self.activation_var())
        p.check_valid()
        return p

    def to_string(self) -> str:
        if self.get_align_rot_num() == 0 and self.state_var() == 0 and self.activation_var() == 0:
            return f"{self.color()}"
        elif self.state_var() == 0 and self.activation_var() == 0:
            return f"{self.color()}:{self.get_align_rot_num()}"
        else:
            return f"{self.color()}:{self.get_align_rot_num()}:{self.state_var()}:{self.activation_var()}"

    def check_valid(self):
        assert self.color()
        assert np.dot(self.direction(), self.alignDir()) == 0

    def can_bind(self, other: BasePatchType):
        return self.color() == other.color()

    def has_torsion(self):
        return self.num_key_points() == 2


class PolycubeRuleCubeType(PatchyBaseParticleType):
    def __init__(self,
                 ct_id: int,
                 patches: list[PolycubesPatch],
                 stateSize=1,
                 effects: Union[None, list[Effect]] = None,
                 name=""):
        super().__init__(ct_id, patches)
        if effects is None:
            effects = []
        self._name = name if name else f"CT{ct_id}"
        self._stateSize = stateSize

        self._effects = []
        # handle potential issues with improperly-indexed string conditionals
        for effect in effects:
            self.add_effect(effect)

    def name(self) -> str:
        return self._name

    def set_name(self, newName: str):
        self._name = newName

    def set_state_size(self, newVal: int):
        self._stateSize = newVal

    def state_size(self) -> int:
        return self._stateSize

    def add_state_var(self) -> int:
        """
        Increases the number of state variables by 1 and returns the newly-created
        state variable
        """
        self._stateSize += 1
        return self._stateSize - 1

    def get_patch_by_diridx(self, dirIdx: int) -> PolycubesPatch:
        return [p for p in self.patches() if p.dirIdx() == dirIdx][0]

    def get_patch_by_state_var(self, state_var: int) -> Union[None, PolycubesPatch]:
        """

        """
        patches = [p for p in self.patches() if p.state_var() == state_var]
        assert len(patches) < 2, "No two patches should have the same state variable!"
        if len(patches) == 0:
            return None
        else:
            return patches[0]

    def has_patch(self, arg: Union[int, np.ndarray]) -> bool:
        if isinstance(arg, int):  # direction index
            return any([p.dirIdx() == arg for p in self.patches()])
        else:
            assert isinstance(arg, np.ndarray)
            return any([(RULE_ORDER[p.dirIdx()] == arg).all() for p in self.patches()])

    def patch(self, direction: Union[int, np.ndarray]) -> PolycubesPatch:
        if isinstance(direction, int):
            return self.get_patch_by_diridx(direction)
        else:
            return [p for p in self.patches() if (RULE_ORDER[p.dirIdx()] == direction).all()][0]

    def diridxs(self) -> set[int]:
        return {p.dirIdx() for p in self.patches()}

    def get_patch_by_idx(self, i: int) -> PolycubesPatch:
        """
        Does NOT use direction indexes!
        Returns:
            the ith patch in the particle's patch list.
        """
        return self._patches[i]

    def patch_index_in(self, p: PolycubesPatch) -> int:
        """
        Returns the index in self._patches of the provided patch
        """
        for i, pp in enumerate(self.patches()):
            if p.get_id() == pp.get_id():
                return i
        return -1

    def add_patch(self, patch: PolycubesPatch):
        self._patches.append(patch)
        assert len(self.patches()) <= 6

    def remove_patch(self, identifier: Union[int, np.ndarray]):
        if isinstance(identifier, int):
            self.remove_patch(RULE_ORDER[identifier])
        else:
            for p in self.patches():
                if np.array_equal(p.direction(), identifier):
                    self._patches.remove(p)
                    return
            raise Exception(f"{self} does not have a patch at position {identifier}")

    def get_patch_state_var(self, key: Union[int, str, np.ndarray],
                            make_if_0=False) -> int:
        """
        returns the state variable associated with a patch

        Parameters:
            key a int, string, or vector (as an np array) that indicates a patch
        """
        # process index
        idx = key if isinstance(key, int) else int(key) if isinstance(key, str) else diridx(key)
        # if we need to make a new variable, and should do that
        if make_if_0 and self.get_patch_by_diridx(idx).state_var() == 0:
            # make one
            self.get_patch_by_diridx(idx).set_state_var(self.add_state_var())
        # return patch state variable
        return self.get_patch_by_diridx(idx).state_var()

    def effects(self) -> list[EFFECT_CLASSES]:
        return self._effects

    def effects_targeting(self, v: int) -> list[EFFECT_CLASSES]:
        """
        Returns a list containing all the effects on this cube type that target a given variable
        """
        return [e for e in self._effects if e.target() == abs(v)]

    def add_effect(self, effect: EFFECT_CLASSES):
        # if isinstance(effect, StringConditionalEffect):
        #     if effect.conditional() != "(true)":
        #         effect.setStr(re.sub(r'b\[(\d+)]',
        #                              # lambda match: str(self.get_patch_by_diridx(int(match.group(1))).state_var()),
        #                              lambda match: str(self.get_patch_state_var(int(match.group(1)), True)),
        #                              effect.conditional())
        #                       )
        #     else:
        #         effect.setStr("0")
        self._effects.append(effect)

    def patch_conditional(self, patch: PolycubesPatch, minimize=False) -> str:
        """
        Expresses the conditional for the provided patch in traditional string form
        Parameters:
            patch: the patch for which to derive a conditional
            minimize: true if the patches should be indexed ignoring zero-color patches, false otherwise
        """
        effects_lists = self.effects_targeting(patch.activation_var())
        if len(effects_lists) == 0:
            return ""
        if isinstance(effects_lists[0], StringConditionalEffect):
            assert len(effects_lists) == 1, "Multiple string conditionals for one patch is not supported"
            if minimize:
                def replacer(match):
                    n = int(match.group(1))  # Extract the integer n
                    value = self.patch_index_in(self.patch(n))
                    return str(value)

                return re.sub(r'b\[(\d+)]', replacer, effects_lists[0].conditional())
            else:
                return effects_lists[0].conditional()
        return self.var_conditional(patch.activation_var())

    def var_conditional(self, var: int, minimize=True):
        # loop dynamic effects that contribute to the activation variable
        in_strs: list[str] = []
        if self.is_patch_state_var(var):
            if minimize:
                if var > 0:
                    state_var_str = str(self.patches().index(self.get_patch_by_state_var(abs(var))))
                else:
                    state_var_str = "!" + str(self.patches().index(self.get_patch_by_state_var(abs(var))))
            else:
                if var > 0:
                    state_var_str = str(self.get_patch_by_state_var(abs(var)).dirIdx())
                else:
                    state_var_str = "!" + str(self.get_patch_by_state_var(abs(var)).dirIdx())
            in_strs.append(state_var_str)
        # ideally state vars controlled by patches and by effects should be mutually exclusive but...
        for e in self.effects_targeting(var):
            # forward-proof for environmental effects, which should be ignored for our purposes here
            if isinstance(e, DynamicEffect):
                state_var_strs: list[str] = []
                # if the input variable is set from a patch
                for input_state_var in e.sources():
                    in_strs.append(self.var_conditional(input_state_var, minimize))

        if len(in_strs) > 1:
            var_effects_str = "(" + " | ".join(in_strs) + ")"
        else:
            var_effects_str = in_strs[0]

        if var < 0:
            var_effects_str = "!" + var_effects_str
        return var_effects_str

    def is_patch_state_var(self, var: int) -> bool:
        return self.get_patch_by_state_var(var) is not None

    def __str__(self) -> str:
        return self.to_string()

    def __repr__(self) -> str:
        return self.to_string()

    def to_string(self, force_static=False) -> str:
        # option to force legacy "static representation"
        if force_static or any([isinstance(e, StringConditionalEffect) for e in self.effects()]):
            sz = "#".join([self.patch_string_static(idx) for (idx, _) in enumerate(RULE_ORDER)])
        # default to dynamic model repr
        else:
            # lmao
            sz = "|".join([
                self.get_patch_by_diridx(i).to_string() if self.has_patch(i) else ""
                for (i, _) in enumerate(RULE_ORDER)])
            if len(self.effects()) > 0:
                sz += "@" + ";".join([str(e) for e in self.effects()])
        return sz

    def patch_string_static(self, patch_dir_idx: int) -> str:
        if self.has_patch(patch_dir_idx):
            p = self.get_patch_by_diridx(patch_dir_idx)
            return f"{p.color()}:{p.get_align_rot_num()}:{self.patch_conditional(p)}"
        else:
            return ""

    def count_start_on_patches(self) -> int:
        return len([p for p in self.patches() if p.state_var() <= 0])

    def radius(self, normal: np.ndarray = np.zeros(shape=(3,))) -> float:
        # TODO: consider making this more specific
        # each patch is distance 1.0 units from center but that's not the radius per se
        return 1.0

    # def __deepcopy__(self, memo) -> PolycubeRuleCubeType:
    #     return PolycubeRuleCubeType(self.type_id(),
    #                                 [
    #         copy.deepcopy(p) for p in self.patches()
    #     ], self.state_size(), [copy.deepcopy(e) for e in self.effects()], self.name())

    # def draw_polyomino(self, d: draw.Drawing, x: int, y: int):
    #     group = draw.Group(transform=f"translate({x},{y}")
    #     d.append(group)
    #     arrow_end = draw.Marker(minx=0, miny=0, maxx=10, maxy=10)
    #     arrow_end.append(draw.Path(d="M 0 0 L 10 5 L 0 10 z", fill="black"))
    #     for f, _ in enumerate(RULE_ORDER):
    #         patchx, patchy = FACE_SVG_COORDS[f]
    #         group.append(draw.Rectangle(patchx * FACE_SQUARE_W,
    #                                     patchy * FACE_SQUARE_W,
    #                                     FACE_SQUARE_W,
    #                                     FACE_SQUARE_W,
    #                                     fill=selectColor(self.type_id())))
    #         if self.has_patch(f):
    #             patch = self.get_patch_by_diridx(f)
    #             rotation = patch.get_align_rot_num() * 90
    #             patch_path = draw.Path(
    #                 d="m 30,85 0,-45 10,0 0,-15 22,0 0,15 10,0 0, 45 z",
    #                 fill=selectColor(patch.color()),
    #                 stroke="black" if patch.color > 0 else "white",
    #                 stroke-width=3.5,
    #                 transform=f"scale(0.5) translate({patchx * FACE_SQUARE_W - 20},{patchy * FACE_SQUARE_W - 20}) rotate({rotation})"
    #             )
    # x_cum = FACE_SQUARE_W * 3 + 2 * FACE_FIG_PAD
    # def activates(x1, y1, x2, y2, deactivates=False, arc_r=1.25):
    #     radius = math.sqrt(2 ** (x2 - x1) + 2 ** (y2 - y1, 2)) * arc_r
    #     color = "red" if deactivates else "green"
    #     group.append(draw.Path(d=f"M {x1} {y1} A {radius} {radius} 0 0 1 {x2}${y2}",
    #                stroke="red" if deactivates else "green",
    #                fill="none",
    #                markerend=arrow_end
    #     ))
    #
    #     def coordspace(n):
    #         return (n + 0.5) * FACE_SQUARE_W
    #
    # if self.num_effects() > 0 {
    #     state_box_w = row_h / (2 * cubeType.state_size() + 3);
    #     row_center = row_h / 2 - 1.5 * state_box_w;
    #     x_cum += + 2 * FACE_FIG_PAD;
    #     state_box_x = x_cum;
    #     x_cum += state_box_w;
    #     # loop state variables
    #     for s in range(-self.state_size(), self.state_size() + 1):
    #         # draw state var
    #         group.append(draw.Rectangle(state_box_x, row_center + s * state_box_w, state_box_w, state_box_w,
    #                                     stroke="black",
    #                                     fill="white"))
    #         group.append(draw.Text(f"{s}",
    #                                state_box_x + state_box_w / 2,
    #                                row_center + (s + 0.5) * state_box_w,
    #                                center=True,
    #                                fontsize=10))
    #
    #     def state_box_intr_coords(s, side="left"):
    #         assert s > -self.state_size() - 1
    #         assert s < + self.state_size() + 1
    #         x = state_box_x + (state_box_w * (0 if side == "left" else 1))
    #         x += 6 * (-1 if side == "left" else 1)
    #         y = (s + 0.5) * state_box_w + row_center
    #         return x, y
    #
    #     def sets_state(x1, y1, state_idx, x2=None)
    #         approach_dir =  "left" if x1 < state_box_x else "right"
    #         x3, y3 = state_box_intr_coords(state_idx, approach_dir)
    #         if x2 is None:
    #             x2 = (x1 + x3) / 2
    #         y2 = (y1 + y3) / 2
    #         group.append(draw.Path(f"M {x1} {y1} S {x2} {y1} {x2} {y2} S {x2} {y3} {x3} {y3}",
    #             stroke="green",
    #             fill="none",
    #             marker_end=arrow_end
    #         )
    #
    #     for f in enumerate(RULE_ORDER):
    #         if self.has_patch(f):
    #             patch = self.get_patch_by_diridx(f)
    #             if self.get_patch_by_diridx(f).state_var():
    #                 x1, y1 = FACE_SVG_COORDS[f]
    #                 sets_state(coordspace(x1), coordspace(y1), patch.state_var())
    #             if patch.activation_var():
    #                 [x1, y1] = state_box_intr_coords(cubeType.patches[f].activation_var)
    #                 [x2, y2] = FACE_SVG_COORDS[f]
    #                 activates(x1, y1, coordspace(x2), coordspace(y2), cubeType.patches[f].activation_var < 0)
    #     x_cum += 2 * FACE_FIG_PAD;
    #     for j, e in enumerate(self.effects()):
    #         origins_coords_list = [state_box_intr_coords(s, "right") for s in e.sources]
    #         if len(origins_coords_list) == 1:
    #             e_draw_size = state_box_w;
    #             e_draw_y = origins_coords_list[0][1] - state_box_w / 2
    #         else:
    #             e_draw_size = state_box_w * 0.75 * origins_coords_list.length
    #             e_draw_y = origins_coords_list.reduce((a, p) = > {
    #                 return a + (p[1] / origins_coords_list.length);
    #             }, 0);
    #         group.path(f"M ${x_cum} {e_draw_y} h {e_draw_size * 0.25} a {e_draw_size / 2} {e_draw_size / 2} 0 0 1 0 {e_draw_size} h {-e_draw_size * 0.25} z",
    #                    stroke="black",
    #                    fill="white")
    #         for k, xy in enumerate(origins_coords_list):
    #             [x1, y1] = xy
    #             x2 = x_cum - 5
    #             y2 = e_draw_y + (k + 1) * e_draw_size / (origins_coords_list.length + 1)
    #             activates(x1, y1, x2, y2, e.sources[k] < 0, 4)
    #         x_cum += e_draw_size * 0.75
    #         y1 = e_draw_y + e_draw_size / 2
    #         x_mid = x_cum + Math.sqrt(Math.abs(e_draw_y + e_draw_size / 2 - dest_y))
    #         sets_state(x_cum, y1, e.target, x_mid)
    #         x_cum += 22
    #         if (e.sources.length == 1:
    #             x_cum -= 10
    #     group.append(draw.Text(
    #         self.typeName,
    #         96, 275,
    #         center=True,
    #         font-size=20
    #     big_cum_x += x_cum


    def shift_state(self, nvars: int):
        """
        Shifts the entire state to the right by nvars
        """
        self._stateSize += nvars

        for p in self.patches():
            if p.state_var():
                p.set_state_var(p.state_var() + nvars)
            if p.activation_var():
                p.set_activation_var(
                    p.activation_var() + nvars if p.activation_var() > 0 else p.activation_var() - nvars)

        for e in self.effects():
            e.set_target(e.target() + nvars)
            e.set_sources([i + nvars if i > 0 else i - nvars for i in e.sources()])

    def get_state_transition_graph(self, filter_patches_settable=True) -> nx.DiGraph:
        # construct empty DiGraph
        g = nx.DiGraph()
        num_states = 2 ** self.state_size()
        # loop states
        for state_num in range(num_states):
            # since the identity variable s0 is always true, only odd states are actually reachable
            if state_num % 2:
                # compute state variable values
                state = [state_num & 1 << n != 0 for n in range(self.state_size())]
                # keys are state var idxs, vars are list of effect firing probabilities
                effect_map: dict[int, list[float]] = {}
                # loop effects
                for e in self.effects():
                    # effects that have already fired are irrelevant
                    if not state[e.target()]:
                        # if effect can fire
                        if all([state[src] if src > 0 else not state[-src] for src in e.sources()]):
                            # state transition probabilities haven't been implemented yet but incl forward proofing
                            effect_prob = 1
                            if e.target() not in effect_map:
                                effect_map[e.target()] = []
                            effect_map[e.target()].append(effect_prob)
                # prob of state var transition is prob that any of the effects will fire
                # so 1 - the prob that none of the effects will fire
                # = 1 - the product of (1 - the probability that each effect wiill fire)
                state_var_transition_prob = {v: 1 - np.prod([1 - p for p in effect_map[v]]) for v in effect_map}
                # compile effects into state transition map
                # set of possible destination states is the power set of the keys of effect_map
                for dest_state in powerset(state_var_transition_prob.keys()):
                    # compute destination state
                    dest_state_num = state_num + sum([2 ** v for v in dest_state])
                    prob_set_state = 1
                    # prob of transitioning to state = the prob that all state vars that need to be set WILL
                    # be set and all that don't need to be set won't
                    # loop vars
                    for v in state_var_transition_prob:
                        if v in dest_state:
                            prob_set_state *= state_var_transition_prob[v]
                        else:
                            prob_set_state *= (1 - state_var_transition_prob[v])

                    # arbitrary probability cutoff
                    if prob_set_state > 1e-8:
                        g.add_edge(state_num, dest_state_num, p=prob_set_state)

        # filter state transition graph
        if filter_patches_settable:
            patch_state_vars = set()
            for p in self.patches():
                if p.state_var():
                    patch_state_vars.add(p.state_var())
            patch_control_states = {1 + sum([2 ** v for v in state]) for state in powerset(patch_state_vars)}
            downstream = set()
            for ctlstate in patch_control_states:
                to_explore = [ctlstate]

                while to_explore:
                    current_node = to_explore.pop()
                    downstream.add(current_node)

                    neighbors = list(g.successors(current_node))
                    for neighbor in neighbors:
                        if neighbor not in downstream:
                            to_explore.append(neighbor)
            nodes_to_remove = [node for node in g if node not in downstream]
            for node in nodes_to_remove:
                g.remove_node(node)
        return g

    def rotate(self, r: np.ndarray):
        """
        Applies a rotation in place to this cube type
        This isn't normally very useful but it's important for merging cube types
        """
        newPatches = []
        for p in self.patches():
            newPatches.append(p.rotate(r))
        self._patches = newPatches


class PolycubesRule(BaseParticleSet):
    def __init__(self, **kwargs):
        """
        Extremely flexable and occassionally-misbehaving constructor for a Polycubes rule
        Kwargs:
            rule_str: a string representation of the rule
            rule_json: a list of dict representations of cube types

        """
        # TODO: break this method up!
        super().__init__()
        # WARNING: I actually have no idea if this code will always behave correctly if given
        # static formulation strings!! for this reason you should play it safe and Not Do That
        if "rule_str" in kwargs:
            rule_str = kwargs["rule_str"]

            for i, particle_type_str in enumerate(rule_str.split("_")):
                vars_set = {0}
                effects = []
                # if this rule string has effects
                if particle_type_str.find("@") > -1:
                    # seperate patches and effects
                    patches_str, effects_str = particle_type_str.split("@")
                    # loop effects
                    for effect_str in effects_str.split(";"):
                        # split sources, target
                        sources, target = effect_str.split(">")
                        # parse sources
                        sources = [int(match.group(0)) for match in re.finditer("-?\d+", sources)]
                        # parse target
                        target = int(target)
                        # append to effects
                        effects.append(DynamicEffect(sources, target))
                        # add vars to vars set
                        vars_set = vars_set.union({target, *[abs(i) for i in sources]})
                else:
                    patches_str = particle_type_str
                if re.match("^([^|]*\|){5}[^|]*$", particle_type_str):
                    patch_strs = patches_str.split("|")
                else:
                    patch_strs = patches_str.split("#")

                # PLEASE for the LOVE of GOD do NOT combine static and dynamic patches in the same rule string!!!!!
                cube_type = PolycubeRuleCubeType(i, [], max(vars_set) + 1, effects)

                string_effects = []

                for j, patch_str in enumerate(patch_strs):
                    # ignore empty faces
                    if len(patch_str.strip()) == 0:
                        continue
                    patchRotation = 0

                    patch_components = patch_str.split(":")
                    color = int(patch_components[0])
                    if color == 0:
                        continue
                    stateVar = 0
                    activationVar = 0
                    if len(patch_components) > 1:
                        patchRotation = int(patch_components[1])
                        if len(patch_components) == 3 and patch_components[2]:
                            conditionalStr = patch_components[2]
                            activationVar = cube_type.add_state_var()
                            e = StringConditionalEffect(conditionalStr, activationVar)
                            string_effects.append(e)
                        elif len(patch_components) == 4:
                            # NOTE: avoid mixing static and dynamic formulations in a single cube type!
                            stateVar = int(patch_components[2])
                            activationVar = int(patch_components[3])
                    vars_set.add(stateVar)  # if stateVar is already in the set this is fine
                    vars_set.add(activationVar)  # ditto for activationVar
                    cube_type.set_state_size(max(cube_type.state_size(), stateVar + 1, abs(activationVar) + 1))

                    patch_ori = get_orientation(j, patchRotation)

                    # patch position is determined by order of rule str
                    patch = PolycubesPatch(self.num_patches(), color, j, diridx(patch_ori), stateVar, activationVar)
                    patch.check_valid()
                    cube_type.add_patch(patch)
                    self.add_patch(patch)

                for e in string_effects:
                    cube_type.add_effect(e)

                # self._cubeTypeList.append(PolycubeRuleCubeType(i,
                #                                                patches_list,
                #                                                len(vars_set),
                #                                                effects))
                self.add_particle(cube_type)

        # TODO: pull tag info from C++ to maintain consistancy?
        elif "rule_json" in kwargs:
            for i, ct_dict in enumerate(kwargs["rule_json"]):
                if "name" in ct_dict:
                    name = ct_dict['name']
                elif "typeName" in ct_dict:
                    name = ct_dict["typeName"]
                else:
                    name = f"CT{i}"

                # iv'e messed these up so badly omg
                patches = []
                effects: list[Effect] = []
                vars_set = {1}
                if "effects" in ct_dict:
                    effects = [DynamicEffect(e_json["sources"], e_json["target"]) for e_json in ct_dict["effects"]]
                if "patches" in ct_dict:
                    for j, patch_json in enumerate(ct_dict["patches"]):
                        if "dir" in patch_json:
                            dirIdx = diridx(from_xyz(patch_json["dir"]))
                        else:
                            dirIdx = j
                        # get patch attributes from json object
                        activation_var = patch_json["activation_var"]
                        state_var = patch_json["state_var"]
                        vars_set.add(activation_var)
                        vars_set.add(state_var)
                        color = patch_json["color"]
                        if color:
                            alignDir = diridx(from_xyz(patch_json["alignDir"]))
                            # add patch
                            patch = PolycubesPatch(self.num_patches(),
                                                   color,
                                                   dirIdx,
                                                   alignDir,
                                                   state_var,
                                                   activation_var)
                            patches.append(patch)
                            self.add_patch(patch)
                            # handle conditionals
                            if "conditionals" in ct_dict and ct_dict["conditionals"][dirIdx]:
                                # if conditionals are inluded alongside a patch list, they'll be indexed by RULE_ORDER
                                # corresponding to the patch_list
                                effects.append(StringConditionalEffect(ct_dict["conditionals"][j], activation_var))
                else:
                    # shittiest version. should always be len(RULE_ORDER) of these
                    activationVarCounter = 0
                    for j, color, alignDir, direction in zip(range(len(RULE_ORDER)), ct_dict['colors'],
                                                             ct_dict['alignments'], ct_dict["conditionals"]):
                        if color:
                            alignDir = diridx(from_xyz(alignDir))
                            if ct_dict["conditionals"][j]:
                                effects.append(StringConditionalEffect(ct_dict["conditionals"][j], activation_var))
                                activation_var = 1 + len(patches) + activationVarCounter
                                activationVarCounter += 1
                            else:
                                activation_var = 0
                            vars_set.add(activation_var)
                            vars_set.add(state_var)
                            patch = PolycubesPatch(self.num_patches(),
                                                   color,
                                                   j,
                                                   alignDir,
                                                   1 + len(patches),
                                                   activation_var)
                            patches.append(patch)
                            self.add_patch(patch)

                self._particle_types.append(PolycubeRuleCubeType(self.num_particle_types(),
                                                                 patches,
                                                                 len(vars_set),
                                                                 effects,
                                                                 name))
        elif "nS" in kwargs:
            # if a number of species is provided, initialize an empty rule
            for i in range(kwargs["nS"]):
                self._particle_types.append(PolycubeRuleCubeType(i, []))

    def toJSON(self) -> list:
        return [{
            "typeName": ct.name(),
            "patches": [{
                "dir": to_xyz(p.direction()),
                "alignDir": to_xyz(p.alignDir()),
                "color": p.color(),
                "state_var": p.state_var(),
                "activation_var": p.activation_var()
            } for p in ct.patches()],
            "effects": [e.toJSON() for e in ct.effects()],
            "state_size": ct.state_size()
        }
            for ct in self.particles()]

    def add_particle_patch(self, particle: Union[int], patch: PolycubesPatch):
        if isinstance(particle, int):
            particle = self.particle(particle)
        self.add_patch(patch)
        particle.add_patch(patch)

    def add_patch(self, patch):
        if patch.get_id() is not None and patch.get_id() != -1:
            assert patch.get_id() == self.num_patches()
        else:
            patch.set_type_id(self.num_patches())
        self._patch_types.append(patch)

    def sort_by_num_patches(self):
        self._particle_types.sort(key=lambda ct: ct.num_patches())

    def remove_cube_type(self, cubeType: PolycubeRuleCubeType):
        # remove cube type from cube type lists
        self._particle_types = [ct for ct in self.particles() if ct.type_id() != cubeType.type_id()]
        # remove patches that are no longer needed
        self._patch_types = [p for p in self.patches() if any([p in ct.patches() for ct in self.particles()])]
        # TODO: handle particle and patch IDs!!!

    def color_set(self) -> set[int]:
        return {p.color() for p in self.patches()}

    def reindex(self):
        for i, particle_type in enumerate(self._particle_types):
            particle_type.set_type_id(i)

    def draw_rule(self, draw_width=800) -> drawsvg.Drawing:
        """
        translation of draw code from polycubes js
        """
        draw_height = math.ceil(self.num_particle_types() / 4 * 325)
        d = drawsvg.Drawing(width=draw_width, height=draw_height)
        x = 0
        y = 0
        for particle in self.particles():
            x, y = particle.draw_polyomino(d, x, y)
        return d

    def __len__(self) -> int:
        return self.num_particle_types()

    def __str__(self) -> str:
        return "_".join(str(ct) for ct in self.particles())

    def __eq__(self, other: Union[PolycubesRule, str]):
        if isinstance(other, PolycubesRule):
            return self == str(other)
        else:
            return str(self) == other

    def concat(self, other: PolycubesRule):
        """
        Joins two polycubes rules by shifting one and concatinating
        """
        other = other >> (self.num_colors() + 1)
        return self + other

    def __add__(self, other: PolycubesRule) -> PolycubesRule:
        return PolycubesRule(rule_str=str(self) + "_" + str(other))

    def __rshift__(self, n: int) -> PolycubesRule:
        """
        "left-shifts" a rule by increasing all colors by n
        """
        cpy: PolycubesRule = copy.deepcopy(self)
        for patch in cpy.patches():
            if patch.color() > 0:
                patch.set_color(patch.color() + n)
            else:
                patch.set_color(patch.color() - n)
        return cpy

    def num_colors(self) -> int:
        return len(self.color_set())


# TODO: integrate with C++ TLM / Polycubes
