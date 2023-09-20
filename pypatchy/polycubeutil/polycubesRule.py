from __future__ import annotations

import copy
import math
import typing
from typing import Union

import numpy as np

from .effect import Effect, StringConditionalEffect, DynamicEffect, EnvironmentalEffect, EFFECT_CLASSES
from ..util import rotAroundAxis
import re

from pypatchy.util import rotation_matrix, from_xyz, to_xyz, getSignedAngle
from pypatchy.patchy_base_particle import PatchyBaseParticleType, BasePatchType, BaseParticleSet

FACE_NAMES = ("left", "right", "bottom", "top", "back", "front")
RULE_ORDER = (
    np.array((-1, 0, 0)),
    np.array((1, 0, 0)),
    np.array((0, -1, 0)),
    np.array((0, 1, 0)),
    np.array((0, 0, -1)),
    np.array((0, 0, 1))
)


# old diridx - bad/buggy
# def diridx(a):
#     return np.all(np.array(RULE_ORDER) == np.array(list(a))[np.newaxis, :], axis=1).nonzero()[0][0]

# new diridx - written by chatgpt
def diridx(a: np.ndarray) -> int:
    for idx, rule in enumerate(RULE_ORDER):
        if np.array_equal(a, rule):
            return idx
    assert False, f"{a} is not in RULE_ORDER"


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
        face_dir = self.direction()
        face_align_dir = self.alignDir()
        face_align_zero = RULE_ORDER[(self.dirIdx() + 2) % 6]
        angleFromTo = getSignedAngle(face_align_zero, face_align_dir, face_dir)
        return int(4 * angleFromTo / (math.pi * 2))

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
        return [e for e in self._effects if e.target() == v]

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
        effect_strs = []
        for e in effects_lists:
            # forward-proof for environmental effects, which should be ignored for our purposes here
            if isinstance(e, DynamicEffect):
                if minimize:
                    patches_in = [str(self.patches().index(self.get_patch_by_state_var(abs(input_state_var)))) if input_state_var > 0
                                  else "!" + str(self.patches().index(self.get_patch_by_state_var(abs(input_state_var))))
                                  for input_state_var in e.sources()]
                else:
                    patches_in = [str(self.get_patch_by_state_var(abs(input_state_var)).dirIdx()) if input_state_var > 0
                                  else "!"+str(self.get_patch_by_state_var(abs(input_state_var)).dirIdx())
                                  for input_state_var in e.sources()]
                if len(patches_in) > 1:
                    effect_strs.append("(" + "&".join(patches_in) + ")")
                else:
                    effect_strs.append(patches_in[0])
        return "|".join(effect_strs)

    def __str__(self) -> str:
        return self.to_string()

    def __repr__(self) -> str:
        return self.to_string()

    def to_string(self, force_static=False) -> str:
        if force_static or any([isinstance(e, StringConditionalEffect) for e in self.effects()]):
            sz = "#".join([self.patch_string_static(idx) for (idx, _) in enumerate(RULE_ORDER)])
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

    def shift_state(self, nvars: int):
        """
        Shifts the entire state to the right by nvars
        """
        self._stateSize += nvars

        for p in self.patches():
            if p.state_var():
                p.set_state_var(p.state_var() + nvars)
            if p.activation_var():
                p.set_activation_var(p.activation_var() + nvars if p.activation_var() > 0 else p.activation_var() - nvars)

        for e in self.effects():
            e.set_target(e.target() + nvars)
            e.set_sources([i + nvars if i > 0 else i - nvars for i in e.sources()])


class PolycubesRule(BaseParticleSet):
    def __init__(self, **kwargs):
        """
        Extremely flexable and occassionally-misbehaving constructor for a Polycubes rule
        Kwargs:
            rule_str: a string representation of the rule
            rule_json: a list of dict representations of cube types

        """
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
                cube_type = PolycubeRuleCubeType(i, [], max(vars_set)+1, effects)

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
            patch.set_id(self.num_patches())
        self._patch_types.append(patch)

    def sort_by_num_patches(self):
        self._particle_types.sort(key=lambda ct: ct.num_patches())

    def remove_cube_type(self, cubeType: PolycubeRuleCubeType):
        # remove cube type from cube type lists
        self._particle_types = [ct for ct in self.particles() if ct.type_id() != cubeType.type_id()]
        # remove patches that are no longer needed
        self._patch_types = [p for p in self.patches() if any([p in ct.patches() for ct in self.particles()])]
        # TODO: handle particle and patch IDs!!!

    def reindex(self):
        for i, particle_type in enumerate(self._particle_types):
            particle_type.set_id(i)

    def __len__(self) -> int:
        return self.num_particle_types()

    def __str__(self) -> str:
        return "_".join(str(ct) for ct in self.particles())


# TODO: integrate with C++ TLM / Polycubes


