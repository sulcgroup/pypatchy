import math

import numpy as np
import pandas as pd
from itertools import chain
import re

from patchy.plpatchy import Patch
from util import rotation_matrix, from_xyz, to_xyz, getSignedAngle

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
def diridx(a):
    for idx, rule in enumerate(RULE_ORDER):
        if np.array_equal(a, rule):
            return idx
    assert False, f"{a} is not in RULE_ORDER"


def get_orientation(face_idx, ori_idx):
    zero_rot_idx = (face_idx + 2) % len(RULE_ORDER)  # feels like it should be +4 but trust me bro
    zero_rotation = RULE_ORDER[zero_rot_idx]
    rot_mat = rotation_matrix(RULE_ORDER[face_idx], ori_idx * math.pi / 2)
    return (rot_mat @ zero_rotation).round()


# TODO: make this extend some kind of generic klossar / patchy particle rule class
class PolycubesRule:
    def __init__(self, **kwargs):
        self._cubeTypeList = []
        self._patchList = []
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
                        sources = [int(match) for match in re.finditer("-?\d+", sources)]
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
                cube_type = PolycubeRuleCubeType(i, [], max(vars_set), effects)

                string_effects = []

                for j, patch_str in enumerate(patch_strs):
                    # ignore empty faces
                    if len(patch_str.strip()) == 0:
                        continue
                    patchRotation = 0

                    patch_components = patch_str.split(":")
                    color = int(patch_components[0])
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
                    cube_type.set_state_size(max(cube_type.state_size(), stateVar+1, abs(activationVar) + 1))

                    patch_ori = get_orientation(j, patchRotation)

                    # patch position is determined by order of rule str
                    patch = PolycubesPatch(self.numPatches(), color, j, diridx(patch_ori), stateVar, activationVar)
                    patch.check_valid()
                    cube_type.add_patch(patch)
                    self._patchList.append(patch)

                for e in string_effects:
                    cube_type.add_effect(e)

                # self._cubeTypeList.append(PolycubeRuleCubeType(i,
                #                                                patches_list,
                #                                                len(vars_set),
                #                                                effects))
                self._cubeTypeList.append(cube_type)

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
                effects = []
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
                            patches.append(PolycubesPatch(self.numPatches(),
                                                          color,
                                                          dirIdx,
                                                          alignDir,
                                                          state_var,
                                                          activation_var))
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
                            patches.append(PolycubesPatch(self.numPatches(),
                                                          color,
                                                          j,
                                                          alignDir,
                                                          1 + len(patches),
                                                          activation_var))

                self._cubeTypeList.append(PolycubeRuleCubeType(self.numCubeTypes(),
                                                               patches,
                                                               len(vars_set),
                                                               effects,
                                                               name))
                self._patchList += patches

    def toJSON(self):
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

    def particle(self, i):
        return self._cubeTypeList[i]

    def patch(self, i):
        return self._patchList[i]

    def numPatches(self):
        return len(self._patchList)

    def numCubeTypes(self):
        return len(self._cubeTypeList)

    def cubeTypes(self):
        return self._cubeTypeList

    def patches(self):
        return self._patchList

    def particles(self):
        return self._cubeTypeList

    def remove_cube_type(self, cubeType):
        # remove cube type from cube type lists
        self._cubeTypeList = [ct for ct in self.particles() if ct.getID() != cubeType.getID()]
        # remove patches that are no longer needed
        self._patchList = [p for p in self.patches() if any([p in ct.patches() for ct in self.particles()])]
        # TODO: handle particle and patch IDs!!!

    def particle(self, i):
        assert -1 < i < self.numCubeTypes()
        return self._cubeTypeList[i]

    def __len__(self):
        return self.numCubeTypes()

    def __str__(self):
        return "_".join(str(ct) for ct in self.particles())


class PolycubeRuleCubeType:
    def __init__(self, ct_id, patches, stateSize=1, effects=[], name=""):
        self._id = ct_id
        self._patches = patches
        self._name = name if name else f"CT{ct_id}"
        self._stateSize = stateSize

        self._effects = []
        # handle potential issues with improperly-indexed string conditionals
        for effect in effects:
            self.add_effect(effect)

    def name(self):
        return self._name

    def getID(self):
        return self._id

    def set_state_size(self, newVal):
        self._stateSize = newVal

    def state_size(self):
        return self._stateSize

    def add_state_var(self):
        self._stateSize += 1
        return self._stateSize - 1

    def get_patch_by_diridx(self, dirIdx):
        return [p for p in self._patches if p.dirIdx() == dirIdx][0]

    def has_patch(self, arg):
        if isinstance(arg, int):  # direction index
            return any([p.dirIdx() == arg for p in self._patches])
        else:
            assert isinstance(arg, np.ndarray)
            return any([(RULE_ORDER[p.dirIdx()] == arg).all() for p in self._patches])

    def patch(self, direction):
        return [p for p in self._patches if (RULE_ORDER[p.dirIdx()] == direction).all()][0]

    def diridxs(self):
        return {p.dirIdx() for p in self._patches}

    def get_patch_by_idx(self, i):
        return self._patches[i]

    def patches(self):
        return self._patches

    def add_patch(self, patch):
        self._patches.append(patch)

    def num_patches(self):
        return len(self._patches)

    def get_patch_state_var(self, key, make_if_0=False):
        idx = key if isinstance(key, int) else int(key) if isinstance(key, str) else diridx(key)
        if make_if_0 and self.get_patch_by_diridx(idx).state_var() == 0:
            self.get_patch_by_diridx(idx).set_state_var(self.add_state_var())
        return self.get_patch_by_diridx(idx).state_var()

    def effects(self):
        return self._effects

    def effects_targeting(self, activation_var):
        return [e for e in self._effects if e.target() == activation_var]

    def add_effect(self, effect):
        if isinstance(effect, StringConditionalEffect):
            if effect.conditional() != "(true)":
                effect.setStr(re.sub(r'b\[(\d+)]',
                                     # lambda match: str(self.get_patch_by_diridx(int(match.group(1))).state_var()),
                                     lambda match: str(self.get_patch_state_var(int(match.group(1)), True)),
                                     effect.conditional())
                              )
            else:
                effect.setStr("0")
        self._effects.append(effect)

    def patch_conditional(self, patch):
        return "|".join(f"({e.conditional()})" for e in self.effects_targeting(patch.activation_var()))

    def __str__(self):
        return self.to_string()

    def to_string(self, force_static=False):
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

    def patch_string_static(self, patch_dir_idx):
        if self.has_patch(patch_dir_idx):
            p = self.get_patch_by_diridx()
            return f"{p.color()}:{p.get_align_rot_num()}:{self.patch_conditional(p)}"
        else:
            return ""

class PolycubesPatch:
    def __init__(self, id, color, direction, orientation, stateVar, activationVar):
        self._id = id
        self._color = color
        self._dirIdx = direction if isinstance(direction, int) else diridx(direction)
        self._oriIdx = orientation if isinstance(orientation, int) else (diridx(orientation))
        self._stateVar = stateVar
        self._activationVar = activationVar

    def color(self):
        return self._color

    def set_state_var(self, newVal):
        self._stateVar = newVal

    def set_activation_var(self, newVal):
        self._activationVar = newVal

    def dirIdx(self):
        return self._dirIdx

    def direction(self):
        return RULE_ORDER[self.dirIdx()]

    def alignDir(self):
        return RULE_ORDER[self._oriIdx]

    def get_align_rot_num(self):
        face_dir = self.direction()
        face_align_dir = self.alignDir()
        face_align_zero = RULE_ORDER[(self.dirIdx() + 2) % 6]
        angleFromTo = getSignedAngle(face_align_zero, face_align_dir, face_dir)
        return int(4 * angleFromTo / (math.pi * 2))

    def state_var(self):
        return self._stateVar

    def set_state_var(self, val):
        self._stateVar = val

    def activation_var(self):
        return self._activationVar

    def set_activation_var(self, val):
        self._activationVar = val

    def to_pl_patch(self):
        relPosition = self.direction() / 2
        return Patch(self._id, self._color, relPosition, self.direction(), self.alignDir())

    def rotate(self, rotation):
        p = PolycubesPatch(self._id,
                           self._color,
                           np.matmul(rotation, self.direction()).round(),
                           np.matmul(rotation, self.alignDir()).round(),
                           self.state_var(),
                           self.activation_var())
        p.check_valid()
        return p


    def to_string(self):
        if self.get_align_rot_num() == 0 and self.state_var() == 0 and self.activation_var() == 0:
            return f"{self.color()}"
        elif self.state_var() == 0 and self.activation_var() == 0:
            return f"{self.color()}:{self.get_align_rot_num()}"
        else:
            return f"{self.color()}:{self.get_align_rot_num()}:{self.state_var()}:{self.activation_var()}"

    def check_valid(self):
        assert self.color()
        assert np.dot(self.direction(), self.alignDir()) == 0


# TODO: integrate with C++ TLM / Polycubes
class Effect:
    def __init__(self, target):
        self._target = target

    def target(self):
        return self._target

    def toJSON(self):
        return {
            "target": self.target()
        }


class StringConditionalEffect(Effect):
    def __init__(self, conditional, target):
        super().__init__(target)
        self._conditional = conditional

    def conditional(self):
        return self._conditional

    def setStr(self, newStr):
        self._conditional = newStr

    def toJSON(self):
        return {
            **super(StringConditionalEffect, self).toJSON(),
            "conditional": self.target()
        }


class DynamicEffect(Effect):
    def __init__(self, vars, target):
        super().__init__(target)
        self._vars = vars

    def conditional(self):
        return "&".join([f"v" if v > 0 else f"!{-v}" for v in self._vars])

    def sources(self):
        return self._vars

    def __str__(self):
        return f"[{','.join([str(s) for s in self.sources()])}]>{str(self.target())}"

    def toJSON(self):
        return {
            **super(DynamicEffect, self).toJSON(),
            "sources": self._vars
        }


def rule_from_string(rule_str):
    """
    It's actually really tricky to break up this function into smaller functions,
    because we need to keep track of particle and patch IDs globally
    """
    particles_list = []
    id_counter = 0


def ruleToDataframe(rule):
    """
    takes a list of PolycubeRuleCubeType objects and returns a
    stylized dataframe of the cube types
    """
    df = pd.DataFrame([chain.from_iterable([
        [
            ct.colors[i],
            diridx(ct.alignments[i].values()),
            ct.conditionals[i]
        ]
        for (i, dirname) in enumerate(FACE_NAMES)
    ])

        for ct in rule
    ], columns=pd.MultiIndex.from_product([FACE_NAMES, ("Color", "Align", "Cond.")], names=("Face", "Attr.")),
        index=map(lambda ct: ct.name, rule))

    return df
