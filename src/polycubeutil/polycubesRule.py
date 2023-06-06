import math

import numpy as np
import pandas as pd
from itertools import chain
import re

from patchy.plpatchy import Patch, PLPatchyParticle
from patchy.util import rotation_matrix, from_xyz


FACE_NAMES = ("left", "right", "bottom", "top", "back", "front")
RULE_ORDER = (
    np.array((-1, 0, 0)),
    np.array((1, 0, 0)),
    np.array((0, -1, 0)),
    np.array((0, 1, 0)),
    np.array((0, 0, -1)),
    np.array((0, 0, 1))
)


def diridx(a):
    return np.all(np.array(RULE_ORDER) == np.array(list(a))[np.newaxis, :], axis=1).nonzero()[0][0]


def get_orientation(face_idx, ori_idx):
    zero_rotation = RULE_ORDER[(face_idx + 4) % len(RULE_ORDER)]  # rotation is rule order offset by + 4
    return zero_rotation * rotation_matrix(RULE_ORDER[face_idx], ori_idx * math.pi / 2)

# TODO: make this extend some kind of generic klossar / patchy particle rule class
class PolycubesRule:
    def __init__(self, **kwargs):
        self._cubeTypeList = []
        self._patchList = []
        if "rule_str" in kwargs:
            rule_str = kwargs["rule_str"]

            for i, particle_type_str in enumerate(rule_str.split("_")):
                if re.match("^([^|]*\|){5}[^|]*$", particle_type_str):
                    patch_strs = particle_type_str.split("|")
                else:
                    patch_strs = particle_type_str.split("#")

                patches_list = []
                effects = []
                activationVarCounter = 0
                vars_set = {*range(len(patch_strs) + 1)}

                for j, patch_str in enumerate(patch_strs):
                    # ignore empty faces
                    if len(patch_str.strip()) == 0:
                        continue
                    oriIdx = 0

                    patch_components = patch_str.split(":")
                    color = int(patch_components[0])
                    stateVar = j
                    activationVar = 0
                    if len(patch_components) > 1:
                        oriIdx = int(patch_components[1])
                        if len(patch_components) == 3 and patch_components[2]:
                            conditionalStr = patch_components[2]
                            activationVar = len(vars_set)
                            effects.append(StringConditionalEffect(conditionalStr, activationVar))
                        elif len(patch_components) == 4:
                            # NOTE: avoid mixing static and dynamic formulations in a single cube type!
                            stateVar = int(patch_components[2])
                            activationVar = int(patch_components[3])
                    vars_set.add(stateVar)  # if stateVar is already in the set this is fine
                    vars_set.add(activationVar)

                    # patch position is determined by order of rule str
                    patch = PolycubesPatch(self.numPatches(), color, j, oriIdx, stateVar, activationVar)
                    patches_list.append(patch)
                    self._patchList.append(patch)

                self._cubeTypeList.append(PolycubeRuleCubeType(i,
                                                               patches_list,
                                                               len(vars_set),
                                                               effects))

        # TODO: pull tag info from C++ to maintain consistancy?
        elif "rule_json" in kwargs:
            for i, ct_dict in enumerate(kwargs["rule_json"]):
                conditionals = ct_dict['conditionals']
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
                        alignDir = dirIdx(from_xyz(patch_json["alignDir"]))
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

    def __len__(self):
        return self.numCubeTypes()


class PolycubeRuleCubeType:
    def __init__(self, id, patches, stateSize=1, effects=[], name=""):
        self._id = id
        self._patches = patches
        self._name = name if name else f"CT{id}"
        self._stateSize = stateSize
        # handle potential issues with improperly-indexed string conditionals

        for effect in effects:
            if isinstance(effect, StringConditionalEffect):
                if effect.conditional() != "(true)":
                    effect.setStr(re.sub(r'b\[(\d+)]',
                                         lambda match: str(self.get_patch_by_diridx(int(match.group(1))).stateVar()),
                                         effect.conditional())
                                  )
                else:
                    effect.setStr("0")
        self._effects = effects

    def name(self):
        return self._name

    def getID(self):
        return self._id

    def stateSize(self):
        return self._stateSize

    def get_patch_by_diridx(self, dirIdx):
        return [p for p in self._patches if p.dirIdx() == dirIdx][0]

    def get_patch_by_idx(self, i):
        return self._patches[i]

    def patches(self):
        return self._patches

    def effectsTargeting(self, activation_var):
        return [e for e in self._effects if e.target() == activation_var]

    def patchConditional(self, patch):
        return "|".join(f"({e.conditional()})" for e in self.effectsTargeting(patch.activationVar()))


class PolycubesPatch:
    def __init__(self, id, color, direction, orientation, stateVar, activationVar):
        self._id = id
        self._color = color
        self._dirIdx = direction if isinstance(direction, int) else diridx(direction)
        self._oriIdx = orientation if isinstance(orientation, int) else (diridx(orientation))
        self._stateVar = stateVar
        self._activationVar = activationVar

    def dirIdx(self):
        return self._dirIdx

    def direction(self):
        return RULE_ORDER[self.dirIdx()]

    def alignDir(self):
        return RULE_ORDER[self._oriIdx]

    def stateVar(self):
        return self._stateVar

    def activationVar(self):
        return self._activationVar

    def to_pl_patch(self):
        relPosition = self.direction() / 2
        return Patch(self._id, self._color, relPosition, self.direction(), self.alignDir())



# TODO: integrate with C++ TLM / Polycubes
class Effect:
    def __init__(self, target):
        self._target = target

    def target(self):
        return self._target


class StringConditionalEffect(Effect):
    def __init__(self, conditional, target):
        super().__init__(target)
        self._conditional = conditional

    def conditional(self):
        return self._conditional

    def setStr(self, newStr):
        self._conditional = newStr
class DynamicEffect(Effect):
    def __init__(self, vars, target):
        super().__init__(target)
        self._vars = target

    def conditional(self):
        return "&".join([f"v" if v > 0 else f"!{-v}" for v in self._vars])


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
