import numpy as np
import pandas as pd
from itertools import chain

FACE_NAMES = ("left", "right", "bottom", "top", "back", "front")
RULE_ORDER = (
    np.array((-1, 0, 0)),
    np.array((1, 0, 0)),
    np.array((0, -1, 0)),
    np.array((0, 1, 0)),
    np.array((0, 0, -1)),
    np.array((0, 0, 1))
)


def load_rule(rule_str):
    return [
        {
            "patches": [
                parse_patch(patch, j)
                for j,patch in enumerate(cubeType.split("#")) if patch
            ],
            "effects": [],
            "state_size": 13,  # hardcode 13 species
            "typeName": f"CT{i}"
        }
        for i, cubeType in enumerate(rule_str.split("_"))
    ]


def parse_patch(patch_str, j):
    static = patch_str.count(":") == 2
    patch_components = patch_str.split(":")
    patch_dir = {
        "x": RULE_ORDER[j][0],
        "y": RULE_ORDER[j][1],
        "z": RULE_ORDER[j][2]
    }
    alignDir = {
        "x": RULE_ORDER[int(patch_components[1])][0],
        "y": RULE_ORDER[int(patch_components[1])][1],
        "z": RULE_ORDER[int(patch_components[1])][2]
    }
    return {
        "activation_var": patch_components[2] if not static else
        (j + len(RULE_ORDER) + 1 if patch_components[2].strip() else 0),
        "state_var": patch_components[3] if not static else j + 1,
        "color": patch_components[0],
        "dir": patch_dir,
        "alignDir": alignDir
    }


def diridx(a):
    return np.all(np.array(RULE_ORDER) == np.array(list(a))[np.newaxis, :], axis=1).nonzero()[0][0]


class PolycubeRuleCubeType:
    def __init__(self, ct_dict):
        # if old format (regretting so many decisions rn)
        self.conditionals = ct_dict['conditionals']
        if "colors" in ct_dict:
            self.name = ct_dict['name']
            self.colors = ct_dict['colors']
            self.alignments = ct_dict['alignments']
        else:
            # TODO: this, properly
            self.name = ct_dict['typeName']
            self.colors = [p['color'] for p in ct_dict['patches']]
            self.alignments = [p['alignDir'] for p in ct_dict['patches']]


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
