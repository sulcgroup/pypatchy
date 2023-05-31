import numpy as np
import pandas as pd
from itertools import chain

FACE_NAMES = ("left", "right", "bottom", "top", "back", "front")
RULE_ORDER = (
    np.array((-1,  0,  0)),
    np.array(( 1,  0,  0)),
    np.array(( 0, -1,  0)),
    np.array(( 0,  1,  0)),
    np.array(( 0,  0, -1)),
    np.array(( 0,  0,  1))
)

def load_rule():
    pass

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
    index=map(lambda ct: ct.name, rule) )
    
    return df