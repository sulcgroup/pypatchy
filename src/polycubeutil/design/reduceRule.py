import copy
import itertools
import json
import os.path
import sys

import numpy as np

import util
from polycubeutil.polycubesRule import diridx, PolycubesRule, DynamicEffect


class RuleReducer:
    def __init__(self, rule):
        self.rule = rule

    def minimize(self):
        """
        Recursive function that introduces allostery to minimzie the number of particle types
        required for this rule
        """
        extra_space = lambda cts: 6 - (cts[0].num_patches() + cts[1].num_patches())
        # weed out particles that can't overlap or which already have allostery
        cube_type_pairs = [(ct1, ct2) for ct1, ct2 in itertools.combinations(self.rule.particles(), 2)
                           if extra_space((ct1, ct2)) >= 0 and ct1.state_size() == 1 and ct2.state_size() == 1]
        # sort cube type pairs
        cube_type_pairs.sort(key=extra_space)
        for ct1, ct2 in cube_type_pairs:
            r = no_overlap_rotation(ct1, ct2)
            if r is not False:
                # add patches from ct1 (rotated by r) to ct2

                # TODO: how to handle multi-multi-types?

                # for this formulation, i'm assigning each patch a unique state and activation variable
                # and creating two new intermediate variables for our two cube type behaviors
                state_any_ct1 = ct2.add_state_var()
                state_any_ct2 = ct2.add_state_var()
                #
                # if self.singlevars:
                #     # loop existing ct2 patches
                #     for patch in ct2.patches():
                #         # assign state variables for new patch
                #         patch.set_state_var(ct2.add_state_var())
                #         # add activation variable. make it repressable
                #         patch.set_activation_var(-ct2.add_state_var())
                #         # new dynamic effect to set state var any ct2 to true when patch is bound
                #         ct2.add_effect(DynamicEffect([patch.state_var()], state_any_ct2))
                #         # new dynamic effect to set activation var when c2 var functionality goes off
                #         ct2.add_effect(DynamicEffect([state_any_ct1], -patch.activation_var()))
                #
                #     # reassign patches from ct1 new state/activation vars
                #     for p in ct1.patches():
                #         # rotate patch
                #         patch = p.rotate(r)
                #         # assign state variables for new patch
                #         patch.set_state_var(ct2.add_state_var())
                #         # add activation variable
                #         patch.set_activation_var(-ct2.add_state_var())
                #         # new dynamic effect to set state var any ct1 to true when patch is bound
                #         ct2.add_effect(DynamicEffect([patch.state_var()], state_any_ct1))
                #         # new dynamic effect to set activation var when c2 var functionality goes off
                #         ct2.add_effect(DynamicEffect([state_any_ct2], -patch.activation_var()))
                #
                #         # add patch
                #         ct2.add_patch(patch)
                # else:

                # loop existing ct2 patches
                for patch in ct2.patches():
                    # assign state variables for new patch
                    patch.set_state_var(ct2.add_state_var())
                    # add activation variable. make it repressable
                    patch.set_activation_var(-state_any_ct1)
                    # new dynamic effect to set state var any ct2 to true when patch is bound
                    ct2.add_effect(DynamicEffect([patch.state_var()], state_any_ct2))

                # reassign patches from ct1 new state/activation vars
                for p in ct1.patches():
                    # rotate patch
                    patch = p.rotate(r)
                    # assign state variables for new patch
                    patch.set_state_var(ct2.add_state_var())
                    # add activation variable
                    patch.set_activation_var(-state_any_ct2)
                    # new dynamic effect to set state var any ct1 to true when patch is bound
                    ct2.add_effect(DynamicEffect([patch.state_var()], state_any_ct1))
                    # add patch
                    ct2.add_patch(patch)

                self.rule.remove_cube_type(ct1)
                yield self.rule
                copy.deepcopy(self).minimize()

def no_overlap_rotation(ct1, ct2):
    ct2_diridxs = ct2.diridxs() # don't recompute this unnessecarily
    # loop all rotations
    for rot in util.getRotations():
        overlaps = False
        # check each patch in ct1
        for p in ct1.patches():
            # if this patch overlaps with a patch on ct2 when this rotation is applied
            # might be possible to optimize this with linear algebra
            d = np.matmul(rot, p.direction()).round()
            if diridx(d) in ct2_diridxs:
                # set overlap flag
                overlaps = True
                # break
                break
        # if the program didn't find any overlapping patches
        if not overlaps:
            return rot
    return False


if __name__ == "__main__":
    if os.path.isfile(sys.argv[1]):
        rule = PolycubesRule(rule_json=json.load(sys.argv[1]))
    else:
        rule = PolycubesRule(rule_str=sys.argv[1])
    reducer = RuleReducer(rule)
    # print(json.dumps(rule.toJSON()))
    print("Starting Rule")
    print(rule)
    for r in reducer.minimize():
        print(r)
        # print(json.dumps(newRule))

