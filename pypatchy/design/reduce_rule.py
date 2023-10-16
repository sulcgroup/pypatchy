import copy
import itertools
import json
import os.path
import sys
from typing import Union, Generator

import numpy as np

from pypatchy import util
from pypatchy.polycubeutil.polycubesRule import diridx, PolycubesRule, DynamicEffect, PolycubeRuleCubeType, PolycubesPatch


class RuleReducer:
    rule: PolycubesRule
    control_vars = dict[str, set[int]]
    rename_counter: int

    def __init__(self, rule: PolycubesRule):
        self.rule = rule
        self.control_vars: dict[str, set[int]] = {ct.name(): set() for ct in self.rule.particles()}
        self.rename_counter = 0

    def minimize(self) -> Generator[PolycubesRule, None, None]:
        """
        Recursive function that introduces allostery to minimzie the number of particle types
        required for this rule
        """
        extra_space = lambda cts: 6 - (cts[0].num_patches() + cts[1].num_patches())
        # weed out particles that can't overlap or which already have allostery
        cube_type_pairs = [(ct1, ct2) for ct1, ct2 in itertools.combinations(self.rule.particles(), 2)
                           if extra_space((ct1, ct2)) >= 0] # and ct1.state_size() == 1 and ct2.state_size() == 1]
        # sort cube type pairs
        cube_type_pairs.sort(key=extra_space)
        for ct1, ct2 in cube_type_pairs:
            if self.try_reduce_cube_pair(ct1, ct2):
                yield self.rule
                yield from self.minimize()
                break

    def add_allosteric_control(self,
                               patch: PolycubesPatch,
                               cube_type: PolycubeRuleCubeType,
                               state_patch_off: set[int],
                               state_patch_on: set[int]):
        # if patch doesn't already have a state var, assign one
        if not patch.state_var():
            state_var = cube_type.add_state_var()
            patch.set_state_var(state_var)
        else:
            state_var = patch.state_var()
        # add an effect to turn on all state vars except the ones that repress this patch
        # newcube.add_effect(DynamicEffect([state_var], state_any_ct1))
        for v in self.control_vars[cube_type.name()].difference(state_patch_off):
            cube_type.add_effect(DynamicEffect([state_var], v))

        # if the patch does not already have an activation var
        if not patch.activation_var():
            assert len(state_patch_off) == 1
            # we can safely presume on-var list length is 1?
            patch.set_activation_var(-list(state_patch_off)[0])
        elif -patch.activation_var() not in self.control_vars[cube_type.name()]:
            # do De Morgan's laws come into play here? i think so.
            old_activation_var = patch.activation_var()
            new_activation_var = cube_type.add_state_var()
            if old_activation_var < 0:
                # repressable
                # either the old activation var or is_any_ct1 should repress
                cube_type.add_effect(DynamicEffect([old_activation_var], new_activation_var))
                for v in state_patch_on:
                    cube_type.add_effect(DynamicEffect([v], new_activation_var))
                patch.set_activation_var(-new_activation_var)
            else:
                # smashing into limits of dynamic model? cannot flip a patch twice
                patch.set_activation_var(new_activation_var)
                cube_type.add_effect(DynamicEffect([old_activation_var, *[-v for v in state_patch_on]], new_activation_var))

    # def add_allosteric_control(self,
    #                            patch: PolycubesPatch,
    #                            cube_type: PolycubeRuleCubeType,
    #                            state_patch_on,
    #                            state_patch_off,
    #                            state_var = 0):
    #
    #     # handle state variable
    #     if not patch.state_var() and state_var:
    #         patch.set_state_var(state_var)
    #     elif not patch.state_var() and not state_var:
    #         patch.set_state_var(cube_type.add_state_var())
    #         cube_type.add_effect(DynamicEffect([patch.state_var()], state_patch_on))
    #     elif patch.state_var() and state_var:
    #         # and-gated?
    #         cube_type.add_effect(DynamicEffect([patch.state_var(), state_var], state_patch_on))
    #
    #     # handle activation variable
    #     # grab existing patch activation var (0 if non allosteric)
    #     old_activation_var = patch.activation_var()
    #
    #     # if the patch isn't already allosterically controlled, this becomes easy!
    #     if not old_activation_var:
    #         # add activation variable. make it repressable
    #         patch.set_activation_var(-state_patch_off)
    #         # new dynamic effect to set state var any ct2 to true when patch is bound
    #     # if patch is allosterically controlled an NOT by an existing behavior
    #     elif -old_activation_var not in self.control_vars[cube_type.name()]:
    #         # set new activation var
    #         new_activation_var = cube_type.add_state_var()
    #         patch.set_activation_var(new_activation_var)
    #         # make new activation var req old activation var and not other state being on
    #         cube_type.add_effect(DynamicEffect([-state_patch_off, old_activation_var], new_activation_var))
    #     # binding on patch should inactivate other on-state

    # def add_allosteric_control(self,
    #                            patch: PolycubesPatch,
    #                            cube_type: PolycubeRuleCubeType,
    #                            off_state_var: int,
    #                            other_off_state: int,
    #                            ) -> dict[int, int]:
    #     """
    #
    #     need to deal with two things: patches with existing state vars
    #     and patches with existing activation vars
    #
    #     Args:
    #         patch:
    #         cube_type:
    #         off_state_var:
    #         other_off_state:
    #
    #     Returns:
    #         a dict of state variables that have been reassigned
    #     """
    #
    #     var_remap: dict[int, int] = dict()
    #
    #     # handle state variable
    #     if patch.state_var() == 0:
    #         patch_state_var = cube_type.add_state_var()
    #     else:
    #         # *kill bill sirens*
    #         patch_old_state_var = patch.state_var()
    #         patch_state_var = cube_type.add_state_var()
    #         # handle this later
    #         var_remap[patch_old_state_var] = patch_state_var
    #
    #     patch.set_state_var(patch_state_var)
    #
    #     # handle activation variable
    #     # grab existing patch activation var (0 if non allosteric)
    #     old_activation_var = patch.activation_var()
    #     # assign state variables for new patch
    #     new_activation_var = cube_type.add_state_var()
    #     # assign state variables for new patch
    #
    #     # if the patch isn't already allosterically controlled, this becomes easy!
    #     if old_activation_var == 0:
    #         # add activation variable. make it repressable
    #         patch.set_activation_var(-off_state_var)
    #         # new dynamic effect to set state var any ct2 to true when patch is bound
    #     else:
    #         # set new activation var
    #         patch.set_activation_var(new_activation_var)
    #         # make new activation var req old activation var and not other state being on
    #         cube_type.add_effect(DynamicEffect([-off_state_var, old_activation_var], new_activation_var))
    #     # binding on patch should inactivate other on-state
    #     cube_type.add_effect(DynamicEffect([patch_state_var], other_off_state))
    #     return var_remap

    def try_reduce_cube_pair(self, ct1: Union[PolycubeRuleCubeType, int], ct2: Union[PolycubeRuleCubeType, int]):
        if isinstance(ct1, int):
            ct1 = self.rule.particle(ct1)
        if isinstance(ct2, int):
            ct2 = self.rule.particle(ct2)
        r = no_overlap_rotation(ct1, ct2)
        if r is not False:
            # clone cube type 1 and apply changes
            # applying in place has potential... issues
            newcube: PolycubeRuleCubeType = copy.deepcopy(ct2)
            newcube.set_name(f"CCT{self.rename_counter}")
            self.rename_counter += 1
            assert ct2.state_size() == newcube.state_size()

            # ct1_vars_remap: dict[int, int] = dict()
            # ct2_vars_remap: dict[int, int] = dict()

            # loop existing ct2 patches
            # for patch in ct2.patches():
            #
            #     newcube.add_patch(patch)
            #
            #     varupdate = self.add_allosteric_control(patch, newcube, state_any_ct1, state_any_ct2)
            #     ct1_vars_remap.update(varupdate)

            # for e in ct1.effects():
            #     newcube.add_effect(copy.deepcopy(e))
            # ct2.shift_state(ct1.state_size())
            # don't need to add the tautology state
            newcube.shift_state(ct1.state_size() - 1)
            newcube.set_name("newcube")

            self.control_vars[newcube.name()] = {v + ct1.state_size() - 1 for v in self.control_vars[ct2.name()]}

            # for this formulation, i'm assigning each patch a unique state and activation variable
            # and creating two new intermediate variables for our two cube type behaviors

            # add control vars from cubes to merge, shifting as approppriate
            ct1_control_vars = {v for v in self.control_vars[ct1.name()]}
            ct2_control_vars = {v + ct1.state_size() - 1 for v in self.control_vars[ct2.name()]}

            self.control_vars[newcube.name()] = ct1_control_vars.union(ct2_control_vars)

            # if ct1 does not already have control vars
            if len(ct1_control_vars) == 0:
                state_any_ct1 = newcube.add_state_var()
                self.control_vars[newcube.name()].add(state_any_ct1)
                ct1_control_vars = {state_any_ct1}

            # self.control_vars[newcube]
            if len(ct2_control_vars) == 0:
                state_any_ct2 = newcube.add_state_var()
                self.control_vars[newcube.name()].add(state_any_ct2)
                ct2_control_vars = {state_any_ct2}

            # self.control_vars["newcube"].add(v + ct1.state_size() - 1)

            for e in ct1.effects():
                newcube.add_effect(copy.deepcopy(e))

            # iter existing patches (inherited from ct2)
            for cube2_patch in newcube.patches():
                self.add_allosteric_control(cube2_patch, newcube, ct2_control_vars, ct1_control_vars)

            # reassign patches from ct1 new state/activation vars

            for cube1_patch in ct1.patches():
                # rotate patch
                patch = cube1_patch.rotate(r)

                newcube.add_patch(patch)
                self.add_allosteric_control(patch, newcube, ct1_control_vars, ct2_control_vars)

            # # remap patches
            # for cube, vars_remap in zip([ct1, ct2], [ct1_vars_remap, ct2_vars_remap]):
            #     for pold, pnew in zip(cube.patches(), newcube.patches()):
            #         # activation vars
            #         if pold.activation_var() in vars_remap:
            #             pnew.set_activation_var(vars_remap[pold.activation_var()])
            #         # state vars
            #         if pold.activation_var() in vars_remap:
            #             pnew.set_state_var(vars_remap[pold.state_var()])
            #     # remap dynamic effects
            #     for eold, enew in zip(cube.effects(), newcube.effects()):
            #         if eold.target() in vars_remap:
            #             enew.set_target(vars_remap[eold.target()])
            #         for var in eold.sources():
            #             if var in vars_remap:
            #                 new_sources = eold.sources()
            #                 new_sources.remove(var)
            #                 new_sources.append(vars_remap[var])
            #                 enew.set_sources(new_sources)
            self.rule.remove_cube_type(ct1)
            self.rule.remove_cube_type(ct2)
            self.rule.add_particle(newcube)
            self.rule.reindex()
            return True
        else:
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

