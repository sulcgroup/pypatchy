#!/usr/bin/env python
from __future__ import annotations
import itertools
import math
from pathlib import Path
from typing import Union
# This file loads patchy particle file from topology and Configuration
import copy

from pypatchy.patchy.pl.plparticle import PLPatch, PLPatchyParticle, PLParticleSet

"""
Library for handling patchy-lock (PL) systems in pypatchy
Where constants etc. exist they should be the same as / similar to the Patchy interactions
in C++
"""

myepsilon = 0.00001

# cutoff patch-patch distance, after which interaction cannot occur
PATCHY_CUTOFF = 0.18
# copied from oxDNA defs.h
# bounded arc-cosine?
LRACOS = lambda x: 0 if x > 1 else math.pi if x < -1 else math.acos(x)


def load_patches(filename: Union[str, Path],
                 num_patches=0) -> list[PLPatch]:
    if isinstance(filename, str):
        filename = Path(filename)
    j = 0
    Np = 0
    patches = [PLPatch() for _ in range(num_patches)]

    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if len(line) > 1 and line[0] != '#':
                if 'patch_' and '{' in line:
                    strargs = []
                    k = j + 1
                    while '}' not in lines[k]:
                        strargs.append(lines[k].strip())
                        k = k + 1
                    patch = PLPatch()
                    # print 'Loaded patch',strargs
                    patch.init_from_string(strargs)
                    index = patch.type_id()
                    # flexable patch indexing
                    # probably not optimized speed wise but optimized for flexibility
                    if index >= len(patches):
                        patches += [None for _ in range(index - len(patches) + 1)]
                    patches[index] = patch
                    Np += 1
            j = j + 1

    if num_patches != 0 and Np != num_patches:
        raise IOError('Loaded %d patches, as opposed to the desired %d types ' % (Np, num_patches))
    return patches


def load_particles(filename: Union[str, Path],
                   patch_types: list[PLPatch],
                   num_particles=0) -> PLParticleSet:
    particles: list[PLPatchyParticle] = [PLPatchyParticle() for _ in range(num_particles)]
    Np = 0
    with open(filename, 'r') as f:
        lines = f.readlines()
        j = 0
        for line in lines:
            line = line.strip()
            if len(line) > 1 and line[0] != '#':
                if 'particle_' and '{' in line:
                    strargs = []
                    k = j + 1
                    while '}' not in lines[k]:
                        strargs.append(lines[k].strip())
                        k = k + 1
                    particle = PLPatchyParticle()
                    # print 'Loaded particle ',strargs
                    particle.init_from_string(strargs)
                    particle.fill_patches(patch_types)
                    index: int = particle.type_id()
                    # flexable particle indexing
                    # probably not optimized speed wise but optimized for flexibility
                    if index >= len(particles):
                        particles += [None for _ in range(index - len(particles) + 1)]
                    particles[index] = copy.deepcopy(particle)
                    Np += 1
            j = j + 1
    return PLParticleSet(particles)


def export_interaction_matrix(patches, filename="interactions.txt"):
    with open(filename, 'w') as f:
        f.writelines(
            [
                f"patchy_eps[{p1.type_id()}][{p2.type_id()}] = 1.0\n"
                for p1, p2 in itertools.combinations(patches, 2)
                if p1.color() == p2.color()
            ]
        )

# TODO: something with this class
