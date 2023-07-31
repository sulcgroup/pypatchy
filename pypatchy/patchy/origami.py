import numpy as np
from copy import deepcopy
from oxDNA_analysis_tools.UTILS.RyeReader import *
from Bio.SVDSuperimposer import SVDSuperimposer
from sympy import sequence
from tools.helix import generate_helix_cords
from tools.mgl_particle import parse_mgl_file
from pypatchy.oxutil import Base, merge_write_confs, merge_tops, read_top, write_top, write_oxview
from json import loads
import os
from random import choice
import itertools
# import nupack

dist = np.linalg.norm
normalize = lambda v: v / dist(v)


def get_patch_cmss(conf, patch_positions):
    # get the centers of mass of the patches
    patch_cmss = []
    for patch in patch_positions:
        cms = np.zeros(3)
        for i in patch:
            cms += conf.positions[i]
        patch_cmss.append(
            cms / len(patch)
        )
    return np.array(patch_cmss)


def position_particles(conf, particles, scale_factor):
    placed_confs = []  # output prom_p
    # everyone's most favorite aligning tool
    sup = SVDSuperimposer()
    pl = len(particles)
    for i, particle in enumerate(particles):
        print(f"{i + 1}/{pl}", end="\r")
        m1 = []
        for patch in particle.patches[:len(patch_cmss)]:
            m1.append(patch.pos / scale_factor)
        m1 = np.array(m1)  # generate refference

        m2 = []
        # compile list of patch centers of mass
        for pcms in patch_cmss:
            m2.append([pcms[0], pcms[1], pcms[2]])
        m2 = np.array(m2)  # generate thing to rotateclusters

        sup.set(m1, m2)
        sup.run()
        rot, tran = sup.get_rotran()  # NOTE: our system has to be positioned at cms = (0,0,0)

        pconf = deepcopy(conf)
        pconf.box = box / scale_factor  # magic numbers needed again for things not to clash
        pconf.positions = np.einsum('ij, ki -> kj', rot, pconf.positions) + particle.cms / scale_factor  # + tran
        pconf.a1s = np.einsum('ij, ki -> kj', rot, pconf.a1s)
        pconf.a3s = np.einsum('ij, ki -> kj', rot, pconf.a3s)

        # we finished the positioning
        placed_confs.append(pconf)
    print()
    return placed_confs


def asign_cords(conf, indices, cords):
    for cds, id in zip(cords, indices):
        conf.positions[id] = cds[0]
        conf.a1s[id] = cds[1]
        conf.a3s[id] = cds[2]


def particle_pair_candidates(particles, particle_bond_delta=1.2):
    handeled_candidates = []
    for i, p1 in enumerate(particles):
        for j, p2 in enumerate(particles):
            if i != j and dist(p1.cms - p2.cms) <= particle_bond_delta:
                if (i, j) not in handeled_candidates and not (j, i) in handeled_candidates:  # prevent repeats
                    handeled_candidates.append((i, j))
                    yield (i, p1), (j, p2)


def colors_pair(patch1, patch2):
    if colorpairings is None:
        return patch1.color == patch2.color
    else:
        return colorpairings[patch1.color] == patch2.color  # relationship assumed to be reflexive


sequence_storage = {}


# def patches_to_bind(p1, p2, patch_delta, cos_theta_max):
#     for q,patch_1 in enumerate(p1.patches):
#         for z,patch_2 in enumerate(p2.patches):
#             if dist(p1.cms + patch_1.pos - (p2.cms+patch_2.pos)) <= patch_delta and colors_pair(patch_1, patch_2):
#                 #https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
#                 patch1norm = patch_1.pos / np.linalg.norm(patch_1.pos)
#                 patch2norm = patch_2.pos / np.linalg.norm(patch_2.pos)
#                 costheta = abs(float(np.clip(np.dot(patch1norm, patch2norm), -1.0, 1.0)))
#                  # confusing,ly, cos theta-max is the cosine of the maximum angle
#                  # so we check if cos theta is LESS than cos theta-max
#                 if costheta >= cos_theta_max:
#                     yield (q, patch_1), (z, patch_2)

def patches_to_bind(particle_1, particle_2, patch_delta, cos_theta_max):
    possible_bindings = list(itertools.product(enumerate(particle_1.patches), enumerate(particle_2.patches)))
    possible_bindings = [pair for pair in possible_bindings if colors_pair(pair[0][1], pair[1][1])]
    possible_bindings.sort(key=lambda pair: dist(particle_1.cms + pair[0][1].pos - (particle_2.cms + pair[1][1].pos)))
    handled_p1 = [False for _ in particle_1.patches]
    handled_p2 = [False for _ in particle_2.patches]
    for (q, patch_1), (z, patch_2) in possible_bindings:
        if not handled_p1[q] and not handled_p2[z]:
            if dist(particle_1.cms + patch_1.pos - (particle_2.cms + patch_2.pos)) <= patch_delta and colors_pair(
                    patch_1, patch_2):
                # https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
                patch1norm = patch_1.pos / np.linalg.norm(patch_1.pos)
                patch2norm = patch_2.pos / np.linalg.norm(patch_2.pos)
                costheta = abs(float(np.clip(np.dot(patch1norm, patch2norm), -1.0, 1.0)))
                # confusing,ly, cos theta-max is the cosine of the maximum angle
                # so we check if cos theta is LESS than cos theta-max
                if costheta >= cos_theta_max:
                    handled_p1[q] = handled_p2[z] = True
                    yield (q, patch_1), (z, patch_2)
            else:  # can do this because list is sorted
                break


def patch_positions_to_bind(patch_positions1, patch_positions2, conf1, conf2):
    pps2 = np.array([conf2.positions[pid] for pid in patch_positions2])

    perms = np.array(list(itertools.permutations([conf1.positions[pid] for pid in patch_positions1])))

    diffs = pps2[np.newaxis, :, :] - perms  # vector difference
    distancesqrd = np.sum(np.multiply(diffs, diffs), axis=(2,))  # dot product, gives us distance squared
    sums = np.sum(distancesqrd, axis=(1,))

    bestmatch = np.argmin(sums)
    return zip(
        list(itertools.permutations(patch_positions1))[bestmatch],
        patch_positions2)

    # pairs = [] # we store pairs of corresponding indicesreverse
    # pp_pos2 = deepcopy(patch_positions2) #to be expendable
    # # go over every patch in the list for 1st particle
    # for patch_id1 in patch_positions1:
    #     # calculate the distances to all patches of particle 2
    #     dists = [dist(conf1.positions[patch_id1] - conf2.positions[id]) for id in pp_pos2]
    #     # figure out the index of the min particle 75
    #     min_dist = min(dists)
    #     id_min = dists.index(min_dist)
    #     # eliminate that index from the positions to scan against
    #     pairs.append(
    #         (patch_id1, pp_pos2.pop(id_min))
    #     )
    # return pairs # the closest patches on the 2 particles provided


def generate_3p_ids(patch_id, sticky_length):
    return range(patch_id, patch_id + sticky_length)


def rc(s):
    mp = {"A": "T", "T": "A", "C": "G", "G": "C"}
    return "".join([mp[c] for c in s][::-1])


def modify_strand3p(top, strand_id, seq):
    connection_lenght = len(seq)
    # we know we work with the 3 end so it's the beginning + sticky_length we have to modify
    new_bases = [Base(t, b.p3, b.p5) for t, b in zip(seq, top.strands[strand_id].bases[0:connection_lenght])]
    new_bases.extend(top.strands[strand_id].bases[connection_lenght:])
    # empty previous bases as we work with tuples and we can't just reasign
    top.strands[strand_id].bases.clear()
    top.strands[strand_id].bases.extend(new_bases)


'''
Generates a set of reverse-complimentary sequences for each pair of colors
TODO: incorporate Petr's script to make it automatically avoid generating sequences that are
similar to other sticky endss
'''


def generateSequences(sticky_length):
    seqs = {}
    # iterate through color pairings
    for color in colorpairings:
        compliment = colorpairings[color]
        # if the compliment of this color has already been assigned a sequence
        if compliment in seqs:
            # set this sequences's color to the reverse compliment of its complement's sequence
            seqs[color] = rc(seqs[compliment])
        else:
            # assign this sequence a random color
            seqs[color] = "".join(choice(["A", "T", "C", "G"]) for i in range(sticky_length))
    return seqs


def bind_particles3p(particles,
                     patch_positions,
                     placed_confs,
                     topologies,
                     base2strand,
                     spacer_length=16,
                     particle_delta=1.2,
                     bond_length=0.4,
                     cos_theta_max=0.95,
                     seqs=None):
    # # todo: specify binding sequence
    # bss ="".join(choice(["A","T","C","G"]) for i in range(sticky_length)) # our bs sequence
    # rc_bss = rc(bss) # reverse compliment
    # # add spacer to bss

    # loop possible pairs of particles
    bondcount = 0
    patch_occupancies = [[False for _ in p.patches] for p in particles]
    # loop particles
    for (i, p1), (j, p2) in particle_pair_candidates(particles, particle_delta):
        # loop through the patch pairs on each particle that can bind
        for (q, patch1), (z, patch2) in patches_to_bind(p1, p2, bond_length, cos_theta_max):
            if patch_occupancies[i][q] or patch_occupancies[j][z]:
                continue
            conf1, conf2 = placed_confs[i], placed_confs[j]
            top1, top2 = topologies[i], topologies[j]
            # for the positions of each of the 2 patches that are about to bind to each other
            for patch_id1, patch_id2 in patch_positions_to_bind(patch_positions[q],
                                                                patch_positions[z],
                                                                conf1, conf2):
                # retrieve sequences from map & add spacers
                seq1 = seqs[patch1.color] + "T" * spacer_length
                seq2 = seqs[patch2.color] + "T" * spacer_length
                sticky_length = len(seq1)
                # generate bonded helix between the particles
                cords1, cords2 = generate_helix_cords(sticky_length,
                                                      start_pos=conf1.positions[patch_id1],
                                                      dir=conf1.a3s[patch_id1],
                                                      perp=conf1.a1s[patch_id1])
                id1s = generate_3p_ids(patch_id1, sticky_length)
                id2s = generate_3p_ids(patch_id2, sticky_length)
                # make sure overhangs comply with the helix
                asign_cords(conf1, id1s, cords1)
                asign_cords(conf2, id2s, cords2)
                # lets modify the topology somehow
                # 1st figure out the strand index
                sid1 = base2strand[patch_id1]
                sid2 = base2strand[patch_id2]

                modify_strand3p(top1, sid1, seq1)
                modify_strand3p(top2, sid2, seq2)
                # add bond helix nucleotides to cluster lists
                for x in id1s:
                    clusters[i][x] = bondcount + len(particles)
                for x in id2s:
                    clusters[j][x] = bondcount + len(particles)
                patch_occupancies[i][q] = patch_occupancies[j][z] = True
                bondcount += 1
    print(f"Created {bondcount} bonds between particles");
    return {"bondcount": bondcount}