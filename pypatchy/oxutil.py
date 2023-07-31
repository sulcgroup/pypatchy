# This file was written by god-knows-who god-knows-when for god-knows-what reason, god-knows-when
# Some of this might be redundant with oxpy or oat
# some of this might be candidates for merging with oxpy or oat
# who knows? the universe is a complecated and confusing place

import numpy as np
from collections import namedtuple
import json
from datetime import datetime

TopInfo = namedtuple('TopInfo', ['nbases', 'nstrands', 'strands'])
Strand = namedtuple('Strand', ['id', 'bases'])
Base = namedtuple('Base', ['type', 'p3', 'p5'])


def write_configuration_header(f, conf):
    f.write('t = {}\n'.format(int(conf.time)))
    f.write('b = {}\n'.format(' '.join(conf.box.astype(str))))
    f.write('E = {}\n'.format(' '.join(conf.energy.astype(str))))


def write_configuration(f, conf):
    for p, a1, a3 in zip(conf.positions, conf.a1s, conf.a3s):
        f.write('{} {} {} 0.0 0.0 0.0 0.0 0.0 0.0\n'.format(' '.join(p.astype(str)), ' '.join(a1.astype(str)),
                                                            ' '.join(a3.astype(str))))


def merge_write_confs(confs, path: str="out_f_merged.dat"):
    with open(path, "w") as file:
        write_configuration_header(file, confs[0])
        cl = len(confs)
        for i, conf in enumerate(confs):
            print(f"{i + 1}/{cl}", end="\r")
            write_configuration(file, conf)
    print()


def read_top(path):
    base2strand = {}
    with open(path) as file:
        lines = file.readlines()
    # get line info
    nbases, nstrands = map(int, lines[0].split())
    # generate placeholder for bases
    strands = [Strand(i, []) for i in range(1, nstrands + 1)]
    # generate the return object
    top_info = TopInfo(nbases, nstrands, strands)
    i = 0
    for line in lines[1:]:
        sid, t, p3, p5 = line.split()
        sid, p3, p5 = map(int, [sid, p3, p5])
        b = Base(t, p3, p5)
        top_info.strands[sid - 1].bases.append(b)
        base2strand[i] = sid - 1
        i += 1
    return top_info, base2strand


def merge_tops(tops):
    # as we can have arbitrary topologies
    # we need to iterate over them to figure out the number of bases \ strands
    nbases, nstrands = 0, 0
    for ti in tops:
        nbases += ti.nbases
        nstrands += ti.nstrands

        # generate placeholder for bases
    strands = [Strand(id, []) for id in range(1, nstrands + 1)]
    # generate the return object
    top_info = TopInfo(nbases, nstrands, strands)
    # now we have to update the bases
    offset = 0
    sid = 0
    # fill in the bases with the new base \ strand offset
    tl = len(tops)
    for id, ti in enumerate(tops):
        print(f"{id + 1}/{tl}", end="\r")
        if id > 0:
            offset += ti.nbases
        for strand in ti.strands:
            top_info.strands[sid].bases.extend([Base(b.type,
                                                     b.p3 + offset if b.p3 != -1 else -1,
                                                     b.p5 + offset if b.p5 != -1 else -1)
                                                for b in strand.bases])
            sid += 1
    print()
    return top_info


def write_top(top_info, path="out_f_merged.top"):
    with open(path, "w") as file:
        file.write(f"{top_info.nstrands} {top_info.nbases}\n")
        for id, strand in enumerate(top_info.strands):
            for b in strand.bases:
                file.write(f"{id + 1} {b.type} {b.p3} {b.p5}\n")


def write_oxview(tops, confs, clusters, path="out.oxview"):
    oxview_json = {}
    oxview_json["box"] = confs[
        0].box.tolist()  # at least in my experience, boxes will be the same for every conf        bid = 0

    oxview_json["date"] = datetime.now().isoformat()
    oxview_json["forces"] = []
    oxview_json["selections"] = []
    system = {
        "id": 0,
        "strands": []
    }
    sid = 0
    offset = 0
    for particle_idx, (top_info, conf_info) in enumerate(zip(tops, confs)):
        bid = 0
        for strand in top_info.strands:
            strand_json = {
                "class": "NucleicAcidStrand",
                "id": sid,
                "end3": float('inf'),  # set later
                "end5": 0,  # set later
                "monomers": []
            }
            for b in strand.bases:
                nucleotide = {
                    "a1": conf_info.a1s[bid].tolist(),
                    "a3": conf_info.a3s[bid].tolist(),
                    "class": "DNA",
                    # "cluster": 0,
                    "cluster": particle_idx if bid not in clusters[particle_idx] else clusters[particle_idx][bid],
                    "color": 0,  # NOT AN IMPORTANT BIT
                    "id": bid + offset,
                    "n3": b.p3 + offset if b.p3 != -1 else -1,
                    "n5": b.p5 + offset if b.p5 != -1 else -1,
                    "p": conf_info.positions[bid].tolist(),
                    "type": b.type
                }
                strand_json["end3"] = min(strand_json["end3"], bid + offset)
                strand_json["end5"] = max(strand_json["end5"], bid + offset)
                strand_json["monomers"].append(nucleotide)
                bid += 1
            system["strands"].append(strand_json)
            sid += 1
        offset += top_info.nbases
    oxview_json["systems"] = [system]
    with open(path, "w") as f:
        json.dump(oxview_json, f)


# this shit was originally in a file called helix.py
# i cannot be resposible for this
BASE_BASE = 0.3897628551303122
POS_BASE = 0.4
CM_CENTER_DS = POS_BASE + 0.2
POS_BACK = -0.4
FENE_EPS = 2.0


def get_rotation_matrix(axis, anglest):
    if not isinstance(anglest, (np.float64, np.float32, float, int)):
        if len(anglest) > 1:
            if anglest[1] in ["degrees", "deg", "o"]:
                angle = (np.pi / 180.) * anglest[0]
                # angle = np.deg2rad (anglest[0])
            elif anglest[1] in ["bp"]:
                # Notice that the choice of 35.9 DOES NOT correspond to the minimum free energy configuration.
                # This is usually not a problem, since the structure will istantly relax during simulation, but it can be
                # if you need a configuration with an equilibrium value for the linking number.
                # The minimum free energy angle depends on a number of factors (salt, temperature, length, and possibly more),
                # so if you need a configuration with 0 twist make sure to carefully choose a value for this angle
                # and force it in some way (e.g. by changing the angle value below to something else in your local copy).
                # Allow partial bp turns
                angle = float(anglest[0]) * (np.pi / 180.) * 35.9
                # Older versions of numpy don't implement deg2rad()
                # angle = int(anglest[0]) * np.deg2rad(35.9)
            else:
                angle = float(anglest[0])
        else:
            angle = float(anglest[0])
    else:
        angle = float(anglest)  # in degrees, I think

    axis = np.array(axis)
    axis /= np.sqrt(np.dot(axis, axis))

    ct = np.cos(angle)
    st = np.sin(angle)
    olc = 1. - ct
    x, y, z = axis

    return np.array([[olc * x * x + ct, olc * x * y - st * z, olc * x * z + st * y],
                     [olc * x * y + st * z, olc * y * y + ct, olc * y * z - st * x],
                     [olc * x * z - st * y, olc * y * z + st * x, olc * z * z + ct]])


def generate_helix_cords(bp, start_pos=np.array([0, 0, 0]), dir=np.array([0, 0, 1]), perp=None, rot=0., double=True,
                         circular=False, DELTA_LK=0, BP_PER_TURN=10.34, ds_start=None, ds_end=None,
                         force_helicity=False):
    # we need numpy array for these
    start_pos = np.array(start_pos, dtype=float)
    dir = np.array(dir, dtype=float)
    if ds_start == None:
        ds_start = 0
    if ds_end == None:
        ds_end = bp

    # we need to find a vector orthogonal to dir
    dir_norm = np.sqrt(np.dot(dir, dir))
    if dir_norm < 1e-10:
        dir = np.array([0, 0, 1])
    else:
        dir /= dir_norm

    if perp is None or perp is False:
        v1 = np.random.random_sample(3)
        v1 -= dir * (np.dot(dir, v1))
        v1 /= np.sqrt(sum(v1 * v1))
    else:
        v1 = perp

    # Setup initial parameters
    ns1 = []
    # and we need to generate a rotational matrix
    R0 = get_rotation_matrix(dir, rot)
    R = get_rotation_matrix(dir, [1, "bp"])
    a1 = v1
    a1 = np.dot(R0, a1)
    rb = np.array(start_pos)
    a3 = dir

    # Add nt in canonical double helix
    for i in range(bp):
        ns1.append([rb - CM_CENTER_DS * a1, a1, a3])  # , sequence[i]))
        if i != bp - 1:
            a1 = np.dot(R, a1)
            rb += a3 * BASE_BASE

    # Fill in complement strand
    if double == True:
        ns2 = []
        for i in reversed(range(ds_start, ds_end)):
            # Note that the complement strand is built in reverse order
            nt = ns1[i]
            a1 = -nt[1]
            a3 = -nt[2]
            nt2_cm_pos = -(FENE_EPS + 2 * POS_BACK) * a1 + nt[0]
            ns2.append([nt2_cm_pos, a1, a3])  # 3-sequence[i]))
        return ns1, ns2
    else:
        return ns1
