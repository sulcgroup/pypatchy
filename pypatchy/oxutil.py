# This file was written by god-knows-who god-knows-when for god-knows-what reason, god-knows-when
# Some of this might be redundant with oxpy or oat
# some of this might be candidates for merging with oxpy or oat
# who knows? the universe is a complecated and confusing place
from pathlib import Path
from typing import IO, Iterable, Union

import numpy as np
from collections import namedtuple
import json
from datetime import datetime

from pypatchy.util import rotation_matrix

TopInfo = namedtuple('TopInfo', ['nbases', 'nstrands', 'strands'])
Strand = namedtuple('Strand', ['id', 'bases'])
Base = namedtuple('Base', ['type', 'p3', 'p5'])


def write_configuration_header(f: IO, conf):
    f.write(f't = {int(conf.time)}\n')
    f.write(f"b = {' '.join(conf.box.astype(str))}\n")
    f.write(f"E = {' '.join(conf.energy.astype(str))}\n")


def write_configuration(f, conf):
    for p, a1, a3 in zip(conf.positions, conf.a1s, conf.a3s):
        f.write('{} {} {} 0.0 0.0 0.0 0.0 0.0 0.0\n'.format(' '.join(p.astype(str)), ' '.join(a1.astype(str)),
                                                            ' '.join(a3.astype(str))))


def read_top(path: Union[str, Path]):
    base2strand = {}
    if isinstance(path, str):
        path = Path(path)  # i swear i'm a real scientist
    assert path.is_file()
    with path.open("r") as file:
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
    strands = [Strand(strand_id, []) for strand_id in range(1, nstrands + 1)]
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


def write_oxview(tops, confs, clusters, file_path: Path):
    assert file_path.parent.exists()
    oxview_json = {
        "box": confs[0].box.tolist(),
        "date": datetime.now().isoformat(),
        "forces": [],
        "selections": []
    }

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
    with file_path.open("w") as f:
        json.dump(oxview_json, f)


# this shit was originally in a file called helix.py
# i cannot be held resposible for this
BASE_BASE = 0.3897628551303122
POS_BASE = 0.4  # I think this is the conversion factor from base pairs to oxDNA units????
CM_CENTER_DS = POS_BASE + 0.2
POS_BACK = -0.4
FENE_EPS = 2.0


def get_rotation_matrix(axis: np.ndarray,
                        angle: Union[int, float, np.float64, np.float32],
                        units: Union[None, str] = None):
    """
    Gets. a rotation matrix around an axis? using rodreguis' formula?
    The main thing this provides is flexability with multiple types
    """
    # if additional angle info provided
    if units:
        if units in ["degrees", "deg", "o"]:
            angle = (np.pi / 180.) * angle  # degrees to radians
            # angle = np.deg2rad (anglest[0])
        elif units == "bp":
            # Notice that the choice of 35.9 DOES NOT correspond to the minimum free energy configuration.
            # This is usually not a problem, since the structure will istantly relax during simulation, but it can be
            # if you need a configuration with an equilibrium value for the linking number.
            # The minimum free energy angle depends on a number of factors (salt, temperature, length, and possibly more),
            # so if you need a configuration with 0 twist make sure to carefully choose a value for this angle
            # and force it in some way (e.g. by changing the angle value below to something else in your local copy).
            # Allow partial bp turns
            angle = angle * (np.pi / 180.) * 35.9
            # Older versions of numpy don't implement deg2rad()
            # angle = int(anglest[0]) * np.deg2rad(35.9)
        # otherwise assume radians
    return rotation_matrix(axis, angle)
    # axis = np.array(axis)
    # axis /= np.sqrt(np.dot(axis, axis))
    #
    # ct = np.cos(angle)
    # st = np.sin(angle)
    # olc = 1. - ct
    # x, y, z = axis
    #
    # return np.array([[olc * x * x + ct, olc * x * y - st * z, olc * x * z + st * y],
    #                  [olc * x * y + st * z, olc * y * y + ct, olc * y * z - st * x],
    #                  [olc * x * z - st * y, olc * y * z + st * x, olc * z * z + ct]])


# the position of a series of DNA nucleotides, where each position is a tuple of numpy arrays
# the first element of the tuple is the position vector, the second and third elements are the a1 and a3 vectors
SequenceConfList = list[tuple[np.ndarray, np.ndarray, np.ndarray]]


def generate_helix_coords(bp: int,
                          start_pos: np.ndarray = np.array([0, 0, 0]),
                          helix_direction: np.ndarray = np.array([0, 0, 1]),
                          perp: Union[None, bool, np.ndarray] = None,
                          rot: float = 0.,
                          double: bool = True,
                          # circular=False,
                          # DELTA_LK=0,
                          # BP_PER_TURN=10.34,
                          ds_start=None,
                          ds_end=None,
                          # force_helicity=False
                          ) -> Union[tuple[SequenceConfList, SequenceConfList],
                                     SequenceConfList]:
    """
    Generates oxview coordinates for a DNA helix

    Parameters:
        bp: the length of the sequence for which to generate coordinates
        start_pos: xyz coords of the start position of the helix. default: 0,0,0 (origin)
        helix_direction: a vector for the direction of the helix
        perp: ???????
        rot: ?????
        double: whether to create coordinates for a double helix. defaults to true
        ds_start: the point where the sequence becomes double-stranded???
        ds_end: the point where the sequence stops being double-stranded???


    """
    # we need numpy array for these
    start_pos = np.array(start_pos, dtype=float)
    helix_direction = np.array(helix_direction, dtype=float)
    if ds_start is None:
        ds_start = 0
    if ds_end is None:
        ds_end = bp

    # we need to find a vector orthogonal to dir
    # compute magnitude of helix direction vector
    dir_norm = np.sqrt(np.dot(helix_direction, helix_direction))
    # if the vector is zero-length
    if dir_norm < 1e-10:
        # arbitrary default helix direction
        helix_direction = np.array([0, 0, 1])
    else:
        # normalize helix direction vector
        helix_direction /= dir_norm

    if perp is None or perp is False:
        v1 = np.random.random_sample(3)
        v1 -= helix_direction * (np.dot(helix_direction, v1))
        v1 /= np.sqrt(sum(v1 * v1))
    else:
        v1 = perp

    # Setup initial parameters
    ns1: SequenceConfList = []
    # and we need to generate a rotational matrix
    R0 = get_rotation_matrix(helix_direction, rot)
    R = get_rotation_matrix(helix_direction, 1, "bp")
    a1 = v1
    a1 = np.dot(R0, a1)
    rb = np.array(start_pos)
    a3 = helix_direction

    # Add nt in canonical double helix
    for i in range(bp):
        ns1.append((rb - CM_CENTER_DS * a1, a1, a3))  # , sequence[i]))
        if i != bp - 1:
            a1 = np.dot(R, a1)
            rb += a3 * BASE_BASE

    # Fill in complement strand
    if double:
        ns2: SequenceConfList = []
        for i in reversed(range(ds_start, ds_end)):
            # Note that the complement strand is built in reverse order
            nt = ns1[i]
            a1 = -nt[1]
            a3 = -nt[2]
            nt2_cm_pos = -(FENE_EPS + 2 * POS_BACK) * a1 + nt[0]
            ns2.append((nt2_cm_pos, a1, a3,))  # 3-sequence[i]))
        return ns1, ns2
    else:
        return ns1


def generate_spacer(spacer_length: int,
                    start_position: np.ndarray,
                    end_position: np.ndarray,
                    perp: Union[None, bool, np.ndarray] = None,
                    rot: float = 0.,
                    stretch: bool = False
                    ) -> SequenceConfList:
    """
    Generates coords for a single-stranded spacer sequence between two points
    """
    coords: SequenceConfList = generate_helix_coords(spacer_length,
                                                     start_pos=start_position,
                                                     helix_direction=end_position - start_position,
                                                     perp=perp,
                                                     rot=rot,
                                                     double=False)

    seq_coords_magnitude = np.linalg.norm(coords[0][0] - coords[-1][0])
    assert abs(seq_coords_magnitude - spacer_length * POS_BASE) < 1, \
        "Length of generated sequence does not match expected length!"
    if stretch:
        stretch_factor = spacer_length * POS_BASE / seq_coords_magnitude
        coords = [
            (
                (pos - start_position) * stretch_factor + + start_position,
                a1,
                a3
            ) for pos, a1, a3 in coords
        ]
    return coords


def generate_3p_ids(patch_id, seq_length):
    """
    Returns the 3-prime residue IDs of nucleotides in the sticky end starting at the provided residue IDs
    """
    return range(patch_id, patch_id + seq_length)


def assign_coords(conf,
                  indices: Iterable[int],
                  coords: SequenceConfList):
    """
    Updates a conf to set positions, a1s, and a3s at the given index to the values
    specified in the passed set of coords
    """
    for cds, idx in zip(coords, indices):
        conf.positions[idx] = cds[0]
        conf.a1s[idx] = cds[1]
        conf.a3s[idx] = cds[2]

