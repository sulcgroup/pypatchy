import itertools
import logging
from datetime import datetime

import libtlm
import networkx as nx
import numpy as np

from scipy.spatial.transform import Rotation as R


from ..polycubeutil.polycube_structure import PolycubeStructure
from ..structure import TypedStructure
from ..util import get_log_dir, enumerateRotations, getSignedAngle
from ..polycubeutil.polycubesRule import RULE_ORDER, PolycubesRule

logging.basicConfig(filename=get_log_dir() / "SAT" / datetime.now().strftime("log_%Y-%m-%d-%H:%M.txt"))
logging.root.setLevel(logging.INFO)


def to_diridx(x: int) -> libtlm.DirIdx:
    return [libtlm.LEFT, libtlm.RIGHT,
            libtlm.BOTTOM, libtlm.TOP,
            libtlm.BACK, libtlm.FRONT][x]


def setup_logger(logger_name, file_path=None):
    if file_path is None:
        file_path = get_log_dir() / "SAT" / f"{logger_name}.log"
    logger = logging.getLogger(str(logger_name))
    logger.setLevel(logging.INFO)

    # create a file handler
    handler = logging.FileHandler(file_path)
    handler.setLevel(logging.INFO)

    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)

    return logger


## Utility functions
def patchRotToVec(i, rot):
    """ Get vector indicating patch rotation, given rotation state and face index

    Args:
        i (int): Index of patch [0...5]
        rot (int): Rotation state [0,1,2,3] = North , East, South, West

    Returns:
        vector: Patch rotation vector
    """
    v = getFACE_ROTATIONS()[i]
    axis = RULE_ORDER[i]
    angle = rot * np.pi / 2
    r = R.from_rotvec(angle * axis)
    return r.apply(v).round()


def getFACE_ROTATIONS():
    return [
        np.array([0, -1, 0]), np.array([0, 1, 0]),
        np.array([0, 0, -1]), np.array([0, 0, 1]),
        np.array([-1, 0, 0]), np.array([1, 0, 0])
    ]


def getFlatFaceRot():
    return [1, 1, 2, 0, 0, 0]


def getIndexOf(elem, array):
    for i, e in enumerate(array):
        if (e == elem).all():
            return i
    return -1


def coordsFromFile(path):
    with open(path) as f:
        return [[int(c) for c in line.strip('()\n').split(',')] for line in f]


def patchVecToRot(i, v):
    """ Get rotation state, given face index and patch rotation vector

    Args:
        i (int): Index of patch [0...5]
        v (vector): Patch rotation vector

    Returns:
        int: Rotation state [0,1,2,3] = North , East, South, West
    """
    angle = getSignedAngle(
        getFACE_ROTATIONS()[i],
        v,
        RULE_ORDER[i]
    )
    return int((angle * (2 / np.pi) + 4) % 4)


def forbidden_symmetries():
    # begin with an arbitrary reflective symmetry
    asym = {
        0: 1,
        1: 0,
        2: 2,
        3: 3,
        4: 4,
        5: 5
    }
    # apply rotations to our seed symmetry
    forbidden = [
        {
            idx_key: rot_dict[asym[idx_key]] for idx_key in rot_dict
        }
        for sym_key, rot_dict in enumerateRotations().items()
    ]
    # perform tests (TODO: remove once I'm more confident)
    assert len(forbidden) == 24
    assert all([
        not any([
            all(
                f[key] == sym[key] for key in range(6)
            ) for sym in enumerateRotations().values()]
        ) for f in forbidden]
    )
    return forbidden


def rotation_mapping_to_matrix(rotation_map: dict[int, int]) -> np.ndarray:
    """
    Written by chatGPT
    Convert a rotation map to a rotation matrix.
    """

    # Create a rotation matrix from the rotation map
    rotation_matrix = np.zeros((3, 3))

    for i in range(3):
        # Find the corresponding vector from the mapping
        # Note: we only need to consider vectors with positive first component (1, 3, 5 in RULE_ORDER)
        # Because a rotation is completely determined by where it sends these three vectors
        vector = RULE_ORDER[rotation_map[2 * i + 1]]

        # Add the vector as a column in the rotation matrix
        rotation_matrix[:, i] = vector
    assert np.linalg.det(rotation_matrix) == 1

    # Calculate the quaternion using the function from the previous part
    return rotation_matrix


def rotation_matrix_to_quaternion(rotation):
    """
    chatGPT wrote this code
    Convert a 3x3 rotation matrix to a quaternion.
    Is there a neater way to do this? unclear!!!
    """

    assert rotation.shape == (3, 3)
    trace = np.trace(rotation)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (rotation[2, 1] - rotation[1, 2]) * s
        y = (rotation[0, 2] - rotation[2, 0]) * s
        z = (rotation[1, 0] - rotation[0, 1]) * s
    else:
        if rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
            w = (rotation[2, 1] - rotation[1, 2]) / s
            x = 0.25 * s
            y = (rotation[0, 1] + rotation[1, 0]) / s
            z = (rotation[0, 2] + rotation[2, 0]) / s
        elif rotation[1, 1] > rotation[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
            w = (rotation[0, 2] - rotation[2, 0]) / s
            x = (rotation[0, 1] + rotation[1, 0]) / s
            y = 0.25 * s
            z = (rotation[1, 2] + rotation[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
            w = (rotation[1, 0] - rotation[0, 1]) / s
            x = (rotation[0, 2] + rotation[2, 0]) / s
            y = (rotation[1, 2] + rotation[2, 1]) / s
            z = 0.25 * s

    q = np.array([w, x, y, z])
    return q


def quaternion_inverse(q):
    # q is a numpy array of four elements [w, x, y, z]

    norm_squared = np.dot(q, q)  # Compute the square of the norm

    if norm_squared == 0:
        raise ValueError("The input quaternion is zero, which cannot be inverted.")

    # Compute the conjugate of the quaternion
    conjugate = np.array([q[0], -q[1], -q[2], -q[3]])

    # Compute the inverse by dividing the conjugate by the norm squared
    inverse = conjugate / norm_squared

    return inverse


def rot_idx_to_quaternion(idx: int) -> np.ndarray:
    return rot_map_to_quat(enumerateRotations()[idx])


def rot_map_to_quat(rot_map: dict[int, int]) -> np.ndarray:
    return rotation_matrix_to_quaternion(rotation_mapping_to_matrix(rot_map))


def countParticlesAndBindings(topology):
    """
    Returns the number of bindings and particles in a topology
    Doesn't take into account external bindings (for crystals) or nanoparticles
    Args:
        topology:

    Returns: a tuple of the number of particles and the number of bindings
    """
    pidsa = [x[0] for x in topology]
    pidsb = [x[2] for x in topology]
    particles = pidsa + pidsb
    return max(particles) + 1, len(topology)


def ruleToDec(ruleset):
    return '_'.join(
        '|'.join(
            "{}:{}".format(
                f['color'], f['orientation']
            ) for f in s
        ) for s in ruleset
    )


def parseDecRule(decRule):
    rule = []
    for s in decRule.split('_'):
        faces = []
        for face in s.split('|'):
            if face != '':
                color, orientation, conditional = [int(v) for v in face.split(':')]
            else:
                color = 0
                orientation = 0
            faces.append({
                'color': color,
                'orientation': orientation
            })
        rule.append(faces)
    return rule

def parseHexRule(hexRule):
    ruleset = []
    faces = []
    for i in range(0, len(hexRule), 2):
        if i % 12 == 0 and i != 0:
            ruleset.append(faces)
            faces = []
        face_hex = hexRule[i:i + 2]
        face_int = int(face_hex, 16)
        face_bin = bin(face_int)[2:].zfill(8)
        face = {}
        sign = int(face_bin[0], 2)
        face['color'] = int(face_bin[1:6], 2) * (-1 if sign else 1)
        face['orientation'] = int(face_bin[6:8], 2)
        face['conditional'] = ""  # hex conditionals can't be expressed in hex (TODO: yet?)
        faces.append(face)
    ruleset.append(faces)
    return ruleset


def ruleToHex(ruleset):
    hexRule = ''
    for rule in ruleset:
        for face in rule:
            sign = bin(face['color'] < 0)[2:]
            color = bin(abs(face['color']))[2:].zfill(5)
            orientation = bin(abs(face['orientation']))[2:].zfill(2)
            binStr = sign + color + orientation
            hexStr = hex(int(binStr, 2))[2:].zfill(2)
            hexRule += hexStr
    return hexRule


def to_xyz(vector):
    return {k: int(v) for k, v in zip(["x", "y", "z"], vector)}


def compute_coordinates(topology: frozenset) -> dict[int, np.array]:
    """
    Args:
        topology:

    Returns:

    """
    top_queue = list(topology)

    # Initialize the first coordinate at (0,0,0) and create dict to hold coordinates
    coord_dict = {top_queue[0][0]: np.zeros((3,))}

    # top_queue = top_queue[1:]  # pop zeroth element

    giveupcount = 0

    while len(top_queue) > 0 and giveupcount < 1e7:
        queue_len = len(top_queue)
        for j in range(queue_len):
            i = queue_len - j - 1
            loc1, dir1, loc2, dir2 = top_queue[i]
            assert -1 < dir1 < len(RULE_ORDER), f"Invalid direction index {dir1}"
            assert -1 < dir2 < len(RULE_ORDER), f"Invalid direction index {dir2}"
            if loc1 in coord_dict and loc2 in coord_dict:  # unclear how this happened
                top_queue = top_queue[:i] + top_queue[i + 1:]
                continue
            # assert loc1 not in coord_dict or loc2 not in coord_dict, \
            #     f"Both locations {loc1} and {loc2} have already been handled!"
            assert (RULE_ORDER[dir1] + RULE_ORDER[dir2] == np.zeros((3,))).all(), \
                f"Particles at {loc1} and {loc2} are not bound by opposite directional patches!!! " \
                f"Directions are {RULE_ORDER[dir1]} and {RULE_ORDER[dir2]}"
            if loc1 in coord_dict:
                # Compute the new coordinate by adding the direction vector to the current coordinate
                coord_dict[loc2] = coord_dict[loc1] + RULE_ORDER[dir1]
            elif loc2 in coord_dict:
                # Compute the new coordinate by adding the direction vector to the current coordinate
                coord_dict[loc1] = coord_dict[loc2] + RULE_ORDER[dir2]
            else:
                continue
            top_queue = top_queue[:i] + top_queue[i + 1:]

    if len(top_queue) > 0:
        raise Exception("Unable to construct a coordinate map! Perhaps topology is not connected?")

    assert len(coord_dict) == len(set(itertools.chain.from_iterable([(b[0], b[2]) for b in topology])))

    # Ensure minimum distance of 1 between structures by sorting and offsetting coordinates
    sorted_locs = sorted(coord_dict.keys())
    for i in range(1, len(sorted_locs)):
        diff = coord_dict[sorted_locs[i]] - coord_dict[sorted_locs[i - 1]]
        if np.abs(diff).max() < 1:
            coord_dict[sorted_locs[i]] += np.sign(diff) * (1 - np.abs(diff))

    return coord_dict


def build_graphs(topology):
    """
    warning: the following function was written by ChatGPT
    Args:
        topology:

    Returns:

    """
    # Initialize an empty graph and list to hold all graphs
    G = nx.Graph()
    graph_list = []

    for binding in topology:
        loc1, _, loc2, _ = binding
        # Add an edge to the graph for this binding
        G.add_edge(loc1, loc2)

    # Generate subgraphs for each connected component
    for component in nx.connected_components(G):
        subgraph = G.subgraph(component).copy()
        graph_list.append(subgraph)

    return graph_list


def toPolycube(rule: PolycubesRule, pc: libtlm.TLMPolycube) -> PolycubeStructure:
    return PolycubeStructure(rule=rule, cubes=[
        {
            "position": tlm_cube_info.getPosition(),
            "rotation": tlm_cube_info.getRotation(),
            "state": tlm_cube_info.getState(),
            "type": tlm_cube_info.getTypeID(),
            "personalName": f"cube_{tlm_cube_info.getUID()}"
        } for tlm_cube_info in pc.getCubes()
    ])


"""
implementation of TypedStructure without full cube types
"""


class TypedTopology(TypedStructure):
    special_types: dict[int, int]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "nanoparticles" in kwargs and len(kwargs["nanoparticles"]) > 0:
            self.special_types = {int(key): val for key, val in kwargs["nanoparticles"].items()}
        else:
            self.special_types = dict()

    def particle_type(self, particle_id: int) -> int:
        return self.special_types[particle_id] if particle_id in self.special_types else -1

    def get_particle_types(self) -> dict[int, int]:
        return self.special_types

    def get_topology(self) -> libtlm.TLMTopology:
        """
        converts to C++ library object
        """
        return libtlm.TLMTopology({
            (
                (i, to_diridx(di)),
                (j, to_diridx(dj))
            ) for i, di, j, dj in self.bindings_list},
            self.get_particle_types())
