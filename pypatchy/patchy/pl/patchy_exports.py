# module created Jun 30 2025
# purpose: functions to export (and import?) patchy particles,
# requiring libraries not otherwise needed for pypatchy
from pygltflib import *

from .plscene import PLPSimulation

from pygltflib import *

import numpy as np
import struct

from math import sin, cos, pi
from scipy.spatial.transform import Rotation as R
from ...util import selectColor


def generate_uv_sphere(radius=1.0, segments=16, rings=16):
    vertices = []
    indices = []

    # Add north pole
    vertices.extend([0.0, radius, 0.0])

    # Add middle vertices
    for y in range(1, rings):
        theta = np.pi * y / rings
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        for x in range(segments):
            phi = 2 * np.pi * x / segments
            x_pos = radius * np.cos(phi) * sin_theta
            y_pos = radius * cos_theta
            z_pos = radius * np.sin(phi) * sin_theta
            vertices.extend([x_pos, y_pos, z_pos])

    # Add south pole
    vertices.extend([0.0, -radius, 0.0])

    north_idx = 0
    south_idx = 1 + (rings - 2) * segments

    # Top cap
    for x in range(segments):
        next_x = (x + 1) % segments
        indices.extend([north_idx, 1 + x, 1 + next_x])

    # Middle bands
    for y in range(rings - 2):
        for x in range(segments):
            next_x = (x + 1) % segments
            i0 = 1 + y * segments + x
            i1 = 1 + y * segments + next_x
            i2 = i0 + segments
            i3 = i1 + segments
            indices.extend([i0, i2, i1])
            indices.extend([i1, i2, i3])

    # Bottom cap
    base_idx = 1 + (rings - 2) * segments
    for x in range(segments):
        next_x = (x + 1) % segments
        indices.extend([base_idx + x, base_idx + next_x, south_idx])

    return vertices, indices



def generate_unit_cube():
    """
    Returns vertices and indices for a unit cube centered at the origin with side length 1.
    Vertices are flat list of floats.
    Indices are triangle indices using counter-clockwise winding.
    """
    # 8 cube vertices
    verts = [
        [-0.5, -0.5, -0.5],
        [ 0.5, -0.5, -0.5],
        [ 0.5,  0.5, -0.5],
        [-0.5,  0.5, -0.5],
        [-0.5, -0.5,  0.5],
        [ 0.5, -0.5,  0.5],
        [ 0.5,  0.5,  0.5],
        [-0.5,  0.5,  0.5],
    ]
    # 12 triangles (two per face)
    indices = [
        0, 1, 2, 2, 3, 0,  # back
        4, 5, 6, 6, 7, 4,  # front
        0, 4, 7, 7, 3, 0,  # left
        1, 5, 6, 6, 2, 1,  # right
        3, 2, 6, 6, 7, 3,  # top
        0, 1, 5, 5, 4, 0   # bottom
    ]
    # flatten
    flat_verts = [coord for v in verts for coord in v]
    return flat_verts, indices



def to_gltf(scene: PLPSimulation, export_to: Path, particle_shape="sphere", center=True, color_saturation=85, color_value=80):
    """
    exports patchy scene as glft file. written with help of chatGPT

    Parameters:
        scene: scene to export
        export_to: path to export file
        particle_shape: shape of particle to export. can be "sphere" or "cube" to automatically generate spheres / cubes, or a list of (verts, faces) tuples
    """
    gltf = GLTF2(asset=Asset(version="2.0"))

    # for centering, if nessecary
    offset = scene.box_size() / 2

    # --- Geometry selection ---
    if particle_shape == "sphere":
        geometry_vertices, geometry_indices = generate_uv_sphere(radius=0.5)
    elif particle_shape == "cube":
        geometry_vertices, geometry_indices = generate_unit_cube()
    else:
        raise ValueError(f"Unsupported particle_shape: {particle_shape}")

    vertex_bytes = struct.pack(f"<{len(geometry_vertices)}f", *geometry_vertices)
    index_bytes = struct.pack(f"<{len(geometry_indices)}H", *geometry_indices)
    vertex_len = len(vertex_bytes)
    index_len = len(index_bytes)
    full_blob = vertex_bytes + index_bytes

    verts_np = np.array(geometry_vertices).reshape(-1, 3)
    vmin = verts_np.min(axis=0).tolist()
    vmax = verts_np.max(axis=0).tolist()

    # --- Buffer + Accessors ---
    gltf.buffers = [Buffer(byteLength=len(full_blob))]
    gltf.bufferViews = [
        BufferView(buffer=0, byteOffset=0, byteLength=vertex_len, target=ARRAY_BUFFER),
        BufferView(buffer=0, byteOffset=vertex_len, byteLength=index_len, target=ELEMENT_ARRAY_BUFFER)
    ]
    gltf.accessors = [
        Accessor(bufferView=0, byteOffset=0, componentType=FLOAT, count=len(geometry_vertices) // 3, type=VEC3, min=vmin, max=vmax),
        Accessor(bufferView=1, byteOffset=0, componentType=UNSIGNED_SHORT, count=len(geometry_indices), type="SCALAR")
    ]

    # --- Meshes + Materials per particle type ---
    gltf.materials = []
    gltf.meshes = []
    type_to_mesh_idx = {}
    for type_id in range(scene.num_particle_types()):
        rgb = selectColor(type_id, saturation=color_saturation, value=color_value, fmt="rgb")
        if isinstance(rgb, str) and rgb.startswith("rgb"):
            rgb_vals = [float(x) for x in rgb[4:-1].split(",")]
        else:
            rgb_vals = rgb
        mat = Material(pbrMetallicRoughness={
            "baseColorFactor": rgb_vals + [1.0],
            "metallicFactor": 0.1,
            "roughnessFactor": 0.8
        })
        gltf.materials.append(mat)

        prim = Primitive(attributes={"POSITION": 0}, indices=1, material=type_id)
        if particle_shape != "cube":
            print("Warning: faces will not be shaded smooth, fix in Blender")
        mesh = Mesh(primitives=[prim])
        gltf.meshes.append(mesh)
        type_to_mesh_idx[type_id] = type_id

    # --- Nodes per particle ---
    gltf.nodes = []
    for p in scene.particles():
        rotation = None
        if particle_shape == "cube":
            # particle rotation in pypatchy doesn't match glTF rotation basis, take transpose
            # to account for this
            rot_matrix = p.rotmatrix().T
            quat = R.from_matrix(rot_matrix).as_quat(scalar_first=False)  # [x, y, z, w]
            rotation = quat.tolist()

        position = p.position()
        if center:
            position -= offset
        gltf.nodes.append(Node(
            translation=position.tolist(),
            rotation=rotation,
            mesh=type_to_mesh_idx[p.get_type()],
            name=p.name()
        ))

    # --- Scene ---
    gltf.scene = 0
    gltf.scenes = [Scene(nodes=list(range(len(gltf.nodes))))]

    # --- Save ---
    if export_to.suffix == ".glb":
        gltf.set_binary_blob(full_blob)
        gltf.save_binary(str(export_to))
    else:
        bin_path = export_to.with_suffix(".bin")
        bin_path.write_bytes(full_blob)
        gltf.buffers[0].uri = bin_path.name
        gltf.save(str(export_to))
