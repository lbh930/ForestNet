# mesh_gen.py

import math

from tools.common import vec3


class Mesh:
    """
    A simple container class to store mesh data.
    - vertices: a list of vec3 (positions)
    - faces: a list of triangles, each triangle is a tuple (i1, i2, i3) 
             referencing vertex indices
    """
    def __init__(self):
        self.vertices = []  # list of vec3
        self.faces = []     # list of (i1, i2, i3)


def export_meshes_to_obj(path: str, mesh_list: list[tuple[str, Mesh]]):
    with open(path, "w") as f:
        vert_offset = 0
        for name, mesh in mesh_list:
            for v in mesh.vertices:
                f.write(f"v {v.x} {v.y} {v.z}\n")
            f.write(f"g {name}\n")
            for face in mesh.faces:
                a, b, c = [idx + vert_offset + 1 for idx in face]
                f.write(f"f {a} {b} {c}\n")
            vert_offset += len(mesh.vertices)


def export_mesh_to_obj(filename, vertices, faces):
    """
    Exports mesh data to a simple OBJ file.

    :param filename:  The output OBJ filename
    :param vertices:  List of vec3 positions
    :param faces:     List of (i1, i2, i3) face indices (0-based)
    """
    with open(filename, 'w') as f:
        # Write out vertex positions
        for v in vertices:
            f.write(f"v {v.x} {v.y} {v.z}\n")
        # Write out faces
        # Note: OBJ format uses 1-based indexing
        for (i1, i2, i3) in faces:
            f.write(f"f {i1 + 1} {i2 + 1} {i3 + 1}\n")

def _build_local_frame(direction: vec3):
    """
    Builds a local orthonormal frame (forward, side1, side2) from the given direction.
    
    - forward is the normalized direction
    - side1, side2 are perpendicular to forward and to each other
    """
    forward = direction.normalized()
    if forward.length() < 1e-6:
        # If the direction is almost zero, return a default frame
        return vec3(0, 0, 1), vec3(1, 0, 0), vec3(0, 1, 0)

    # Attempt to find a perpendicular vector by crossing with an up_candidate
    up_candidate = vec3(0, 1, 0)
    # If they are nearly parallel, change the candidate
    if abs(forward.dot(up_candidate)) > 0.999:
        up_candidate = vec3(1, 0, 0)

    side1 = forward.cross(up_candidate).normalized()
    side2 = forward.cross(side1).normalized()
    return forward, side1, side2

def _create_ring(mesh: Mesh, node, radial_segments=8):
    """
    Creates a circular ring around the node's position, 
    oriented perpendicular to the node's direction, 
    and with the node's radius.
    
    Returns a list of vertex indices corresponding to the ring.
    """
    forward, side1, side2 = _build_local_frame(node.direction)
    ring_indices = []

    for i in range(radial_segments):
        angle = 2.0 * math.pi * i / radial_segments
        x = math.cos(angle) * node.radius
        y = math.sin(angle) * node.radius
        # Position on the ring
        pos = node.position + side1 * x + side2 * y
        mesh.vertices.append(pos)
        ring_indices.append(len(mesh.vertices) - 1)

    return ring_indices

def _connect_rings(mesh: Mesh, ringA, ringB):
    """
    Connects two rings with the same segment count to form a cylindrical 
    or conical side surface using two triangles per segment.

    :param mesh:   The Mesh to which new faces are added
    :param ringA:  List of vertex indices for the first ring
    :param ringB:  List of vertex indices for the second ring
    """
    radial_segments = len(ringA)
    for i in range(radial_segments):
        i1 = ringA[i]
        i2 = ringA[(i + 1) % radial_segments]
        i3 = ringB[i]
        i4 = ringB[(i + 1) % radial_segments]

        # Two triangles for each segment
        # Triangle 1: (i1, i2, i4)
        mesh.faces.append((i1, i2, i4))
        # Triangle 2: (i1, i4, i3)
        mesh.faces.append((i1, i4, i3))

def _cap_ring(mesh: Mesh, ring, center_pos, invert=False):
    """
    Adds a circular cap (filled disk) to the ring.

    :param mesh:        The Mesh to which the cap is added
    :param ring:        The list of vertex indices for the ring
    :param center_pos:  The position of the cap center
    :param invert:      If True, invert triangle order (flip normals)
    """
    center_index = len(mesh.vertices)
    mesh.vertices.append(center_pos)

    radial_segments = len(ring)
    for i in range(radial_segments):
        i1 = ring[i]
        i2 = ring[(i + 1) % radial_segments]
        if not invert:
            mesh.faces.append((center_index, i1, i2))  # (center, ring[i], ring[i+1])
        else:
            mesh.faces.append((center_index, i2, i1))  # flip order

def generate_tree_mesh(root_node, radial_segments=8):
    """
    Generates a mesh from the given root TreeNode (which may contain a hierarchy of children).
    
    Each node corresponds to a circular ring. 
    The parent ring and child ring are connected to form a 'cylindrical' or 'conical' segment.
    
    - If node.is_root == True, add a bottom cap.
    - If a node has no children, add a top cap on its ring.
    - If a node has multiple children, each child ring will connect 
      to the same parent ring (this can create overlapping geometry at the branching node).

    :param root_node:       A TreeNode, possibly with children
    :param radial_segments: Number of segments to approximate the circular cross-section
    :return:                A Mesh object containing the vertices and faces
    """
    mesh = Mesh()

    # A dictionary to store ring vertex indices for each node, avoiding duplicate creation
    ring_map = {}

    def traverse(node):
        if node not in ring_map:
            # Create and cache the ring for this node
            ring_map[node] = _create_ring(mesh, node, radial_segments)
            
            # If this node is root, optionally cap the bottom
            if node.is_root:
                _cap_ring(mesh, ring_map[node], node.position, invert=True)

        # The current node's ring
        ringA = ring_map[node]

        # If there are no children, it is a leaf. Cap the top.
        if len(node.children) == 0:
            _cap_ring(mesh, ringA, node.position, invert=False)

        # For each child, connect the parent's ring to the child's ring
        for child in node.children:
            if child not in ring_map:
                ring_map[child] = _create_ring(mesh, child, radial_segments)
            ringB = ring_map[child]

            _connect_rings(mesh, ringA, ringB)
            
            # Traverse the child to build its segment(s)
            traverse(child)

    # Start recursion from root
    traverse(root_node)

    return mesh