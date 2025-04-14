import math
import matplotlib.pyplot as plt
import numpy as np
from tools.common import vec3, process_config
from tools.gen_mesh import Mesh, export_mesh_to_obj, generate_tree_mesh
from tools.gen_nodes import TreeNode, gen_tree
from mpl_toolkits.mplot3d import Axes3D
import random

def _draw_circle_at_node(ax, node: TreeNode, circle_points=32):
    """Draw a circular cross-section of the node."""
    d = node.direction
    if d.length() < 1e-6:
        d = vec3(0, 0, 1)
    d = d.normalized()

    up = vec3(0, 0, 1)
    if abs(d.dot(up)) > 0.99:
        up = vec3(1, 0, 0)
    e1 = d.cross(up)
    if e1.length() < 1e-6:
        up = vec3(0, 1, 0)
        e1 = d.cross(up)
    e1 = e1.normalized()
    e2 = d.cross(e1).normalized()

    theta = np.linspace(0, 2 * np.pi, circle_points)
    X, Y, Z = [], [], []
    r = node.radius
    c = node.position
    for t in theta:
        pos = c + e1 * (r * math.cos(t)) + e2 * (r * math.sin(t))
        X.append(pos.x)
        Y.append(pos.y)
        Z.append(pos.z)
    X.append(X[0])
    Y.append(Y[0])
    Z.append(Z[0])
    ax.plot(X, Y, Z, color='r', linewidth=0.8)


def visualize_tree(node: TreeNode, ax, parent_pos=None):
    """Recursively visualize the tree."""
    ax.scatter(node.position.x, node.position.y, node.position.z, c='g', s=10)
    if parent_pos:
        ax.plot(
            [parent_pos.x, node.position.x],
            [parent_pos.y, node.position.y],
            [parent_pos.z, node.position.z],
            'k-', linewidth=0.5
        )
    _draw_circle_at_node(ax, node)
    for child in node.children:
        visualize_tree(child, ax, node.position)


def _offset_tree_positions(node: TreeNode, offset: vec3):
    """Offset all positions of the tree by a vector."""
    node.position = node.position + offset
    for child in node.children:
        _offset_tree_positions(child, offset)


def _accumulate_limits(node: TreeNode, limits_dict):
    """Store node positions in limits dict."""
    limits_dict["x"].append(node.position.x)
    limits_dict["y"].append(node.position.y)
    limits_dict["z"].append(node.position.z)
    for child in node.children:
        _accumulate_limits(child, limits_dict)


def run_forest_simulation(forest_config):
    """Generate a forest of trees based on the given config."""
    # Process tree config
    tree_config_raw = forest_config.get("tree_config", {})
    tree_config = process_config(tree_config_raw)

    # Read forest params
    width = forest_config.get("width", 50)
    length = forest_config.get("length", 50)
    tree_count = forest_config.get("tree_count", 100)
    minimal_distance = forest_config.get("minimal_distance", 5)
    
    print ("generating forest with config:")
    print (forest_config)

    # Decide how to set tree height
    use_random_height = "average_height" in forest_config
    if use_random_height:
        avg_height = forest_config["average_height"]
        var_height = forest_config.get("variance_height", 1.0)
    else:
        default_height = tree_config.get("Height", 10.0)

    # Try placing trees so they respect minimal distance
    positions = []
    max_tries = tree_count * 10
    attempts = 0
    
    print ("---starting generation---")
    
    while len(positions) < tree_count and attempts < max_tries:
        attempts += 1
        x = random.uniform(0, width)
        y = random.uniform(0, length)
        candidate = vec3(x, y, 0)
        too_close = any((candidate - p).length() < minimal_distance for p in positions)
        if not too_close:
            positions.append(candidate)

    print(f"Actually placed {len(positions)} trees (requested {tree_count}).")

    # Set up figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("3D Forest Visualization")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Collect axis limits
    limits = {"x": [], "y": [], "z": []}

    # Combine meshes
    forest_mesh = Mesh()
    
    tree_gen_count = 0

    for pos in positions:
        # Pick a tree height
        if use_random_height:
            h = max(0.5, random.gauss(avg_height, var_height))
        else:
            h = default_height

        # Update config with current height
        tree_config["Height"] = h

        # Gather other params
        starting_radius = tree_config.get("starting_radius", 0.2)
        max_levels = tree_config.get("maximum_levels", 4)
        angle_limit = tree_config.get("Branching_Angle_Range:", (15, 45))
        up_straight = tree_config.get("Up_Straightness", 0.6)
        step_sz = tree_config.get("Step_Size", 1)
        branch_prob = tree_config.get("Branching_Probability", 0.4)
        curve_range = tree_config.get("Curvature_Range", (0.0, 0.25))
        rad_coeff = tree_config.get("Radius_Coefficient", 0.5)
        len_coeff = tree_config.get("Length_Coefficient", 0.33)

        # Generate a tree
        tree = gen_tree(
            starting_radius=starting_radius,
            starting_direction=vec3(0, 0, 1),
            total_length=h,
            maximum_levels=max_levels,
            branching_angle_limit=angle_limit,
            step_size=step_sz,
            base_branching_probability=branch_prob,
            curvature_range=curve_range,
            up_straightness=up_straight,
            radius_coefficient=rad_coeff,
            length_coefficient=len_coeff
        )

        # Offset tree
        _offset_tree_positions(tree, pos)

        # # The `visualize_tree` function is recursively visualizing a tree structure in a 3D plot.
        # Here's a breakdown of what it does:
        
        visualize_tree(tree, ax)
        _accumulate_limits(tree, limits)

        # Generate mesh for this tree
        tree_mesh = generate_tree_mesh(tree, radial_segments=16)

        # Merge meshes
        base_vert_count = len(forest_mesh.vertices)
        forest_mesh.vertices.extend(tree_mesh.vertices)
        for face in tree_mesh.faces:
            forest_mesh.faces.append([idx + base_vert_count for idx in face])
            
        tree_gen_count += 1
        if tree_gen_count % 10 == 0:
            print ("generated ", tree_gen_count, "trees")
            
    print ("generated ", tree_gen_count, " trees")

    # Set axis limits
    if limits["x"] and limits["y"] and limits["z"]:
        x_min, x_max = min(limits["x"]), max(limits["x"])
        y_min, y_max = min(limits["y"]), max(limits["y"])
        z_min, z_max = min(limits["z"]), max(limits["z"])
        max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
        x_mid = (x_max + x_min) / 2
        y_mid = (y_max + y_min) / 2
        z_mid = (z_max + z_min) / 2

        ax.set_xlim(x_mid - max_range / 2, x_mid + max_range / 2)
        ax.set_ylim(y_mid - max_range / 2, y_mid + max_range / 2)
        ax.set_zlim(z_mid - max_range / 2, z_mid + max_range / 2)

    plt.show()

    print ("---exporting obj file---")
    # Export the forest mesh
    export_mesh_to_obj("test_forest.obj", forest_mesh.vertices, forest_mesh.faces)
    print("Forest mesh exported to 'test_forest.obj'.")


def main():
    from presets.forests import broadleaf_forest_config
    run_forest_simulation(broadleaf_forest_config)


if __name__ == "__main__":
    main()
