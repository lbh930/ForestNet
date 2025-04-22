import math
import random

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from tools.common import process_config, vec3
from tools.gen_mesh import Mesh, generate_tree_mesh, export_meshes_to_obj
from tools.gen_nodes import TreeNode, gen_tree


def _draw_circle_at_node(ax, node: TreeNode, circle_points: int = 32):
    """Draw a circular cross‑section of the node."""
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

    theta = np.linspace(0, 2 * math.pi, circle_points)
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
    ax.plot(X, Y, Z, color="r", linewidth=0.8)


def visualize_tree(node: TreeNode, ax, parent_pos=None):
    """Recursively visualize the tree."""
    if node.is_main:
        ax.scatter(node.position.x, node.position.y, node.position.z, c="g", s=10)
    else:
        ax.scatter(node.position.x, node.position.y, node.position.z, c="b", s=10)
    if parent_pos is not None:
        ax.plot([parent_pos.x, node.position.x], [parent_pos.y, node.position.y], [parent_pos.z, node.position.z], "k-", linewidth=0.5)
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
    tree_config = forest_config.get("tree_config", {})
    width = forest_config.get("width")
    length = forest_config.get("length")
    tree_count = forest_config.get("tree_count")
    minimal_distance = forest_config.get("minimal_distance")

    print("generating forest with config:")
    print(forest_config)

    use_random_height = "average_height" in forest_config
    if use_random_height:
        avg_height = forest_config["average_height"]
        var_height = forest_config.get("variance_height", 1.0)
    else:
        default_height = tree_config.get("Height", 10.0)

    positions = []
    max_tries = tree_count * 10
    attempts = 0

    print("---starting generation---")

    while len(positions) < tree_count and attempts < max_tries:
        attempts += 1
        x = random.uniform(0, width)
        y = random.uniform(0, length)
        candidate = vec3(x, y, 0)
        too_close = any((candidate - p).length() < minimal_distance for p in positions)
        if not too_close:
            positions.append(candidate)

    print(f"Actually placed {len(positions)} trees (requested {tree_count}).")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("3D Forest Visualization")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    limits = {"x": [], "y": [], "z": []}
    tree_meshes: list[tuple[str, Mesh]] = []
    tree_gen_count = 0

    for i, pos in enumerate(positions):
        if use_random_height:
            h = max(0.5, random.gauss(avg_height, var_height))
        else:
            h = default_height

        cur_tree_config = tree_config.copy()
        cur_tree_config["Height"] = h
        cur_tree_config = process_config(cur_tree_config)

        starting_radius = cur_tree_config.get("DBH", 0.5) / 2
        max_levels = cur_tree_config.get("maximum_levels", 4)
        angle_limit = cur_tree_config.get("Branching_Angle_Range:", (15, 45))
        up_straight = cur_tree_config.get("Up_Straightness", 0.6)
        step_sz = cur_tree_config.get("Step_Size", 1)
        branch_prob = cur_tree_config.get("Branching_Probability", 0.4)
        curve_range = cur_tree_config.get("Curvature_Range", (0.0, 0.25))
        rad_coeff = cur_tree_config.get("Radius_Coefficient", 0.5)
        len_coeff = cur_tree_config.get("Length_Coefficient", 0.33)
        sympodial_chance = cur_tree_config.get("Sympodial_Chance", 0.3)
        side_branch_decay = cur_tree_config.get("Side_Branch_Decay", 1.2)

        tree_gen_count += 1

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
            length_coefficient=len_coeff,
            sympodial_chance=sympodial_chance,
            max_tree_height=h,
            side_branch_decay=side_branch_decay,
        )

        print("generated tree with height ", h, " and starting radius ", starting_radius)

        _offset_tree_positions(tree, pos)
        _accumulate_limits(tree, limits)

        tree_mesh = generate_tree_mesh(tree, radial_segments=16)

        print("built mesh with ", len(tree_mesh.vertices), " vertices and ", len(tree_mesh.faces), " faces")

        tree_meshes.append((f"tree_{i:04d}", tree_mesh))

        if tree_gen_count % 5 == 0:
            print("generated ", tree_gen_count, "trees")

    print("generated total of ", tree_gen_count, " trees")

    if limits["x"]:
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

    total_vertices = sum(len(m.vertices) for _, m in tree_meshes)
    total_faces = sum(len(m.faces) for _, m in tree_meshes)
    print(f"Forest mesh has {total_vertices} vertices and {total_faces} faces.")

    x_min, x_max = min(limits["x"]), max(limits["x"])
    y_min, y_max = min(limits["y"]), max(limits["y"])
    z_min, z_max = min(limits["z"]), max(limits["z"])
    print(f"Bounding box: x({x_min}, {x_max}), y({y_min}, {y_max}), z({z_min}, {z_max})")

    print("---exporting obj file---")
    export_meshes_to_obj("test_forest.obj", tree_meshes)
    print("Forest mesh exported to 'test_forest.obj'.")


def main():
    from presets.forests import L1W_forest_config
    run_forest_simulation(L1W_forest_config)


if __name__ == "__main__":
    main()