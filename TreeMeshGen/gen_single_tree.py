import math

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from presets.trees import broadleaf_config, oak_config, pine_config
from tools.common import process_config, vec3
from tools.gen_mesh import Mesh, export_mesh_to_obj, generate_tree_mesh
from tools.gen_nodes import TreeNode, gen_tree

def _draw_circle_at_node(ax, node: TreeNode, circle_points=32):
    """
    Draw a circle representing the node's cross section, perpendicular to its direction vector.
    :param ax: Matplotlib 3D axis.
    :param node: TreeNode that contains position, direction, and radius.
    :param circle_points: Number of points used to approximate the circle.
    """
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
    center = node.position
    for t in theta:
        pos = center + e1 * (r * math.cos(t)) + e2 * (r * math.sin(t))
        X.append(pos.x)
        Y.append(pos.y)
        Z.append(pos.z)
    X.append(X[0])
    Y.append(Y[0])
    Z.append(Z[0])
    ax.plot(X, Y, Z, color='r', linewidth=0.8)


def visualize_tree(node: TreeNode, ax, parent_position=None):
    """
    Recursively visualize the tree structure in 3D.
    Also draw a circle representing the node's radius, perpendicular to its direction.
    :param node: Current TreeNode to draw.
    :param ax: Matplotlib 3D axis.
    :param parent_position: Position of the parent node (for connecting lines).
    """
    if node.is_main:
        ax.scatter(node.position.x, node.position.y, node.position.z, c='g', s=10)
    else:
        ax.scatter(node.position.x, node.position.y, node.position.z, c='b', s=10)
        
    if parent_position is not None:
        ax.plot(
            [parent_position.x, node.position.x],
            [parent_position.y, node.position.y],
            [parent_position.z, node.position.z],
            'k-',
            linewidth=0.5
        )
    _draw_circle_at_node(ax, node)
    for child in node.children:
        visualize_tree(child, ax, node.position)


def run_tree_simulation(config):
    """
    Using the provided config, perform the following steps:
    
    1. Process the config to ensure consistency of Height and DBH.
    2. Set tree generation parameters based on config.
    3. Generate the tree, visualize it in 3D, and adjust the plot limits.
    4. Generate the tree mesh and export it as an OBJ file.
    """
    
    config = process_config(config)
    
    # Extract parameters from config.
    starting_radius = config.get("DBH", 0.25) / 2
    starting_direction = vec3(0, 0, 1)  # Vertical growth.
    total_length = config.get("Height", 10.0)
    maximum_levels = config.get("maximum_levels", 4)
    branching_angle_limit = config.get("Branching_Angle_Range:", (15, 45))
    up_straightness = config.get("Up_Straightness", 0.6)
    step_size = config.get("Step_Size", 1)
    base_branching_probability = config.get("Branching_Probability", 0.4)
    curvature_range = config.get("Curvature_Range", (0.0, 0.25))
    
    # Use the new allometry parameters from config.
    radius_coefficient = config.get("Radius_Coefficient", 0.5)
    length_coefficient = config.get("Length_Coefficient", 0.33)
    
    # New parameters for sympodial logic and height bounding.
    sympodial_chance = config.get("Sympodial_Chance", 0.3)  # My comment: added new param for sympodial branching.
    max_tree_height = config.get("Max_Tree_Height", total_length)  # My comment: bound tree height.
    side_branch_decay = config.get("Side_Branch_Decay", 1.2)  # My comment: added decay factor for side branch length.

    # Generate the tree using the updated gen_tree signature.
    tree = gen_tree(
        starting_radius=starting_radius,
        starting_direction=starting_direction,
        total_length=total_length,
        maximum_levels=maximum_levels,
        branching_angle_limit=branching_angle_limit,
        step_size=step_size,
        base_branching_probability=base_branching_probability,
        curvature_range=curvature_range,
        up_straightness=up_straightness,
        radius_coefficient=radius_coefficient,
        length_coefficient=length_coefficient,
        sympodial_chance=sympodial_chance,
        max_tree_height=max_tree_height,
        side_branch_decay=side_branch_decay
    )
    
    # Set up a 3D plot.
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("3D Tree Visualization")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    visualize_tree(tree, ax)
    
    # Adjust axis limits.
    def get_limits(node, limits):
        limits["x"].append(node.position.x)
        limits["y"].append(node.position.y)
        limits["z"].append(node.position.z)
        for child in node.children:
            get_limits(child, limits)
    limits = {"x": [], "y": [], "z": []}
    get_limits(tree, limits)
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
    
    #plt.show()
    plt.savefig("test_tree_node_structure.png", dpi=300)
    
    # Generate the mesh and export to OBJ.
    mesh = generate_tree_mesh(tree, radial_segments=16)
    export_mesh_to_obj("test_tree.obj", mesh.vertices, mesh.faces)
    print("Mesh exported to 'test_tree.obj'.")


def main():
    # You can choose any configuration: pine_config, oak_config, or broadleaf_config.
    config = broadleaf_config
    run_tree_simulation(config)


if __name__ == "__main__":
    main()
