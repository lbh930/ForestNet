import math
import matplotlib.pyplot as plt
import numpy as np
from common import vec3
from gen_mesh import Mesh, export_mesh_to_obj, generate_tree_mesh
from gen_nodes import TreeNode, gen_tree
from mpl_toolkits.mplot3d import Axes3D

# Pine tree configuration
pine_config = {
    "DBH": 1.0,                     # Diameter at breast height (in chosen units)
    "K": 12,                        # Allometric parameter K (typically higher for slender, tall pines)
    "Y": 0.67,                      # Allometric exponent Y (WBE theory predicts 2/3)
    # Height can be provided directly or computed from DBH if missing.
    "Height": 12 * (1.0 ** 0.67),     # Height = K * DBH^Y (for DBH = 1.0, Height = 12)
    
    "Radius_Coefficient": 0.5,        # Exponent for branch radius scaling (theoretical default ~0.5)
    "Length_Coefficient": 0.33,       # Exponent for branch length scaling (theoretical default ~0.33)
    
    "Branching_Angle_Range:": (15, 30),  # Pine branches tend to have a narrower angle (in degrees)
    "Step_Size": 1,                   # Step size for branch generation
    "Branching_Probability": 0.3,     # Lower branching probability for a sparser pine crown
    "Curvature_Range": (0.0, 0.2),      # Range of branch curvature
    "Up_Straightness": 0.8,           # High up-straightness for pine trees (more vertical growth)
    
    # Optional simulation parameters:
    "starting_radius": 0.2,           # Initial trunk radius
    "maximum_levels": 4,              # Maximum branching levels
}


# Oak tree configuration
oak_config = {
    "DBH": 1.0,                     # Diameter at breast height
    "K": 8,                         # Oak trees generally have a lower K (shorter relative height)
    "Y": 0.67,                      # Exponent for allometry remains around 2/3
    "Height": 8 * (1.0 ** 0.67),      # Height = K * DBH^Y (for DBH = 1.0, Height = 8)
    
    "Radius_Coefficient": 0.5,
    "Length_Coefficient": 0.33,
    
    "Branching_Angle_Range:": (30, 60),  # Oaks have a wider branching angle for an open crown
    "Step_Size": 1,
    "Branching_Probability": 0.5,     # Higher branching probability for denser crown formation
    "Curvature_Range": (0.0, 0.3),
    "Up_Straightness": 0.6,           # Moderately straight upward growth
    
    # Optional simulation parameters:
    "starting_radius": 0.3,           # Oaks often start with a thicker trunk
    "maximum_levels": 5,
}


# Generic broadleaf tree configuration (e.g., a maple tree)
broadleaf_config = {
    "DBH": 1.0,                     # Diameter at breast height
    "K": 9,                         # A mid-range K value for broadleaf trees
    "Y": 0.67,                      # Allometric exponent (2/3)
    "Height": 9 * (1.0 ** 0.67),      # Height = K * DBH^Y (for DBH = 1.0, Height = 9)
    
    "Radius_Coefficient": 0.5,
    "Length_Coefficient": 0.4,        # Slightly larger length exponent for a different branching style
    
    "Branching_Angle_Range:": (20, 60),  # A moderate branching angle range for generic broadleaf trees
    "Step_Size": 1,
    "Branching_Probability": 0.45,    # A balanced branching probability
    "Curvature_Range": (0.0, 0.25),
    "Up_Straightness": 0.65,          # Fairly straight upward growth
    
    # Optional simulation parameters:
    "starting_radius": 0.25,
    "maximum_levels": 6,
}


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
    ax.scatter(node.position.x, node.position.y, node.position.z, c='g', s=10)
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


def process_config(config):
    """
    Check and update the config parameters for Height and DBH.
    
    - If both Height and DBH are provided, use them directly.
    - If Height is missing but DBH is provided, compute Height using H = K * DBH^Y.
    """
    if "Height" not in config and "DBH" in config:
        if "K" not in config or "Y" not in config:
            raise ValueError("Missing parameters K and Y for height computation.")
        config["Height"] = config["K"] * (config["DBH"] ** config["Y"])
    elif "Height" in config and "DBH" not in config:
        if "K" not in config or "Y" not in config:
            raise ValueError("Missing parameters K and Y for DBH computation.")
        config["DBH"] = (config["Height"] / config["K"]) ** (1 / config["Y"])
    return config


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
    starting_radius = config.get("starting_radius", 0.2)
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
        length_coefficient=length_coefficient
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
    
    plt.show()
    
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
