import math
import matplotlib.pyplot as plt
import numpy as np
from common import vec3
from gen_mesh import Mesh, export_mesh_to_obj, generate_tree_mesh

# 注意：这里假设你已经将新的Allometry版本的代码依然命名为 gen_nodes.py，并且保留了 gen_tree() 接口
from gen_nodes import TreeNode, gen_tree

from mpl_toolkits.mplot3d import Axes3D


def _draw_circle_at_node(ax, node: TreeNode, circle_points=32):
    """
    Draw a circle representing the node's cross section, perpendicular to its direction vector.
    :param ax: Matplotlib 3D axis.
    :param node: TreeNode that contains position, direction, and radius.
    :param circle_points: Number of points used to approximate the circle.
    """
    # If the direction vector is too small, assign a default direction
    d = node.direction
    if d.length() < 1e-6:
        d = vec3(0, 0, 1)  
    d = d.normalized()

    # Choose an arbitrary vector `up` that is not parallel to `d`
    up = vec3(0, 0, 1)
    if abs(d.dot(up)) > 0.99:
        # If too parallel, choose another vector
        up = vec3(1, 0, 0)

    # Construct two orthogonal unit vectors e1, e2 that are perpendicular to d
    e1 = d.cross(up)
    if e1.length() < 1e-6:
        # If the cross product is too small, switch to another `up` vector
        up = vec3(0, 1, 0)
        e1 = d.cross(up)
    e1 = e1.normalized()
    e2 = d.cross(e1).normalized()

    # Generate points along the circle
    theta = np.linspace(0, 2 * np.pi, circle_points)
    X = []
    Y = []
    Z = []

    r = node.radius
    center = node.position
    for t in theta:
        # Parametric form: center + r*cos(t)*e1 + r*sin(t)*e2
        pos = center + e1 * (r * math.cos(t)) + e2 * (r * math.sin(t))
        X.append(pos.x)
        Y.append(pos.y)
        Z.append(pos.z)

    # Close the circle
    X.append(X[0])
    Y.append(Y[0])
    Z.append(Z[0])

    # Plot the circle in 3D
    ax.plot(X, Y, Z, color='r', linewidth=0.8)


def visualize_tree(node: TreeNode, ax, parent_position=None):
    """
    Recursively visualize the tree structure in 3D.
    Also draw a circle representing the node's radius, perpendicular to its direction.
    :param node: Current TreeNode to draw.
    :param ax: Matplotlib 3D axis.
    :param parent_position: Position of the parent node (for connecting lines).
    """
    # Plot the current node as a point
    ax.scatter(node.position.x, node.position.y, node.position.z, c='g', s=10)

    # If there is a parent, draw a line connecting to it
    if parent_position is not None:
        ax.plot(
            [parent_position.x, node.position.x],
            [parent_position.y, node.position.y],
            [parent_position.z, node.position.z],
            'k-',
            linewidth=0.5
        )

    # Draw a circle representing the radius at the current node
    _draw_circle_at_node(ax, node)

    # Recursively visualize children
    for child in node.children:
        visualize_tree(child, ax, node.position)


def main():
    """Generate and visualize a tree."""
    # Define tree parameters
    starting_radius = 0.2
    starting_direction = vec3(0, 0, 1)  # Grow along the Z-axis
    total_length = 8
    maximum_levels = 4
    branching_angle_limit = (15, 45)  # Branching angle range in degrees
    up_straightness = 0.6

    # Additional parameters for step-based generation
    step_size = 1
    base_branching_probability = 0.4
    curvature_range = (0.0, 0.25)

    # ----- NEW: Allometry-related hyperparameters -----
    # You can adjust these to see how the tree shape changes.
    area_fraction = 0.8       # fraction of parent cross-sectional area allocated to child branches
    end_radius_ratio = 0.3    # ratio of trunk end radius to start radius
    radius_decay_exp = 1.5    # exponent controlling how quickly radius tapers

    # Generate the tree (calling the updated gen_tree with allometric parameters)
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

        # The new allometric hyperparameters:
        area_fraction=area_fraction,
        end_radius_ratio=end_radius_ratio,
        radius_decay_exp=radius_decay_exp
    )

    # Set up a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Customize the 3D plot appearance
    ax.set_title("3D Tree Visualization")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Visualize the tree
    visualize_tree(tree, ax)

    # Adjust the aspect ratio
    # Gather all the points in the tree to calculate limits
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

    # Center the axes
    x_mid = (x_max + x_min) / 2
    y_mid = (y_max + y_min) / 2
    z_mid = (z_max + z_min) / 2

    ax.set_xlim(x_mid - max_range / 2, x_mid + max_range / 2)
    ax.set_ylim(y_mid - max_range / 2, y_mid + max_range / 2)
    ax.set_zlim(z_mid - max_range / 2, z_mid + max_range / 2)

    # Show the plot
    plt.show()
    
    # Generate the mesh
    mesh = generate_tree_mesh(tree, radial_segments=16)
    export_mesh_to_obj("test_tree.obj", mesh.vertices, mesh.faces)
    print("Mesh exported to 'tree_mesh.obj'.")


if __name__ == "__main__":
    main()
