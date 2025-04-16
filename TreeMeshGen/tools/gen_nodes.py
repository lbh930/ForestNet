import math
import random
import matplotlib.pyplot as plt  # 3D visualization library

from tools.common import vec3

class TreeNode:
    """
    A tree node.
    
    Parameters:
      - is_root (bool): Whether the node is the starting point (root) of a branch or trunk.
      - position (vec3): The position of the node in 3D space.
      - direction (vec3): The growth direction of this node.
      - radius (float): The radius (thickness) of the trunk/branch at this node.
      - level (int): The hierarchical level (depth) of the node.
      - children: A list of child TreeNode instances.
    """
    def __init__(self, is_root, position, direction, radius, level):
        self.is_root = is_root
        self.position = position
        self.direction = direction
        self.radius = radius
        self.level = level
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def __str__(self):
        return f"TreeNode({self.is_root}, {self.position}, {self.direction}, {self.radius})"

    def __repr__(self):
        return f"TreeNode({self.is_root}, {self.position}, {self.direction}, {self.radius})"


def _random_unit_vector():
    """
    Generate a random unit vector uniformly distributed on the sphere.
    """
    theta = random.random() * 2.0 * math.pi
    z = random.uniform(-1.0, 1.0)
    r = math.sqrt(1.0 - z * z)
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    return vec3(x, y, z).normalized()


def _random_angle_in_range(angle_limit: tuple):
    """
    Convert a random angle (in degrees) within 'angle_limit' to radians.
    
    :param angle_limit: A tuple (min_deg, max_deg)
    :return: Random angle in radians.
    """
    min_angle_deg, max_angle_deg = angle_limit
    angle_deg = random.uniform(min_angle_deg, max_angle_deg)
    return math.radians(angle_deg)


def _rotate_direction(base_dir: vec3, angle_degs: tuple) -> vec3:
    """
    Rotate 'base_dir' by a random angle within 'angle_degs', 
    around a random axis approximately perpendicular to 'base_dir'.
    
    :param base_dir: The original direction vector.
    :param angle_degs: Tuple containing (min_deg, max_deg) for the rotation angle.
    :return: A new rotated unit vector.
    """
    angle = _random_angle_in_range(angle_degs)
    if base_dir.length() < 1e-6:
        return _random_unit_vector()

    rv = _random_unit_vector()
    tries = 5
    while abs(rv.dot(base_dir.normalized())) > 0.95 and tries > 0:
        rv = _random_unit_vector()
        tries -= 1

    ax = base_dir.cross(rv)
    if ax.length() < 1e-6:
        return _random_unit_vector()

    ax = ax.normalized()
    v = base_dir.normalized()
    cos_theta = math.cos(angle)
    sin_theta = math.sin(angle)
    kdotv = ax.dot(v)

    part1 = v * cos_theta
    part2 = ax.cross(v) * sin_theta
    part3 = ax * (kdotv * (1.0 - cos_theta))
    return (part1 + part2 + part3).normalized()


def _random_curvature_dir(base_dir: vec3, curvature_range: tuple) -> vec3:
    """
    Add a small random perpendicular offset to 'base_dir' to simulate curvature.
    
    :param curvature_range: Tuple (curv_min, curv_max) controlling the curvature.
    :return: A new direction vector with curvature applied.
    """
    curv_min, curv_max = curvature_range
    if base_dir.length() < 1e-6:
        return _random_unit_vector()

    curvature = random.uniform(curv_min, curv_max)
    rand_vec = _random_unit_vector()
    parallel_comp = base_dir.normalized() * rand_vec.dot(base_dir.normalized())
    perp_vec = rand_vec - parallel_comp
    if perp_vec.length() < 1e-6:
        return base_dir.normalized()
    perp_dir = perp_vec.normalized() * curvature
    new_dir = (base_dir + perp_dir).normalized()
    return new_dir


def _dynamic_branch_probability(depth: int, max_depth: int, base_probability: float) -> float:
    """
    Compute the branching probability based on the current depth and a baseline probability.
    
    :param depth: Current hierarchical depth.
    :param max_depth: Maximum allowed depth.
    :param base_probability: The base branching probability.
    :return: Adjusted branching probability.
    """
    depth_factor = (1.0 - depth / max_depth)
    return max(0.0, base_probability * depth_factor)


def _generate_branch_stepwise_allometric(
    parent_node: TreeNode,
    total_length: float,
    step_size: float,
    max_depth: int,
    base_branching_probability: float,
    curvature_range: tuple,
    up_straightness: float,
    min_radius: float,
    branching_angle_limit: tuple,
    radius_coefficient: float,   # Exponent for radius scaling (default 0.5)
    length_coefficient: float,   # Exponent for branch length scaling (default 0.33)
    sympodial_chance: float,      # New param: chance for sympodial switch
    max_tree_height: float,       # New param: maximum allowed tree height
    side_branch_decay: float      # New param: side branch decay factor
):
    """
    Generate trunk or branch segments in a stepwise manner while applying allometric rules.
    
    Original comments retained.
    # Added sympodial_chance to enable sympodial-like splitting.
    # New max_tree_height bounds the tree's vertical growth.
    # Added side_branch_decay to accelerate side branch length attenuation.
    
    Changes:
      - For the main trunk, we estimate a final radius using a power law based on max_depth.
      - For side branches, compute:
            child branch radius = parent's radius × ( (branch_count+1)^(–radius_coefficient) )
            side branch length = total_length × ( (branch_count+1)^(–length_coefficient) ) / side_branch_decay
        where branch_count is the number of side branches generated.
      - Implement sympodial switch: sometimes the main trunk is replaced by one of its side branches.
      - Growth stops if maximum height is reached.
    """
    start_radius = parent_node.radius
    distance_covered = 0.0

    # Estimate final radius at the tip of the branch using the power law.
    final_radius = start_radius * (max_depth ** (-radius_coefficient))

    side_branches = []  # collect side branches for possible sympodial switch

    # While loop also checks that tree height is bounded.
    while distance_covered < total_length and parent_node.radius > min_radius and parent_node.position.z < max_tree_height:
        distance_covered += step_size
        if distance_covered > total_length:
            distance_covered = total_length

        # Compute fraction of the total branch length covered.
        frac = distance_covered / total_length
        # Linearly interpolate between the start and estimated final radius.
        current_radius = start_radius * (1 - frac) + final_radius * frac
        if current_radius < min_radius:
            current_radius = min_radius

        # Move forward along the current direction.
        new_position = parent_node.position + parent_node.direction.normalized() * step_size
        
        # Stop branch if new_position.z exceeds max_tree_height.
        if new_position.z > max_tree_height:
            break  # stop growth for this branch

        # Apply curvature and upward bias.
        up = vec3(0, 0, 1)
        curved_dir = _random_curvature_dir(parent_node.direction, curvature_range)
        new_direction = (up * up_straightness + curved_dir * (1.0 - up_straightness)).normalized()

        # Create a new node along the main branch.
        child_node = TreeNode(
            is_root=False,
            position=new_position,
            direction=new_direction,
            radius=current_radius,
            level=parent_node.level
        )
        parent_node.add_child(child_node)
        # Continue along the main branch by default.
        main_continuation = child_node

        # Compute dynamic branching probability based on current depth.
        branching_probability = _dynamic_branch_probability(parent_node.level, max_depth, base_branching_probability)

        # Attempt to generate side branches.
        if random.random() < branching_probability and current_radius > min_radius:
            branch_count = random.randint(1, 3)  # Randomly choose the number of side branches.

            for _ in range(branch_count):
                # Calculate child branch radius using power-law scaling:
                # child_radius = parent's radius × ((branch_count+1)^(–radius_coefficient))
                child_r = parent_node.radius * ((branch_count + 1) ** (-radius_coefficient))
                child_r *= random.uniform(0.9, 1.1)  # slight random variation.
                if child_r < min_radius:
                    child_r = min_radius

                # Rotate the direction within the specified branching angle.
                branch_dir = _rotate_direction(parent_node.direction, branching_angle_limit)

                # Create the side branch node.
                branch_node = TreeNode(
                    is_root=False,
                    position=parent_node.position + parent_node.direction.normalized() * random.uniform(0, step_size),
                    direction=branch_dir,
                    radius=child_r,
                    level=parent_node.level + 1
                )
                parent_node.add_child(branch_node)
                side_branches.append(branch_node)

                # Compute side branch length with extra decay factor:
                sub_branch_length = total_length * ((branch_count + 1) ** (-length_coefficient)) / side_branch_decay

                # Recursively generate the side branch.
                _generate_branch_stepwise_allometric(
                    parent_node=branch_node,
                    total_length=sub_branch_length,
                    step_size=step_size,
                    max_depth=max_depth,
                    base_branching_probability=base_branching_probability * 0.8,  # reduced probability for side branches
                    curvature_range=curvature_range,
                    up_straightness=up_straightness * random.uniform(0.6, 0.9),
                    min_radius=min_radius,
                    branching_angle_limit=branching_angle_limit,
                    radius_coefficient=radius_coefficient,
                    length_coefficient=length_coefficient,
                    sympodial_chance=sympodial_chance,
                    max_tree_height=max_tree_height,
                    side_branch_decay=side_branch_decay
                )
        
        # Attempt sympodial switch: if any side branch exists and chance condition is met,
        # switch main trunk to one of the side branches.
        if side_branches and random.random() < sympodial_chance:
            # New comment: switch main branch to a randomly chosen side branch.
            main_continuation = random.choice(side_branches)
            break  # stop current trunk; sympodial branch takes over

        # Continue along the main branch normally.
        parent_node = main_continuation


def gen_tree(
    starting_radius: float,
    starting_direction: vec3,
    total_length: float,
    maximum_levels: int,
    branching_angle_limit: tuple,
    step_size: float = 0.5,
    base_branching_probability: float = 0.5,
    curvature_range: tuple = (0.0, 0.2),
    up_straightness: float = 0.0,
    radius_coefficient: float = 0.5,   # Exponent for branch radius scaling (default 0.5)
    length_coefficient: float = 0.33,  # Exponent for branch length scaling (default ~0.33)
    sympodial_chance: float = 0.3,      # New param: chance for sympodial splitting
    max_tree_height: float = 30.0,       # New param: maximum tree height
    side_branch_decay: float = 1.2     # New param: side branch decay factor
):
    """
    Generate a tree structure using node-based representation with allometric rules.
    
    Parameters:
      - starting_radius: Initial radius at the root.
      - starting_direction: Initial growth direction (e.g., vec3(0,0,1) for vertical growth).
      - total_length: Approximate total length of the main trunk.
      - maximum_levels: Maximum depth (number of branching levels).
      - branching_angle_limit: Tuple (min_deg, max_deg) for random branching angles.
      - step_size: Distance between generated nodes along a branch.
      - base_branching_probability: Initial branching probability (decreases with depth).
      - curvature_range: Tuple (curv_min, curv_max) for random bending per step.
      - up_straightness: Factor [0,1] biasing growth toward the vertical direction.
      - radius_coefficient: Exponent for branch radius scaling (default 0.5).
      - length_coefficient: Exponent for branch length scaling (default ~0.33).
      - sympodial_chance: New parameter for sympodial branching (chance to switch main branch).
      - max_tree_height: New parameter to bound overall tree height.
      - side_branch_decay: New parameter to accelerate side branch length attenuation.
    
    Returns:
      - The root TreeNode of the generated tree.
    """
    if maximum_levels < 1:
        raise ValueError("maximum_levels must be >= 1")

    # Set a minimum radius to prevent branches from becoming infinitely thin.
    min_radius = 0.01 * starting_radius

    # Create the root node at the origin.
    root = TreeNode(
        is_root=True,
        position=vec3(0, 0, 0),
        direction=starting_direction.normalized(),
        radius=starting_radius,
        level=0
    )

    # Generate the main trunk and subsequent branches.
    _generate_branch_stepwise_allometric(
        parent_node=root,
        total_length=total_length,
        step_size=step_size,
        max_depth=maximum_levels,
        base_branching_probability=base_branching_probability,
        curvature_range=curvature_range,
        up_straightness=up_straightness,
        min_radius=min_radius,
        branching_angle_limit=branching_angle_limit,
        radius_coefficient=radius_coefficient,
        length_coefficient=length_coefficient,
        sympodial_chance=sympodial_chance,
        max_tree_height=max_tree_height,
        side_branch_decay=side_branch_decay
    )

    return root
