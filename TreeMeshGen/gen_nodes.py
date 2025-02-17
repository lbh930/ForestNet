import math
import random

from common import vec3

class TreeNode:
    '''
    A tree node.
    Parameters:
    - bool is_root:    Whether the node is the starting point (root) of a branch or trunk
    - vec3 position:   The position of the node in 3D space
    - vec3 direction:  The growth direction of this node
    - float radius:    The radius (thickness) of the trunk/branch at this node
    - int level:       The hierarchical level (depth) of the node
    - children:        A list of child TreeNode instances
    '''

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
    angle_limit is a tuple (min_deg, max_deg).
    """
    min_angle_deg, max_angle_deg = angle_limit
    angle_deg = random.uniform(min_angle_deg, max_angle_deg)
    return math.radians(angle_deg)


def _rotate_direction(base_dir: vec3, angle_degs: tuple) -> vec3:
    """
    Rotate 'base_dir' by a random angle within 'angle_degs',
    around a random axis approximately perpendicular to 'base_dir'.
    """
    angle = _random_angle_in_range(angle_degs)
    if base_dir.length() < 1e-6:
        return _random_unit_vector()

    rv = _random_unit_vector()
    tries = 5
    # Ensure that the random vector is not too parallel to base_dir
    while abs(rv.dot(base_dir.normalized())) > 0.95 and tries > 0:
        rv = _random_unit_vector()
        tries -= 1

    ax = base_dir.cross(rv)
    ax_len = ax.length()
    if ax_len < 1e-6:
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
    curvature_range is (curv_min, curv_max).
    """
    curv_min, curv_max = curvature_range
    if base_dir.length() < 1e-6:
        return _random_unit_vector()

    curvature = random.uniform(curv_min, curv_max)

    rand_vec = _random_unit_vector()
    parallel_comp = base_dir.normalized() * rand_vec.dot(base_dir.normalized())
    perp_vec = rand_vec - parallel_comp
    perp_len = perp_vec.length()

    if perp_len < 1e-6:
        return base_dir.normalized()

    perp_dir = perp_vec.normalized() * curvature
    new_dir = (base_dir + perp_dir).normalized()
    return new_dir


def _dynamic_branch_probability(depth: int, max_depth: int, base_probability: float) -> float:
    """
    Compute the branching probability based on current depth and a baseline probability.
    Probability decreases linearly as depth -> max_depth.
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

    # Below are newly introduced hyperparameters for allometry:

    area_fraction: float,
    #'''
    #area_fraction: float
    #------------------------------------------
    #Fraction of the parent's cross-sectional area that is distributed among child branches.
    #For example, an area_fraction = 0.8 means the total cross-sectional area of children
    #will be about 80% of the parent's area. This simulates the hydraulic or structural
    #branching constraints in real trees.
    #'''

    end_radius_ratio: float,
    #"""
    #end_radius_ratio: float
    #------------------------------------------
    #Controls the tapering (or narrowing) of the main trunk/branch from base to tip.
    #If set to 0.2, the radius at the branch end can be about 20% of the initial radius,
    #ensuring realistic trunk/branch taper.
    #"""

    radius_decay_exp: float,
    #"""
    #radius_decay_exp: float
    #------------------------------------------
    #The exponent in the non-linear decay function controlling how rapidly the radius
    #decreases along the length. A higher exponent means the radius stays thicker for
    #a longer portion of the branch and then tapers quickly near the end.
    #"""
):
    """
    Generate trunk or branch segments in a stepwise manner, applying allometric rules
    for diameter taper and branching area distribution.
    """

    start_radius = parent_node.radius
    distance_covered = 0.0

    # The final radius if the entire length is grown:
    final_radius = start_radius * end_radius_ratio

    # Continue stepping along the branch/trunk
    while distance_covered < total_length and parent_node.radius > min_radius:
        distance_covered += step_size
        if distance_covered > total_length:
            distance_covered = total_length

        # fraction of completed distance
        frac = distance_covered / total_length

        # Non-linear decay for radius:
        # radius transitions from start_radius -> final_radius using an exponent
        decay_factor = (1.0 - frac**radius_decay_exp)
        current_radius = final_radius + (start_radius - final_radius) * decay_factor
        if current_radius < min_radius:
            current_radius = min_radius

        # Move forward along the direction
        new_position = parent_node.position + parent_node.direction.normalized() * step_size

        # Apply curvature + up_straightness
        up = vec3(0, 0, 1)
        curved_dir = _random_curvature_dir(parent_node.direction, curvature_range)
        new_direction = (up * up_straightness + curved_dir * (1.0 - up_straightness)).normalized()

        # Create a new child node
        child_node = TreeNode(
            is_root=False,
            position=new_position,
            direction=new_direction,
            radius=current_radius,
            level=parent_node.level
        )
        parent_node.add_child(child_node)
        parent_node = child_node  # Continue the main branch from the newly created node

        # Dynamically compute branching probability
        branching_probability = _dynamic_branch_probability(parent_node.level, max_depth, base_branching_probability)

        # Attempt to branch out
        if random.random() < branching_probability and current_radius > min_radius:
            # Decide how many side branches to generate
            branch_count = random.randint(1, 3)

            # Calculate parent's cross-sectional area
            parent_area = math.pi * (current_radius ** 2)
            # The total area allocated to children
            allocated_area = parent_area * area_fraction
            # Each child gets an equal share of the allocated area
            if branch_count > 0:
                child_area_each = allocated_area / branch_count
            else:
                child_area_each = allocated_area

            for _ in range(branch_count):
                # Slight random variation to avoid identical child radii
                local_child_area = child_area_each * random.uniform(0.8, 1.2)
                if local_child_area < (math.pi * min_radius**2):
                    local_child_area = math.pi * (min_radius**2)

                # Convert cross-sectional area -> radius
                child_r = math.sqrt(local_child_area / math.pi)

                # Rotate the direction within the specified branching angle
                branch_dir = _rotate_direction(parent_node.direction, branching_angle_limit)

                # Create the branch node at the same position as the parent's node
                branch_node = TreeNode(
                    is_root=False,
                    position=parent_node.position,
                    direction=branch_dir,
                    radius=child_r,
                    level=parent_node.level + 1
                )
                parent_node.add_child(branch_node)

                # Reduce total_length for side branches
                sub_branch_length = total_length * random.uniform(0.4, 0.6)

                # Recursively generate the sub-branch using the same allometric parameters
                _generate_branch_stepwise_allometric(
                    parent_node=branch_node,
                    total_length=sub_branch_length,
                    step_size=step_size,
                    max_depth=max_depth,
                    base_branching_probability=base_branching_probability * 0.8,  # Lower probability in child branches
                    curvature_range=curvature_range,
                    up_straightness=up_straightness * random.uniform(0.6, 0.9),
                    min_radius=min_radius,
                    branching_angle_limit=branching_angle_limit,

                    area_fraction=area_fraction,
                    end_radius_ratio=end_radius_ratio,
                    radius_decay_exp=radius_decay_exp
                )


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

    # Below are our allometric hyperparameters with docstrings describing their meaning.

    area_fraction: float = 0.8,
    #"""
    #area_fraction (default=0.8)
    #-----------------------------------------------------------
    #Fraction of cross-sectional area that children can inherit
    #from the parent branch at each branching point. 1.0 means
    #'area conservation' (all parent's area is distributed among
    #the children), while <1.0 means partial distribution.
    #""",

    end_radius_ratio: float = 0.2,
    #"""
    #end_radius_ratio (default=0.2)
    #-----------------------------------------------------------
    #Defines how much thinner the end of a trunk or major branch
    #is compared to its base. If the trunk starts with radius R,
    #then at full length it ends with R * end_radius_ratio.
    #""",

    radius_decay_exp: float = 1.5
    #"""
    #radius_decay_exp (default=1.5)
    #-----------------------------------------------------------
    #The exponent that controls how quickly the radius tapers
    #from the base to the tip of the trunk/branch.
    #A larger exponent = radius remains thicker for longer,
    #then rapidly decreases near the end.
    #"""
):
    """
    Generate a tree structure with node-based representation, using allometric rules.

    :param starting_radius:           Initial radius at the root or trunk base
    :param starting_direction:        Initial direction of growth (e.g., vec3(0,0,1) for vertical)
    :param total_length:              Approximate total length of the main trunk
    :param maximum_levels:            Maximum depth of recursion / branching
    :param branching_angle_limit:     (min_deg, max_deg) for random branching angles
    :param step_size:                 The distance between generated nodes along a branch
    :param base_branching_probability:Initial branching probability at the root (decreases with depth)
    :param curvature_range:           (curv_min, curv_max) controlling small random bending each step
    :param up_straightness:           Factor [0..1] to bias branches toward the vertical 'up' direction
    :param area_fraction:             See docstring above
    :param end_radius_ratio:          See docstring above
    :param radius_decay_exp:          See docstring above

    :return: Root TreeNode of the entire generated tree.
    """

    if maximum_levels < 1:
        raise ValueError("maximum_levels must be >= 1")

    # Minimum radius to avoid infinitely thin branches
    min_radius = 0.01 * starting_radius

    # Create the root node
    root = TreeNode(
        is_root=True,
        position=vec3(0, 0, 0),
        direction=starting_direction.normalized(),
        radius=starting_radius,
        level=0
    )

    # Generate the main trunk (and subsequent branches) in a stepwise manner
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

        # Pass along allometric hyperparameters
        area_fraction=area_fraction,
        end_radius_ratio=end_radius_ratio,
        radius_decay_exp=radius_decay_exp
    )

    return root
