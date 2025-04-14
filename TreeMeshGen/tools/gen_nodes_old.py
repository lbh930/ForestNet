import math
import random

from common import vec3


class TreeNode:
    '''
    A tree node
    Parameters:
    # bool is_root: Whether the node is the starting point of a branch
    # vec3 position: The position of the node
    # vec3 direction: The direction of the node
    # float radius: The radius of the node
    # int level: The level of the node
    # TreeNode[] children: The children of the node
    '''
    # A tree node
    # Parameters:
    ## bool is_root: Whether the node is the starting point of a branch
    ## vec3 position: The position of the node
    ## vec3 direction: The direction of the node
    ## float radius: The radius of the node
    ## int level: The level of the node
    ## TreeNode[] children: The children of the node
    
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


# Random unit vector (on sphere)
def _random_unit_vector():
    theta = random.random() * 2.0 * math.pi
    z = random.uniform(-1.0, 1.0)
    r = math.sqrt(1.0 - z * z)
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    return vec3(x, y, z).normalized()


# Random angle in radians from range (degrees)
def _random_angle_in_range(angle_limit: tuple):
    min_angle_deg, max_angle_deg = angle_limit
    angle_deg = random.uniform(min_angle_deg, max_angle_deg)
    return math.radians(angle_deg)


# Rotate direction randomly within angle range
# (Kept for reference from original code; may be superseded by step-based random curvature)
def _rotate_direction(base_dir: vec3, angle_degs: tuple) -> vec3:
    angle = _random_angle_in_range(angle_degs)
    if base_dir.length() < 1e-6:
        return _random_unit_vector()

    rv = _random_unit_vector()
    tries = 5
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
    Apply a small random offset in the plane perpendicular to 'base_dir'.
    The offset magnitude is chosen randomly in [curv_min, curv_max].
    """
    curv_min, curv_max = curvature_range
    if base_dir.length() < 1e-6:
        # If base_dir is nearly zero,  ret a random unit vector
        return _random_unit_vector()

    # Choose a random magnitude for curvature
    curvature = random.uniform(curv_min, curv_max)

    # Create a random perpendicular direction
    rand_vec = _random_unit_vector()
    parallel_comp = base_dir.normalized() * rand_vec.dot(base_dir.normalized())
    perp_vec = rand_vec - parallel_comp
    perp_len = perp_vec.length()

    if perp_len < 1e-6:
        # Fallback
        return base_dir.normalized()

    # Scale perpendicular vector by the curvature magnitude
    perp_dir = perp_vec.normalized() * curvature

    # Add to the base direction, then normalize
    new_dir = (base_dir + perp_dir).normalized()
    return new_dir

def _dynamic_branch_probability(depth: int, max_depth: int, base_probability: float) -> float:
    """
    Compute the branching probability based on the current depth and a baseline probability.
    The branching probability decreases as the depth increases, scaled by the baseline probability.
    :param depth: Current depth of the tree node.
    :param max_depth: Maximum depth of the tree.
    :param base_probability: Baseline branching probability at the root.
    :return: Adjusted branching probability for the current depth.
    """
    depth_factor = (1.0 - depth / max_depth)  # Linearly decrease probability with depth
    return max(0.0, base_probability * depth_factor)  # Ensure a minimum branching probability


def _generate_branch_stepwise(
    parent_node: TreeNode,
    total_length: float,
    step_size: float,
    max_depth: int,
    base_branching_probability: float,
    curvature_range: tuple,
    up_straightness: float,
    min_radius: float,
    branching_angle_limit: tuple
):
    """
    Generate the trunk/branch by stepping forward until reaching total_length or radius ~ 0.
    At each step:
    - Move forward by step_size
    - Radius decreases according to a formula that slows initial decay and accelerates near the end
    - Direction changes slightly (curvature)
    - Branching probability decreases as depth increases (dynamic branching probability)
    """

    # The initial radius for this branch
    start_radius = parent_node.radius
    distance_covered = 0.0

    # Keep stepping as long as we haven't reached total_length or shrunk below min_radius
    while distance_covered < total_length and parent_node.radius > min_radius:
        distance_covered += step_size
        if distance_covered > total_length:
            distance_covered = total_length

        # fraction of completed distance
        frac = distance_covered / total_length
        # Non-linear decay: slow at the beginning, faster near the end
        current_radius = start_radius * (1.0 - frac**3)
        if current_radius < min_radius:
            current_radius = min_radius

        # Step forward
        new_position = parent_node.position + parent_node.direction.normalized() * step_size
        
        # calculate children node direction
        up = vec3(0, 0, 1)
        rand_direction : vec3 = _random_curvature_dir(parent_node.direction, curvature_range)
        #print(f"up: {up}, type: {type(up)}")
        #print(f"rand_direction: {rand_direction}, type: {type(rand_direction)}")

        new_direction = (up*up_straightness + rand_direction*(1.0-up_straightness)).normalized()

        # Create child node for the next segment
        child_node = TreeNode(
            is_root=False,
            position=new_position,
            direction=new_direction,
            radius=current_radius,
            level=parent_node.level
        )
        parent_node.add_child(child_node)
        parent_node = child_node  # continue growing from this child in the next iteration

        # Compute branching probability dynamically based on the depth
        branching_probability = _dynamic_branch_probability(parent_node.level, max_depth, base_branching_probability)
        #print (f"branching_probability: {branching_probability}")

        # Attempt to branch
        if random.random() < branching_probability and current_radius > min_radius:
            # Decide how many branches to split into
            branch_count = random.randint(1, 4)

            # Each child's radius is parent.radius / sqrt(branch_count)
            child_r = current_radius / math.sqrt(branch_count)

            # For each branch, create a new node at the same position,
            # but with a different direction
            for _ in range(branch_count):
                branch_direction = _rotate_direction(parent_node.direction, branching_angle_limit)
                branch_node = TreeNode(
                    is_root=False,
                    position=parent_node.position,
                    direction=branch_direction,
                    radius=child_r * random.uniform(0.6, 0.8),  # Slightly randomize radius
                    level=parent_node.level + 1
                )
                parent_node.add_child(branch_node)

                # Recursively generate the sub-branch
                _generate_branch_stepwise(
                    branch_node,
                    total_length=total_length * random.uniform(0.4, 0.6),  # Reduce length for sub-branches
                    step_size=step_size,
                    max_depth=max_depth,
                    base_branching_probability=base_branching_probability * 0.8,  # Reduce probability for sub-branches
                    curvature_range=curvature_range,
                    up_straightness=up_straightness*random.uniform(0.6, 0.8),
                    min_radius=min_radius,
                    branching_angle_limit=branching_angle_limit
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
    up_straightness: float = 0.0
):
    """
    Generate tree structure
    :param starting_radius: Radius at the root
    :param starting_direction: Initial direction of growth
    :param total_length: Approximate total length of the trunk
    :param maximum_levels: Maximum depth of the tree
    :param branching_angle_limit: Angle range for branching direction
    :param step_size: Each step forward in the trunk or branch
    :param base_branching_probability: Baseline branching probability at the root
    :param curvature_range: (min_curvature, max_curvature) to randomly bend direction each step
    :param up_straightness: control the tendency the tree grows upwards
    :return: Root TreeNode of the generated tree
    """

    if maximum_levels < 1:
        raise ValueError("maximum_levels must be >= 1")

    # For demonstration, we'll define a 'min_radius' fraction of the starting radius
    min_radius = 0.01 * starting_radius

    # Create the root
    root = TreeNode(
        is_root=True,
        position=vec3(0, 0, 0),
        direction=starting_direction.normalized(),
        radius=starting_radius,
        level=0
    )

    # Stepwise generation of the main trunk
    _generate_branch_stepwise(
        root,
        total_length=total_length,
        step_size=step_size,
        max_depth=maximum_levels,
        base_branching_probability=base_branching_probability,
        curvature_range=curvature_range,
        min_radius=min_radius,
        branching_angle_limit=branching_angle_limit,
        up_straightness=up_straightness
    )

    return root

