
# Pine tree configuration
pine_config = {
    "K": 12,                        # Allometric parameter K (typically higher for slender, tall pines)
    "Y": 0.67,                      # Allometric exponent Y (WBE theory predicts 2/3)
    "Height": 12,                   # Height = K * DBH^Y (for DBH = 1.0, Height = 12)
    
    "Radius_Coefficient": 0.5,      # Exponent for branch radius scaling
    "Length_Coefficient": 0.33,     # Exponent for branch length scaling
    
    "Branching_Angle_Range:": (15, 30),  # Pine branches narrow (in degrees)
    "Step_Size": 1,                 # Step size for branch generation
    "Branching_Probability": 0.3,   # Sparser pine crown
    "Curvature_Range": (0.0, 0.2),    # Range of branch curvature
    "Up_Straightness": 0.8,         # High up-straightness for vertical growth
    
    # Optional parameters:
    #"DBH": 0.3,                  # Diameter at breast height
    "maximum_levels": 4,          # Maximum branching levels

    # New parameters for sympodial simulation.
    "Sympodial_Chance": 0,        # Increase chance for sympodial switching in pine
    "Max_Tree_Height": 12,         # Maximum tree height equals trunk target height.
    "Side_Branch_Decay": 1.2       # Side branches decay factor.
}


# Oak tree configuration
oak_config = {
    "K": 8,                         # Oak trees generally have a lower K (shorter relative height)
    "Y": 0.67,                      # Exponent for allometry remains around 2/3
    "Height": 8,                    # Height = K * DBH^Y (for DBH = 1.0, Height = 8)
    
    "Radius_Coefficient": 0.5,
    "Length_Coefficient": 0.33,
    
    "Branching_Angle_Range:": (30, 60),  # Oaks have a wider branching angle for an open crown
    "Step_Size": 1,
    "Branching_Probability": 0.5,   # Denser crown formation
    "Curvature_Range": (0.0, 0.3),
    "Up_Straightness": 0.6,         # Moderately straight upward growth
    
    # Optional parameters:
    #"DBH": 0.3,
    "maximum_levels": 5,

    # New parameters for sympodial simulation.
    "Sympodial_Chance": 0.3,        # Lower sympodial chance for oak
    "Max_Tree_Height": 8,          # Tree height is bounded by target height.
    "Side_Branch_Decay": 1.3       # Slightly stronger decay for side branches.
}


# Generic broadleaf tree configuration
broadleaf_config = {
    "K": 30,                        # Mid-range K value for broadleaf trees
    "Y": 0.67,                      # Allometric exponent (2/3)
    "Height": 25,                   # Height = K * DBH^Y (for DBH = 1.0, Height = 25)
    
    "Radius_Coefficient": 0.5,
    "Length_Coefficient": 0.5,      # Slightly larger length exponent for different branching style
    
    "Branching_Angle_Range:": (30, 90),  # Broadleaf trees have wider branch angles
    "Step_Size": 1,
    "Branching_Probability": 0.4,   # Lower branching probability for open crown
    "Curvature_Range": (0.0, 0.3),
    "Up_Straightness": 0.25,        # Fairly weak vertical bias
    
    # Optional parameters:
    #"DBH": 0.5,
    "maximum_levels": 7,

    # New parameters for sympodial simulation.
    "Sympodial_Chance": 0.2,       # Moderate chance for sympodial switch in broadleaf
    "Max_Tree_Height": 25,         # Bound height to target Height
    "Side_Branch_Decay": 1.2       # Slightly stronger decay for side branches.
}