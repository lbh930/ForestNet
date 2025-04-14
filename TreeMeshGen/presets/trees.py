
# Pine tree configuration
pine_config = {
    "K": 12,                        # Allometric parameter K (typically higher for slender, tall pines)
    "Y": 0.67,                      # Allometric exponent Y (WBE theory predicts 2/3)
    # Height can be provided directly or computed from DBH if missing.
    "Height": 12,     # Height = K * DBH^Y (for DBH = 1.0, Height = 12)
    
    "Radius_Coefficient": 0.5,        # Exponent for branch radius scaling (theoretical default ~0.5)
    "Length_Coefficient": 0.33,       # Exponent for branch length scaling (theoretical default ~0.33)
    
    "Branching_Angle_Range:": (15, 30),  # Pine branches tend to have a narrower angle (in degrees)
    "Step_Size": 1,                   # Step size for branch generation
    "Branching_Probability": 0.3,     # Lower branching probability for a sparser pine crown
    "Curvature_Range": (0.0, 0.2),      # Range of branch curvature
    "Up_Straightness": 0.8,           # High up-straightness for pine trees (more vertical growth)
    
    # Optional parameters:
    #"DBH": 0.3,                     # Diameter at breast height (in chosen units)
    "maximum_levels": 4,              # Maximum branching levels
}


# Oak tree configuration
oak_config = {
    "K": 8,                         # Oak trees generally have a lower K (shorter relative height)
    "Y": 0.67,                      # Exponent for allometry remains around 2/3
    "Height": 8,      # Height = K * DBH^Y (for DBH = 1.0, Height = 8)
    
    "Radius_Coefficient": 0.5,
    "Length_Coefficient": 0.33,
    
    "Branching_Angle_Range:": (30, 60),  # Oaks have a wider branching angle for an open crown
    "Step_Size": 1,
    "Branching_Probability": 0.5,     # Higher branching probability for denser crown formation
    "Curvature_Range": (0.0, 0.3),
    "Up_Straightness": 0.6,           # Moderately straight upward growth
    
    # Optional parameters:
    #"DBH": 0.3,                     # Diameter at breast height (in chosen units)
    "maximum_levels": 5,
}


# Generic broadleaf tree configuration
broadleaf_config = {
    "K": 30,                         # A mid-range K value for broadleaf trees
    "Y": 0.67,                      # Allometric exponent (2/3)
    "Height": 25,      # Height = K * DBH^Y (for DBH = 1.0, Height = 9)
    
    "Radius_Coefficient": 0.5,
    "Length_Coefficient": 0.5,        # Slightly larger length exponent for a different branching style
    
    "Branching_Angle_Range:": (45, 90),  # A moderate branching angle range for generic broadleaf trees
    "Step_Size": 1.5,
    "Branching_Probability": 0.3,    # A balanced branching probability
    "Curvature_Range": (0.0, 0.3),
    "Up_Straightness": 0.1,          # Fairly straight upward growth
    
    # Optional  parameters:
    #"DBH": 0.5,                     # Diameter at breast height (in chosen units)
    "maximum_levels": 5,
}