from presets.trees import pine_config, broadleaf_config, oak_config
broadleaf_forest_config = {
    "tree_config": broadleaf_config,
    "width": 50, #in meters, range on x axis
    "length": 50, #in meters, range on y axis
    "average_height": 8,
    "variance_height": 1, #height randomized based on normal distribution
    "tree_count": 5,
    "minimal_distance": 5, # minimal distance between distinct trees.
}